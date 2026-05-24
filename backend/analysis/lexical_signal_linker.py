"""Algorithmic lexical linker for stock-news relevance.

This module replaces embedding retrieval (e.g., bge-m3) with a lightweight
pipeline:
1) Alias dictionary expansion
2) BM25 relevance
3) TF-IDF character n-gram similarity
4) SimHash near-duplicate clustering / dedup
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_ALIAS_SUFFIXES = (
    "股份有限公司",
    "集团股份有限公司",
    "集团有限公司",
    "有限公司",
    "股份",
    "控股集团",
    "控股",
    "集团",
    "公司",
    "a股",
    "b股",
    "h股",
)

_TEXT_KEYS = (
    "title",
    "content",
    "summary",
    "snippet",
    "desc",
    "description",
)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_text(text: str) -> str:
    text = _safe_text(text).strip().lower()
    if not text:
        return ""
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")
    text = text.replace("，", ",").replace("。", ".")
    text = re.sub(r"\s+", " ", text)
    return text


def _char_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> Counter[str]:
    src = "".join(ch for ch in text if not ch.isspace())
    out: Counter[str] = Counter()
    for n in range(n_min, n_max + 1):
        if len(src) < n:
            continue
        for i in range(0, len(src) - n + 1):
            out[src[i : i + n]] += 1
    return out


def _tokenize_for_bm25(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    tokens: List[str] = []

    latin = re.findall(r"[a-z0-9][a-z0-9._-]*", text)
    tokens.extend(latin)

    zh = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    if zh:
        for n in (2, 3):
            if len(zh) >= n:
                for i in range(0, len(zh) - n + 1):
                    tokens.append(zh[i : i + n])

    if not tokens:
        tokens = list(text)
    return tokens


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv is not None:
            dot += av * bv
    if dot <= 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _minmax_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _hamming_distance(x: int, y: int) -> int:
    return (x ^ y).bit_count()


def _simhash(features: Counter[str], bits: int = 64) -> int:
    if not features:
        return 0
    vec = [0.0] * bits
    for token, weight in features.items():
        h = _hash64(token)
        w = float(weight)
        for i in range(bits):
            if (h >> i) & 1:
                vec[i] += w
            else:
                vec[i] -= w
    out = 0
    for i, v in enumerate(vec):
        if v >= 0:
            out |= 1 << i
    return out


def _extract_record_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in _TEXT_KEYS:
        parts.append(_safe_text(record.get(key, "")))

    for list_key in ("keywords", "tags", "entities", "entity_mentions", "stock_aliases"):
        raw = record.get(list_key)
        if isinstance(raw, list):
            for item in raw:
                parts.append(_safe_text(item))
        elif isinstance(raw, str):
            parts.append(raw)

    return _normalize_text(" ".join(p for p in parts if p))


@dataclass
class _LexicalDoc:
    idx: int
    record: Dict[str, Any]
    text: str
    tokens: List[str]
    tf_tokens: Counter[str]
    tf_char: Counter[str]
    simhash_fp: int
    bm25_raw: float = 0.0
    bm25_norm: float = 0.0
    tfidf_raw: float = 0.0
    tfidf_norm: float = 0.0
    alias_norm: float = 0.0
    score_pre: float = 0.0
    score_final: float = 0.0
    cluster_id: int = -1
    cluster_size: int = 1
    debug: Dict[str, Any] = field(default_factory=dict)


class LexicalSignalLinker:
    """Lexical ranking + dedup for stock-related alternative/news signals."""

    def __init__(self) -> None:
        self.bm25_k1 = 1.4
        self.bm25_b = 0.75
        self.weight_bm25 = 0.48
        self.weight_tfidf = 0.34
        self.weight_alias = 0.18
        self.simhash_bits = 64
        self.simhash_dist_threshold = 6

    def _build_aliases(
        self,
        stock_name: str,
        records: Sequence[Dict[str, Any]],
        extra_aliases: Optional[Iterable[str]] = None,
    ) -> List[str]:
        base = _normalize_text(stock_name)
        aliases = {base, base.replace(" ", "")}

        for suffix in _ALIAS_SUFFIXES:
            if base.endswith(suffix):
                cut = base[: -len(suffix)].strip()
                if len(cut) >= 2:
                    aliases.add(cut)

        for token in re.split(r"[\s,;|/()（）\[\]【】\-_.]+", base):
            token = token.strip()
            if len(token) >= 2:
                aliases.add(token)

        if extra_aliases:
            for alias in extra_aliases:
                v = _normalize_text(_safe_text(alias))
                if len(v) >= 2:
                    aliases.add(v)

        for item in records:
            for key in ("stock_code", "symbol", "ticker"):
                raw = item.get(key)
                if raw is None:
                    continue
                v = _normalize_text(_safe_text(raw))
                if len(v) >= 2:
                    aliases.add(v)

        cleaned = sorted(a for a in aliases if len(a) >= 2)
        return cleaned[:64]

    def _alias_match_score(self, text: str, aliases: Sequence[str]) -> float:
        if not text or not aliases:
            return 0.0
        hits = [a for a in aliases if a in text]
        if not hits:
            return 0.0
        hit_ratio = len(hits) / max(1.0, min(float(len(aliases)), 6.0))
        longest = max(len(h) for h in hits)
        len_boost = min(1.0, longest / 10.0)
        return _clip(0.65 * hit_ratio + 0.35 * len_boost)

    def _score_bm25(self, docs: Sequence[_LexicalDoc], query_tokens: Sequence[str]) -> None:
        if not docs:
            return
        token_df: Counter[str] = Counter()
        for d in docs:
            token_df.update(set(d.tokens))
        avgdl = sum(len(d.tokens) for d in docs) / max(len(docs), 1)
        n_docs = len(docs)
        uniq_query = list(dict.fromkeys(query_tokens))

        for d in docs:
            score = 0.0
            dl = max(len(d.tokens), 1)
            for term in uniq_query:
                tf = d.tf_tokens.get(term, 0)
                if tf <= 0:
                    continue
                df = token_df.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * dl / max(avgdl, 1e-9))
                score += idf * (tf * (self.bm25_k1 + 1.0) / max(denom, 1e-12))
            d.bm25_raw = score

        scaled = _minmax_scale([d.bm25_raw for d in docs])
        for d, val in zip(docs, scaled):
            d.bm25_norm = _clip(val)

    def _score_tfidf(self, docs: Sequence[_LexicalDoc], query_char_tf: Counter[str]) -> None:
        if not docs or not query_char_tf:
            return
        n_docs = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d.tf_char.keys()))
        df.update(set(query_char_tf.keys()))

        idf: Dict[str, float] = {}
        for tok, cnt in df.items():
            idf[tok] = math.log((1.0 + n_docs) / (1.0 + cnt)) + 1.0

        q_vec: Dict[str, float] = {}
        q_total = sum(query_char_tf.values()) or 1
        for tok, c in query_char_tf.items():
            q_vec[tok] = (c / q_total) * idf.get(tok, 1.0)

        for d in docs:
            vec: Dict[str, float] = {}
            t_total = sum(d.tf_char.values()) or 1
            for tok, c in d.tf_char.items():
                vec[tok] = (c / t_total) * idf.get(tok, 1.0)
            d.tfidf_raw = _cosine_sparse(q_vec, vec)

        scaled = _minmax_scale([d.tfidf_raw for d in docs])
        for d, val in zip(docs, scaled):
            d.tfidf_norm = _clip(val)

    def _cluster_docs(self, docs: Sequence[_LexicalDoc]) -> List[List[_LexicalDoc]]:
        clusters: List[List[_LexicalDoc]] = []
        reps: List[int] = []
        for doc in docs:
            assigned = False
            for ci, rep_fp in enumerate(reps):
                if _hamming_distance(doc.simhash_fp, rep_fp) <= self.simhash_dist_threshold:
                    clusters[ci].append(doc)
                    assigned = True
                    break
            if not assigned:
                reps.append(doc.simhash_fp)
                clusters.append([doc])
        return clusters

    def rank_records(
        self,
        stock_name: str,
        records: Sequence[Dict[str, Any]],
        top_k: int = 80,
        min_score: float = 0.0,
        extra_aliases: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        docs: List[_LexicalDoc] = []
        for i, item in enumerate(records or []):
            if not isinstance(item, dict):
                continue
            text = _extract_record_text(item)
            if not text:
                continue
            tokens = _tokenize_for_bm25(text)
            tf_tokens = Counter(tokens)
            tf_char = _char_ngrams(text)
            docs.append(
                _LexicalDoc(
                    idx=i,
                    record=item,
                    text=text,
                    tokens=tokens,
                    tf_tokens=tf_tokens,
                    tf_char=tf_char,
                    simhash_fp=_simhash(tf_char, bits=self.simhash_bits),
                )
            )

        if not docs:
            return [], {
                "method": "alias_bm25_tfidf_charngram_simhash_v1",
                "query": stock_name,
                "raw_record_count": len(records or []),
                "usable_record_count": 0,
                "cluster_count": 0,
                "deduplicated_count": 0,
                "returned_count": 0,
            }

        aliases = self._build_aliases(stock_name, docs and [d.record for d in docs] or [], extra_aliases)
        query_text = _normalize_text(" ".join(aliases)) or _normalize_text(stock_name)
        query_tokens = _tokenize_for_bm25(query_text)
        query_char_tf = _char_ngrams(query_text)

        for d in docs:
            d.alias_norm = self._alias_match_score(d.text, aliases)

        self._score_bm25(docs, query_tokens)
        self._score_tfidf(docs, query_char_tf)

        for d in docs:
            d.score_pre = _clip(
                self.weight_bm25 * d.bm25_norm
                + self.weight_tfidf * d.tfidf_norm
                + self.weight_alias * d.alias_norm
            )

        # SimHash dedup clustering, keep highest-ranked representative per cluster.
        docs_sorted = sorted(docs, key=lambda x: x.score_pre, reverse=True)
        clusters = self._cluster_docs(docs_sorted)
        representatives: List[_LexicalDoc] = []
        for cid, members in enumerate(clusters):
            members.sort(key=lambda x: x.score_pre, reverse=True)
            for m in members:
                m.cluster_id = cid
                m.cluster_size = len(members)
            representatives.append(members[0])

        for d in representatives:
            crowding_penalty = min(0.12, 0.03 * (d.cluster_size - 1))
            d.score_final = _clip(d.score_pre - crowding_penalty)

        ranked = [d for d in sorted(representatives, key=lambda x: x.score_final, reverse=True) if d.score_final >= min_score]
        ranked = ranked[: max(1, int(top_k))]

        output: List[Dict[str, Any]] = []
        for d in ranked:
            row = dict(d.record)
            row["lexical_link_score"] = round(d.score_final, 6)
            row["lexical_link_components"] = {
                "alias": round(d.alias_norm, 6),
                "bm25": round(d.bm25_norm, 6),
                "tfidf_char_ngram": round(d.tfidf_norm, 6),
            }
            row["lexical_cluster_id"] = d.cluster_id
            row["lexical_cluster_size"] = d.cluster_size
            output.append(row)

        meta = {
            "method": "alias_bm25_tfidf_charngram_simhash_v1",
            "query": stock_name,
            "query_alias_count": len(aliases),
            "raw_record_count": len(records or []),
            "usable_record_count": len(docs),
            "cluster_count": len(clusters),
            "deduplicated_count": max(0, len(docs) - len(representatives)),
            "returned_count": len(output),
            "weights": {
                "bm25": self.weight_bm25,
                "tfidf_char_ngram": self.weight_tfidf,
                "alias": self.weight_alias,
            },
        }
        return output, meta


lexical_signal_linker = LexicalSignalLinker()

