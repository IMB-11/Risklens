"""
数据集整合工具 - 高质量股票筛选版

筛选标准：
1. 排除ETF、ADR、优先股等非普通股票
2. 日均成交量 >= 100,000股（流动性好）
3. 平均股价 >= $5（避免低价垃圾股）
4. 数据周期 >= 1500个交易日（约6年完整数据）
5. 排除高波动股票（日波动率 > 5%）
6. 知名股票优先（包含主要指数成分股）
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob

class HighQualityStockFilter:
    """高质量股票筛选器"""

    def __init__(self):
        # 知名股票列表（主要指数成分股）
        self.BLUE_CHIPS = {
            # S&P 500 主要成分股
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'BRK-B', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS',
            'BAC', 'XOM', 'CVX', 'KO', 'PFE', 'MRK', 'PEP', 'COST', 'WBA',
            'MCD', 'BA', 'CAT', 'IBM', 'INTC', 'ORCL', 'CSCO', 'CRM', 'ADBE',
            # 其他知名股票
            'NFLX', 'BABA', 'TCEHY', 'JD', 'PDD', 'BIDU', 'NVDA', 'AMD', 'NKE'
        }

        # ETF关键词（严格排除）
        self.ETF_KEYWORDS = {
            'SPY', 'QQQ', 'DIA', 'VTI', 'VOO', 'IVV', 'IWM', 'EFA', 'EEM',
            'TQQQ', 'SQQQ', 'UVXY', 'TVIX', 'SPXL', 'SDS', 'SH', 'FAZ', 'FAS'
        }

        # 非股票后缀
        self.NON_STOCK_SUFFIXES = {
            'ETF', 'ETN', 'ADR', 'PR', 'PREF', 'UNIT', 'CLASS', 'INC',
            '-U', '-L', '-R', '-S', '-A', '-B', '-C', '-D', '-E'
        }

    def is_high_quality(self, symbol, df):
        """判断是否为高质量股票"""
        symbol_upper = symbol.upper()

        # 1. 严格排除ETF
        if symbol_upper in self.ETF_KEYWORDS:
            return False, "排除ETF"

        # 2. 排除非股票后缀
        for suffix in self.NON_STOCK_SUFFIXES:
            if symbol_upper.endswith(suffix):
                return False, f"排除后缀: {suffix}"

        # 3. 排除数字结尾
        if symbol_upper[-1].isdigit() and len(symbol_upper) > 1:
            return False, "排除数字结尾"

        # 4. 检查数据量（至少1500个交易日）
        if len(df) < 1500:
            return False, f"数据量不足: {len(df)} < 1500"

        # 5. 检查平均价格（至少$5）
        avg_price = df['Close'].mean()
        if avg_price < 5:
            return False, f"价格过低: ${avg_price:.2f}"

        # 6. 检查日均成交量（至少10万股）
        avg_volume = df['Volume'].mean()
        if avg_volume < 100000:
            return False, f"成交量不足: {int(avg_volume):,} < 100,000"

        # 7. 检查波动率（日波动率不超过5%）
        daily_returns = df['Close'].pct_change().dropna()
        if len(daily_returns) > 0:
            volatility = daily_returns.std() * 100
            if volatility > 5:
                return False, f"波动率过高: {volatility:.2f}% > 5%"

        # 8. 知名股票直接通过
        if symbol_upper in self.BLUE_CHIPS:
            return True, "✅ 蓝筹股"

        # 9. 普通股票通过基本筛选
        return True, "✅ 通过筛选"

class DatasetCombiner:
    def __init__(self, stock_dir, news_path, output_dir):
        self.stock_dir = stock_dir
        self.news_path = news_path
        self.output_dir = output_dir
        self.stock_filter = HighQualityStockFilter()
        os.makedirs(output_dir, exist_ok=True)

    def load_stock_data(self):
        """加载高质量股票数据"""
        print("="*60)
        print("[1/6] 正在筛选高质量股票")
        print("="*60)

        stock_files = glob(os.path.join(self.stock_dir, '*.csv'))
        print(f"发现 {len(stock_files):,} 个股票文件")

        all_stocks = []
        valid_symbols = set()
        excluded_reasons = {}

        for idx, file_path in enumerate(stock_files, 1):
            if idx % 300 == 0:
                print(f"已检查: {idx}/{len(stock_files)} | 有效: {len(valid_symbols)} | 排除: {len(excluded_reasons)}")

            symbol = os.path.basename(file_path).replace('.csv', '').strip()

            if not symbol or symbol in valid_symbols:
                continue

            try:
                df = pd.read_csv(file_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'])

                # 限制日期范围
                df = df[(df['Date'] >= '2012-01-01') & (df['Date'] <= '2019-12-31')]

                # 应用高质量筛选
                is_qualified, reason = self.stock_filter.is_high_quality(symbol, df)

                if is_qualified:
                    valid_symbols.add(symbol)
                    all_stocks.append(df.assign(symbol=symbol))
                    if "✅" in reason:
                        print(f"  {symbol}: {reason}")
                else:
                    excluded_reasons[symbol] = reason

            except Exception as e:
                excluded_reasons[symbol] = f"读取错误: {str(e)[:30]}"

        if not all_stocks:
            raise ValueError("未找到符合条件的高质量股票")

        combined = pd.concat(all_stocks, ignore_index=True)
        combined = combined.sort_values(['symbol', 'Date']).reset_index(drop=True)

        # 打印筛选统计
        print("\n" + "="*60)
        print("筛选结果统计")
        print("="*60)
        print(f"总文件数: {len(stock_files):,}")
        print(f"有效股票: {len(valid_symbols):,} 只")
        print(f"排除股票: {len(excluded_reasons):,} 只")
        print(f"有效数据: {len(combined):,} 条")
        print(f"日期范围: {combined['Date'].min().date()} 至 {combined['Date'].max().date()}")

        # 输出排除原因统计
        reason_counts = {}
        for reason in excluded_reasons.values():
            key = reason.split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1

        print("\n排除原因分布:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {reason}: {count:,} 只")

        return combined

    def extract_features(self, df):
        """提取技术指标"""
        print("\n" + "="*60)
        print("[2/6] 正在提取技术指标")
        print("="*60)

        df = df.copy().sort_values(['symbol', 'Date']).reset_index(drop=True)

        # 收益率
        df['return_1d'] = df.groupby('symbol')['Close'].pct_change(1)

        # 过滤异常收益率
        df.loc[df['return_1d'].abs() > 0.3, 'return_1d'] = np.nan

        # 移动平均线
        df['ma5'] = df.groupby('symbol')['Close'].rolling(5).mean().reset_index(0, drop=True)
        df['ma20'] = df.groupby('symbol')['Close'].rolling(20).mean().reset_index(0, drop=True)
        df['ma60'] = df.groupby('symbol')['Close'].rolling(60).mean().reset_index(0, drop=True)

        # RSI
        delta = df.groupby('symbol')['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        gain = gain.groupby(df['symbol']).rolling(14).mean().reset_index(0, drop=True)
        loss = loss.groupby(df['symbol']).rolling(14).mean().reset_index(0, drop=True)
        df['rsi'] = 100 - (100 / (1 + (gain / loss).fillna(1)))

        print(f"技术指标提取完成")
        return df

    def load_and_merge_news(self, stock_df):
        """加载并合并新闻数据 - 建立股票与舆情的深度联系"""
        print("\n" + "="*60)
        print("[3/6] 正在加载新闻数据并建立股票-舆情关联")
        print("="*60)

        news_data = []
        with open(self.news_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    news = json.loads(line)
                    news_data.append(news)

        print(f"发现 {len(news_data):,} 条新闻")

        # 处理新闻数据
        news_df = pd.DataFrame(news_data)
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df[(news_df['date'] >= '2012-01-01') & (news_df['date'] <= '2019-12-31')]

        # 获取有新闻的所有日期集合
        news_dates = set(news_df['date'].dt.date)
        print(f"有新闻的日期数量: {len(news_dates):,} 天")

        # 为股票数据添加日期键
        stock_df['date_key'] = stock_df['Date'].dt.date

        # 核心筛选：检查前后两天是否有新闻
        def has_news_in_window(row):
            check_date = row['date_key']
            # 检查当天、前一天、后一天是否有新闻
            for delta in [-1, 0, 1]:
                check = check_date + pd.Timedelta(days=delta)
                if check in news_dates:
                    return True
            return False

        # 标记有舆情关联的股票数据
        stock_df['has_news_nearby'] = stock_df.apply(has_news_in_window, axis=1)

        # 统计筛选前的数据量
        before_count = len(stock_df)
        before_stocks = stock_df['symbol'].nunique()

        # 保留前后两天内有新闻的股票数据
        filtered = stock_df[stock_df['has_news_nearby']].copy()

        # 统计筛选后的数据量
        after_count = len(filtered)
        after_stocks = filtered['symbol'].nunique()

        print(f"\n📊 舆情关联筛选结果:")
        print(f"  筛选前: {before_count:,} 条记录, {before_stocks:,} 只股票")
        print(f"  筛选后: {after_count:,} 条记录, {after_stocks:,} 只股票")
        print(f"  保留比例: {after_count / before_count * 100:.1f}%")

        # 聚合新闻特征（用于后续分析）
        daily_news = news_df.groupby(news_df['date'].dt.date).agg({
            'headline': 'count'
        }).reset_index()
        daily_news.columns = ['date', 'news_count']
        daily_news['date_key'] = daily_news['date']

        # 合并新闻数据
        merged = pd.merge(
            filtered,
            daily_news[['date_key', 'news_count']],
            on='date_key',
            how='left'
        )

        merged['news_count'] = merged['news_count'].fillna(0).astype(int)

        # 添加前后两天的新闻数量
        daily_news_dict = daily_news.set_index('date_key')['news_count'].to_dict()

        def get_news_count(date_key, delta):
            check_date = date_key + pd.Timedelta(days=delta)
            return daily_news_dict.get(check_date, 0)

        merged['news_prev_day'] = merged['date_key'].apply(lambda x: get_news_count(x, -1))
        merged['news_next_day'] = merged['date_key'].apply(lambda x: get_news_count(x, 1))
        merged['total_news_window'] = merged['news_count'] + merged['news_prev_day'] + merged['news_next_day']

        return merged.drop('date_key', axis=1)

    def create_targets(self, df):
        """创建预测目标"""
        print("\n" + "="*60)
        print("[4/6] 正在创建预测目标")
        print("="*60)

        df = df.copy().sort_values(['symbol', 'Date']).reset_index(drop=True)
        df['next_day_return'] = df.groupby('symbol')['return_1d'].shift(-1)
        df['up_down'] = (df['next_day_return'] > 0).astype(int)

        # 过滤无效数据
        df = df.dropna(subset=['next_day_return', 'return_1d'])
        df = df[(df['next_day_return'] >= -0.3) & (df['next_day_return'] <= 0.3)]

        print(f"有效样本数: {len(df):,} 条")
        return df

    def add_time_features(self, df):
        """添加时间特征"""
        print("\n" + "="*60)
        print("[5/6] 正在添加时间特征")
        print("="*60)

        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['quarter'] = df['Date'].dt.quarter

        print("时间特征添加完成")
        return df

    def save_dataset(self, df):
        """保存数据集"""
        print("\n" + "="*60)
        print("[6/6] 正在保存数据集")
        print("="*60)

        # 保存主数据集
        output_path = os.path.join(self.output_dir, 'training_data.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # 保存股票列表
        with open(os.path.join(self.output_dir, 'stock_list.txt'), 'w', encoding='utf-8') as f:
            for s in sorted(df['symbol'].unique()):
                f.write(f"{s}\n")

        # 保存统计信息
        stats = {
            '股票数量': df['symbol'].nunique(),
            '样本总数': len(df),
            '时间范围': f"{df['Date'].min().date()} 至 {df['Date'].max().date()}",
            '上涨比例': f"{df['up_down'].mean() * 100:.1f}%",
            '收益率均值': f"{df['next_day_return'].mean() * 100:.2f}%",
            '收益率标准差': f"{df['next_day_return'].std() * 100:.2f}%",
            '收益率范围': f"[{df['next_day_return'].min() * 100:.2f}%, {df['next_day_return'].max() * 100:.2f}%]"
        }

        with open(os.path.join(self.output_dir, 'stats.txt'), 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("高质量股票训练数据集统计\n")
            f.write("="*60 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        print(f"数据集已保存: {output_path}")
        return stats

    def process(self):
        """完整处理流程"""
        try:
            # 1. 加载高质量股票
            stock_data = self.load_stock_data()

            # 2. 提取特征
            stock_with_features = self.extract_features(stock_data)

            # 3. 合并新闻
            merged = self.load_and_merge_news(stock_with_features)

            # 4. 创建目标
            with_targets = self.create_targets(merged)

            # 5. 添加时间特征
            final = self.add_time_features(with_targets)

            # 6. 保存
            stats = self.save_dataset(final)

            # 最终统计
            print("\n" + "="*60)
            print("高质量股票训练数据集 - 最终统计")
            print("="*60)
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("="*60)
            print("\n🎉 高质量股票数据集创建完成！")

            return stats

        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    STOCK_DIR = r"C:\Users\魏子翔\Desktop\黑客松\黑客松\dataset_splits\dataset_2012-2019"
    NEWS_PATH = r"C:\Users\魏子翔\Desktop\黑客松\黑客松\news_dataset_splits\financial_news_2012-2019.json"
    OUTPUT_DIR = r"C:\Users\魏子翔\Desktop\黑客松\黑客松\combined_dataset"

    # 清理旧数据
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # 处理
    combiner = DatasetCombiner(STOCK_DIR, NEWS_PATH, OUTPUT_DIR)
    combiner.process()

if __name__ == "__main__":
    main()
