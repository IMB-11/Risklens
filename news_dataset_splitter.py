"""
Kaggle新闻分类数据集截取工具

数据集信息：
- 来源：https://www.kaggle.com/datasets/rmisra/news-category-dataset
- 文件名：News_Category_Dataset_v3.json
- 格式：JSON Lines（每行一条JSON记录）
- 时间范围：2012年1月 - 2022年5月

数据结构：
{
    "category": "BUSINESS",
    "headline": "新闻标题",
    "authors": "作者",
    "link": "新闻链接",
    "short_description": "简短描述",
    "date": "2020-01-05"
}

功能：
1. 读取JSON格式新闻数据集
2. 按年份范围截取新闻数据
3. 统计各分类新闻数量
4. 保持原始数据格式
"""

import os
import json
import glob
from datetime import datetime
from collections import Counter

class NewsDatasetSplitter:
    def __init__(self, source_file: str, output_dir: str = './news_dataset_splits'):
        self.source_file = source_file
        self.output_dir = output_dir
        self.all_news = []

    def load_dataset(self):
        """
        加载新闻数据集（JSON Lines格式）
        """
        print(f"[扫描] 正在读取新闻数据集: {self.source_file}")

        self.all_news = []
        with open(self.source_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        news = json.loads(line)
                        self.all_news.append(news)
                    except:
                        continue

        # 按日期排序
        self.all_news.sort(key=lambda x: self._parse_date(x.get('date', '')))

        # 统计信息
        if self.all_news:
            first_date = self._parse_date(self.all_news[0].get('date', ''))
            last_date = self._parse_date(self.all_news[-1].get('date', ''))
            total_size = os.path.getsize(self.source_file)

            print(f"✅ 读取完成！")
            print(f"   总新闻数: {len(self.all_news):,} 条")
            print(f"   时间范围: {first_date.strftime('%Y-%m-%d') if first_date else '未知'} 至 {last_date.strftime('%Y-%m-%d') if last_date else '未知'}")
            print(f"   文件大小: {self._format_size(total_size)}")
        else:
            print(f"❌ 未读取到任何数据！")

        return self.all_news

    def _parse_date(self, date_str: str):
        """解析日期字符串"""
        if not date_str:
            return datetime.min

        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%B %d, %Y'  # 如 January 01, 2020
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        return datetime.min

    def _format_size(self, bytes_size: int) -> str:
        """格式化文件大小"""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        else:
            return f"{bytes_size / (1024 * 1024):.1f} MB"

    # 金融相关分类（仅保留最相关的商业和财经新闻）
    FINANCIAL_CATEGORIES = {
        'BUSINESS',  # 商业新闻
        'MONEY'      # 财经新闻
    }

    def extract_by_year_range(self, start_year: int, end_year: int, output_name: str = None,
                              financial_only: bool = False) -> dict:
        """
        按年份范围截取新闻数据（可选按金融分类筛选）

        参数：
            start_year: 开始年份（包含）
            end_year: 结束年份（包含）
            output_name: 输出文件名
            financial_only: 是否只保留金融类新闻（默认False）

        返回：
            截取结果字典
        """
        if output_name is None:
            if financial_only:
                output_name = f'financial_news_{start_year}-{end_year}.json'
            else:
                output_name = f'news_{start_year}-{end_year}.json'

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_name)

        if financial_only:
            print(f"\n[时间范围] 正在截取 {start_year} 年到 {end_year} 年的金融类新闻...")
            print(f"   金融分类: {', '.join(sorted(self.FINANCIAL_CATEGORIES))}")
        else:
            print(f"\n[时间范围] 正在截取 {start_year} 年到 {end_year} 年的新闻数据...")

        # 按年份范围和分类筛选
        filtered_news = []
        for idx, news in enumerate(self.all_news):
            date = self._parse_date(news.get('date', ''))
            category = news.get('category', '').strip().upper()

            # 年份筛选
            if not (date.year >= start_year and date.year <= end_year):
                continue

            # 金融分类筛选
            if financial_only and category not in self.FINANCIAL_CATEGORIES:
                continue

            filtered_news.append(news)

            # 进度显示
            if (idx + 1) % 50000 == 0:
                print(f"   已处理 {idx + 1:,} 条...")

        # 统计分类
        category_counts = Counter(news.get('category', 'UNKNOWN') for news in filtered_news)

        # 写入输出文件（JSON Lines格式，每行一条记录）
        with open(output_path, 'w', encoding='utf-8') as f:
            for news in filtered_news:
                f.write(json.dumps(news, ensure_ascii=False) + '\n')

        output_size = os.path.getsize(output_path)

        print(f"✅ {start_year}-{end_year} 年份范围数据截取完成！")
        print(f"   输出目录: {self.output_dir}")
        print(f"   截取新闻数: {len(filtered_news):,} 条")
        print(f"   实际大小: {self._format_size(output_size)}")

        # 输出前5个分类统计
        print(f"   分类统计（前5）:")
        for category, count in category_counts.most_common(5):
            print(f"      - {category}: {count:,} 条")

        return {
            'output_path': output_path,
            'news_count': len(filtered_news),
            'actual_size': output_size,
            'category_counts': dict(category_counts)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Kaggle新闻分类数据集截取工具')
    parser.add_argument('--source', required=True, help='原始新闻数据集JSON文件路径')
    parser.add_argument('--output', default='./news_dataset_splits', help='输出目录')
    parser.add_argument('--start-year', type=int, default=2012, help='开始年份（默认2012）')
    parser.add_argument('--end-year', type=int, default=2019, help='结束年份（默认2019）')
    parser.add_argument('--financial-only', action='store_true', help='只保留金融类新闻（BUSINESS, MONEY等）')
    args = parser.parse_args()

    # 初始化工具
    splitter = NewsDatasetSplitter(args.source, args.output)

    # 加载数据集
    splitter.load_dataset()

    # 按年份范围截取（可选金融类）
    result = splitter.extract_by_year_range(
        args.start_year,
        args.end_year,
        financial_only=args.financial_only
    )

    # 输出汇总
    print("\n" + "="*60)
    print("[汇总] 新闻数据集截取汇总")
    print("="*60)
    print(f"\n{args.start_year}-{args.end_year} 年份范围版本:")
    print(f"   输出路径: {result['output_path']}")
    print(f"   新闻数量: {result['news_count']:,} 条")
    print(f"   实际大小: {splitter._format_size(result['actual_size'])}")

    print("\n[完成] 所有任务完成！")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("""
Kaggle新闻分类数据集截取工具
============================

使用方法：
python news_dataset_splitter.py --source <JSON文件路径> [--output <输出目录>] [--start-year <年份>] [--end-year <年份>] [--financial-only]

示例：
# 截取所有新闻（2012-2019年）
python news_dataset_splitter.py --source "C:/Users/魏子翔/Desktop/News_Category_Dataset_v3.json"

# 仅截取金融类新闻（推荐用于股票预测）
python news_dataset_splitter.py --source "C:/Users/魏子翔/Desktop/News_Category_Dataset_v3.json" --financial-only

# 自定义年份范围 + 金融类
python news_dataset_splitter.py --source "C:/Users/魏子翔/Desktop/News_Category_Dataset_v3.json" --start-year 2015 --end-year 2020 --financial-only

参数说明：
--source         : 原始新闻数据集JSON文件路径（必需）
--output         : 输出目录（默认: ./news_dataset_splits）
--start-year     : 开始年份（默认: 2012）
--end-year       : 结束年份（默认: 2019）
--financial-only : 只保留金融类新闻（BUSINESS, MONEY, POLITICS等）

功能：
1. 读取JSON Lines格式新闻数据集
2. 按年份范围截取新闻数据
3. 支持仅截取金融类新闻（用于股票预测）
4. 统计各分类新闻数量
5. 保持原始数据格式

金融类新闻分类（仅保留最相关的两类）：
- BUSINESS    : 商业新闻
- MONEY       : 财经新闻

数据集说明：
- 来源：https://www.kaggle.com/datasets/rmisra/news-category-dataset
- 格式：JSON Lines（每行一条JSON记录）
- 字段：category, headline, authors, link, short_description, date
- 时间范围：2012年1月 - 2022年5月

输出格式：
- 保持JSON Lines格式（每行一条记录）
- 方便后续处理和分析
- 金融类新闻输出文件名：financial_news_{start}-{end}.json
        """)
        sys.exit(1)

    main()
