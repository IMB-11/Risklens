"""
Kaggle股票数据集截取工具

数据集信息：
- 来源：https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
- 结构：
  - archive/
    - etfs/      # ETF数据（每只ETF一个CSV文件）
    - stocks/    # 股票数据（每只股票一个CSV文件）
    - symbols_valid_meta.csv  # 股票元数据
- 格式：Date,Open,High,Low,Close,Adj Close,Volume
- 时间顺序：是，按日期升序排列

功能：
1. 按年份范围截取所有股票/ETF数据
2. 完整保留每只股票的指定年份数据
3. 支持2012-2019年等自定义年份范围
"""

import os
import glob
import csv
from pathlib import Path
from datetime import datetime

class StockDatasetSplitter:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.file_info = []

    def scan_dataset(self, include_etfs: bool = True):
        """
        扫描数据集，获取所有CSV文件信息

        参数：
            include_etfs: 是否包含ETF数据（默认True）
        """
        print(f"[扫描] 正在扫描数据集: {self.source_dir}")

        # 定义数据目录
        stocks_dir = os.path.join(self.source_dir, 'stocks')
        etfs_dir = os.path.join(self.source_dir, 'etfs')

        # 查找所有CSV文件
        csv_patterns = []

        # 股票目录
        if os.path.exists(stocks_dir):
            csv_patterns.append(os.path.join(stocks_dir, '*.csv'))
            csv_patterns.append(os.path.join(stocks_dir, '*.CSV'))
            print(f"   - 已识别股票目录: {stocks_dir}")

        # ETF目录（可选）
        if include_etfs and os.path.exists(etfs_dir):
            csv_patterns.append(os.path.join(etfs_dir, '*.csv'))
            csv_patterns.append(os.path.join(etfs_dir, '*.CSV'))
            print(f"   - 已识别ETF目录: {etfs_dir}")

        all_files = []
        for pattern in csv_patterns:
            all_files.extend(glob.glob(pattern, recursive=False))

        # 过滤可能的元数据文件
        exclude_names = ['all_stocks_5yr', 'symbols', 'readme', 'README', '_metadata']
        data_files = []

        for filepath in all_files:
            filename = os.path.basename(filepath).lower()
            if any(exclude in filename for exclude in exclude_names):
                continue
            if os.path.getsize(filepath) > 0:
                data_files.append(filepath)

        # 收集文件信息
        self.file_info = []
        for filepath in data_files:
            size = os.path.getsize(filepath)
            stock_name = os.path.splitext(os.path.basename(filepath))[0]
            data_type = 'etf' if 'etfs' in filepath.lower() else 'stock'

            self.file_info.append({
                'path': filepath,
                'size': size,
                'stock_name': stock_name,
                'data_type': data_type
            })

        # 按股票名称排序
        self.file_info.sort(key=lambda x: x['stock_name'])

        # 统计信息
        total_size = sum(f['size'] for f in self.file_info)
        stock_count = sum(1 for f in self.file_info if f['data_type'] == 'stock')
        etf_count = sum(1 for f in self.file_info if f['data_type'] == 'etf')

        print(f"[完成] 扫描完成！")
        print(f"   发现 {len(self.file_info)} 个数据文件")
        print(f"   - 股票: {stock_count} 只")
        print(f"   - ETF: {etf_count} 只")
        print(f"   总大小: {self._format_size(total_size)}")

        return self.file_info

    def _format_size(self, bytes_size: int) -> str:
        """格式化文件大小"""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        else:
            return f"{bytes_size / (1024 * 1024):.1f} MB"

    def extract_by_year_range(self, start_year: int, end_year: int, output_name: str = None) -> dict:
        """
        按年份范围截取所有股票数据（完整保留每只股票的指定年份数据）

        参数：
            start_year: 开始年份（包含）
            end_year: 结束年份（包含）
            output_name: 输出目录名（默认为 dataset_{start_year}-{end_year}）

        返回：
            截取结果字典
        """
        if output_name is None:
            output_name = f'dataset_{start_year}-{end_year}'
        output_path = os.path.join(self.output_dir, output_name)
        os.makedirs(output_path, exist_ok=True)

        # 清空已有文件
        for f in glob.glob(os.path.join(output_path, '*.csv')):
            os.remove(f)

        current_size = 0
        file_count = 0
        skipped_count = 0

        print(f"\n[时间范围] 正在截取 {start_year} 年到 {end_year} 年的股票数据...")

        for idx, file_info in enumerate(self.file_info):
            stock_name = file_info['stock_name']
            output_file = os.path.join(output_path, f"{stock_name}.csv")

            actual_size = self._copy_by_year_range(
                file_info['path'],
                output_file,
                start_year,
                end_year
            )

            if actual_size > 0:
                current_size += actual_size
                file_count += 1
            else:
                skipped_count += 1

            # 进度显示
            if (idx + 1) % 100 == 0:
                print(f"   已处理 {idx + 1}/{len(self.file_info)} 个文件...")

        print(f"[完成] {start_year}-{end_year} 年份范围数据截取完成！")
        print(f"   输出目录: {output_path}")
        print(f"   成功处理: {file_count} 个股票文件")
        print(f"   跳过（无数据）: {skipped_count} 个文件")
        print(f"   实际大小: {self._format_size(current_size)}")

        return {
            'output_path': output_path,
            'file_count': file_count,
            'skipped_count': skipped_count,
            'actual_size': current_size
        }

    def _copy_by_year_range(self, source_path: str, output_path: str, start_year: int, end_year: int) -> int:
        """
        复制CSV文件中指定年份范围的数据

        参数：
            source_path: 源文件路径
            output_path: 输出文件路径
            start_year: 开始年份
            end_year: 结束年份

        返回：
            实际截取的文件大小
        """
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as src:
                reader = csv.reader(src)

                # 读取表头
                header = next(reader)

                # 查找日期列
                date_col_idx = None
                for i, col in enumerate(header):
                    if 'date' in col.lower():
                        date_col_idx = i
                        break

                if date_col_idx is None:
                    return 0

                # 读取所有数据行并按年份筛选
                filtered_rows = []
                for row in reader:
                    if len(row) > date_col_idx:
                        try:
                            date_str = row[date_col_idx]
                            # 尝试多种日期格式
                            date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
                            date = None
                            for fmt in date_formats:
                                try:
                                    date = datetime.strptime(date_str, fmt)
                                    break
                                except:
                                    continue

                            if date and start_year <= date.year <= end_year:
                                filtered_rows.append(row)
                        except:
                            continue

                if not filtered_rows:
                    return 0

                # 按日期排序（升序）
                def get_date(row):
                    date_str = row[date_col_idx]
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                    return datetime.min

                filtered_rows.sort(key=get_date)

                # 写入输出文件
                with open(output_path, 'w', encoding='utf-8', newline='') as dst:
                    writer = csv.writer(dst)
                    writer.writerow(header)
                    for row in filtered_rows:
                        writer.writerow(row)

                return os.path.getsize(output_path)
        except Exception as e:
            return 0

def main():
    # 配置
    import argparse
    parser = argparse.ArgumentParser(description='Kaggle股票数据集年份范围截取工具')
    parser.add_argument('--source', required=True, help='原始数据集目录')
    parser.add_argument('--output', default='./dataset_splits', help='输出目录')
    parser.add_argument('--start-year', type=int, default=2012, help='开始年份（默认2012）')
    parser.add_argument('--end-year', type=int, default=2019, help='结束年份（默认2019）')
    args = parser.parse_args()

    # 初始化工具
    splitter = StockDatasetSplitter(args.source, args.output)

    # 扫描数据集
    splitter.scan_dataset()

    # 按年份范围截取
    result = splitter.extract_by_year_range(args.start_year, args.end_year)

    # 输出汇总
    print("\n" + "="*60)
    print("[汇总] 数据集截取汇总")
    print("="*60)
    print(f"\n{args.start_year}-{args.end_year} 年份范围版本:")
    print(f"   输出路径: {result['output_path']}")
    print(f"   成功处理: {result['file_count']} 个股票文件")
    print(f"   跳过（无数据）: {result['skipped_count']} 个文件")
    print(f"   实际大小: {splitter._format_size(result['actual_size'])}")

    print("\n[完成] 所有任务完成！")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("""
Kaggle股票数据集截取工具
========================

使用方法：
python dataset_splitter.py --source <数据集目录> [--output <输出目录>] [--start-year <年份>] [--end-year <年份>]

示例：
python dataset_splitter.py --source "C:/Users/魏子翔/Desktop/archive"
python dataset_splitter.py --source "C:/Users/魏子翔/Desktop/archive" --start-year 2015 --end-year 2020

参数说明：
--source      : 原始数据集目录（必需，包含stocks/和etfs/子目录）
--output      : 输出目录（默认: ./dataset_splits）
--start-year  : 开始年份（默认: 2012）
--end-year    : 结束年份（默认: 2019）

功能：
1. 扫描数据集目录（自动识别stocks/和etfs/子目录）
2. 按年份范围截取所有股票/ETF数据
3. 完整保留每只股票的指定年份数据
4. 自动跳过无匹配数据的股票
5. 保持CSV表头和数据格式

数据集结构（适用于您的数据）：
archive/
├── stocks/                    # 股票数据（每只股票一个CSV文件）
├── etfs/                      # ETF数据（每只ETF一个CSV文件）
└── symbols_valid_meta.csv     # 元数据文件（会被自动跳过）

数据格式：
- CSV字段：Date,Open,High,Low,Close,Adj Close,Volume
- 时间顺序：按日期升序排列
        """)
        sys.exit(1)

    main()
