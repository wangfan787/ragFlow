from pathlib import Path
from datasets import load_dataset

# 数据保存目录
output_dir = Path("./T2Retrieval")
cache_dir = output_dir / "cache"

output_dir.mkdir(parents=True, exist_ok=True)

print("正在下载 corpus...")
corpus = load_dataset(
    "mteb/T2Retrieval",
    "corpus",
    split="dev",
    cache_dir=str(cache_dir)
)

print("正在下载 queries...")
queries = load_dataset(
    "mteb/T2Retrieval",
    "queries",
    split="dev",
    cache_dir=str(cache_dir)
)

print("正在下载 qrels...")
qrels = load_dataset(
    "mteb/T2Retrieval",
    "default",
    split="dev",
    cache_dir=str(cache_dir)
)

# 显示基本信息
print("\n===== 数据集信息 =====")
print("corpus:", corpus)
print("queries:", queries)
print("qrels:", qrels)

print("\n===== 第一条文档 =====")
print(corpus[0])

print("\n===== 第一条查询 =====")
print(queries[0])

print("\n===== 第一条相关性标注 =====")
print(qrels[0])

# 导出为普通 Parquet 文件，方便本地查看
print("\n正在导出 Parquet 文件...")
corpus.to_parquet(output_dir / "corpus.parquet")
queries.to_parquet(output_dir / "queries.parquet")
qrels.to_parquet(output_dir / "qrels.parquet")

# 另外导出少量 CSV 预览，方便用 Excel 打开
corpus.select(range(min(1000, len(corpus)))).to_csv(
    output_dir / "corpus_preview.csv",
    index=False
)

queries.select(range(min(1000, len(queries)))).to_csv(
    output_dir / "queries_preview.csv",
    index=False
)

qrels.select(range(min(1000, len(qrels)))).to_csv(
    output_dir / "qrels_preview.csv",
    index=False
)

print("\n下载和导出完成。")
print("保存位置：", output_dir.resolve())