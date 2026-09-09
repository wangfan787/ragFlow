# T2Retrieval 10k 评测子集（持久化 artifact）

本目录是后续所有 T2 评测的唯一数据依赖；`dataset/T2Retrieval/`（全量 118,605 篇）只是原料，
可随时删除并重新下载，不影响本子集。

## 内容

| 文件 | 说明 |
|---|---|
| `corpus.parquet` | 10,000 篇候选池文档（`_id` / `text` / `title`，schema 与全量一致） |
| `queries.parquet` | 500 条评测 query |
| `qrels.parquet` | 2,614 对相关性标注（仅覆盖被抽的 500 条 query） |
| `manifest.json` | 抽样种子、数量、**全量源文件的 sha256 与大小指纹** |

## 生成方式（可复现）

```bash
cd dataset
python subsample_t2retrieval.py --num-queries 500 --pool-docs 10000 --seed 42
```

抽样策略为 query 优先：从有正例标注的 query 中随机抽 500 条（seed=42）；
被抽 query 的**全部正例文档强制进池**（保证指标不因正例缺席而系统性偏低），
剩余名额由随机干扰文档补齐到 10,000 篇。
同 seed + 同源文件（指纹见 manifest）必然重抽出完全相同的子集。

## 管线接入

embed / evaluate 脚本默认指向全量目录，跑子集时必须显式导出环境变量：

```bash
export T2_DATASET_DIR="$(pwd)/dataset/T2Retrieval-subset"
```

embedding 产物将写入本目录下的 `embeddings-v2/`（mmap 向量 + chunk 映射 + 断点续跑指纹），
与本子集自包含，不会污染全量目录。

## 已测成本（生产链路精确口径，cl100k 计数）

- 81,353 个 Child（平均 8.14 个/篇），**总计 9,595,852 token**
- 检索指标为 **against 10k-pool**，与全量语料上的绝对值不可直接比较；
  用于本项目内部消融对比（同池同 query）完全有效。
