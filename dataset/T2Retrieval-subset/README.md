# T2Retrieval 10k 评测子集

本目录保存 T2 评测的本地数据与向量资产。Git 只保留本文和 `manifest.json`；Parquet、`embeddings-v2/` 和生成的 `reports/` 均忽略。已生成的本地文件保留，避免重新调用 embedding。

`dataset/T2Retrieval/`（全量 118,605 篇）是抽样原料。重新下载后的源文件必须与 manifest 中的指纹一致，才能复现相同子集。

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

embed / evaluate 脚本默认指向全量目录，跑子集时在 `config/local.yaml` 中设置：

```yaml
dataset:
  t2_dir: dataset/T2Retrieval-subset
```

embedding 产物将写入本目录下的 `embeddings-v2/`（mmap 向量 + chunk 映射 + 断点续跑指纹），
与本子集自包含，不会污染全量目录。

## 完整复现入口

在仓库根目录激活 Conda `agent` 并安装 `backend/requirements.txt`、`dataset/requirements.txt`。首次使用且没有原始数据时，下载并抽样：

```bash
cd dataset
python download_t2retrieval.py
python subsample_t2retrieval.py --num-queries 500 --pool-docs 10000 --seed 42
cd ..
```

配置本地 `config/local.yaml` 的 embedding 凭据和上述数据目录后，生成向量并评测：

```bash
# 只解析、分块、生成待嵌入文本和指纹，不调用模型。
python -m dataset.embed_t2retrieval --prepare-only
# 首次执行会调用 embedding；已有完整产物时按指纹校验后复用。
python -m dataset.embed_t2retrieval
# 只读取本地向量，500 条 query 对 10,000 篇候选文档评测。
python -m dataset.evaluate_t2retrieval --expected-query-count 500
```

评测输出 JSON 到标准输出，可自行保存到本地 `reports/`。`--rerank-model` 可选，依赖 `backend/requirements-rerank.txt`，不传则仅做向量检索。历史上游排序适配实验已从维护范围移除，已有比较报告仍留在本地 `reports/`，不属于正式 RAGFlow benchmark。

## 已测成本（生产链路精确口径，cl100k 计数）

- 81,353 个 Child（平均 8.14 个/篇），**总计 9,595,852 token**
- 检索指标为 **against 10k-pool**，与全量语料上的绝对值不可直接比较；
  用于本项目内部消融对比（同池同 query）完全有效。
