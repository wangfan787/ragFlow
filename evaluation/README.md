# Evaluation

该目录提供独立于生产代码的可复现评测闭环。原始 CRUD-RAG、生成的数据集和真实运行结果均为本地资产，不提交 Git；构建器、校验器、评分器、报告器和测试进入版本管理。

## 1. 构建并校验数据集

使用项目固定环境：

```bash
conda activate agent
python -m evaluation.build_dataset
python -m evaluation.validate_dataset
```

默认读取 `dataset/CRUD_RAG/`，生成 `evaluation/datasets/crud_rag_mini_v1/`：1,000 篇文档，150 条查询（1/2/3 文档任务各 50 条），其中 Dev 30 条、Test 120 条。

构建器只读取上游 JSON/文本，不导入 CRUD-RAG 源码，也不需要安装其旧版 LlamaIndex、LangChain 或 Milvus 依赖。

## 2. Run JSONL 契约

生产 runner 后续应为目标 split 的每个 query 录制一行：

```json
{
  "query_id": "crud_q_...",
  "retrieved": [
    {"doc_id": "crud_...", "rank": 1, "score": 0.82, "channel": "hybrid"}
  ],
  "evidence_doc_ids": ["crud_..."],
  "answer": "基于证据生成的回答",
  "citations": [{"doc_id": "crud_..."}],
  "status": "ok",
  "latency_ms": {"retrieval": 35.2, "generation": 410.0, "total": 445.2},
  "usage": {"input_tokens": 800, "output_tokens": 120, "cost": 0.0012}
}
```

约束：rank 从 1 连续递增；检索、evidence 和引用中的 doc_id 必须存在于冻结 corpus；run 必须覆盖所选 split 的每个 query 且不得重复。`latency_ms` 与 `usage` 可以省略；省略时报告明确显示 unavailable，不补零、不猜测。

## 3. 评分和报告

```bash
python -m evaluation.score_run evaluation/runs/baseline.jsonl \
  --split test \
  --output evaluation/scores/baseline.json

python -m evaluation.report evaluation/scores/baseline.json \
  --output evaluation/reports/generated/baseline.md
```

当前 scorer 计算 Hit/Recall@1/3/5/10、MRR@10、nDCG@10、Evidence Recall/Precision、Citation Validity/Document Precision，以及可得的 P50/P95、token 和成本。它不会把文档级引用精度冒充为 claim-level 引用正确性，也不会在未调用 Judge 时报告语义指标。

## 4. 测试

```bash
conda activate agent
python -m pytest evaluation/tests -q
python -m pytest backend/tests evaluation/tests -q
```

详细范围、P0/P1/P2 与面试口径见 [`docs/plan/评测计划.md`](../docs/plan/评测计划.md)。

