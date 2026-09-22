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

生产 runner 为目标 split 的每个 query 录制一行：

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

## 3. 生产 Runner

Runner 只编排与录制：索引、检索、问答全部调用 `backend/` 生产实现（`IngestionPipeline` / `HybridRouter` / `QAService`），评测代码不复制算法。前置条件：本地 Elasticsearch 已启动，`config/local.yaml` 已配置 embedding/LLM。

```bash
conda activate agent

# 1) 全量索引 1,000 篇到独立评测索引（--limit N 可先冒烟）
python -m evaluation.runner.index_dataset \
  --dataset evaluation/datasets/crud_rag_mini_v1 \
  --index-name rag-eval-crud_rag_mini_v1

# 2) 检索 baseline（不调 Chat）：vector / keyword / hybrid / hybrid_rerank
python -m evaluation.runner.run_retrieval \
  --dataset evaluation/datasets/crud_rag_mini_v1 \
  --index-name rag-eval-crud_rag_mini_v1 --split dev

# 3) Dev 参数扫描 + 锁定（证据层会调用 Chat，--skip-evidence 可跳过）
python -m evaluation.runner.sweep \
  --dataset evaluation/datasets/crud_rag_mini_v1 \
  --index-name rag-eval-crud_rag_mini_v1

# 4) 锁定后 Test 各运行一次（不得再依据 Test 调参）
python -m evaluation.runner.run_retrieval \
  --dataset evaluation/datasets/crud_rag_mini_v1 \
  --index-name rag-eval-crud_rag_mini_v1 --split test
python -m evaluation.runner.run_qa \
  --dataset evaluation/datasets/crud_rag_mini_v1 \
  --index-name rag-eval-crud_rag_mini_v1 --split test
```

产物：`runs/*.jsonl`（每行带 dataset/Git/config/model 指纹）、`scores/*.json`、`reports/generated/*.md`（Runner 自动评分出报告）、`indexes/*.index_manifest.json`、`locks/locked_config.json` + `sweep_summary.md`（Dev 扫描与锁定结论）。实验纪律见 [`docs/plan/评测.md`](../docs/plan/评测.md) E1-C：扫描只在 Dev；Rerank 无稳定收益（recall@10 增益 < 1pp 或 mrr@10 下降）保持默认关闭。

历史四组消融的原始 run/score/report 保留在本地；一次性消融脚本及 `rule-jieba` 后端已移除。

## 4. 评分和报告

```bash
python -m evaluation.score_run evaluation/runs/baseline.jsonl \
  --split test \
  --output evaluation/scores/baseline.json

python -m evaluation.report evaluation/scores/baseline.json \
  --output evaluation/reports/generated/baseline.md
```

当前 scorer 计算 Hit/Recall@1/3/5/10、MRR@10、nDCG@10、Evidence Recall/Precision、Citation Validity/Document Precision，以及可得的 P50/P95、token 和成本。它不会把文档级引用精度冒充为 claim-level 引用正确性，也不会在未调用 Judge 时报告语义指标。

## 5. 测试

```bash
conda activate agent
python -m pytest evaluation/tests -q
python -m pytest backend/tests evaluation/tests -q
```

详细范围、E0–E3 阶段与真实 run 记录见 [`docs/plan/评测.md`](../docs/plan/评测.md)；面试口径见 [`docs/interview/评测与调参口径.md`](../docs/interview/评测与调参口径.md)。
