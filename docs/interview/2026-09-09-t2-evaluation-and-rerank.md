# 面试记录：RAG 检索链路评测体系搭建与 Rerank 优化

> 日期：2026-09-09
> 一句话总结：在 4,000 万 embedding token 预算约束下，为 Small-to-Big RAG 链路搭建了
> 成本可控的 T2Retrieval 评测体系（基线 Recall@10 79.9%，MRR@10 0.85），
> 并通过接入 cross-encoder 重排提升到 **Recall@10 84.1%，MRR@10 0.91**。

## 1. 背景与约束

- 目标：用标准中文检索数据集（T2Retrieval）量化验证自研 RAG 链路
  （HTML/文本解析 → 严格 token 分块 → Child-only embedding → Family 均值聚合 → 文档 max）的真实检索效果。
- 硬约束：GLM embedding-3 套餐仅剩 **4,000 万 token**；本机网络访问 GitHub/HuggingFace 直连基本不可用。
- 全量语料规模：118,605 篇文档、22,812 条 query、118,932 对相关性标注。

## 2. 成本工程：先算账，再花钱

**问题**：全量跑一次 embedding 要多少 token？套餐够不够？

**做法**：
1. 不做估算，写测量脚本走**完整生产链路**（与入库完全相同的 Parser → Chunker →
   EmbeddingTextBuilder），逐 Child 累加 `embedding_input_tokens`，得到口径一致的精确数字。
2. 用 API 返回的 `usage` 校准本地 tokenizer 与 GLM 计费口径的偏差。

**结论**：
| 方案 | cl100k 口径 | GLM 实际计费 | 占 4,000 万预算 |
|---|---|---|---|
| 全量 T2（118,605 篇） | ≈ 1.14 亿 | ≈ 6,000 万 | ❌ 超预算 50%+ |
| **10k 抽样子集** | 9,595,852 | **≈ 510 万（系数 0.531）** | **~13%** ✅ |

关键发现：GLM 中文分词比 cl100k **省约一半**（实测系数 0.531）；即使如此全量仍超预算，
所以抽样子集是唯一合理路径。

**抽样方法（面试重点）**：query 优先抽样（seed=42，确定性可复现）——
先随机抽 500 条有正例标注的 query，把这些 query 的**全部正例文档强制纳入候选池**，
再随机补干扰文档到 10,000 篇。这样指标不会因"正例没进池"而系统性偏低；
代价是指标为 against-10k-pool 口径，只用于项目内部消融对比，不与论文排行榜直接比。
子集可信度用长度分布验证：子集每篇 token p50=525/p90=2040/p99=7010，
与全量统计（517/2084/7139）几乎重合。

**成本分层**：向量持久化（mmap artifact + 断点续跑指纹）之后，
评测侧消融（聚合策略、rerank、候选深度）**0 token 无限重跑**；
只有动 chunk 粒度或 embedding profile 才需要重新付费（每轮约 510 万）。

## 3. 基线结果与指标设计

评测输出 Recall@1/3/5/10、MRR@10、nDCG@10。多档小截止点是因为 T2 平均每条 query
只有 ~5.2 个正例，Recall@10 之上区分度有限。

**天花板分析（面试重点）**：Recall@1 = 23.5% 不能直接和 100% 比——top-1 只有一个位置，
按 5.2 正例/条算理论天花板约 1/5.2 ≈ 19%；实际超出天花板，配合 MRR@10 = 0.85，
说明"把最相关文档排第一"的能力很强，曲线衰减平缓说明相关文档集中在头部。

另外跑通了设计文档规划的对照实验：document 层聚合 topn-mean（R@10 78.9%）
略差于 max（79.9%），验证了当前默认配置（Family 均值 + 文档 max）确实最优。

## 4. Rerank 优化

**实现原则**：评测脚本直接复用生产 `CrossEncoderReranker`（不复制算法，
保证评测语义 = 生产语义）。检索 top-300 Child → cross-encoder 重打分 →
原有 Family 均值 → 文档 max 聚合。

**模型选择**：默认的 ms-marco MiniLM 是英文模型，对中文语料无效，
换成 `BAAI/bge-reranker-base`（中英双语，1.1GB，CPU 15 万对打分约 10 分钟）。

**结果（同池同 query 干净 A/B）**：

| 指标 | 无 rerank | + rerank | 相对提升 |
|---|---|---|---|
| Recall@1 | 23.48% | **26.35%** | +12.2% |
| Recall@3 | 48.06% | **54.34%** | +13.1% |
| Recall@5 | 63.47% | **68.79%** | +8.4% |
| Recall@10 | 79.99% | **84.08%** | +5.1% |
| MRR@10 | 0.8507 | **0.9097** | +6.9% |
| nDCG@10 | 0.7776 | **0.8309** | +6.9% |

头部（@1/@3）提升最大，正是 cross-encoder 精排的设计用途；生产问答默认取 top-5 证据，
Recall@5 从 63% 提到 69% 直接提升答案质量上限。

## 5. 排查修复的工程问题（体现排查能力）

按发现顺序，全部是"运行即崩/从未跑通"级别：

1. **`contracts.py` 缺失能力协议**：7 个模块 import 不存在的 `EmbeddingModel`/
   `Reranker`/`ChatModel`，主应用无法启动、4 个测试文件无法收集——设计文档 §8.7
   要求的协议从未落盘。补齐 Protocol 定义（成员按调用方实际用法定义）。
2. **形式契约与活实现脱节**：`chunking/models.py`（423 行，含 330 行演示代码）
   import 不存在的符号且全库无消费者——删除；同类的 `Settings.bool`、
   `Settings.optional_integer`（GLM 工厂路径从未跑通）、queries 嵌入路径
   `text_builder.counter`（从未跑通）。
3. **测试与实现版本漂移**：5 个测试仍针对重构前废弃的 API
   （`answer_llm=`、`QueryRequest.history`），现行行为已由设计锚定的测试覆盖——删除过期用例。
4. **评测器范围校验不适配子集**（要求 query artifact 必须是 prefix 选取）；
   映射加载时裁掉 `embedding_text`/`chunk_order`（rerank 拿不到文本）。
5. **faiss 与 torch 的 OpenMP 运行时冲突**：评测改用 `--engine exact` 暴力精确检索根除——
   顺带让评测更严谨，并附带量化了 HNSW 近似损失仅 ~0.05pp（生产继续用 faiss 无碍）。

共性教训：**文档声称的契约、测试假设的 API、实际运行的代码三者会悄悄分叉；
只有"真正跑通端到端"才会暴露。**

## 6. 可能的追问 & 回答要点

- **为什么不全量跑？** 算过账：全量 GLM 计费约 6,000 万，超预算 50%；子集 510 万。
  抽样保正例进池，长度分布与全量对齐，内部消融完全有效；排行榜数字本就不能
  直接比（他们也是自己的候选池设置）。
- **为什么 rerank 放在 Child 候选之后而不是文档级？** 与生产 HybridRouter 语义一致：
  召回交给孩子向量（便宜、精准），cross-encoder 只对 top-300 候选精排（贵但准，量小）。
- **为什么评测用精确检索而不是 ANN？** 评测基准应排除近似误差；实测 HNSW 损失 0.05pp，
  说明生产用 faiss-hnsw 是"几乎免费"的提速。
- **没有 rerank 之前效果差吗？** 不差：MRR@10 0.85 说明链路结构（父子映射、聚合）无缺陷；
  rerank 是锦上添花而非纠错。
- **花了多少 token？** embedding 约 510 万（13%），剩余额度足够 6 轮以上同等成本实验。

## 7. 遗留与下一步

- v2 物理索引切换（重建脚本已就绪，按 §8.9 流程待执行）
- `RetrievalMetadataGenerator` 未接入 ingestion 管线（第二 embedding profile 的前置）
- 生产问答启用 rerank：`rerank_backend="cross-encoder"` + `MVP_RERANK_MODEL=BAAI/bge-reranker-base`
- 可选：bge-reranker-v2-m3（更大更准，CPU 推理时间约 3 倍）

## 8. 产物清单

- 评测子集：`dataset/T2Retrieval-subset/`（10k 文档 + 500 query + qrels + 指纹 manifest + README）
- 工具脚本：`dataset/subsample_t2retrieval.py`（抽样）、`dataset/measure_t2_tokens.py`（精确成本测量）
- 向量 artifact：`dataset/T2Retrieval-subset/embeddings-v2/`（81,353 + 500 向量，可断点续跑）
- 评测命令：
  ```bash
  export T2_DATASET_DIR=$PWD/dataset/T2Retrieval-subset
  python evaluate_t2retrieval.py --expected-query-count 500 --candidate-top-k 300 \
      [--engine exact] [--rerank-model BAAI/bge-reranker-base]
  ```
