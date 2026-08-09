# 正式 RAG 架构：当前实现与 RAGFlow 源码对照及修复方案

> 状态：讨论稿，尚未批准实施。已确认：（1）Parent/Child 采用 RAGFlow 核心语义，Parent/Mother 只存上下文且不参与普通召回，Child 负责召回；（2）所有长文本必须持续分段并保证内容覆盖，不能因超限丢弃正文，现有 Parent 512/768、Child 128/192 token 作为首轮 baseline；（3）Family 按 Parent/Mother 聚合命中的 Child、取 Child 均值作为分数并返回 Parent/Mother，Child 明细仅作为溯源增强保留；（4）接受新建 v2 物理索引并全量重新 embedding，现有 embedding 套餐可承担该成本。本文先分析正式项目，再讨论 T2Retrieval；测试放在最后。

## 0. 本文范围

核心问题不是 `embed_t2retrieval.py` 如何避免 400，而是正式项目的 RAG 设计是否合理：

```text
Parsing
  → Chunking
  → Metadata / Embedding text
  → Indexing / Storage
  → Retrieval / Rerank
  → Parent context
  → Citation
```

T2Retrieval 应当用来评估这条正式链路的核心能力，而不能反过来决定正式架构。

本文每个模块都按以下顺序展开：

1. 当前项目如何实现；
2. RAGFlow 如何实现；
3. 两者差异和实际风险；
4. 建议保留、修改或不参考什么。

## 1. 证据范围与两条正式链路

### 1.1 RAGFlow 哪条源码才是本文依据

本地 `ragflow-main/pyproject.toml:1-4` 标识版本为 **0.26.4**；该目录没有 `.git` 元数据，因此本文不能声称知道具体 commit。

更重要的是，RAGFlow 同时保留了新旧两套 task executor。`ragflow-main/rag/svr/task_executor.py:1753-1771` 的默认值是 `TE_RUN_MODE=0`，含义如下：

| `TE_RUN_MODE` | 实际路径 | 本文证据地位 |
|---|---|---|
| `0`（默认） | `TaskManager.run_refactored_task()` | **生产主证据** |
| `1` | 旧实现执行 + 新实现 dry-run 对比 | 迁移验证路径 |
| 其他值 | `do_handle_task()` 旧实现 | legacy/fallback 对照证据 |

默认调用继续进入：

```text
task_executor.py
  → task_executor_refactor/task_manager.py:58-106
  → task_executor_refactor/task_handler.py:548-648
  → ChunkService.build_chunks()
  → EmbeddingService.embed_chunks()
  → ChunkService.insert_chunks()
```

因此，本文后续以 `task_executor_refactor/` 为 embedding、插入和编排的主证据；旧 `task_executor.py:712-762,1291-1331` 只用于确认新旧行为是否延续。Parser/Chunker 的底层算法仍由 refactor 的 `chunk_builder.py` 调用 `rag/app/*` 与 `rag/nlp/*`，所以这些文件仍是 chunk 算法的直接证据。

### 1.2 当前项目正式链路

当前生产调用链位于 `backend/src/apps/services/ingestion_pipeline.py:21-118`：

```text
file_path
  → ParserFactory(md/pdf)
  → ParseResultBlock
  → MarkdownChunker
      → Parent + Child
  → RetrievalMetadataGenerator
  → EmbeddingIndexer
      → Parent 和 Child 全部 embedding
  → ElasticsearchStore
  → HybridRouter
      → vector + BM25 + optional rerank
      → Child 命中后展开 Parent
  → QAService
      → top-k + 每块前 2500 字符
  → CitationService
```

### 1.3 RAGFlow 默认正式链路

对应主链位于：

- parser/chunker 路由：`ragflow-main/rag/app/naive.py:950-1316`
- chunk 文档构造：`ragflow-main/rag/nlp/__init__.py:355-496`
- 默认 task 编排：`ragflow-main/rag/svr/task_executor.py:1753-1767`
- refactor 主流程：`ragflow-main/rag/svr/task_executor_refactor/task_handler.py:548-648`
- embedding：`ragflow-main/rag/svr/task_executor_refactor/embedding_service.py:58-117`
- embedding 文本和标题加权：`ragflow-main/rag/svr/task_executor_refactor/embedding_utils.py:53-83,161-201`
- mother/child 插入：`ragflow-main/rag/svr/task_executor_refactor/chunk_service.py:239-305`
- 检索：`ragflow-main/rag/nlp/search.py:134-245,549-746`
- children 回溯 mother：`ragflow-main/rag/nlp/search.py:903-957`
- citation 匹配：`ragflow-main/rag/nlp/search.py:251-328`

```text
file/binary
  → 按文件类型和 parser 配置解析
  → sections / tables / positions
  → token-aware chunk merge
  → 可选 children delimiter
      → Child 保存 mom_with_weight
  → 关键词/问题元数据
  → chunk 内容向量 + 文档名向量加权
  → document store
      → Child available
      → Mother available_int=0，仅存上下文
  → term + vector hybrid retrieval
  → optional rerank
  → children 按 mom_id 聚合并展开 Mother
  → chunk/doc/position 引用信息
```

### 1.4 总体对照：现状、RAGFlow、应修改什么

| 模块 | 当前正式实现 | RAGFlow 0.26.4 默认实现 | 当前设计问题 | 本项目拟修改（待批准） | 取舍 |
|---|---|---|---|---|---|
| Parsing | 只路由 Markdown、PDF；统一 `ParseResultBlock` | 按格式选择 parser，保留 section/table/page/position/image | Text/HTML benchmark 与内存文本无法复用正式入口；PDF 版面能力弱 | 保留当前强类型 block；增加 Text/HTML 与 `ParseSource`，暂不复制 OCR/provider 矩阵 | 适配 |
| Chunking | Parent 512/768、Child 128/192；部分超长 block/code/table 可越过 max | 普通路径用 token budget 合并，超长 atom 继续按字符二分；custom delimiter 路径仍有绕过上限的缺陷 | 配置不是硬约束；最终只能靠 embedding adapter 截断，正文尾部会丢失 | **已确认：引入全出口严格 splitter，任何可 embedding Child 都不得超限且正文不得丢弃；现有粒度作为首轮 baseline；adapter 截断只作告警/兜底** | 参考并补齐缺口 |
| Parent/Child | Parent、Child 全部 embedding 且同时召回 | Child 召回；Mother 存储但 `available_int=0`；按 `mom_id` 展开 | 双粒度竞争 top-k、重复成本、family 分数含义混乱 | **已确认：Parent 仅作上下文记录，Child 唯一进入 KNN/BM25；保留 parent_id/child_ids** | 采用核心语义 |
| Metadata/Embedding | `doc_name + section + (questions or content)` | 默认 refactor 同样优先 `question_kwd`，title 单独 embedding 后加权 | 有 questions 时正文完全消失；把溯源字段和召回特征混为一谈 | 正文永不被问题替换；溯源只存 payload；标题/section/questions 作为可评测 profile | 不照抄替换正文 |
| Index/Storage | `VectorRecord` 假设每条记录有向量 | Mother 与 Child 同库，Mother `available_int=0` 且只留必要字段 | 无法表达“存储但不可召回”的 Parent | 引入 `retrieval_eligible`/record role；Parent 无向量；所有检索入口强制过滤 | 简化采用 |
| Retrieval | KNN + BM25 线性融合；family max；Child 命中后展开 Parent | hybrid/rerank；同 Mother 的已命中 Child 分数取均值并返回 Mother | Parent 修复前候选池污染；当前 max 与目标语义不一致 | 采用 RAGFlow 的 Family 均值聚合与返回 Parent/Mother 语义；保留分项 score trace | 采用核心语义 |
| QA Context | 取前 top-k，每个展开 Parent 只保留前 2500 字符 | `kb_prompt()` 按排序放入完整 chunk；首个使累计量越界的 chunk 仍被保留，只丢更后的 chunk，故不是严格预算；position 主要用于 reference | 两边都没有以命中 Child 为锚组织 Parent 上下文 | 本项目新增 matched-Child 锚定 Parent window，再按 chat 模型 token budget 选择证据 | 针对两边局限新增，不是照抄 |
| Citation | LLM `[n]` 按 evidence 顺序绑定；展开后只引用 Parent | 对答案句子二次 embedding/词法匹配，再插引用 | 丢失真正命中的 Child 证据 | 保留稳定的 `[n]` 协议，同时保存 context Parent 与 matched Child 两层 provenance | 保留并增强 |
| T2/测试 | 数据集脚本自行拼接 title/text、直接一文一向量 | 不作为 RAGFlow 正式链路证据 | 没测到正式 parser/chunker/index text；不能评价本项目设计 | 正式架构批准并实现后，adapter 只做数据映射并复用生产组件 | 最后处理 |

这张表是本文的结论索引，后续章节负责给出源码依据、边界和具体修改位置。它不是编码授权。

### 1.5 到底参考 RAGFlow 哪些、没有参考哪些

| 类别 | RAGFlow 机制 | 本项目态度 | 原因 |
|---|---|---|---|
| 直接参考核心语义 | 超长 unit 继续拆到 token budget 内 | 采用并覆盖其 custom delimiter 例外 | 解决长文不丢失与模型输入合法性 |
| 直接参考核心语义 | Child 召回、Mother 不参与普通检索、命中后展开 | 采用，字段沿用本项目 parent/child 命名 | 避免双粒度竞争和重复 embedding |
| 适配 | 多格式 parser 路由 | 只实现 Text/HTML + 现有 Markdown/PDF | 补正式入口，不复制企业格式矩阵 |
| 适配 | 文档标题向量与内容向量加权 | 先作为实验对照，不立即成为默认 | 额外 API 成本，收益需评测 |
| 不照抄 | `question_kwd` 替换正文 | 拒绝 | 生成问题质量差会让正文语义消失 |
| 不照抄 | custom delimiter 绕过 chunk token max | 拒绝 | 与严格输入上限冲突 |
| 不照抄 | 答案句子二次 embedding 后自动插 citation | 暂不采用 | 当前 `[n]` 更确定；先补 matched Child provenance |
| 不复制企业设施 | 多租户、多存储后端、MinIO、计费、任务队列、OCR/provider 矩阵 | 不采用 | 不解决当前个人项目的核心算法问题 |
| 保留本项目特点 | `ParseResultBlock`、section_path、source_span、显式 `[n]` | 保留并增强 | 强类型与溯源能力比机械照抄更适合本项目 |

所以目标不是“把项目改成缩小版 RAGFlow”，而是把 RAGFlow 已验证的长文切分和 mother/child 检索语义抽出来，再保留本项目更清晰的 block/provenance 契约。

## 2. Parsing：格式解析与位置保留

### 2.1 当前实现

`backend/src/parsing/parser_factory.py:9-21` 只路由：

```text
md  → MarkdownParserIt
pdf → PdfParser
```

统一输出 `ParseResultBlock`，字段包括：

```text
text / block_type / page_no / bbox
section_path / order / source_span / metadata
```

优点：

- block 类型和 section path 明确；
- Markdown 可保留 heading、paragraph、code、table；
- Citation 已能使用 section/page；
- Parser 与 Chunker 有清晰边界。

问题：

- 不支持 `.txt`、内存 text、HTML fragment；
- PDF 只用 `pypdf.extract_text()` 并按行生成 paragraph，版面和表格能力较弱；
- Parser API 假设输入是 `file_path`，不适合数据库字段或 benchmark 内存文本。

### 2.2 RAGFlow 实现

`ragflow-main/rag/app/naive.py:1004-1205` 按扩展名调用不同 parser：

- Txt/代码文件：`TxtParser`；
- Markdown：Markdown parser；
- HTML：`HtmlParser`；
- PDF：PlainText/DeepDOC/MinerU/Docling/OCR 等可选；
- Excel/Docx/JSON/EPUB 等另有专用路径。

PDF parser 还可产生 table、image、page position；`tokenize_chunks()` 在 `ragflow-main/rag/nlp/__init__.py:391-416` 将位置写入 chunk。

RAGFlow 解决的是多格式、复杂版面和产品配置问题。个人项目不需要复制它的所有 parser provider、OCR 模型路由和嵌入文件递归处理。

### 2.3 应如何取舍

| 设计点 | 当前项目 | RAGFlow | 建议 |
|---|---|---|---|
| 统一 block 模型 | 有 | 更偏松散 dict/sections | 保留当前 `ParseResultBlock` |
| Markdown section_path | 有 | 有标题 token，但接口不完全相同 | 保留当前设计 |
| PDF 多 parser | 只有 pypdf | 多种企业选项 | 暂不复制，只承认当前 PDF 能力边界 |
| Text parser | 无 | 有 | 增加正式 `TextParser` |
| HTML fragment | 无 | 有 HTML parser | 增加轻量 HTML normalizer/parser |
| 内存输入 | 不自然 | binary 为主 | 重构输入模型以支持 path/text，不照抄 binary 编排 |

### 2.4 拟修改方向（待批准）

在 `parsing/` 增加统一 `ParseSource`，允许 `file_path` 或 `text`；增加 TextParser 和 HTML fragment normalizer。继续输出现有 block 模型，使 Markdown/PDF/Text/HTML 进入相同 Chunker。

这不是为了 T2 临时加 parser，而是补正式项目输入能力；T2 后续只是调用它。

## 3. Chunking：长文的核心设计

### 3.1 当前实现

配置位于 `backend/src/chunking/chunk_config.py`：

```text
Parent target/max = 512/768 token
Child  target/max = 128/192 token
preserve code/table = True
```

这里必须把仓库里的四类数字彻底分开。配置证据在 `backend/config.yaml:13-19`，GLM 输入上限映射在 `backend/src/infrastructure/embedding_factory.py:16-22`：

| 数字 | 实际含义 | 是否参与相减 |
|---|---|---|
| 2048 | `embedding-3` 的默认**输出向量维度**（配置注释） | 不与 token 数相减 |
| 1024 | 当前 `backend/config.yaml` 选择的实际**输出向量维度** | 所有 chunk 都输出同维向量，才能计算余弦相似度 |
| 3072 token | 当前 GLM adapter 对单条输入设置的保护上限 | 是 embedding 输入合法性边界，不是 chunk 目标大小 |
| Parent 512/768、Child 128/192 token | 当前 chunk target/max 文本预算 | 决定切分粒度，不决定向量维度 |

所以 `2048 - 768` 或 `2048 - 192` 没有算法意义。每个 Child 都独立映射为同一个 1024 维向量，余弦相似度可以正常计算；更小的 chunk 只是使用较短的文本表达更局部的语义。真正的“丢失”发生在源文档没有被后续 Child 覆盖、最终 embedding 文本被 adapter 截断，或命中的内容在 QA prompt 组装时被再次裁掉。

因此评判 chunk 设计需要同时检查两个独立条件：

1. **覆盖完整性**：长文必须被拆成多个 Child，所有正文都进入某个 Child；
2. **单块合法性**：每个最终 embedding 文本必须小于模型输入上限。

“只取前 3072 token”只满足第二个条件的一部分，并破坏第一个条件，所以不能作为长文方案。

`BlockMergeStrategy` 两阶段处理：

1. block 按 heading/target token 合并成 Parent；
2. Parent 内部按 block/句子拆成 Child；
3. `MarkdownChunker` 同时输出 Parent 与 Child，并建立 parent_id/child_ids。

设计意图是 Small-to-Big：Child 精准召回，Parent 提供上下文。这个方向本身成立。

但当前“硬上限”并不严格：

- `block_merge.py:258-268`：超长 block 被完整放进 Parent；
- `block_merge.py:340-349`：code/table 整体作为 Child；
- `block_merge.py:351-362,408-415`：无句子边界的长段整体作为 Child。

所以 768/192 在一些真实输入上只是配置说明，不是算法保证。

### 3.2 RAGFlow 实现

RAGFlow 默认 `naive_merge()` 位于 `ragflow-main/rag/nlp/__init__.py:1285-1360`：

1. 按配置 delimiter 切 section；
2. 小片段按 token 预算合并；
3. 单个片段超限时调用 `_split_oversized_unit()`；
4. 优先按空白 atom 拆分；
5. 单个 atom 仍超限时 `_split_atom_by_token_budget()` 用字符二分查找满足 token 预算的边界；
6. 可配置 overlap，且 overlap + 新块超过预算时放弃 overlap。

这套实现解决了当前项目最关键的缺口：无标点中文、长 URL、连续字符也不能突破 token 上限。

RAGFlow 也有缺陷：`naive_merge()` 的 custom delimiter 分支在 `ragflow-main/rag/nlp/__init__.py:1307-1326` 明确不再应用 `chunk_token_num`。因此不能整段复制后声称所有路径都有硬上限。

### 3.3 RAGFlow Parent/Child 实际做法

配置了 children delimiter 时，`tokenize_chunks()`：

- 把原 chunk 保存在 `mom_with_weight`；
- 输出可检索的 Child。

默认 refactor 插入阶段 `task_executor_refactor/chunk_service.py:239-305`：

- 为 Mother 生成 `mom_id`；
- Mother 写入 document store；
- Mother 设置 `available_int=0`，不进入正常召回；
- Child 保存 `mom_id`。

检索后 `search.py:903-957`：

- 按 `mom_id` 收集 Child；
- 读取 Mother 内容；
- 同 Mother 的 Child 分数取均值；
- 返回 Mother 上下文；
- Mother 缺失时退回 Child。

这与当前“Parent 和 Child 都生成向量并进入候选池”有实质区别。

### 3.4 对照与决策

| 问题 | 当前项目 | RAGFlow | 建议 |
|---|---|---|---|
| 长文完整覆盖 | 无标点/特殊块可能超限并最终截断 | 普通路径严格拆分 | **已确认：参考其 oversized splitter，并修复 custom delimiter 绕过问题；超限只能继续分段，不能丢弃正文** |
| Parent 是否召回 | 是 | Mother `available_int=0` | **已确认：Parent 只存上下文，不参与普通召回** |
| Child 是否召回 | 是 | 是 | 保留 |
| Parent/Child 关系 | parent_id/child_ids | mom_id/mom_with_weight | 保留当前更清晰的数据结构，参考其可用性策略 |
| family 分数 | 当前取最高成员分 | RAGFlow 取已命中 Child 均值 | 已确认采用 RAGFlow 的均值语义；max 仅保留为评测对照，不作为目标默认值 |
| overlap | 不使用 | flat chunk 可配置 | Parent/Child 模式可先不加；单层模式再考虑 |

### 3.5 已确认的 Chunking/Parent-Child 方向（尚未授权实现）

已确认保留 Parent/Child，并采用 RAGFlow 的核心召回语义，具体改成：

```text
Parent：存完整上下文和溯源，不生成向量，不进入召回
Child：严格 token 上限，生成向量，负责召回
检索：按 parent_id 聚合 Child，再展开 Parent
```

同时提取严格 `split_oversized_unit()`，应用到 paragraph、code、table 和自定义 delimiter 的所有最终出口。

修改后必须满足以下算法不变量，而不是只看几个示例：

- 除明确记录的空白/格式 normalization 外，所有 source span 都被一个或多个 Child 覆盖；
- 任何 Child 的**最终 embedding text**（正文加上获批的 title/section/question 特征）都不超过 profile 的 token budget；
- overlap 只能重复内容，不能造成正文缺口；
- code/table 不能以“保持完整”为由越过硬上限：先尝试结构边界，仍超限再走安全硬切，并保留 continuation/provenance；
- model adapter 若仍发生截断，必须记录 chunk_id、原 token 数与截断数；它是异常兜底，不是正常数据路径。

参数 512/768、128/192 是否合理不能从模型 3072 上限推导。已确认先把它们作为首轮 baseline，再通过检索实验调整；向量维度（当前 1024）与 chunk token 数无相减关系。

## 4. Metadata 与 Embedding 文本

### 4.1 当前实现

`backend/src/indexing/retrieval_metadata.py` 为每个 chunk 生成：

```text
important_kwd / important_tks
question_kwd / question_tks
title_tks
retrieval_metadata_trace
```

生成 prompt 同时使用 doc_name、section_path 和 content。没有 LLM 时有规则 fallback。

`backend/src/indexing/embedding_indexer.py:77-82` 的 embedding 输入是：

```text
doc_name
section_path
question_kwd 或 content
```

最大风险是 `questions or content`：一旦生成问题存在，原始正文完全不参与向量。

### 4.2 RAGFlow 实现

默认 refactor 路径由 `task_executor_refactor/embedding_service.py:58-117` 与
`task_executor_refactor/embedding_utils.py:53-83,161-201` 共同完成：

1. 文档名单独 embedding；
2. chunk 使用 `question_kwd`，没有问题时才用 `content_with_weight`；
3. 内容在发送前按 `embedding_model.max_length - 10` 截断；
4. 最终 `title_weight * title_vector + (1-title_weight) * content_vector`，默认 title 权重约 0.1。

RAGFlow 的优点是标题影响可控，不必把标题文本硬拼进每个 chunk。缺点是同样采用 `question_kwd or content`，问题生成质量差时正文语义会被替换；这不应因为是 RAGFlow 就直接照搬。

### 4.3 溯源与 embedding 必须分开

RAGFlow 并非不考虑溯源。它把以下字段存入 chunk/document store 并在检索结果返回：

```text
chunk_id / doc_id / docnm_kwd / kb_id
position_int（最终 reference 通常投影为 positions）/ mom_id / image_id
```

当前项目保存：

```text
Chunk 嵌套 meta：source_span / source_block_ids / block_types
ES payload：chunk_id / doc_id / doc_name / section_path / page_no
            parent_id / child_ids / chunk_order
RetrievedChunk/Citation：没有 source_span
```

因此 `doc_name + section_path` 是否进入 embedding 是召回特征问题，不是能否 citation 的条件。但必须纠正现状判断：当前 `source_span` 只停留在 chunk 的嵌套 `meta`，`retrieval_metadata_fields()` 没有把它写入 ES payload，`RetrievedChunk` 和 `Citation` 也没有该字段，所以现在只能按 section/page 粗粒度溯源，不能声称已实现 source-span citation。

此外，Markdown parser 的部分 `start_char/end_char` 是 block 内局部偏移；若一个超长 block 被拆成多个 Child，不能让所有 Child 继续共用整块 span。严格 splitter 必须同步产生子片段相对 offset，parser/chunker 再把它换算为稳定的行列或绝对字符范围。

需要明确两套不能混用的坐标：

| 坐标 | 用途 | 建议字段 |
|---|---|---|
| 原始 source 坐标 | 回到 Markdown/HTML/PDF 原文 | `source_span`，并带 `accuracy=exact/line_only/unavailable` |
| 规范化文本坐标 | 在 Parent 文本中定位 Child、构造 QA window | `parent_char_start/end`，必要时增加 token span |

Parser 对 HTML entity、标签删除、Markdown marker 清理等 normalization 应输出 segment map，将 normalized block 范围映射回 raw source；无法精确映射时必须降低 `accuracy`，不能伪造精确字符位置。Chunker 根据 fragment offset 计算 `parent_char_start/end`，QA window 只依赖这套 Parent 内坐标，Citation 再同时返回 normalized window 和可用的 raw-source span。

### 4.4 拟修改方向（待批准）

保留正文作为 embedding 的主输入，禁止元数据完全替换正文。比较三个方案：

| 方案 | 特点 | 建议 |
|---|---|---|
| `doc + section + content + questions` 拼接 | 简单，一次 API | 首个可解释 baseline |
| RAGFlow 标题向量与内容向量加权 | 权重可控，API 成本更高 | 作为对照实验 |
| question 替换 content | 依赖生成质量，正文丢失 | 不采用 |

Metadata 的 provenance 字段始终存 payload；是否进入 embedding 由实验配置控制。

因为 metadata 在 chunking 之后生成，最终输入超预算时必须有确定策略。建议的优先级是：

1. 正文 Child 使用已批准的 `content_budget`，**不裁正文**；
2. 每类可选特征预先设独立上限，例如 title、section、questions 各自限额；
3. 超出 `embedding_input_budget` 时，按 `questions → title → section` 的既定低优先级依次缩短/移除可选特征；
4. 如果只剩正文仍超限，说明 chunker 或 tokenizer 口径违反契约，拒绝入库并报告 chunk_id，而不是继续截正文；
5. infrastructure adapter 的截断只防止供应商请求失败，同时必须产生可观察告警；正式验收要求正常路径截断数为 0。

具体的特征限额和优先级仍需你批准；但“正文不可被动态 metadata 挤掉”应作为不可变约束。

完整 provenance 字段必须按以下链路逐层透传并校验，任何一层丢字段都不能算完成：

```text
Parser block source span
  → Child fragment offset/span
  → ChunkRecord top-level provenance
  → ES payload
  → RetrievedChunk (matched Child + context Parent)
  → QA evidence actually sent to LLM
  → Citation
```

## 5. Indexing 与 Storage

### 5.1 当前实现

`EmbeddingIndexer.index()` 对 Chunker 输出的全部记录统一 embedding，所以 Parent 和 Child 都成为 `VectorRecord`。ES 按 `q_{dimension}_vec` 建 dense vector，并保存模型、维度、父子字段。

优点：

- 数据结构直接；
- 文档重新入库用 `delete_by_doc_id + upsert`；
- 支持多 embedding dimension 字段；
- payload 已包含较完整溯源。

问题：

- Parent/Child 重复 embedding；
- 两种粒度竞争 candidate top-k；
- `SearchStore.upsert()` 假设每条记录都有向量，不能表达“只存上下文的 Parent”。

### 5.2 RAGFlow 实现

RAGFlow 默认 refactor 的 `ChunkService.insert_chunks()` 先调用 `_create_mother_chunks()`，把 mother 设置为 `available_int=0` 并裁剪到必要字段，再插入 main chunks（`task_executor_refactor/chunk_service.py:239-305`）。因此可召回 Child 与不可召回 Mother 可以共存于 document store，而 Mother 不作为正常搜索候选。向量字段只服务于 main chunks。

RAGFlow 还支持 ES/Infinity/OceanBase 等后端，这是企业扩展需求，当前项目不必复制多后端抽象。

### 5.3 拟修改方向（待批准）

- 保留 Elasticsearch 单后端；
- 将存储记录区分为“可检索向量 chunk”和“不可检索 context Parent”；
- KNN/BM25 都过滤 `retrieval_eligible=true`；
- Parent 只保存 content 与 provenance；
- 保留当前模型/维度过滤；
- embedding text builder 从 `EmbeddingIndexer` 中提取，供正式入库和后续 benchmark 共用。

## 6. Retrieval 与 Parent 展开

### 6.1 当前实现

`HybridRouter`：

1. ES KNN 取 `candidate_top_k`；
2. BM25 取 `candidate_top_k`；
3. `HybridFusion` 按配置权重线性融合；
4. 可选 rule/cross-encoder rerank；
5. similarity threshold；
6. `_expand_parent_context()` 按 family 去重；
7. Child 命中后读取 Parent，并保留 `matched_child_id`。

优点是结构清晰，已经实现 Small-to-Big 闭环。核心问题仍是展开之前 Parent 和 Child 都在候选池。

另一个需要验证的问题是分数尺度：BM25 在 `ElasticsearchStore.keyword_search()` 中按本次结果最大值归一化，KNN 分数裁到 `[0,1]`，然后直接线性相加。这个方案简单，但分数会随候选集合变化。

### 6.2 RAGFlow 实现

`ragflow-main/rag/nlp/search.py`：

- store 侧同时构造 term match、dense match 和 weighted fusion；
- ES 路径再取得纯 KNN cosine；
- 本地以 term/vector 权重重新组合；
- 可选外部 reranker；
- threshold 后分页；
- 返回 chunk、doc、position 和多路分数；
- children 模式再通过 `mom_id` 展开 Mother。

RAGFlow 更复杂是因为支持多个存储后端、分页、rank feature 和多租户。当前项目不需要复制这些编排。

### 6.3 值得参考与不参考

| RAGFlow 机制 | 是否参考 | 原因 |
|---|---|---|
| 候选池大于最终 top-k | 参考 | rerank/parent 聚合需要足够候选 |
| Mother 不参与普通召回 | 参考 | 消除 Parent/Child 重复竞争 |
| Mother 缺失回退 Child | 参考 | 保证结果不丢失 |
| term/vector 分数分别保留 | 已有，保留 | 有利于可解释 trace |
| 多后端分页编排 | 不复制 | 当前项目不需要 |
| 同 Mother Child 分数取均值 | 采用 | 已确认 Family 排序参考 RAGFlow；max 仅作为评测对照 |

### 6.4 已确认的 Family 方向（尚未授权实现）

Family 参考 RAGFlow 的核心方案：

1. 普通候选池只包含 Child，Parent 不参与 KNN/BM25；
2. 按 `parent_id` 聚合同一 Parent 下本次实际命中的 Child；
3. Family 分数取这些已命中 Child 分数的算术均值；
4. 通过 ID lookup 读取并返回 Parent；Parent 缺失时回退对应 Child；
5. 最终按 Family 均值分数排序。

`max(child score)` 只保留为评测对照，不作为目标默认值。均值会受“本次召回到多少个 Child”影响，因此评测仍需记录候选深度和 contributing-child 数量。

RAGFlow 返回 Mother 时不会完整输出每个 contributing Child 的 id、score 和 span。本项目不改变它的 Family 排序/返回语义，但为了后续 matched-Child 锚定 QA window 和 Citation，在 Parent 结果旁保留按分数排序的辅助 `matched_children[]`（id、score、snippet、span），并给出 `primary_matched_child_id`。这份列表只表达溯源，不再参与其他 Family 计分；Family score 与 evidence provenance 的职责保持分离。

## 7. QA Context、Citation 与回答溯源

### 7.1 当前 QA prompt 仍会丢掉已命中的内容

`backend/src/apps/services/qa_service.py:63-68,85-110` 有第二层截断：

1. 只取检索结果前 `context_top_k=5`；
2. 每条展开后的 Parent 只取前 `context_chars_per_chunk=2500` 字符；
3. 截断按字符而不是 chat model token budget；
4. 没有使用 `matched_child_id` 定位命中位置。

因此即使 Child 正确命中了 Parent 后部，Parent 展开后也可能只把开头 2500 字符交给 LLM。这个问题证明长文链路不能在 embedding 处结束，必须一直检查到：

```text
retrieved Child → expanded Parent → prompt window → answer → citation
```

### 7.2 当前 Citation 实现

`CitationService` 解析答案中的 `[1]`，按 evidence 顺序绑定：

```text
chunk_id / doc_id / snippet / page_no / section_path
```

优点是确定、简单、便于测试。问题是 Child 展开成 Parent 后，citation 使用展开后的 Parent chunk_id/snippet；虽然检索结果有 `matched_child_id`，Citation model 没保存它，精确命中证据丢了一层。

### 7.3 RAGFlow 实现

RAGFlow 的 QA prompt 也不是本项目要直接复制的答案。`ragflow-main/rag/prompts/generator.py:139-173` 的 `kb_prompt()` 按检索排序计算完整 chunk token；循环先递增 `chunks_num`，再判断是否越界，随后又按 `kbinfos["chunks"][:chunks_num]` 重建 prompt。因此**首个使累计 token 越界的 chunk 实际仍被保留，只丢弃它后面的 chunk**，并不能保证严格不超预算；空内容还可能让计数与原列表切片错位。它也没有用 `position` 在 Mother 内生成命中锚定窗口。`chunks_format()` 在同文件 `41-66` 将 `positions` 放入返回 reference，主要服务结果展示/引用。

RAGFlow `insert_citations()` 会：

1. 将答案按句子切分；
2. 对答案片段 embedding；
3. 将答案片段与候选 chunk 做词法+向量相似度；
4. 自动插入 `[ID:n]`；
5. reference 中保留检索 chunk、doc 和 position 信息。

它更自动，但额外调用 embedding、阈值启发式较多，引用稳定性和成本更难控制。

### 7.4 拟修改方向（待批准）

不直接复制 RAGFlow 的答案二次 embedding。保留当前 LLM 显式 `[n]` 方案，但增强 provenance：

```text
context_chunk_id        # 提供给 LLM 的 Parent
matched_children[]      # 实际召回的 Child 列表及 score/span/snippet
primary_matched_child_id
doc_id/doc_name
page_no/section_path/source_span
```

这样既保留当前确定性，也补齐 Small-to-Big 的精确证据。

QA prompt 不再做“Parent 从开头截 2500 字符”，而是：

1. 根据 `matched_children[]` 取得各 Child 在 Parent 内的字符/token span；
2. 以每个 Child 为锚，优先保留完整 Child，再向前后扩展 Parent 邻近上下文；重叠窗口合并；
3. 用 QA 模型 tokenizer 按总 prompt budget 分配多个 evidence window，而不是每块固定字符数；
4. 每个 evidence 记录 `prompt_span` 和实际送入 LLM 的文本；
5. Citation 的 snippet 必须来自实际 prompt window，并列出该窗口覆盖的一个或多个 matched Child 原始 span；
6. Parent 或 span 缺失时回退 Child 全文，不能回退到 Parent 开头。

这里的 token budget 不能复用当前 `MVP_QA_LLM_MAX_TOKENS`：它在 `OpenAIChat.complete()` 中作为 API 的 `max_tokens`，含义是**生成输出上限**，不是模型总上下文。目标配置和公式应为：

```text
qa_context_limit_tokens     # 模型总 context，必须按模型显式配置/注册，禁止猜默认值
qa_completion_reserve       # 当前 max_tokens 的语义
qa_prompt_safety_tokens     # tokenizer/消息封装余量

evidence_budget = context_limit
                - completion_reserve
                - tokens(system prompt + question + message envelope)
                - safety_tokens
```

当前 `AnswerLLM = Callable[...]` 无法暴露模型能力，因此应改为 `AnswerModel`/`ChatModel` 协议，由 adapter 暴露 model name、context limit、completion reserve 和 token counter；测试 fake 也实现同一协议。若供应商没有精确 tokenizer，必须选择并声明保守 counter 和 safety margin，不能回退到字符数截断。

## 8. 按当前目录给出的正式项目修改方案

以下是“批准后应该改哪里、改成什么”的目录级计划。它只表达依赖与目标，不构成编码授权；不会为了兼容当前错误产物而保留双轨逻辑。

### 8.1 `backend/src/parsing/`

| 当前文件/行为 | RAGFlow 对应依据 | 拟修改文件 | 拟修改内容 | 验收边界 |
|---|---|---|---|---|
| `parser_factory.py` 只接受 md/pdf | `rag/app/naive.py` 按文件类型路由 parser | `models.py`、`parser_factory.py` | 引入 `ParseSource`（path/text、类型、名称）；factory 按 source type 路由，而不是要求所有输入先落盘 | Markdown/PDF 旧能力不下降；纯文本和 HTML fragment 可走同一正式入口 |
| 无 Text parser | RAGFlow `TxtParser` | 新增 `text_parser.py` | 纯文本按段落/空行产生 block，保留 char/line span | 不在 dataset 脚本里复制清洗逻辑 |
| 无 HTML parser | RAGFlow HTML parser | 新增 `html_parser.py` | 去 script/style，保留 heading/list/table/paragraph 结构；HTML entity 正规化 | 不是简单 `strip_tags + 拼 title`；输出统一 block |
| `markdown_parser_it.py` 多处 `start_char=0` 或 block-local end | RAGFlow position 以可返回的 page/offset 列表保存 | `markdown_parser_it.py`、`models.py` | 产生原始文档绝对 line/char span；normalization 输出 segment map 和 accuracy | 不把局部 offset 冒充文档绝对位置 |
| PDF 仅 pypdf 文本行 | RAGFlow 可选 DeepDOC/MinerU/Docling/OCR | `pdf_parser.py` 暂只补能力声明和稳定 span | 不引入企业级 parser provider 矩阵；把复杂版面/OCR 列为明确未支持 | 不虚称支持扫描件、复杂表格 |

### 8.2 `backend/src/chunking/`

| 当前文件/行为 | RAGFlow 对应依据 | 拟修改文件 | 拟修改内容 | 验收边界 |
|---|---|---|---|---|
| `block_merge.py` 的 oversized parent 原样保留 | `_split_oversized_unit()` | `block_merge.py` 或新增 `oversized_splitter.py` | 实现 sentence → whitespace atom → token-budget 字符二分；返回 fragment offset，并据此计算 `parent_char_start/end` 与 raw-source span | 无标点中文、长 URL、连续字符都不能越过 hard max；Child span 不共用整块 span |
| code/table 因 preserve 可无限长 | RAGFlow 普通超长单元仍会继续拆 | 同上 | preserve 改为“优先保留结构”，不是“允许超限”；记录 continuation 和源 span | 结构完整性不能压过模型输入合法性 |
| `ChunkConfig` 只定义正文预算 | RAGFlow 最终 embedding 仍留安全余量 | `chunk_config.py` | 区分 `content_budget` 与 `embedding_input_budget`；预算必须包含获批的 embedding 前缀/附加特征 | 不再让 chunk 合法但拼 metadata 后超限 |
| `MarkdownChunker` 输出 Parent+Child 混合列表，source_span 只在嵌套 meta | RAGFlow main child + unavailable Mother，并保留 position | `models.py`、`markdown_chunker.py` | 保留两种角色，显式加入 `retrieval_eligible`；把 Child 精确 span/provenance 提升为正式契约字段 | 每个 Child 有 parent_id 和独立 span；每个 Parent 的正文由其 Child 完整覆盖 |
| token counter 和模型 tokenizer 可能不一致 | RAGFlow 也存在通用 tokenizer 与模型上限的边界 | `token_counter.py` 与 embedding profile | 以所选 embedding tokenizer/已验证上限为最终预算口径；通用 tokenizer 只用于近似合并 | 进入 API 前再次校验，但不静默截断 |

### 8.3 `backend/src/indexing/`

| 当前文件/行为 | RAGFlow 对应依据 | 拟修改文件 | 拟修改内容 | 验收边界 |
|---|---|---|---|---|
| `_retrieval_text()` 用 questions 替换 content | `EmbeddingUtils._extract_content()` 也采用 question 优先 | 新增 `embedding_text_builder.py`，修改 `embedding_indexer.py` | `content` 永远是主语义；title/section/questions 由显式 profile 决定拼接或单独向量 | 任何 profile 都不得无提示删除正文 |
| `RetrievalMetadataGenerator` 同时承担召回增强字段，payload helper 未透传 source_span | RAGFlow keyword/question 是可选 post-process，position 单独保存 | `retrieval_metadata.py`、`retrieval/metadata_fields.py` | 输出区分 `provenance`、`retrieval_features`、生成 trace；显式序列化 source_span/source_block_ids | LLM 失败不影响正文入库；provenance 不因 embedding profile 改变 |
| 所有 chunk 统一 encode | RAGFlow mother 不在 main embedding docs 中 | `embedding_indexer.py` | 只对 `retrieval_eligible=true` 的 Child 构造文本与向量；Parent 走 context record 分支 | embedding 数量等于 Child 数，不是 Parent+Child 数 |
| adapter 截断隐藏上游失败 | RAGFlow `truncate(max_length-10)` 是防线但也可能丢正文 | `embedding_indexer.py` 与 infrastructure adapter 的职责边界 | indexer 在调用前验证；adapter 保留最后保护但发出可观察错误/告警 | 正常入库的 truncate_count 必须为 0 |

首轮建议只实现并评测 `content-only` 与 `doc + section + content + questions` 两个 profile；RAGFlow 的 title/content 双向量加权作为第二个实验分支，不先增加线上 API 成本。

### 8.4 `backend/src/contracts.py` 与 `backend/src/infrastructure/`

| 当前文件/行为 | RAGFlow 对应依据 | 拟修改文件 | 拟修改内容 | 验收边界 |
|---|---|---|---|---|
| `VectorRecord`/`SearchStore.upsert` 默认记录都有向量 | RAGFlow document store 同时保存 unavailable Mother 和 main chunks | `contracts.py` | 把“存储记录”和“向量记录”语义拆开，或允许 context record 无 vector；字段名由实现前接口评审确定 | 类型层明确表达 Parent 不可向量召回 |
| ES 索引中 Parent/Child 都可被 KNN/BM25 搜到 | `available_int=0` 过滤 Mother | `elasticsearch_store.py` | mapping 增加 `retrieval_eligible`/role；KNN 和 keyword query 都强制过滤 true；`query_by_ids` 不过滤以便回溯 Parent | Parent 只能 ID lookup，不能进入普通候选 |
| 文档重建 delete + upsert | RAGFlow 批量插入/状态管理更复杂 | `embedding_indexer.py`、`elasticsearch_store.py` | 保留单文档替换语义；不复制 MinIO、多租户、多后端、任务计费 | 个人项目保持简单且原子失败可见 |

### 8.5 `backend/src/retrieval/`

| 当前文件/行为 | RAGFlow 对应依据 | 拟修改文件 | 拟修改内容 | 验收边界 |
|---|---|---|---|---|
| 两个 retriever 接受用户 filter，但角色过滤不内建；`RetrievedChunk` 无 source_span/matched snippet | RAGFlow Mother `available_int=0` 且返回 position | `embedding_retriever.py`、`keyword_retriever.py`、`models.py`、store | `retrieval_eligible=true` 作为不可被覆盖的系统过滤；检索模型透传 Parent 与 `matched_children[]` 的 span/text/score | 任意查询配置都无法召回 Parent；溯源字段不在检索层丢失 |
| `_expand_parent_context()` 在 Parent/Child 混合候选后去重 | RAGFlow 先只召回 Child，再按 mom_id 聚合并返回 Mother | `hybrid_router.py` | family 输入只包含 Child；按 Parent 分组并取已命中 Child 均值，ID lookup Parent；缺失时回退 Child；Parent 结果附带仅用于溯源的 `matched_children[]` 和 primary child | 排序与返回语义采用 RAGFlow，同时不丢失后续 QA/Citation 所需证据 |
| family 固定取 max | RAGFlow 对已命中同 Mother Child 取 mean | `ranking.py` 或新增 family aggregator | 默认固定为 `mean`；`max` 只作为离线评测对照，不进入目标生产默认路径 | 已确认采用 RAGFlow 的 Family 计分语义 |
| BM25 与 KNN 归一后线性相加 | RAGFlow 保留多路分数并可 rerank | `hybrid_fusion.py` | 先保留现设计与完整 trace；只有正式评测证明问题后才替换为 RRF/校准方案 | 避免一次修改多个变量导致无法归因 |

### 8.6 `backend/src/citation/`

| 当前文件/行为 | RAGFlow 对应依据 | 拟修改文件 | 拟修改内容 | 验收边界 |
|---|---|---|---|---|
| `Citation` 只有展开后的 `chunk_id/snippet` | RAGFlow reference 保留 chunk/doc/position，但使用答案二次匹配 | `models.py` | 增加 `context_chunk_id/context_snippet/context_span` 与 `matched_children[]`，继续保留 doc/page/section | 用户既能看实际送给 LLM 的上下文，也能定位一个或多个原始命中证据 |
| `CitationService` 按 `[n]` 绑定 evidence | RAGFlow 对答案句子再做 embedding+lexical 匹配 | `citation_service.py` | 暂保留 `[n]` 的确定性协议；从实际 prompt window 覆盖的 matched children 生成双层 citation | 不引入额外 embedding 成本和阈值漂移 |

### 8.7 `backend/src/apps/services/`

`ingestion_pipeline.py` 仍是唯一正式编排入口，但阶段契约改为：

```text
ParseSource
  → parser.parse() -> ParseResultBlock[]
  → chunker.chunk() -> Parent(context) + Child(retrievable)
  → metadata generator -> provenance + optional retrieval features
  → embedding text builder -> only Child texts
  → indexer -> context records + vector records
```

`qa_service.py` 消费的 evidence 需要同时携带 context/matched 两层标识。新增独立的 context-window builder：对 `matched_children[]` 分别锚定 Parent 窗口、合并重叠窗口，再按 QA tokenizer 与总 prompt budget 选择 evidence。当前固定 `context_chars_per_chunk=2500` 的前缀截断应被删除；Citation 只能引用实际进入 prompt 的 evidence window，而不是事后猜测命中来源。

QA 能力和配置还需要同步修改：

| 文件 | 拟修改 |
|---|---|
| `backend/config.yaml` | 增加 QA `context_limit_tokens` 与 `prompt_safety_tokens`；现有 `llm.qa.max_tokens` 明确改名/解释为 completion reserve；删除 chars-per-chunk 设计 |
| `backend/src/config/env.py` | 增加对应 `MVP_QA_LLM_CONTEXT_TOKENS`、`MVP_QA_PROMPT_SAFETY_TOKENS` 映射 |
| `backend/src/contracts.py` | 定义 `ChatModel`/`AnswerModel` 能力协议与 token counter，而不是只传普通 Callable |
| `backend/src/infrastructure/openai_chat.py` | adapter 暴露 model/context/completion/token-count 能力；发送请求时仍只把 completion reserve 传给 `max_tokens` |
| `backend/src/apps/services/qa_service.py` | 根据模型能力计算 evidence budget，组装锚定窗口，并记录 budget trace（固定开销、证据 token、保留/丢弃窗口） |

模型总 context limit 的来源只能是显式项目配置或经过核对的 model registry；缺失时拒绝启用自动窗口预算，不能把 completion `max_tokens` 当总窗口。

### 8.8 实施依赖顺序

1. **Parsing 契约 + Chunking 不变量**：先保证内容完整覆盖和单块合法；
2. **Index/Store 角色语义**：Parent 存储不可召回、Child 唯一召回；
3. **Embedding text profile**：正文不被替换，最终输入预算闭环；
4. **Retrieval family 聚合 + matched-Child 锚定 QA window + Citation 双层 provenance**；
5. **最后才是 T2 adapter 和测试/评测脚本**。

如果你只批准其中一部分，只实现该部分；遇到依赖尚未批准就停下，不擅自补齐。

### 8.9 Schema 与旧产物迁移（架构批准后、T2 之前）

新增强制 `retrieval_eligible=true` 过滤后，旧 ES 文档没有该字段；如果直接切查询，它们会全部消失。这里不能以“兼容”为由保留错误双索引语义。已确认选择**新 index 全量重建**，用户现有 embedding 套餐可以承担重新 embedding 成本：

```text
rag-mvp-chunks-v2
  → 用新 Parser/Chunker/EmbeddingTextBuilder 重建全部文档
  → 校验 Parent/Child 数、向量数、provenance 与抽样检索
  → 修改 MVP_ELASTICSEARCH_INDEX/backend config 指向 v2，并用新 backend 代码重启服务
  → 失败时把“旧 backend 代码版本 + 旧物理 index 配置”成对恢复并重启
  → 稳定观察期结束后再决定是否删除旧物理 index
```

| 方案 | 优点 | 问题 | 本文建议 |
|---|---|---|---|
| 原 index backfill 字段 | 少算 embedding | 旧 chunk、旧向量文本和旧 span 都已错误，补字段不能修数据 | 不选 |
| 读时兼容“字段缺失视为 true” | 上线快 | Parent 仍会进入候选，延续核心错误 | 不选 |
| 新版本 index 全量重建 | 语义干净，可校验、可短期回退 | 需要重新解析和 embedding | **已确认采用，embedding 成本可接受** |

迁移清单必须固定 index schema version、chunk profile version、embedding profile hash 和模型/维度；任何一项变化都要求重建，不能把旧向量混入新索引。

本项目首轮明确选择**物理 index 配置切换**，不使用 alias：当前 `_ensure_index()` 以 `mapping[self.index_name]` 读取 mapping，对 alias 不安全。重建 runner 必须显式构造 `ElasticsearchStore(index_name="rag-mvp-chunks-v2")`，禁止复用生产默认 store；校验完成后才修改 `MVP_ELASTICSEARCH_INDEX`/YAML 并重启。

这里的“新应用”不是再创建一个产品或服务，而是指**修改后的 backend 代码版本**。建议的个人项目切换流程是：旧 backend + 旧 index 继续可用；在旁边构建并校验 v2；校验通过后，在无请求或低使用时段，用“新 backend + v2 配置”重启。若出现空召回、引用错误、schema 错误或明显指标回退，则恢复“旧 backend + 旧 index 配置”。因为新代码强制 `retrieval_eligible=true`，它读取旧 index 会得到空结果，所以不能只把新代码指回旧 index。

旧 index 的作用只是短期回退保险，不是永久兼容路径。对当前个人项目，本文建议 v2 切换并通过抽样问答、检索指标和日志检查后保留旧 index **7 天**；期间无问题再由用户明确确认删除。若 ES 空间紧张，可缩短观察期，但不得在 v2 刚切换时自动删除旧 index。若未来要做原子 alias swap，必须先单独修改并验证 store 的 alias-aware mapping 逻辑，不与本次架构修复混做。

## 9. T2Retrieval：正式设计确定后的评测适配

T2 目前不是首要修复对象。它的问题来自绕过正式链路。

### 9.1 本地数据实际分布与复杂度判断

对本地 `dataset/T2Retrieval/corpus.parquet` 的 118,605 条记录做了只读统计：将 title 与去除 HTML 标签后的正文拼接，使用项目当前 `cl100k_base` 口径计数。该口径不等同于供应商 embedding tokenizer，只用于判断数据规模和切分必要性。

| 指标 | 结果 |
|---|---:|
| 文本 token 中位数 / P90 / P95 / P99 | 517 / 2,084 / 3,395 / 7,139 |
| 最大 token 数 | 44,254 |
| 超过 Child max 192 token | 101,798（85.829%） |
| 超过 Parent max 768 token | 41,705（35.163%） |
| 超过当前 adapter 保护值 3072 token | 6,880（5.801%） |
| 含 HTML 标签 | 77,974（65.743%） |
| 含 table / code 或 pre | 2,877 / 1,306 |
| 超过 192 token 且缺少常见句末标点 | 878 |

结论：T2 的多数记录都需要切成多个 Child，且存在必须处理的 HTML、超长正文、表格、代码和无标点长段。它不需要 dataset 专用的复杂切分器，但不能继续“一文一向量”，也不能假设不存在超长输入。适配方式是复用正式 HTML/Text Parser、严格 splitter 和 EmbeddingTextBuilder；所有生产兜底仍然生效，dataset 目录不复制算法。

正式 Parser/Chunker/EmbeddingTextBuilder 稳定后，T2 adapter 应：

```text
T2 corpus row（HTML fragment）
  → 正式 HTML/Text Parser
  → 正式严格 Chunker 的 benchmark profile
  → 正式 EmbeddingTextBuilder
  → 正式 EmbeddingModel
  → chunk vectors
  → doc_id 聚合
  → qrels metrics
```

dataset 目录只负责读取 parquet、保存 benchmark artifact 和计算指标，不再实现文本清洗或切块算法。

已确认正式链路采用 RAGFlow 式 Parent/Child，因此 T2 的主评测也使用同一正式 Parent/Child 模式；使用同一 splitter 的 flat chunk 仅作为离线对照，不作为主实现。这样 benchmark 直接评价生产语义，同时仍能量化 Parent/Child 相对 flat baseline 的收益。

在写脚本前还必须固定 artifact 契约，避免再次出现“一文一向量”和“chunk 向量”语义混用：

| 项目 | 约束 |
|---|---|
| `doc_id` | 直接来自 corpus 主键，稳定且唯一 |
| `chunk_id` | 由 `doc_id + chunk profile version + source span/order` 稳定生成；断点续跑不得改变 |
| mapping | 单独保存 `chunk_id → doc_id → parent_id/source_span` |
| vector artifact | 明确 schema version、model、dimension、dtype、shape、profile hash |
| qrels | 保持 query→doc relevance；评测前不能错误地拿 doc_id 当 chunk_id |
| chunk→doc score | 明确 `max`、top-n mean 或其他聚合；必须与 Parent family 聚合区分 |
| resume | checkpoint 同时记录输入指纹、最后完成 chunk_id、向量行数；三者不一致即拒绝续写 |
| 对照组 | production Parent/Child 为主；同 splitter flat baseline 与获批的 embedding text profiles 为离线对照 |

T2 的目标是评估正式组件和已声明的 profile，而不是让 dataset 脚本形成第二套 RAG 实现。

## 10. 测试与评测（最后执行）

测试必须验证上面已经批准的设计，而不是先写测试替代设计讨论。

### 10.1 Production unit tests

- Parser：Markdown/PDF/Text/HTML 的 block 和位置；
- Chunker：长中文、无标点、长 URL、code、table 全部严格限长、不丢内容且 Child span 精确；
- Indexer：只对可召回 Child embedding；可选特征溢出不裁正文；
- Store：Parent 无向量且不进入 KNN/BM25，ID lookup 仍可取回；
- Retrieval：Child→Parent、聚合、缺失回退与 provenance 透传；
- QA Context：matched Child 位于 Parent 开头/中间/末尾时都出现在实际 prompt window；总 token 不超预算；
- Citation：实际 prompt context 与 matched Child 原始 span 同时可追踪。

### 10.2 Production integration test

```text
Parser → Chunker → Metadata → FakeEmbedding
→ Store → HybridRouter → Parent expansion
→ matched-Child context window → FakeAnswerLLM → Citation
```

### 10.3 T2 benchmark

在正式链路稳定后计算 Recall@K、MRR@10、nDCG@10，并比较：

- chunk 大小；
- Parent/Child 与 flat chunk；
- Family 默认使用 RAGFlow 式 mean，并以 max 作为离线对照；
- content-only 与 metadata-enhanced embedding；
- RAGFlow title-vector blend。

## 11. 实现前需要你批准的正式设计决策

| 决策 | 当前项目 | RAGFlow | 本文建议 |
|---|---|---|---|
| 是否保留 Parent/Child | 是，双索引 | 可选 children + unavailable Mother | **已确认：保留，但 Parent 不召回，Child 唯一负责普通召回** |
| 长文本硬切 | 不完整 | 普通路径完整，custom delimiter 有例外 | **已确认：参考并补齐所有出口，超限继续分段且不得丢弃正文** |
| 首轮 chunk 粒度 | Parent 512/768、Child 128/192 token | 按配置的 token budget 切分 | **已确认：现有参数作为首轮 baseline，后续按生产检索与 T2 指标调整** |
| family score | 最高成员 | 已命中 Child 均值 | **已确认：采用 Child 均值；max 仅作离线对照** |
| embedding 正文 | question 存在时丢弃 | 同样可能丢弃 | 正文必须保留 |
| 标题/section | 文本拼接 | title vector 权重 | 保留 section；比较拼接与向量加权 |
| 可选特征超预算 | 最终由 adapter 截断 | 内容 `max_length-10` 截断 | 正文不裁；按批准优先级裁可选特征；正文仍超限则拒绝入库 |
| family evidence | 只保留单个 `matched_child_id` | 按 `mom_id` 聚合多个 Child 并返回 Mother，Mother 缺失时回退 Child | **已确认：采用该返回语义；额外保存仅用于溯源的 `matched_children[]` + primary child，不改变 Family 均值计分** |
| QA Parent 截取 | Parent 固定取前 2500 字符 | 完整 chunk 按序入 prompt；越界 chunk 仍保留、仅丢更后内容；不按 position 锚定 | 本项目新增 matched Children 锚定、QA token 严格总预算窗口 |
| QA 总窗口 | 只有 completion `max_tokens`，无总 context 能力 | 有 token budget，但仍不做 Mother 内锚定 | 增加显式 context limit、completion reserve、counter 与 safety margin |
| Citation | LLM `[n]` | 自动答案片段匹配 | 保留当前方式并补 matched children |
| source span | 只在嵌套 chunk meta，未贯穿检索/citation | position 贯穿 store/retrieval/reference | 建立 parser→Child→ES→retrieval→prompt→citation 全链路 |
| Text/HTML Parser | 无 | 有 | 增加简化正式实现 |
| ES schema 迁移 | 旧 index 无 `retrieval_eligible` | RAGFlow 有 availability 语义 | **已确认新建 v2 并全量重新 embedding；新 backend+v2 成对切换，旧 backend+旧 index 成对回退；不 backfill/alias。建议旧 index 保留 7 天，删除时再明确确认** |

只有这些正式架构决策得到明确批准后，才实施对应类、字段、schema 和迁移；T2 脚本的重写排在正式链路之后。
