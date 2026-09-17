# RAG 项目统一实施计划

更新日期：2026-09-17。

本文件是项目唯一有效的计划与能力盘点。旧版按 01–08 拆分的阶段稿已经删除；后续开发只更新本文件，不再建立互相引用、容易过期的重复计划。

## 1. 项目定位

本项目的目标不是复制一个大而全的 Agent 平台，而是完成一个可运行、可解释、可评测的高可信 RAG 产品：

> 保留文档结构与来源 → 用小块准确召回 → 用父块和证据窗口补充上下文 → 生成有依据的回答 → 用评测证明每次优化有效。

当前最有辨识度的能力是父子分块、来源坐标、证据窗口和离线检索评测。近期建设重点是完成产品闭环、业务评估集与图文解析，而不是提前增加 Agent、MCP、知识图谱或复杂分布式组件。

技术边界：

- 后端：FastAPI、LangChain Document、Elasticsearch、SQLite（首版）。
- 模型：独立的 Chat、Embedding、Vision 配置；不允许静默伪造模型结果。
- 数据：当前支持 Markdown、PDF、HTML、TXT；核心展示场景是 Markdown 图文知识库。
- 交付：上传、入库、提问、引用预览、评测报告形成完整演示闭环。

## 2. 状态与优先级

| 标记 | 含义 |
|---|---|
| `√ 已实现` | 已进入当前生产代码主链；仍需结合“验证状态”判断是否真实验收 |
| `△ 部分实现` | 有代码但未接入主链、未形成产品闭环，或当前工作区仍在重构 |
| `× 未实现` | 当前没有实际能力 |
| `P0` | 当前必须完成，阻塞可运行闭环或可信评测 |
| `P1` | P0 稳定后完成，构成完整 RAG 产品能力 |
| `P2` | 暂缓；只有业务评测证明需要时才实施 |

完成判定必须区分：

1. **已编码**：代码和替身测试存在。
2. **已验证**：目标 Python 环境中测试通过。
3. **已联调**：真实 Elasticsearch、Embedding、Chat/Vision 模型通过。
4. **已评测**：固定数据集有可复现指标和报告。

不能用“文件存在”代替“功能已经接入”，也不能把设计目标写成已实现能力。

## 3. 当前事实基线

- `backend/src` 约 5,000 行 Python，核心链路包含解析、父子分块、Embedding、ES、向量/BM25 混合检索、可选 Rerank、父块恢复、证据窗口、回答和引用。
- 当前工作区正在把手写 Markdown/PDF/HTML/TXT 解析器迁移为 `UnstructuredParser` 统一实现。阶段记录显示迁移后 79 个测试通过；该数字是已有实施记录，本文件更新时没有重新执行测试。
- PDF 当前使用 Unstructured `fast` 策略，不含 OCR 和版面模型；Markdown 表格会被规范化为纯文本，列表内围栏代码可能被拍平。
- `QueryRewriteService` 有独立实现和测试，但未接入 `QAService`。
- `RetrievalMetadataGenerator`（LLM 生成 important_kwd/question_kwd/title_tks 等检索元数据）有独立实现，但 `EmbeddingIndexer` 从未调用它，相关字段目前既未生成也未参与 BM25 检索；RAGFlow 同类机制的源码对比与接入方案见 `docs/compare.md`。
- `/qa/query` 路由文件存在，但尚未在 FastAPI 应用入口挂载。
- T2Retrieval 子集和向量产物作为历史实验保留，不再作为当前主评测集。
- 当前主评测集确定为 `CRUD-RAG-mini-v1`：1,000 篇文档、150 条 1/2/3 文档问答；详细的数据契约、阶段任务和验收标准见 `docs/plan/评测计划.md`。
- 文档、代码与实际运行结果冲突时，以当前代码和实际测试结果为准，并立即回写本文件。

## 4. 算法层

| 功能 | 状态 | 现有优点 | 当前缺点 | 是否需要优化、如何优化 |
|---|---|---|---|---|
| 统一 Document 契约 | √ 已实现 | 解析、分块、索引、检索和引用共用 `Document(page_content, metadata)`，减少重复容器和转换 | metadata 仍保留部分旧字段；缺少严格 schema 校验 | `P0`：整理唯一字段表和 schema version；删除无消费者字段；关键边界增加校验 |
| 统一解析引擎 | △ 重构中 | `UnstructuredParser` 统一处理 md/pdf/html/txt，下游分块不感知格式差异 | 新增依赖较重且需要 NLTK 数据；当前是未提交工作区改动 | `P0`：完成依赖、冷启动和部署验证；保留格式能力测试；记录真实解析耗时 |
| Markdown 结构解析 | △ 部分实现 | 能映射 Title/List/Table/Code 等类别并维护 section path；有原文回定位 | GFM 表格结构丢失；列表内代码可能降级；图片未进入链路 | `P0`：补图片资产和 VLM 描述；评估 `text_as_html` 保留表格结构；用回归样例约束代码块 |
| PDF 解析 | △ 部分实现 | 比旧版逐行 pypdf 更容易识别 Title/List 等结构；保留页码入口 | fast 策略无 OCR、表格推理、bbox 和复杂版面恢复 | Markdown 主线稳定前不扩张；`P2` 再基于失败数据决定 OCR/hi_res，而不是直接引入重模型 |
| HTML/TXT 解析 | √ 已实现 | 统一入口、严格 UTF-8、TXT 可做精确坐标 | 不是核心展示场景；维护成本仍存在 | 暂时保留作为低成本能力，不继续扩展；若长期没有场景可删除 |
| 图片/VLM 解析 | × 未实现 | 已有独立 Vision 模型配置和数据契约设计基础 | 图片没有被提取、描述、分块、索引和引用 | `P0`：按当前用户和 document/asset 查库确认读取权限→图片哈希生成 asset_id→保存原图→VLM 描述→生成 image block→进入分块→鉴权预览原图 |
| 来源坐标 | √ 已实现 | TXT 可精确定位；Markdown/HTML 采用保守的行级定位；子块可追踪父块 | 归一化回定位在重复文本和复杂标记下可能 unavailable；PDF 无精确坐标 | `P0`：建设 span 回归集；报告 exact/line_only/unavailable 比例；不得伪装为逐字精确 |
| 父子分块 | √ 已实现 | 小子块召回、大父块补上下文；父子均有硬 token 上限；保护代码/表格边界；已有 version/hash | 当前没有证据证明默认 profile 不足 | `P2（条件触发）`：仅当 Golden Set 显示代码或表格存在稳定分片失败时，新增一个针对性 profile，与 default 同集 A/B，胜出后才替换 |
| 内容守恒 | √ 已实现 | 子块拼接必须等于规范化父块，能发现文本丢失 | 主要体现在测试断言，没有汇总指标 | `P0`：报告丢失字符、重复字符、越界块、token 分布和每篇文档父子块数量 |
| Embedding 输入 | √ 已实现 | 只嵌入检索子块；校验响应数量、维度、NaN 和零向量 | 没有比较正文、标题+正文、章节+正文 | `P1`：在固定 Golden Set 上做三种输入消融，用 Recall/MRR/成本选择 |
| ES 索引 | √ 已实现 | 子块保存向量，父块保存上下文；重入库清理陈旧 Chunk；保存模型和 profile 元数据 | 没有索引发布、回滚和模型迁移流程 | 首版记录 index/schema/model/profile hash 即可；蓝绿索引和双写放 `P2` |
| 向量检索 | √ 已实现 | ES KNN；强制过滤不可检索父块；有候选池和阈值 | 阈值和 TopK 没有业务数据支撑 | `P0`：用业务 Golden Set 扫描 candidate_top_k、threshold、final_top_k，保存曲线和最优配置 |
| BM25 检索 | √ 已实现 | 可补足型号、术语和精确字符串；搜索文档名、章节和正文 | standard analyzer 对中文可能不足；索引侧增强字段（important/question）未生成未接入 | `P0/P1`：按 `docs/compare.md` 方案接入增强字段并评测；先按问题类型切片评测；仅在中文关键词场景显著失败时更换 analyzer |
| 多路召回 | √ 已实现 | 向量和关键词均能召回并保留通道分数 | 当前串行执行，单通道失败会影响主链 | `P1`：并行执行、分别超时、单通道失败降级；记录每通道耗时和贡献率 |
| 融合排序 | △ 可用但待验证 | 当前加权分数实现简单、可解释 | 向量与 BM25 分数尺度不天然一致，权重可能漂移 | `P0`：实现 Weighted RRF 作为候选，与当前加权融合做同集 A/B；由 Hit/Recall/MRR/nDCG 决定 |
| Rerank | △ 代码存在 | 有规则重排和 Cross Encoder，失败可回退融合结果 | 默认关闭；没有真实增益、延迟和成本报告 | `P0/P1`：固定“召回池→Rerank 池→最终 TopK”漏斗，比较开启前后的 nDCG、Evidence Recall、P95 |
| 父块聚合 | √ 已实现 | 子块召回后恢复父块；同一 family 合并多个贡献子块并保留明细 | family score 当前用均值，可能压低单个强命中 | `P1`：比较 mean、max、top-N mean；同时观察文档级 Recall 和噪声 |
| 证据窗口 | √ 已实现 | 以命中子块为锚点截取父块；远距离命中可拆多个窗口；控制 Prompt token | 默认 384 token 尚无业务最优性证据 | `P0`：比较 child-only、window、full-parent 的事实覆盖、证据密度、Faithfulness 和成本 |
| 回答生成 | √ 内部实现 | 强制依据证据回答；完整计算上下文预算；模型错误不伪造答案 | 单模型、同步调用；无会话上下文；空结果语义不统一 | `P0`：统一 `ok/no_evidence`；完成 API 闭环后再做模型路由 |
| 引用 | √ 已实现 | 校验 `[n]` 范围并返回来源、命中子块和实际 Prompt 窗口 | 只能证明引用编号存在，不能证明引用支持对应事实 | `P0`：增加 Citation Validity、Correctness、Coverage 和 Unsupported Claim 指标 |
| 知识图谱/联网搜索 | × 未实现 | — | 无多跳关系和时效信息通道 | `P2`：当前暂不实现；只有业务失败集中于多跳或时效性问题时再增加 |
| Agent/MCP | × 未实现 | — | 没有规划循环和外部工具调用 | `P2`：RAG 产品和评测闭环稳定前不实现 |

### 4.1 算法层近期目标

1. 完成 Unstructured 迁移与真实环境验证。
2. 完成 Markdown 图片资产、VLM 描述和引用闭环。
3. 建立 Weighted Sum、Weighted RRF、Rerank 的可复现实验。
4. 用业务数据确定 TopK、阈值和证据窗口，而不是继续采用未经验证的默认值。

## 5. 后端层

| 功能 | 状态 | 现有优点 | 当前缺点 | 是否需要优化、如何优化 |
|---|---|---|---|---|
| FastAPI 基础 | √ 已实现 | 有 health、CORS、统一异常入口、request ID 和总延迟响应头 | 同步重任务会阻塞事件循环；API 尚未形成完整闭环 | `P0`：服务调用放线程池或后台任务；增加 API 集成测试 |
| QA API | △ 部分实现 | 路由和 `QAService` 都存在 | Router 未挂载，实际启动服务后无法问答 | `P0`：挂载 `/qa/query`，完成真实 HTTP 端到端测试 |
| 参数校验 | △ 部分实现 | Pydantic 请求模型和服务层空值/长度校验 | 请求约束弱；上传无大小限制；错误状态大量归为 400 | `P0`：在请求模型声明长度；区分 400/404/409/422/502/503；限制上传大小 |
| Demo 认证 | △ Demo 实现 | JWT 生成与校验路径完整 | 默认账号、密码和 secret；无用户表、角色和数据归属 | 演示期明确标记非生产；`P1` 优先做数据归属，完整账户系统可 `P2` |
| 文档上传 | √ 已实现 | 源文件保存与元数据登记分离 | `doc_{count+1}` 有并发/删除冲突；无大小、哈希和重复检测 | `P0`：UUID；保存 SHA256、size、created_at、error、counts |
| 文档生命周期 | △ 部分实现 | 有 pending/processing/success/failed 雏形 | 无详情、删除、源文件、图片接口和节点日志；并发 ingest 保护不足 | `P0`：统一 uploaded/indexing/ready/failed；实现 source/assets/delete/retry；同 doc indexing 返回 409 |
| 知识库实体 | × 未实现 | ES filters 可作为底层基础 | 没有知识库、文档集合、配置和归属 | `P1`：增加 KnowledgeBase；文档、检索配置、权限都绑定 knowledge_base_id |
| 用户数据归属 | × 未实现 | — | 认证用户之间没有文档和会话隔离 | `P1`：服务端根据 owner/tenant 注入过滤；不接受客户端任意权限过滤 |
| 查询改写 | △ 组件存在 | 有历史裁剪、token 预算、严格 JSON、失败回退和测试 | 未接入 QAService；API 无 session/history | `P1`：先持久化会话，再按“load memory→rewrite→retrieve”接入；Trace 保存原问题和改写结果 |
| 子问题拆分 | × 未实现 | — | 复合问题只能整句检索一次 | `P2`：业务集证明多实体问题是主要失败后再做；限制最多 3 个子问题 |
| 会话与消息 | × 未实现 | — | 无 Conversation、Message、session_id | `P1`：先实现最近 N 轮消息，不立即做复杂长期记忆 |
| 会话摘要 | × 未实现 | — | 长对话无法压缩和跨轮回顾 | `P2`：最近 N 轮稳定后再做“持久摘要+近期消息”，摘要需版本化和可追踪 |
| 意图识别 | × 未实现 | — | 所有输入都直接检索，闲聊和模糊问题没有专门路径 | `P1`：首版只做 KB_QUERY、CHAT、CLARIFY 三类；不复制复杂意图树 |
| 问答编排 | △ 简单链路 | 当前“检索→窗口→生成→引用”清晰 | 缺少显式上下文和短路点，难插入记忆、改写、意图与持久化 | `P1`：建立请求级 `QAContext` 和阶段化 Pipeline，禁止共享可变 trace |
| 空结果与拒答 | △ 部分实现 | 不会在无证据时自由生成 | 当前表现为业务错误，与产品响应不一致 | `P0`：统一返回 `no_evidence`；纳入误拒率、正确拒答率评测 |
| SSE 流式输出 | × 未实现 | — | 无 TTFT、取消和实时体验 | `P1`：闭环稳定后实现 answer/citation/done/error 事件；支持客户端断开取消 |
| 后台入库任务 | × 未实现 | 同步流程便于当前调试 | 大文档阻塞请求，无进度和节点重试 | `P1`：先用轻量 worker/任务表，不为个人项目直接引入 MQ |
| 幂等与并发保护 | × 未实现 | — | 重复点击会重复处理；模型调用无并发上限 | `P1`：入库以 doc_id+profile_hash 幂等；模型入口使用 semaphore；多实例后再考虑 Redis |
| 模型路由/熔断 | × 未实现 | Chat、Embedding、Vision 配置已分离 | 无候选模型、健康状态和自动切换 | `P2`：先做 primary+fallback 和超时记录，再决定是否需要三态熔断 |
| Trace | △ 部分实现 | 有 request ID、检索候选、分数、引用和 QA budget | 只在响应/内存；`last_trace` 共享状态不适合并发 | `P0/P1`：请求级 Trace 对象；持久化 run/node、耗时、配置和异常；敏感内容脱敏 |
| 用户反馈 | × 未实现 | — | 真实失败不能沉淀为评估集 | `P1`：点赞/点踩、失败原因和备注；负反馈一键转 regression case |
| 前端问答页 | × 未实现 | 已有 API 与页面目标 | 无成品体验 | `P0`：完成上传、入库状态、提问、答案、引用预览五个核心状态 |
| 管理后台/审计 | × 未实现 | — | 无知识库、Chunk、Trace、评测管理 | `P2`：先做最小文档和 Trace 页面；配置审计等真实需要出现后再补 |

### 5.1 后端目标链路

P0 完成后的单轮链路：

```text
上传文档 → 登记状态 → 解析/分块/索引 → 提问 → 混合检索
→ 父块恢复 → 证据窗口 → 回答 → 引用 → 原文/图片预览
```

P1 完成后的多轮链路：

```text
加载会话 → 查询改写 → 简单意图路由/短路 → 多知识库检索
→ 证据窗口 → 流式回答 → 保存消息/证据/Trace → 用户反馈
```

## 6. 评测层

评测不是交付前的附加步骤，而是所有算法和 Prompt 修改的准入条件。流程固定为：

```text
init（固定数据和索引）
→ run（录制一次真实链路）
→ score（离线重复评分）
→ report（对比、失败归因和回归集）
```

### 6.1 数据集与验证

| 数据集/能力 | 状态 | 当前规模或内容 | 如何验证 | 缺点与下一步 |
|---|---|---|---|---|
| 单元与集成测试 | △ 已有记录 | 当前阶段记录为 79 个测试通过 | 解析、分块、索引、检索、窗口、引用、改写的确定性断言 | `P0`：在固定环境重新运行，报告 collected/passed/failed/skipped 和耗时 |
| T2Retrieval 子集 | √ 历史资产 | 10,000 篇候选文档、500 Query、2,614 qrels；已有向量产物和评分脚本 | 保留已有产物，不删除、不继续扩建 | 只能做检索实验，不能覆盖回答与引用；不再作为当前主线 |
| CRUD-RAG Mini | × 待建设 | 目标 1,000 篇文档、150 条问答，1/2/3 文档各 50 条 | 固定 commit/seed/SHA；30 dev + 120 test；事件级隔离 | `P0`：按 `评测计划.md` 生成、校验并冻结 `crud_rag_mini_v1` |
| 解析评估集 | △ 有样例 | 真实 Markdown/PDF 和 showcase | 检查 expected facts、block types、来源范围、解析失败 | `P0`：固定 20–30 篇代表性文档；统计内容保留率、结构保留率、span 精度分布 |
| 分块评测 | √ 有不变量 | 长中文、代码、表格、边界和坐标样例 | 内容守恒、token 上限、父子引用和坐标合法性 | 增加聚合报告，不只输出 pytest pass/fail |
| 主线 RAG 评测集 | × 未实现 | `CRUD-RAG-mini-v1` 共 150 条，答案与证据来自公开基准的固定子集 | Retrieval、Evidence、Answer、Citation、Latency 分层评分 | `P0`：先完成确定性数据/评分闭环；语义 Judge 放 P1 |
| Evidence 评测 | × 未实现 | 基于 CRUD-RAG 的 reference news 与 qrels | Evidence Recall、Precision、Token Density；检查必要文档是否进入 Prompt | `P0`：优先完成，它能定位问题在检索还是生成 |
| 回答正确性 | × 未实现 | question、expected facts、answer | 先做必要事实覆盖和禁止事实；再接 LLM Judge | `P1`：LLM Judge 结果抽样人工校准，不能当绝对真值 |
| Faithfulness | × 未实现 | answer + 实际 evidence | claim 是否能被 evidence 支持 | `P1`：使用 Ragas 或自建 rubric；保存 judge 原始理由和失败 claim |
| 引用评测 | △ 基础存在 | 当前只能校验编号和范围 | Validity、Correctness、Coverage、Unsupported Claims | `P0`：把回答 claim 与其引用证据对应评分，输出错误引用案例 |
| 拒答评测 | × 未实现 | 业务集中 15%–20% 不可回答问题 | 拒答 Precision、Recall、F1；误拒率和过回答率 | `P0`：统一 no_evidence 协议后接入 |
| Query Rewrite 评测 | △ 只有单测 | 代词、省略、历史预算和失败回退样例 | 实体保留、约束保留、意图不变、独立可检索 | `P1`：接入主链后建设 20–30 组真实多轮对话 |
| Prompt Injection 安全集 | × 未实现 | 文档注入、用户注入、引用欺骗、越权请求 | 系统指令遵循、数据隔离、拒绝危险行为 | `P1`：首版 15–20 条，所有真实安全失败加入回归集 |
| 性能与成本 | △ 少量 trace | 总延迟、QA budget、Embedding token 已有局部记录 | P50/P95、分阶段耗时、输入/输出 token、每成功回答成本 | `P1`：请求级持久化；SSE 后增加 TTFT 和取消率 |
| Agent 评测 | × 未实现 | — | 将来评 Task Success、Tool/Argument Accuracy、Trajectory、Side-effect Safety | `P2`：Agent 尚未实现，当前不建设空框架 |

### 6.2 当前评测集分布

| 类型 | 总数 | Dev | Test | 主要验证 |
|---|---:|---:|---:|---|
| 单文档问答 | 50 | 10 | 40 | 基础召回、答案和引用 |
| 双文档问答 | 50 | 10 | 40 | 多证据覆盖与组合 |
| 三文档问答 | 50 | 10 | 40 | 深召回、证据缺失和排序 |

150 条问题使用互不重复的事件 ID；Dev 与 Test 也按事件隔离。所有正例文档强制进入 1,000 文档语料库，再加入普通与困难干扰项。CRUD-RAG 不覆盖不可回答、图片、权限和 Prompt Injection，这些能力需要以后用小型业务回归集补充，不得把它们包装成已测。

### 6.3 评测产物

建议新增独立顶级目录，避免评测逻辑侵入生产代码：

```text
evaluation/
├── datasets/
│   ├── documents/
│   └── golden_cases.jsonl
├── runner/
│   ├── run_retrieval.py
│   └── run_qa.py
├── scorers/
│   ├── retrieval.py
│   ├── evidence.py
│   ├── answer.py
│   ├── citation.py
│   └── performance.py
├── runs/          # 一次真实调用的完整录制，JSONL
├── scores/        # 可重复离线计算的 JSON
├── reports/       # Markdown + CSV + 图表
└── failures/      # 失败样例与根因分类 JSONL
```

每条 run 至少保存：

- case_id、Git commit、运行时间。
- 解析/分块/Embedding/索引版本。
- 检索参数和模型配置快照（不保存密钥）。
- 原问题、改写问题（启用后）、候选结果、实际 evidence。
- answer、citations、状态和错误。
- 分阶段延迟、token 和成本。

录制结果与评分逻辑分离：昂贵的真实链路运行一次，确定性指标和 Judge 可以基于同一 JSONL 多次重算。

### 6.4 展示方式

项目首页只展示真实跑出的结果：

1. **一页纸看板**：Hit@5、Recall@5、MRR@10、nDCG@10、Evidence Recall、Faithfulness、Citation Correctness、误拒率、P95。
2. **实验对比**：Baseline 与 Candidate 的配置、指标变化、延迟和成本，明确唯一变量。
3. **分片结果**：按问题类型、文档类型和难度展示，不能只看总体平均值。
4. **失败案例**：问题→期望文档/事实→实际召回→Evidence→回答→引用→根因→修复。
5. **完整链路样例**：选 3–5 个代表性问题展示可追溯过程。

CI 只运行稳定且确定性的快速指标；需要真实模型或 LLM Judge 的评测放在夜间或发版前。阈值以第一次 baseline 为依据，不照搬其他项目的目标数字。

## 7. 实施路线

### P0：形成可信的单轮 RAG 成品

按顺序执行：

1. 完成 Unstructured 解析迁移的环境与回归验证。
2. 清理统一 Document/metadata/schema，移除无消费者旧实现。
3. 挂载 QA API，对齐状态码、no_evidence 和响应字段。
4. 补齐文档 UUID、状态、error、counts、source/assets/delete/retry。
5. 实现 Markdown 图片提取、VLM 描述、入库和引用预览。
6. 完成最小前端：上传、入库状态、问答、答案、引用和原文/图片预览。
7. 建设 `CRUD-RAG-mini-v1`（1,000 文档、150 问题）和确定性评测 runner。
8. 产出 Vector、Hybrid Weighted Sum、Weighted RRF、Rerank 的首份对比报告。
9. 用评测确定 TopK、阈值、证据窗口，更新默认配置。

P0 验收：

- 新环境按文档一次启动成功。
- 真实上传→入库→问答→引用预览闭环通过。
- 单元/集成测试全部通过，真实 ES/模型有单独联调记录。
- 至少一份可复现 baseline 报告和失败案例清单。
- README 只陈述已经实测的数据。

### P1：完整的多轮知识库产品

1. KnowledgeBase、Conversation、Message 和数据归属。
2. 查询改写接入及多轮评估集。
3. 简化意图分类和短路处理。
4. 多通道并行、失败隔离和请求级 Trace。
5. SSE 流式输出、客户端取消和轻量后台入库任务。
6. 用户反馈转评估回归样例。
7. Faithfulness、Answer Correctness、Context Precision 等语义评测。

### P2：评测证明必要后再做

- OCR/复杂 PDF 版面恢复。
- 查询拆分和复杂意图树。
- 知识图谱、联网搜索。
- 多供应商模型路由、三态熔断、Redis 公平排队和 MQ。
- 完整管理后台、审计系统。
- Agent、MCP、工具调用及 Agent trajectory 评测。

## 8. 明确暂不做的事情

- 不为了模仿其他项目堆叠功能数量。
- 不在没有业务失败数据时引入知识图谱和复杂 Agent。
- 不用 LLM Judge 替代 Hit/Recall/MRR、引用范围、延迟等确定性指标。
- 不同时引入多套评测框架；首版以自建 Python runner/scorer 为主，Ragas 只补语义指标。
- 不把测试替身结果写成真实模型联调结果。
- 不在文档中保存密钥、完整敏感 Prompt 或用户隐私数据。

## 9. 文档维护规则

1. 本文件是唯一的项目优先级总表；`评测计划.md` 只维护评测数据与执行契约，不另行定义冲突的项目优先级。
2. 每次实施完成后更新状态、验证命令、真实结果和未完成项。
3. `docs/interview/` 只保存已经由代码或评测支持的项目讲解与面试题，不重复维护实施状态。
4. 算法改动必须附 baseline/candidate 结果；没有评测数据时只允许写“候选方案”。
5. 生产代码、测试和文档必须使用同一名词：Document、Parent、Child、Evidence、Citation、KnowledgeBase、Conversation。
6. 如果规划与当前实现冲突，先修正文档，再继续扩功能。

最终交付应同时回答三个问题：

```text
算法层：实现了什么，为什么这样设计？
后端层：如何把能力做成稳定、可使用的产品？
评测层：如何证明改动有效，并防止回归？
```
