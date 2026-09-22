# RAG 项目统一实施计划

更新日期：2026-09-19。

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
- 手写 Markdown/PDF/HTML/TXT 解析器已迁移为 `UnstructuredParser` 统一实现，并修复 PDF/Numba 在只读环境中的冷启动问题。2026-09-21 在 Conda `agent` 环境重跑后端与评测测试，全量 `164 passed`。
- 2026-09-18 完成 §4.2 四项生产评测前置改造：`retrieval_mode` 通道门控、`rerank_top_n` 漏斗、请求级 `QAConfig`、请求级结构化 Trace（config/timings_ms/usage）。默认行为不变（默认 hybrid、Rerank 默认关闭、窗口默认值保持）。Conda `agent` 环境全量 102 个测试通过（含新增 20 个验收测试，现已按功能归并至 `backend/tests/integration/test_retrieval.py`、`backend/tests/services/test_qa_service.py` 与 `backend/tests/foundation/test_request_config.py`）。
- PDF 当前使用 Unstructured `fast` 策略，不含 OCR 和版面模型；Markdown 表格会被规范化为纯文本，列表内围栏代码可能被拍平。
- `QueryRewriteService` 已接入 `QAService` 与同步/SSE API：请求可携带 `history`，服务按完整 user+assistant 轮次做最近窗口和 token 裁剪，并把改写后的独立问题同时用于检索和回答；无历史时可按请求开启口语规范化，Step-back 作为默认关闭的实验开关执行双路检索和去重合并。当前仍没有 Conversation/Message/session_id 持久化、歧义澄清协议和查询侧关键词追加。
- `RetrievalMetadataGenerator`（LLM 生成 important_kwd/question_kwd/title_tks 等检索元数据）有独立实现，但 `EmbeddingIndexer` 从未调用它，相关字段目前既未生成也未参与 BM25 检索。**2026-09-19 决定暂缓接线**：RAGFlow 同类开关（`auto_keywords`/`auto_questions`）默认关闭，逐子块 LLM 成本与收益未经评测证明，当前 Golden Set 也未呈现 BM25 召回缺口；接入方案保留在 `docs/compare.md` §4.1-4.3 备查，失败证据触发后再启动（文件头部已标注用途与 TODO）。
- `/qa/query` 已在 FastAPI 应用入口挂载（`main.py`），并有 HTTP 端到端替身测试（鉴权 401、合法请求 200、非法配置 422）。
- 2026-09-19 完成后端 P0 产品闭环：`ServiceError` 在抛出点声明 HTTP 状态码（400/404/409/413/422/502）；`doc_id` 改 UUID 并登记 SHA256/size/created_at/counts；状态机统一 uploaded/indexing/ready/failed，新增 detail/source/delete/retry 与同文档并发 409；检索无命中返回 `status="no_evidence"` 产品响应（不再抛业务错误，评测 Runner 按新协议映射 run 行）；新增 SSE `/qa/query/stream`（answer/citation/done/error 事件，配置校验挡在流开始前，断开取消尽力而为）；同步重任务端点全部改为线程池执行（def 端点 / `asyncio.to_thread`）。2026-09-21 加入查询改写、口语规范化和 Step-back 请求级接线后，Conda `agent` 环境全量 `backend/tests + evaluation/tests` 为 `164 passed`。
- T2Retrieval 子集和向量产物作为历史实验保留，不再作为当前主评测集。
- 当前主评测集确定为 `CRUD-RAG-mini-v1`：1,000 篇文档、150 条 1/2/3 文档问答；详细的数据契约、阶段任务和验收标准见 `docs/plan/评测.md`。
- 文档、代码与实际运行结果冲突时，以当前代码和实际测试结果为准，并立即回写本文件。

## 4. 算法层

为避免把“已实现但待评测”重复写成多个未完成 P0，能力按生产链路合并如下：

| 能力包 | 当前状态 | 已有证据 | 唯一剩余工作 |
|---|---|---|---|
| 文本解析（md/pdf/html/txt） | √ 已实现并回归验证 | `UnstructuredParser` 统一四种格式；PDF 依赖按需加载并规避只读 Numba 缓存；Markdown/HTML/TXT/PDF、嵌套代码、表格文本、非法 UTF-8 均有测试 | `P1`：补部署 bootstrap 与冷/热耗时；`P2`：失败数据触发后再做 OCR、hi_res、复杂表格结构恢复 |
| Document 与来源追踪 | √ 已实现并回归验证 | 全链路使用 `Document(page_content, metadata)`；`exact/line_only/unavailable` 精度分级、重复文本单调回定位、HTML 实体和 Child 坐标投影已有回归 | `P1`：清理无消费者字段、形成 20–30 篇精度分布报告；不再作为算法 P0 |
| Markdown 图片/VLM | √ 已实现并回归验证（P0-A） | `AssetFileStore` 按 SHA256 去重保存且读取前校验路径边界；`AssetRegistry` 用 asset_id+document_id+owner 单条 JOIN 鉴权；`ImageDescriber` 输出受 schema 约束 JSON 并程序拼装可检索文本；VLM 失败保留资产与 failed 状态、非法引用保留 skipped 占位；image 为原子块且 `asset_id` 透传到 Citation；`/assets/preview` 二次鉴权。权限/穿越/重复/失效均有测试 | `P1`：真实 VLM 模型联调、线上伴随图片上传通道；HTML/PDF 图片与 OCR 维持不做 |
| 父子分块与 Embedding | √ 已实现并回归验证 | Parent/Child 硬 token 上限、内容逐字守恒、坐标投影、稳定 profile hash、只嵌 Child 正文、向量数量/维度/NaN/零向量校验均有测试 | `P2（条件触发）`：只有失败集证明必要时才增加 profile 或比较输入模板，不预建多套方案 |
| ES 索引与重入库 | √ 已实现 | Parent 保存上下文、Child 保存向量；先 upsert 再清理陈旧 ID；记录 index/schema/model/profile 信息 | `P2`：需要真实发布迁移时再做蓝绿索引、双写和回滚 |
| Hybrid 检索与 Rerank | √ 已实现并完成首次真实 baseline（P0-B） | 向量/BM25、Weighted Sum、父块聚合、规则/Cross-Encoder Rerank 均已编码；`retrieval_mode` 严格门控、`rerank_top_n` 漏斗、失败回退和请求级 Trace 已测试；`evaluation/runner` 只调用生产实现，替身测试 8 项通过 | 2026-09-19 真实 run 已执行（详见 [`评测.md`](评测.md)）：全量 1,000 篇索引、Dev 单变量扫描锁定、Test 各运行一次；锁定配置召回 97.5%（默认 hybrid 88.2%）、evidence recall 93.1%、引用正确率 96.7%、P95 2.3s、成本约 ¥2。元数据增强、并行/隔离放 `P1`，Weighted RRF 仅失败触发 `P2` |
| Evidence、回答与引用 | √ 已实现并回归验证 | 父块恢复、锚定窗口、child/window/full-parent 请求级配置、Prompt 总预算、证据约束回答、引用范围校验及文档级 Citation scorer 已实现 | 窗口默认值并入同一个 **P0-B** Dev 扫描，不再单列 P0；`no_evidence` 产品协议已随后端 P0 交付（2026-09-19）；claim-level Correctness/Coverage 归 `P1` Judge |
| 知识图谱、联网搜索、Agent/MCP | × 暂不实现 | 当前单轮 RAG 主线不依赖这些能力 | `P2`：只有多跳、时效性或工具调用失败形成稳定业务类别时再立项 |

### 4.1 算法层近期目标

已完成且不再计入待办：统一解析迁移、来源精度分级与最小回归、父子分块/Embedding 不变量、离线数据集与 scorer，以及 §4.2 的四项生产评测接口。2026-09-21 在 Conda `agent` 环境重跑 `backend/tests + evaluation/tests`：`164 passed`。

算法层的两个 P0 工作包已于 2026-09-19 完成编码并回归；包含随后端与查询预处理新增回归在内，2026-09-21 全量为 `164 passed`：

1. **P0-A Markdown 图片/VLM 最小闭环**：已实现。资产保存/归属鉴权/VLM 描述/原子 image block/`asset_id` 透传/鉴权预览全部打通并有验收测试；真实 VLM 模型联调归 P1。
2. **P0-B 生产 Runner 与真实 baseline**：√ 已完成（2026-09-19）。全量 1,000 篇索引 → Dev 扫描锁定 → Test 正式 run 全部执行，结果见 [`评测.md`](评测.md)：召回 88.2%→97.5%，引用正确率 96.7%，P95 2.3s。

### 4.2 已完成：生产评测接入的四个前置改造

这些是让 `evaluation/runner` 能公平调用生产链路的接口改造，不是在评测目录复制一套 Retriever/QA：

> 里程碑（2026-09-18）：四项改造完成时全量为 102 个测试；Runner、全量索引、Dev 扫描和 Test baseline 已于 2026-09-19 完成，当前结果见 [`评测.md`](评测.md)。涉及文件：`backend/src/config/retrieval_config.py`、`backend/src/config/qa_config.py`、`backend/src/retrieval/hybrid_router.py`、`backend/src/apps/services/qa_service.py`、`backend/src/apps/restful_apis/qa.py`。

1. **真实检索模式**：在 `RetrievalConfig` 增加 `retrieval_mode=vector/keyword/hybrid`；未选中的通道不得执行。只有这样 `Vector-only` 与 `Hybrid` 才是有效消融。
2. **固定 Rerank 漏斗**：保留已有 `rerank_enabled/backend`，增加 `rerank_top_n`；Trace 同时记录召回池、送排池、最终 TopK、模型和 fallback。
3. **请求级证据配置**：给 `QAService.query()` 增加独立 `QAConfig`，至少包含 `context_top_k/evidence_mode/evidence_window_tokens`，不得靠修改全局环境变量完成逐组实验。
4. **结构化可观测性**：生产响应返回请求级 `trace.config`、`trace.timings_ms` 和 `trace.usage`。阶段至少覆盖 retrieval、window、generation、citation；usage 至少保存模型名、input/output/total token，可得时保存成本。供应商未返回的字段写 `unavailable`，不得猜测或补零。

验收时必须证明 Runner 只编排和录制，检索、窗口、生成与引用仍调用 `backend/` 生产实现。

## 5. 后端层

| 功能 | 状态 | 现有优点 | 当前缺点 | 是否需要优化、如何优化 |
|---|---|---|---|---|
| FastAPI 基础 | √ 已实现 | 有 health、CORS、统一异常入口、request ID 和总延迟响应头；同步重任务端点已改为线程池执行（def 端点 / `asyncio.to_thread`），不再阻塞事件循环 | SSE 生成阶段的取消是尽力而为（当前分块完成后停止） | `P1`：真实并发压测与请求超时预算 |
| QA API | √ 已实现 | `/qa/query` 与 `/qa/query/stream` 共用生产 QA 链路；请求级 `retrieval_config/qa_config/history` 覆盖；鉴权、配置 422、空问题、`no_evidence`、SSE 事件和查询改写均有 HTTP 替身测试 | 真实供应商流式返回、断开取消与改写收益尚未联调/评测 | P0 已完成；`P1`：持久化会话、真实 SSE 联调与查询改写 A/B |
| 参数校验 | √ 已实现 | `ServiceError` 抛出点声明状态码，已区分 400/404/409/413/422/502；问题空值/3072 字节上限、文件类型白名单、上传大小限制（`upload.max_bytes` 默认 20 MiB）均有测试 | 上传端点仍先 `await file.read()` 再检查大小，能拒绝超限文件但不能避免大请求先占用内存；请求模型的深度约束仍较宽 | `P1`：改为分块读取并在超过阈值时提前停止；真实滥用样例出现后继续收紧模型约束 |
| Demo 认证 | △ Demo 实现 | JWT 生成与校验路径完整 | 默认账号、密码和 secret；无用户表、角色和数据归属 | 演示期明确标记非生产；`P1` 优先做数据归属，完整账户系统可 `P2` |
| 文档上传 | √ 已实现 | `doc_id` 为 UUID；登记 SHA256/size/created_at，超限 413；源文件保存与元数据登记分离；API 响应不外泄 `file_path` | 大小校验发生在整文件读入内存之后；无同哈希重复上传检测 | `P1`：流式限量读取；`P2`：存储成为瓶颈后再做 SHA256 秒传/去重 |
| 文档生命周期 | √ 主闭环已实现 | 状态机统一 uploaded/indexing/ready/failed；detail/source/delete/retry 端点齐全；同文档并发 ingest 返回 409（进程内锁）；失败保留 error 可重试；删除按索引→源文件→文档记录清理，索引失败保留记录可重删 | 删除尚未清理该文档的 assets 表记录和无引用图片文件；节点级进度、图片列表接口未做 | `P1`：补资产引用清理与孤儿文件回收；进度上报随后台入库任务一起做 |
| 知识库实体 | × 未实现 | ES filters 可作为底层基础 | 没有知识库、文档集合、配置和归属 | `P1`：增加 KnowledgeBase；文档、检索配置、权限都绑定 knowledge_base_id |
| 用户数据归属 | × 未实现 | — | 认证用户之间没有文档和会话隔离 | `P1`：服务端根据 owner/tenant 注入过滤；不接受客户端任意权限过滤 |
| 查询改写 | √ 主链接入，△ 产品化未完成 | `history` 已进入同步/SSE API；多轮指代消解、无历史口语规范化、失败回退和请求级 Trace 已接 `QAService`；Step-back 以默认关闭的实验开关执行双路检索补漏 | 历史由客户端随请求传入，无 Conversation/Message/session_id；尚无歧义澄清；未做口语/多轮/Step-back 切片 A/B，也未实现查询侧关键词追加 | `P1`：持久化最近会话消息、增加澄清短路并完成三个切片的 Recall/MRR、误改写率、P95 与成本评测 |
| 子问题拆分 | × 未实现 | — | 复合问题只能整句检索一次 | `P2`：业务集证明多实体问题是主要失败后再做；限制最多 3 个子问题 |
| 会话与消息 | × 未实现 | 改写组件已定义完整轮次裁剪方式 | 无 Conversation、Message、session_id；当前没有代码负责保存或加载历史 | `P1`：先持久化最近 N 个“user 问题 + assistant 回答”完整轮次；这是会话窗口，不称为长期 Memory，也不把历史回答当知识证据 |
| 会话摘要 | × 未实现 | — | 长对话无法压缩和跨轮回顾 | `P2`：最近 N 轮稳定后再做“持久摘要+近期消息”，摘要需版本化和可追踪 |
| 意图识别 | × 未实现 | — | 所有输入都直接检索，闲聊和模糊问题没有专门路径 | `P1`：首版只做 KB_QUERY、CHAT、CLARIFY 三类；不复制复杂意图树 |
| 问答编排 | △ 阶段化链路 | 当前已有“改写→可选 Step-back→检索→`no_evidence` 短路→窗口→生成→引用”，阶段耗时和可选 Trace 均为请求局部 | 尚无显式 `QAContext`、意图分类、会话加载与消息持久化；同步与 SSE 仍各自编排相似阶段 | `P1`：提取共享请求级 Pipeline/Context，再接意图、会话和持久化，避免两条入口继续重复演进 |
| 空结果与拒答 | √ 已实现 | 检索无命中返回 `status="no_evidence"` 产品响应（answer/citations 为空、trace 保留、usage 如实 unavailable），同步与 SSE 行为一致；评测 Runner 按新协议把该响应映射为 run 行 `no_evidence` 状态 | 拒答评测集（不可回答问题切片）未建 | `P0` 协议已完成；评测集随业务回归集建设（`P1`） |
| SSE 流式输出 | √ 已实现（P0 提前交付） | `/qa/query/stream` 已实现 answer/citation/done/error 事件；配置校验挡在流开始之前（422 而非半途断流）；无证据只发 done；客户端断开即停止发送；与 `/qa/query` 共用同一检索/窗口/引用链路，有 HTTP 替身测试 | 真实模型的流式联调未做；usage 依赖供应商在流分块中返回，否则如实 unavailable；无 TTFT/取消率指标 | `P1`：真实模型联调后补 TTFT、取消率评测 |
| 后台入库任务 | × 未实现 | 同步流程便于当前调试 | 大文档阻塞请求，无进度和节点重试 | `P1`：先用轻量 worker/任务表，不为个人项目直接引入 MQ |
| 幂等与并发保护 | × 未实现 | — | 重复点击会重复处理；模型调用无并发上限 | `P1`：入库以 doc_id+profile_hash 幂等；模型入口使用 semaphore；多实例后再考虑 Redis |
| 模型路由/熔断 | × 未实现 | Chat、Embedding、Vision 配置已分离 | 无候选模型、健康状态和自动切换 | `P2`：先做 primary+fallback 和超时记录，再决定是否需要三态熔断 |
| Trace | √ 已实现 | 有 request ID、检索候选、分数、引用、QA budget 和总请求延迟；QA 响应已返回请求级 `trace.config`（配置快照/模型/索引名）、`trace.timings_ms`（retrieval/window/generation/citation/total）和 `trace.usage`（供应商未返回记 unavailable，不补零） | `HybridRouter.last_trace` 仍保留作调试兼容 | P0（评测）已完成：Runner 已原样录制请求级 trace 并写入 run JSONL（`production_adapter`） |
| 用户反馈 | × 未实现 | — | 真实失败不能沉淀为评估集 | `P1`：点赞/点踩、失败原因和备注；负反馈一键转 regression case |
| 前端问答页 | × 未实现 | 已有 API 与页面目标 | 无成品体验 | `P0`：完成上传、入库状态、提问、答案、引用预览五个核心状态 |
| 管理后台/审计 | × 未实现 | — | 无知识库、Chunk、Trace、评测管理 | `P2`：先做最小文档和 Trace 页面；配置审计等真实需要出现后再补 |

### 5.1 查询预处理设计（P1）

查询预处理需要区分不同问题，不能把“查询改写”当成一个含义模糊的开关：

| 问题 | 示例 | 首选处理 | 当前状态与优先级 |
|---|---|---|---|
| 多轮指代/省略 | 上轮讨论 embedding-3，本轮问“它最多支持多少 token” | 结合最近会话生成无需上下文也能理解的 `standalone_query` | 已接主链；当前历史由客户端随请求传入，服务端会话加载仍属 `P1` |
| 口语与文档文体差异 | “这玩意咋老卡死” vs “线程阻塞、锁等待、死锁” | 在不改变意图的前提下生成规范独立问句 | 已实现请求级开关并接主链，默认关闭；收益 A/B 与两级 `retrieval_query` 仍属 `P1` |
| 文档侧缺少用户问法 | 正文只写专业描述，没有自然问句 | 入库时生成 `question_kwd/question_tks` 并接入 BM25 | 生成器存在但未接入；作为索引侧互补方案评测 |
| 查询与答案体差异仍大 | 简短问题难以靠普通改写命中专业段落 | HyDE 生成假设答案，仅作为检索表示，不作为回答证据 | `P2` 条件候选；普通改写失败集证明必要后 A/B |
| 具体问题缺少背景原理 | “为什么这个锁在此处无效”需要先理解锁和并发模型 | Step-back 生成上位背景问题，与原问题分别检索后融合 | 已实现为默认关闭的实验开关；背景知识切片 A/B 未完成，未证明前不进入默认链路 |

下一阶段的 P1 目标是在现有 `standalone_query` 基础上扩展为一次调用输出“两级问题 + 澄清状态”；下面是目标契约，不是当前已经返回的结构：

```json
{
  "standalone_query": "Python 互斥锁为什么会导致程序卡死？",
  "retrieval_query": "Python 互斥锁导致线程阻塞、锁竞争或死锁的原因",
  "preserved_terms": ["Python", "互斥锁"],
  "rewrite_types": ["coreference", "colloquial"],
  "needs_clarification": false,
  "clarification_question": null
}
```

执行约束：

1. 根据 `session_id` 加载最近 N 个完整轮次；每轮包含一条 user 消息及其后续 assistant 回答，默认沿用现有组件的 6 轮、历史 2,048 token 上限，并同时服从模型总上下文预算。历史只用于理解指代和对话意图，不作为最终事实证据。
2. `standalone_query` 只补全指代、省略和相对日期；`retrieval_query` 再做口语到专业表达的受控规范化。人名、产品名、型号、错误码、函数名、数字、单位、时间、否定词和用户约束必须保留，不得把可能原因改写成确定事实。
3. 指代存在多个合理对象或无历史可解析时，返回 `needs_clarification=true` 并短路检索；不得猜测“她/它/这个”指向谁。
4. 历史按不可信 JSON 数据传给模型，输出必须通过严格 schema 和长度校验；模型不可用、超预算、输出非法或校验失败时回退原问题，并记录原因。
5. 检索使用 `retrieval_query`，回答生成围绕 `standalone_query` 且只依据 Evidence；响应 Trace 保存 `original_query`、两级改写、使用轮次/token、模型、改写类型、澄清状态和 fallback。
6. HyDE 和 Step-back 均不得直接替换基线或充当事实证据；必须以独立实验模式与普通改写同集比较 Recall/MRR、回答指标、P95 和单题成本。HyDE 优先用于文体差异失败，Step-back 优先用于背景原理缺失，未证明收益则不进入默认链路。

### 5.2 后端目标链路

P0 完成后的单轮链路：

```text
上传文档 → 登记状态 → 解析/分块/索引 → 提问 → 混合检索
→ 父块恢复 → 证据窗口 → 回答 → 引用 → 原文/图片预览
```

P1 完成后的多轮链路：

```text
加载最近 user+assistant 会话消息 → 指代消解 + 口语规范化
→ 歧义时澄清短路 → 简单意图路由/短路 → 多知识库检索
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
| 单元与集成测试 | √ 已验证 | 2026-09-21 在 Conda `agent` 重跑后端+评测：`164 passed` | 覆盖解析、分块、索引、检索、窗口、引用、生命周期 API、同步/SSE、`no_evidence`、多轮改写、口语规范化、Step-back 和评测契约 | 继续作为每次改动的回归门槛；真实 ES/模型/流式联调单独记录 |
| T2Retrieval 子集 | √ 历史资产 | 10,000 篇候选文档、500 Query、2,614 qrels；已有向量产物和评分脚本 | 保留已有产物，不删除、不继续扩建 | 只能做检索实验，不能覆盖回答与引用；不再作为当前主线 |
| CRUD-RAG Mini | √ 已冻结 | 1,000 篇文档、150 条问答，1/2/3 文档各 50 条；300 qrels | 固定 commit/seed/SHA；30 dev + 120 test；事件级隔离；validator 通过 | 原始与生成数据不提交 Git；通过构建器可复现 |
| 解析评估集 | √ 最小回归已验证 | Markdown/PDF 真实样例，以及 HTML/TXT、重复文本、实体、嵌套代码和表格文本回归 | 已约束内容存在、block types、来源精度分级、重复坐标和失败语义 | `P1`：扩到 20–30 篇并形成内容/结构/span 精度分布报告；不再重复列为算法 P0 |
| 分块评测 | √ 有不变量 | 长中文、代码、表格、边界和坐标样例 | 内容守恒、token 上限、父子引用和坐标合法性 | 增加聚合报告，不只输出 pytest pass/fail |
| 主线 RAG 评测集 | √ 已实现并完成真实 run（P0-B） | `CRUD-RAG-mini-v1` 共 150 条，答案与证据来自公开基准的固定子集 | Retrieval、Evidence、Citation、Latency 的确定性 scorer 与报告器已完成；真实 ES/模型 baseline 已由生产 Runner 录制（见 `评测.md`） | 剩余工作：语义 Judge 为 `P1` |
| Evidence 评测 | √ 已实现并完成真实 run（P0-B） | 基于 CRUD-RAG 的 reference news 与 qrels；真实生产 evidence 已录制并产出 baseline | Evidence Recall/Precision 按 overall 与 1/2/3docs 分片；锁定配置 evidence recall 93.1% | 剩余工作：claim-level Correctness/Coverage 归 `P1` Judge |
| 回答正确性 | × 未实现 | question、expected facts、answer | 先做必要事实覆盖和禁止事实；再接 LLM Judge | `P1`：LLM Judge 结果抽样人工校准，不能当绝对真值 |
| Faithfulness | × 未实现 | answer + 实际 evidence | claim 是否能被 evidence 支持 | `P1`：使用 Ragas 或自建 rubric；保存 judge 原始理由和失败 claim |
| 引用评测 | △ 文档级 scorer 已实现 | 可计算 Citation Validity 与 Citation Document Precision | 尚不能证明每个回答 claim 被对应引用支持 | `P1`：真实 run 先产出文档级 baseline，再用人工校准 Judge 做 claim-level Correctness/Coverage |
| 拒答评测 | × 未实现 | 业务集中 15%–20% 不可回答问题；生产 `no_evidence` 协议已就绪（2026-09-19），响应可被 Runner 原样录制 | 拒答 Precision、Recall、F1；误拒率和过回答率 | `P1`：不可回答问题切片随业务回归集建设后接入 scorer |
| Query Rewrite 评测 | △ 三策略已实现并接入主链（2026-09-21），单测齐全、切片级收益未测 | 多轮指代消解（`query_rewrite_enabled` 默认开、`history` 入口）、口语规范化（`colloquial_normalization_enabled`，无历史时生效）、Step-back 背景扩展（`step_back_enabled`，生成背景问题双路检索合并补漏、默认关闭）均为请求级开关；`trace.query_rewrite`/`trace.step_back` 记录结果与耗时；默认配置下评测 Runner 行为不变 | 改写前后检索收益未测：尚缺口语/多轮指代/歧义三个评测切片与 Recall/MRR、意图保持、误改写率对比；Step-back 双路合并对 Recall/MRR 与 P95 的影响未测 | `P1`：三个切片开关 A/B，记录 Recall/MRR、误改写率、P95 和成本；HyDE 维持 `P2` 条件候选（触发条件：文体差异失败切片） |
| Prompt Injection 安全集 | × 未实现 | 文档注入、用户注入、引用欺骗、越权请求 | 系统指令遵循、数据隔离、拒绝危险行为 | `P1`：首版 15–20 条，所有真实安全失败加入回归集 |
| 性能与成本 | √ 已实现并完成真实 run（P0-B） | QA 已返回 retrieval/window/generation/citation/total 分阶段耗时、配置快照和真实可得 usage；不可得字段明确为 `unavailable` | 真实 run 已产出 P95 2.3s 与成本（约 ¥2，见 `评测.md`） | `P1`：持久化 run 产物并在 SSE 后增加 TTFT/取消率 |
| Agent 评测 | × 未实现 | — | 将来评 Task Success、Tool/Argument Accuracy、Trajectory、Side-effect Safety | `P2`：Agent 尚未实现，当前不建设空框架 |

### 6.2 当前评测集分布

| 类型 | 总数 | Dev | Test | 主要验证 |
|---|---:|---:|---:|---|
| 单文档问答 | 50 | 10 | 40 | 基础召回、答案和引用 |
| 双文档问答 | 50 | 10 | 40 | 多证据覆盖与组合 |
| 三文档问答 | 50 | 10 | 40 | 深召回、证据缺失和排序 |

150 条问题使用互不重复的事件 ID；Dev 与 Test 也按事件隔离。所有正例文档强制进入 1,000 文档语料库，再加入普通与困难干扰项。CRUD-RAG 不覆盖不可回答、图片、权限和 Prompt Injection，这些能力需要以后用小型业务回归集补充，不得把它们包装成已测。

### 6.3 评测产物

独立顶级目录负责数据、实验编排和评分；生产算法仍保留在 `backend/`：

```text
evaluation/
├── build_dataset.py
├── validate_dataset.py
├── score_run.py
├── report.py
├── datasets/crud_rag_mini_v1/  # 本地生成，不提交 Git
├── runner/
│   ├── production_adapter.py
│   ├── index_dataset.py
│   ├── run_retrieval.py
│   ├── run_qa.py
│   └── sweep.py
├── experiments/                 # 受版本控制的实验配置
├── judge/                       # 固定 50 条、rubric、Judge 与人工校准
├── runs/          # 一次真实调用的完整录制，JSONL
├── scores/        # 可重复离线计算的 JSON
├── reports/generated/
└── failures/generated/
```

每条 run 至少保存：

- case_id、Git commit、运行时间。
- 解析/分块/Embedding/索引版本。
- 检索参数和模型配置快照（不保存密钥）。
- 原问题、独立问题、检索问题（启用后）、改写类型/澄清/fallback、使用的历史轮次与 token、候选结果、实际 evidence。
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

已完成并从待办移除：统一解析迁移与回归、CRUD-RAG Mini 数据/scorer、四项生产评测接口、算法 P0-A/P0-B 和后端 P0 编码闭环。当前不再新增后端 P0 功能；以下区分“代码完成”和“交付验收”：

1. **后端产品闭环**：√ 已完成（2026-09-19）：统一状态码（`ServiceError` 抛出点声明 400/404/409/413/422/502）、`no_evidence` 产品协议、文档 UUID 与 SHA256/size/created_at/counts、uploaded/indexing/ready/failed 状态机、detail/source/delete/retry、并发 409、上传大小限制、线程池改造与 API 集成替身测试。
2. **算法 P0-A**：√ 已完成（2026-09-19）；真实 VLM 模型联调与伴随图片上传通道归 P1。
3. **算法 P0-B / 评测 E1**：√ 已完成（2026-09-19）：全量索引、Dev 扫描锁定、Test baseline 均已真实执行，见 [`评测.md`](评测.md)。
4. **最小前端闭环**：上传、入库状态、问答、答案、引用及原文/图片预览；后端已就绪（含 SSE 流式与 `no_evidence` 协议）。

P0 验收：

- 新环境按文档一次启动成功。
- 真实上传→入库→问答→引用预览闭环通过。
- 单元/集成测试全部通过，真实 ES/模型有单独联调记录。
- 至少一份可复现 baseline 报告和失败案例清单。
- README 只陈述已经实测的数据。

当前状态：自动化回归与历史真实评测已完成；仍需在提交当前工作区后，用当前版本补一次真实 HTTP“上传→入库→问答→引用/图片预览”smoke，并补干净环境 bootstrap。完成这两项之前，只称“P0 已编码并回归”，不称“当前版本已完成部署验收”。

### P1：完整的多轮知识库产品

1. KnowledgeBase、Conversation、Message 和数据归属。
2. 实现 Conversation/Message/session_id，服务端加载最近完整 user+assistant 轮次；现有指代消解与口语规范化已接请求主链，剩余歧义澄清和对应评估集。
3. 简化意图分类和短路处理。
4. 多通道并行、失败隔离和请求级 Trace。
5. SSE 增强（真实模型流式联调、TTFT/取消率评测）、轻量后台入库任务。
6. 用户反馈转评估回归样例。
7. Faithfulness、Answer Correctness、Context Precision 等语义评测。

### P2：评测证明必要后再做

- OCR/复杂 PDF 版面恢复。
- 查询拆分和复杂意图树。
- HyDE：仅在普通查询改写的失败切片证明有稳定收益时实现；Step-back 已实现为默认关闭的实验开关，完成切片 A/B 前不进入默认链路。
- 知识图谱、联网搜索。
- 多供应商模型路由、三态熔断、Redis 公平排队和 MQ。
- 完整管理后台、审计系统。
- Agent、MCP、工具调用及 Agent trajectory 评测。

## 8. 明确暂不做的事情

- 暂缓索引侧 LLM 元数据生成（doc2query：important_kwd/question_kwd）——RAGFlow 同类开关（`auto_keywords`/`auto_questions`）默认关闭、逐子块一次 Chat 成本高，且当前 Golden Set 未证明 BM25 召回缺口；接入方案保留在 `compare.md` §4.1-4.3，同义词/口语切片失败证据触发后再做（2026-09-19 决定）。
- 不为了模仿其他项目堆叠功能数量。
- 不在没有业务失败数据时引入知识图谱和复杂 Agent。
- 不用 LLM Judge 替代 Hit/Recall/MRR、引用范围、延迟等确定性指标。
- 不同时引入多套评测框架；首版以自建 Python runner/scorer 为主，Ragas 只补语义指标。
- 不把测试替身结果写成真实模型联调结果。
- 不在文档中保存密钥、完整敏感 Prompt 或用户隐私数据。

## 9. 文档维护规则

1. 本文件是唯一的项目优先级总表；`评测.md` 只维护评测数据契约、真实运行记录与参数速查，不另行定义冲突的项目优先级。
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
