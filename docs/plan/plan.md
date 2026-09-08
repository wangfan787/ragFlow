# Markdown 图文 RAG：首版实施总约定

版本：v1，2026-09-06。**设计已定。模型调用及主链路 Document 传递已接入；02–06 的重复文本容器已删除。图片解析、md-v1 字段/索引与最终 API 等产物待实施；运行验证待 agent 环境。**

本文件是 8 个阶段共同使用的契约。实施者直接执行这里的首版选择；“代码未实施”是进度说明，不是要求重新讨论设计。新对话使用 README 中的实施提示词即可。

## 1. 目标、范围和实施顺序

面向秋招的 Markdown 图文问答个人项目，重点讲清：信息保留、父子切片、混合检索、证据密度和引用。保留 FastAPI、Elasticsearch，用 LangChain 减少模型调用代码；各阶段独立，不加入 Agent、消息队列、账户系统或通用文件平台。

顺序固定为 01 基础 → 02 图文解析 → 03 切片 → 04 索引 → 05 检索 → 06 回答 → 07 网页 → 08 交付。新对话要求实施某阶段时，先检查前置产物，缺失的前置阶段按编号补齐，不重开设计讨论。

每阶段修改实际调用方并完成必要的内部结构适配，保持应用可导入；不保留两套最终生产链路。阶段 08 清理临时适配和旧代码。登录、权限、多轮记忆、重排、远程图片和非 Markdown 导入均不进入首版。

## 2. 固定技术选择与参数

| 项目 | 首版决定 |
|---|---|
| 运行 | conda agent / Python 3.10；从仓库根目录启动 FastAPI，端口 8000 |
| 模型组件 | langchain-core 的 Document；langchain-openai 的 ChatOpenAI、OpenAIEmbeddings |
| 聊天模型 | 将现有有效 QA 配置显式保留；当前 YAML 的空 QA 字段迁为 GLM-5.1 和既有 coding API 地址，不再运行 metadata 模型回退链 |
| 视觉模型 | glm-4.6v-flash，标准 bigmodel API 地址；独立配置，max_tokens=512、temperature=0、关闭思考 |
| 嵌入 | embedding-3，1024 维，批量 16；发送原始字符串，check_embedding_ctx_length=False |
| token 计数 | cl100k_base 作为统一工程估算；记录其不是 GLM 官方 tokenizer |
| 父子切片 | 父块目标 512 / 最大 768；子块目标 128 / 最大 192；无 overlap |
| 嵌入输入 | 仅子块正文，不生成检索问题/关键词，不拼接大段额外元数据 |
| ES | 沿用现有服务，官方客户端与服务主版本一致；新索引 rag-md-v1 |
| BM25 | ES 自带 standard analyzer；搜索 title^2、section^2、content，无中文插件安装 |
| 检索 | 每路 30 个子块、向量候选池 100；0.75 向量 + 0.25 BM25；阈值沿用 0.1；前 5 个父块 |
| QA 窗口 | 默认 384 tokens，每次最多 5 个证据窗口；完整父块仅作为评测对照模式 |
| QA 预算 | 配置总上下文 32768、输出预留 1024、安全余量 256；完整消息估算后再发送 |
| 网页 | 原生 HTML/CSS/JavaScript；FastAPI 同源；Marked + DOMPurify，静态资源本地保存 |
| 持久化 | 本地 data/md-rag/documents.sqlite3 与 data/md-rag/uploads；ES 保存检索记录 |

384-token 窗口是本次落实的首版决定：它让窗口在较大父块上实际收缩，同时仍容纳一个最大 192-token 子块。它不是“最优参数”或已测提升；不修改父子切片粒度，不自动调参。

环境变量 MVP_* > 仓库根 .env > backend/config.yaml > 上表默认值。聊天、视觉、嵌入使用独立配置，视觉 key 缺省可使用嵌入 key；不再从 metadata 配置隐式回退。已有私有配置迁移时不输出密钥。模型地址或凭据不可用属于外部联调条件，不改变已定设计。

## 3. 跨阶段数据契约

内部统一使用 LangChain Document(page_content, metadata)，不建立一套新的多层领域模型。contracts.py 只保留少量 TypedDict 和简短中文说明，各阶段共用一个 DocumentMetadata 字段提示，不重复定义父/子/检索/证据类或 Document 别名；API 请求由 Pydantic 校验。以下字段名称固定，所有阶段使用同一含义。

| 对象 | 固定字段 |
|---|---|
| DocumentRecord（SQLite JSON） | doc_id、name、source_path、batch_id、status、error、assets、counts |
| 通用 Document.metadata | doc_id、name、source_path、section_path、asset_ids |
| ParsedBlock.metadata 追加 | block_id、kind、language、line_start、line_end；kind 为 heading/text/code/table/image |
| Parent.metadata 追加 | chunk_id、role=parent、parent_id=null、spans |
| Child.metadata 追加 | chunk_id、role=child、parent_id、parent_start、parent_end、spans |
| SourceSpan（spans 元素） | start、end、line_start、line_end、asset_id；start/end 是规范化父块中的字符区间 |
| RetrievedParent.metadata 追加 | score、matched_children；每个命中含 chunk_id、score、vector_score、keyword_score、parent_start、parent_end |
| Evidence.metadata 追加 | evidence_id、window_start、window_end；spans 和 matched_children 只保留与本窗口相交项 |

字符区间统一 0-based、左闭右开；行号统一 1-based、两端包含。来源精度为原始结构块的行范围：部分截取代码或段落时，行范围可以宽于展示片段，不宣称逐字坐标。图片描述的行号定位 Markdown 图片引用，asset_id 指向原图，描述标记为生成内容。

assets 是以 asset_id 为键的字典，每项含 relative_path、mime_type、description；图片文件内容 SHA256 前 16 位作为 asset_id。doc_id 使用 UUID4；一次重新入库沿用 doc_id，一次新上传创建新文档，不自动跨上传合并。

新登记的 status=uploaded、error=null、assets={}、counts 全 0。无图片的 asset_ids=[]；非代码块 language=null。block_id=doc_id+":b:"+ordinal，ordinal 从 0 开始；evidence_id 是本次请求内从 1 开始的整数。

parent_id / chunk_id 使用 SHA256(schema_version + doc_id + role + ordinal + normalized_content) 前 32 位；schema_version=md-v1。同样输入重复切片得到相同 ID；正文或切片顺序改变，重建时覆盖该 doc_id 的全部索引记录。

spans 的 start/end 始终相对父块；解析阶段只有原文行号，阶段 03 才补字符映射。只有展示证据需要的来源放入 spans；检索生成特征与来源不能互相替代。

## 4. 固定模块入口

以下是目标公开 Python 入口，文件在原有目录内精简，禁止额外引入插件注册层：

| 阶段 | 调用约定 |
|---|---|
| 01 | build_chat(kind: qa/vision)、build_embeddings()、count_tokens(text) |
| 02 | DocumentService.upload_file(name, bytes) -> list[DocumentRecord]；parse_document(record) -> list[Document] |
| 03 | MarkdownChunker.chunk(blocks) -> list[Document]，返回 parents + children |
| 04 | EmbeddingIndexer.index(record, chunks) -> counts；IngestionPipeline.run(doc_id) -> DocumentRecord |
| 05 | HybridRouter.retrieve(question, mode=hybrid) -> list[Document]，返回父块及全部贡献子块；mode 支持 vector 供评测 |
| 06 | ContextWindowBuilder.build(parents, mode=window) -> list[Document]；QAService.query(question) -> QAResponse |
| 07 | 仅调用服务入口，不在路由/网页复制解析、检索或引用逻辑 |
| 08 | 复用上述入口；full_parent 模式仅改变窗口选取，不能改变召回结果 |

自动优化已将所有旧文本容器的消费者迁至 Document 并删除对应类型。当前 ES 字段及来源坐标名称保持既有语义，尚未全部改为上表 md-v1 名称。测试替身通过构造参数注入，不单独维护生产 fallback。

## 5. 对外 API 与状态

返回业务 JSON，不再套旧 ok(data) 包装。无登录、无客户端任意 filters。业务错误统一 HTTP 状态码 + {"detail":"中文可理解原因"}。

| 路由 | 请求 | 成功返回 |
|---|---|---|
| GET /health | 无 | {"status":"ok"}；不调用模型/ES |
| POST /documents | multipart file，.md 或 .zip | 201，{"documents":[DocumentSummary]} |
| GET /documents | 无 | {"documents":[DocumentSummary]}，按上传顺序 |
| POST /documents/ingest | {"doc_id":"..."} | {"document":DocumentSummary}；同步等待入库结束 |
| GET /documents/{doc_id}/source | 无 | {"doc_id":"...","name":"...","content":"原始 Markdown"} |
| GET /documents/{doc_id}/assets/{asset_id} | 无 | 对应图片字节与 MIME |
| POST /qa/query | {"question":"非空字符串"} | QAResponse |

DocumentSummary = {doc_id,name,source_path,status,error,counts}；counts = {blocks,parents,children,images}，未入库时均为 0。status 流转 uploaded → indexing → ready；失败为 failed，error 为错误原因；可对 uploaded/ready/failed 重试入库。同 doc_id 正在 indexing 时返回 409。

QAResponse = {status,answer,citations,evidence,elapsed_ms}；status 仅 ok/no_evidence。evidence 是所有实际送入模型的证据，citations 是答案确实引用的有效编号，二者使用同一对象结构：

{id,doc_id,name,section_path,content,line_ranges,images}

id 为本次请求内从 1 开始的编号；line_ranges 是 [[start,end],...]；images 是 [{asset_id,url,description}]。路径信息取存储元数据，不依赖模型编造。

无候选返回 200 + no_evidence，answer 固定“未找到相关依据”，其余列表为空。候选存在但模型返回同一句拒答时，status 同为 no_evidence、citations 为空，evidence 保留实际发送片段供检查。模型/ES/入库外部调用失败返回 502 或 503；缺失文档/图片 404，空问题、问题超过 3072 UTF-8 字节或不支持的文件 400。不建设额外错误码体系。

## 6. 验证、删除和完成判定

每阶段文件给出实施步骤、精确文件范围和核心验收。只运行与该阶段相关的检查；最后做一次真实端到端验收，不建立全面企业测试体系。

首版移除：独立 PDF/TXT/HTML 导入、旧 Markdown 路径、认证/JWT、问题改写、自动关键词/问题、重排器、哈希/自定义向量缓存、旧 T2/FAISS 产物框架和无消费者字段。原始数据、旧 ES 索引和本地 RAGFlow 参考源码不自动删除。

agent 环境、模型凭据或 ES 服务缺失时：仍完成代码、替身验证和启动所需静态检查，记录“外部联调未完成”，不把它写成待设计，也不虚报验收完成。状态使用“待实施 → 实施中 → 已实现/外部联调未完成 → 已验证完成”。

## 7. 源码参考与本次决策

本地 RAGFlow 声明 0.27.1，TE_RUN_MODE 默认 0，走 task_executor_refactor；没有精确 Git commit。只借鉴相关机制，不完整复刻。阶段文件的短表记录采用、简化或不采用的理由。

首版已定：统一 Document 契约；standard analyzer；384-token 窗口；本地图片缺失时入库失败并可重试；回答阶段不再次看图；同步非流式接口；Marked + DOMPurify；精确 API、状态和测试范围。

2026-09-06 实施进度：01 基础已写入；入库/检索直接调用 LangChain 嵌入，QA 调用 invoke，删除七个旧模型/缓存/元数据生成/改写模块，上传/SQLite 接入统一目录。保留并调整核心测试；agent 缺失，pytest、启动、依赖精确锁定和模型/ES 联调未完成。当前仍用旧 ES schema，04 新 mapping 就绪后切换 rag-md-v1；正文嵌入变更需重建相关文档，旧数据未自动操作。净减量与细节见 01。

后续自动优化：Document 已贯穿解析/切片/存储读取/检索结果/窗口/引用；删除重复的容器、转换和闲置 IngestionConfig。父子聚合、分数权重、模型参数和嵌入输入不变，T2 输出算法版本不变。本轮无需因容器变化重建现有索引；未连接或修改外部数据。静态核查完成，新增替身链路测试尚待 agent 执行。

参考：[LangChain 聊天接入](https://docs.langchain.com/oss/python/integrations/chat/openai)、[Embedding-3 输入与维度](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%9C%AC%E5%B5%8C%E5%85%A5)、[ES 向量与分数](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector/)、[Marked 渲染](https://marked.js.org/)。
