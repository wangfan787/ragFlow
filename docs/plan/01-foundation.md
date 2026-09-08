# 01 基础配置与公共契约

v1 / 2026-09-06。**基础、模型调用与主链路 Document 传递已接入；运行验证待 agent 环境，md-v1 字段和各阶段新功能仍待完成。** 无前置阶段，共用约定见 [plan.md](D:/code/ragFlow-main/docs/plan/plan.md)。

## 30 秒复习

模型调用只读 models.py：入库用 embed_documents，检索用 embed_query，QA 用 invoke。文本统一看 Document.page_content 和 metadata；不用再理解 ParseResultBlock、ChunkMeta、ChunkRecord、VectorRecord、RetrievedChunk 等容器。复习重点仍是父子切片、混合检索、命中窗口和来源；完整图文问答页面待后续阶段。

<details>
<summary>实施规格与交接</summary>

## 固定规格

- 文本统一使用 LangChain Document；metadata 字段以 plan.md 为准。contracts.py 只留 DocumentRecord、SourceSpan、MatchedChild 和一个共用 DocumentMetadata 提示，不给每个阶段再建一套类或别名。
- infrastructure/models.py 提供 build_chat(qa/vision)、build_embeddings()。延迟构造，缺 key 明确报错；新入口没有哈希向量或假答案回退，测试通过构造参数注入替身。
- ChatOpenAI：use_responses_api=False、timeout=60、max_retries=1；GLM 的 max_tokens 和 thinking={type:disabled} 放 extra_body，避免输出限制被改成 max_completion_tokens。
- QA：GLM-5.1、https://open.bigmodel.cn/api/coding/paas/v4、temperature=0.2、输出 1024；总上下文 32768、安全余量 256。视觉：glm-4.6v-flash、https://open.bigmodel.cn/api/paas/v4、temperature=0、输出 512。
- OpenAIEmbeddings：embedding-3、标准 bigmodel 地址、dimensions=1024、chunk_size=16、check_embedding_ctx_length=False、model_kwargs={encoding_format:float}。入库只发完整正文，超过 192 估算 tokens 或 3072 UTF-8 字节时拒绝并要求重新切片。
- 配置优先级：进程 MVP_* > 根 .env > backend/config.yaml > 默认值。保留原 MVP_EMBEDDING_*、MVP_QA_LLM_*、MVP_ELASTICSEARCH_* 映射，新增 MVP_VISION_LLM_* 和 MVP_DATA_DIR；只有视觉 key 可以缺省复用嵌入 key。
- 非空 QA 配置保留；原 QA 空字段显式迁入 GLM-5.1/coding 地址，不再从 metadata 运行回退，不输出凭据。旧 metadata 与 query_rewrite 配置及消费者已删除。
- count_tokens(text) 用 cl100k_base 工程估算，首次计数才加载词表；包含空白，异常抛出，不能报成零。不是 GLM 官方 tokenizer。保留有消费者的 SimpleTokenCounter 薄适配。
- GET /health 直接返回 {status:ok}，应用导入和健康检查不构造模型、不请求 ES。auth 暂留至 07；移除 main 对未挂载 QA 路由的导入。
- 数据路径相对仓库根，默认 data/md-rag/uploads 与 data/md-rag/documents.sqlite3；上传和 SQLite 统一使用配置入口。不自动移动或删除旧数据。

## 文件与删除边界

2026-09-06 本轮自动优化范围：按用户要求优先复用组件，完成 Document 在解析、切片、存储读取、检索结果和 QA/引用之间的传递，删除对应自建文本容器及重复字段转换。保留现有解析/分数/窗口算法和当前 ES 记录字段；最终 md-v1 字段、图片解析及 API 仍按后续阶段完成，不借本次重构改变召回规则、删除缓存数据或操作旧索引。现有顺序函数无需引入 LangGraph。

| 当前实现 | 核查的默认上游 | 解决问题 / 差异 | 本轮处理 |
|---|---|---|---|
| 解析 dict → ParseResultBlock → dict；ChunkMeta/ChunkRecord → 重复 meta | task_handler.py:588–618 的 ChunkService 返回 chunk 字典 | 上游不要求多套文本类，也不提供本项目来源坐标 | 采用统一 Document，保留来源字段；内部切片临时范围结构保留 |
| VectorRecord/VectorSearchResult/RetrievedChunk 多轮字段搬运 | rag/nlp/search.py:975 的父块读取、命中子块均分 | 上游直接用 chunk 字典；本项目还要保留所有命中与坐标 | ES 边界读写 Document，保留当前检索均分规则 |
| Citation 类创建后立即转 dict | rag/prompts/generator.py:140 的实际证据组装 | 上游不能替代本项目的命中窗口/原始行号要求 | QA/窗口接收 Document；引用直接构造响应，保留有效编号校验 |

源码 authority 沿用本地 0.27.1、无 Git commit；本轮复核 TE_RUN_MODE 默认 0。用户已要求自动执行上一轮提出的 Document 统一，无新增架构选择待确认。验证重点为正文/来源守恒、分数不变、窗口不修改父块、旧 ES 字段读写和 QA 引用；T2 适配器只随正式入口改表示，不改算法版本或重算产物。

Document 优化实际产物：删除 chunking/models.py、retrieval/models.py、citation/models.py、retrieval/metadata_fields.py；解析内部保留必要输入 ParseSource，切分内部保留范围片段。ES 写入接口为 upsert(documents,vectors_by_chunk_id)，向量不混入正文或 Document.metadata；读取统一恢复 Document。融合计算内部仍用分数字典，业务出口为 Document。answer/citations/trace 主字段沿用；retrieved_chunks 直接导出 Document 内容及现有 metadata，不再逐字段搬运/补齐闲置字段。

本轮生产目录 51→47 个 Python 文件，5209→4009 行；新增 391、删除 1591，净减 1200 行（含注释、空行）。测试/演示净减 854 行，单独统计。test_document_pipeline.py 新增实际解析至 QA 的替身链路、ES 边界读写、来源独立性和窗口输入不变验证；原父子守恒/均分/引用测试已迁移。解析和切片的打印测试改为断言；完整内容与报告合并到 show_results.py（--report），walkthrough_demo.py 保留真实父子输出。均未运行，不宣称行为对照已通过。

2026-09-06 用户再次要求实际简化后，提前完成已定 04/06 中的模型消费者迁移和相关删除：入库/检索用 embed_documents/embed_query，QA 用 invoke；删除旧聊天/嵌入封装、hash/cache、生成问题/关键词、问题改写及其专用测试。公共切片、父块聚合、窗口和来源算法保留，不能将这项删除工作等同于 02–07 全阶段完成。

| 本次替换实现 | 已核查上游 | 问题与差异 | 实际处理 |
|---|---|---|---|
| 旧模型协议、工厂和双层缓存 | task_handler.py:339 的模型绑定 | 本项目已选 LangChain，双入口增加调用层数 | 迁移所有生产调用方，删除旧协议/封装；测试替身遵守 LangChain 方法 |
| RetrievalMetadataGenerator + 元数据增强文本 | embedding_utils.py:54 的正文/问题选择 | 上游可用问题代替正文；首版已定仅完整子块正文 | 删除生成器与特征裁剪；正文长度越界拒绝送模，保留原位置 |
| QA 回退、改写、模型内预算接口 | generator.py:140 的证据组装 | 首版已定单轮、显式预算，失败不能假装回答 | invoke + 统一消息计数；错误显式返回，保留命中窗口和引用 |

保留当前模型/维度过滤及父块恢复；BM25 只查询文档名、章节和正文。当前旧 ES schema/索引不自动操作，04 新 mapping 就绪后再切 rag-md-v1；正文向量需要重建相关文档才能反映新输入，历史效果不能混报。T2 适配器和评测同时升级版本标识，旧向量产物不能继续复用。

核心文件：[models.py](D:/code/ragFlow-main/backend/src/infrastructure/models.py)、[settings.py](D:/code/ragFlow-main/backend/src/config/settings.py)、[contracts.py](D:/code/ragFlow-main/backend/src/contracts.py)、[token_counter.py](D:/code/ragFlow-main/backend/src/chunking/token_counter.py)。配置加载、路径和路由分别在 config/env.py、config/data_paths.py、main.py；backend/config.yaml 保存配置。

实际删除：embedding_factory.py、openai_embedding.py、openai_chat.py、hash_embedding.py、cached_embedding.py、retrieval_metadata.py、query_rewrite.py；删除对应旧协议、专用测试、未使用的 Settings 方法、重复 Document 别名。common_service.py/state_store.py 使用统一数据路径。

本轮简化统计（与本轮修改前快照比较，含注释/空行）：backend/src 从 58 文件、6646 行降至 51 文件、5209 行；新增 159、删除 1596，净减 1437 行。测试净减 332 行；连同未增减行数的脚本，项目自有 Python 总计净减 1769 行。不将删除测试算作生产代码减量。

## 已核查的源码依据

本地 RAGFlow 声明 0.27.1，无可核实 commit；[task_executor.py:1786](D:/code/ragFlow-main/ragflow/ragflow-main/rag/svr/task_executor.py:1786) 默认 TE_RUN_MODE=0 走 refactor，1 为对照运行，其他值为 legacy。

| 当前实现 | RAGFlow 依据 | 解决问题 | 差异/风险 | 本阶段修改 | 取舍 |
|---|---|---|---|---|---|
| 模型配置回退与多层封装 | [task_handler.py:339](D:/code/ragFlow-main/ragflow/ragflow-main/rag/svr/task_executor_refactor/task_handler.py:339) | 绑定模型 | 上游含租户、计费、探测 | 独立配置及 LangChain 入口 | 简化 |
| 导入加载词表、异常零计数 | [token_utils.py:55](D:/code/ragFlow-main/ragflow/ragflow-main/common/token_utils.py:55) | token 预算 | 上游部分异常仍返回 0 | 延迟加载，错误显式传播 | 采用延迟加载 |
| 多套阶段类型 | [task_handler.py:588](D:/code/ragFlow-main/ragflow/ragflow-main/rag/svr/task_executor_refactor/task_handler.py:588) 使用 chunk 字典 | 传递文本/来源 | 无本项目行号/父块坐标的直接等价物 | Document + 少量字段提示 | 不复制 schema |
| /health 间接初始化 QA | task_handler.py:384 绑定时 encode(["ok"]) | 探测 | 入库探测不适用于健康路由 | 健康检查与远端调用分开 | 本地实现 |

当前入库：DocumentService → IngestionPipeline → parser/chunker → EmbeddingIndexer → LangChain/ES。主链路文本均为 Document；解析 token 组装和检索分数计算内部仍可用局部字典。检索用 embed_query，QA 为检索 → 窗口 → 消息预算 → invoke → 引用。md-v1 字段尚待各阶段迁移；QA 路由待 07 挂载。

## 验证与实际状态

- **测试代码**：test_foundation.py 保留配置、GLM 请求体、视觉消息、原文嵌入/批次、路径/SQLite、token 和独立进程健康检查；test_repair_plan.py 调整真实消费者替身，覆盖延迟构造、非法向量禁止写入/查询、模型失败不返回假答案，保留父块、窗口、引用回归。
- **静态核查**：本轮 59 个项目自有 Python 文件 UTF-8 正常，113 处本地导入路径存在，无冲突标记或已删除模块的残留引用，差异空白检查无输出。这不等于 Python 导入或 pytest 通过。
- **未执行**：pytest、实际启动、真实聊天/视觉/嵌入。Windows 未找到 conda agent；WSL Ubuntu 用户为 study，登记 /home/xvxing/miniconda3/envs/agent/bin/python 不存在，常见位置未找到 agent；没有改用 Python 3.11/base。
- **外部条件**：根 .env 不存在、现有 YAML key 为空、localhost:9200 不可达。backend/requirements.txt 暂用 Python 3.10 兼容范围、LangChain 1.x、ES 8.x，尚无运行验证后的精确锁定版本；PyJWT/pypdf/beautifulsoup4 随旧消费者清理。
- **恢复环境**：先核实 agent/Python 3.10，检查已有依赖，只安装缺项；测试通过后仅固定直接依赖版本，不 freeze 整个环境。各模型真实调用一次再记录模型/向量维度，不输出 key。02 可按既定规格实现，但验收前须补齐本阶段运行验证。

恢复 agent 后，从仓库根目录使用其核实后的绝对路径运行（登记路径如下，命令尚未执行；T2 测试另外需要 numpy/pyarrow/faiss 和数据）：

```bash
/home/xvxing/miniconda3/envs/agent/bin/python -m pytest backend/tests/test_foundation.py backend/tests/test_document_pipeline.py backend/tests/test_repair_plan.py backend/tests/parsing_test/test_parsers.py backend/tests/chunking_test/test_chunking.py -k "not t2" -q
/home/xvxing/miniconda3/envs/agent/bin/python -m uvicorn backend.src.main:app --host 127.0.0.1 --port 8000
```

模型参数已查 [ChatOpenAI 源码](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py) 和 [OpenAIEmbeddings 源码](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/embeddings/base.py)。PyPI 的 [langchain-openai 1.6.0](https://pypi.org/project/langchain-openai/1.6.0/) 与 [langchain-core 1.6.2](https://pypi.org/project/langchain-core/1.6.2/) 声明支持 Python 3.10；这是发布元数据，不是本机已安装版本。

</details>
