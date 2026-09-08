# 04 嵌入与索引

v1 / 2026-09-06。**LangChain 嵌入、正文校验与 Document 存储读写已接入；新 ES mapping、md-v1 字段和本阶段验收待完成。** 前置 01–03；公共类型和路径见 [总约定](D:/code/ragFlow-main/docs/plan/plan.md)。

## 30 秒复习

子块向量负责匹配，父块正文负责补充解释。来源只需正确存储和传递，不必全部加入嵌入。新索引重建使规则简单，不维护旧版本兼容。

<details>
<summary>直接实施规格</summary>

## 嵌入输入已定

- 仅 embed_documents(child.page_content)，不生成问题、关键词，不拼接标题。标题和章节保留给 BM25 及引用。
- embedding-3 / 1024 维 / 每批 16，check_embedding_ctx_length=False；接收原始字符串，校验向量数量、维度、非零和有限值。
- 子块不超过 192 个工程估算 tokens；额外用 UTF-8 字节数不超过 3072 作为保守输入护栏。这两个上限由阶段 03 同时保证；索引器发现违反时拒绝入库并提示重新切片，不临时改写子块和位置，更不交给适配器截断。该护栏不是 GLM 官方 token 计数。
- 有凭据时用文本、代码、中文样例核对 API 实际接受情况与 usage；失败时保留完整输入，报告具体限制，不伪称完成输入长度实测。

## ES mapping 与记录

新索引固定 rag-md-v1，配置可显式覆盖；绝不复用旧 rag-mvp-chunks。无旧字段 backfill 和兼容读取。

| 字段 | mapping / 用途 |
|---|---|
| chunk_id、doc_id、parent_id、role、embedding_profile | keyword；标识、过滤和父块读取 |
| title、section、content | text + standard analyzer；title=name，section=section_path 用空格连接 |
| vector | dense_vector，dims=1024，index=true，similarity=cosine；仅子块写入 |
| metadata | object、enabled=false；保存完整 metadata（含位置、图片、spans），只取回不搜索 |

ES _id=chunk_id；embedding_profile=SHA256(model + normalized_base_url + dimensions) 前 16 位。父块也保存该 profile，但没有 vector。查询时使用顶层字段过滤，返回后由 metadata 恢复 Document，避免再造第二份类型。

ESStore 保留 ensure_index、delete_by_doc_id、upsert、query_by_ids、vector_search、keyword_search 六个短方法。用官方 helpers.bulk，写完 refresh 后返回成功。映射维度不匹配时明确报错并提示使用新的配置索引名，不自动删除已有索引。

## 入库流程与重建

1. DocumentService.ingest 按 doc_id 读登记；不存在 404，正在 indexing 409；由该服务设置 indexing、清空 error。Pipeline 负责内容处理，不另起一套状态更新。
2. 调阶段 02 解析和图片描述 → 阶段 03 父子切片 → 本阶段对子块生成向量，先在内存准备完整记录。
3. 准备成功后删除新索引内该 doc_id 的旧记录，再 bulk 写入父子记录并 refresh；Pipeline 返回更新后的 record（含 assets/counts），DocumentService 登记 ready。
4. 解析/模型阶段失败时设 failed；已有上次成功记录可以保留。删除后的 bulk 失败则尽力清理本次该 doc_id 的部分记录，错误保留，用户可重新入库；不建设原子发布或恢复平台。
5. 原始文档、图片和描述缓存不随重入库删除。新上传是新 doc_id，点击“重新入库”才沿用旧 doc_id。

## 实施步骤和文件范围

- 修改 indexing/embedding_text_builder.py 为简短正文/护栏处理；embedding_indexer.py 调 LangChain；elasticsearch_store.py 按上表精简。
- IngestionPipeline.run(doc_id) 串联解析/切片/索引，DocumentService.ingest 负责调用与文档状态；两处只保留一套状态更新职责，状态在 DocumentService 内落库。
- 删除 indexing/retrieval_metadata.py；删除已迁移的 openai_embedding.py、embedding_factory.py、hash_embedding.py、cached_embedding.py；清理其导入和参数。
- 更新旧存储调用方到 metadata / 顶层 mapping；旧重建脚本替换为调用 IngestionPipeline 的简短脚本，不保留旧索引协议。
- 新增 test_indexing.py；只为真实写入契约、子块向量和重建行为写核心测试。

## 局部参考

| 当前实现 | RAGFlow 依据 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| 正文加可选特征、父块无向量 | 默认 embedding_service / embedding_utils | 准备模型输入 | 上游可用生成问题代替正文并预截断 | 仅用完整子块正文，元数据独立存储 | 简化并保留正文 |

[当前索引器](D:/code/ragFlow-main/backend/src/indexing/embedding_indexer.py:14)、[上游嵌入](D:/code/ragFlow-main/ragflow/ragflow-main/rag/svr/task_executor_refactor/embedding_service.py)、[上游输入](D:/code/ragFlow-main/ragflow/ragflow-main/rag/svr/task_executor_refactor/embedding_utils.py)。

## 验收与交接

- 替身确认父块不调用嵌入，返回的所有子块有 1024 维向量，正文未替换。
- 真实 ES 中可按 ID 读父子 metadata；重复入库后该 doc_id 没有旧块残留。
- 服务重启后列表和原文可读；模拟写入失败不会登记 ready。
- 完整入库返回 counts={blocks,parents,children,images}，交给 05/07。当前：Document 容器迁移已实施；本阶段新增能力未完成，测试未运行。

</details>
