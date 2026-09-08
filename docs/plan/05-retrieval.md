# 05 混合检索与父块展开

v1 / 2026-09-06。**检索结果已统一 Document，现有融合与父块均分保留；md-v1 字段、入口精简和本阶段验收待完成。** 前置 01–04；字段按 [总约定](D:/code/ragFlow-main/docs/plan/plan.md)，不重新选择存储或融合算法。

## 30 秒复习

BM25 补术语匹配，向量补语义匹配；同父块多次命中聚合成一个父块结果，并保留子块位置。权重是起点，效果靠同一批问题对照，不是概率保证。

<details>
<summary>直接实施规格</summary>

## 固定检索流程

1. strip 问题，空问题或超过 3072 UTF-8 字节返回 400；用阶段 01 的 embed_query 得到 1024 维向量。
2. ES 向量搜索：field=vector，k=30，num_candidates=100，size=30；过滤 role=child、embedding_profile=当前模型配置。
3. BM25 用 multi_match，fields=[title^2,section^2,content]、type=best_fields、operator=or，size=30；使用同样 role/profile 过滤。standard analyzer 不安装额外插件，不宣称做了中文词级分词优化。
4. 以 chunk_id 合并两路命中；ES cosine 的原始 _score 已为 (1+cosine)/2，直接作为 vector_score，不再次归一化。keyword_score=该命中 BM25 分数/本次 BM25 最大分数，空路为 0。
5. child_score=0.75*vector_score+0.25*keyword_score；缺失通道按 0，去掉 score<0.1 的子块。mode=vector 时只执行向量路，score=vector_score，其他阶段一致。
6. 按 parent_id 分组，parent_score=组内已保留命中子块 score 的均值；matched_children 保存全部贡献子块及两路分数、位置。不得只保存最强子块。
7. 父块按 score 降序、chunk_id 升序打破平局，取 5 个；通过 mget 读取正文/metadata，再添加 score 与 matched_children。

不增加 rerank、查询改写、生成关键词、RRF 或第三路召回。两路可以直接顺序请求，先保持代码短；不专门建设并发调度。

## 返回与失败规则

- retrieve(question,mode=hybrid) 返回 RetrievedParent Document 列表，正文始终是父块完整规范化内容，证据裁剪交给 06。
- 没有候选返回 []。某一路正常返回空结果不影响另一路；ES 请求失败返回明确服务错误，不默默切成另一套算法。
- 索引不存在或父块记录缺失时提示“索引未就绪，请重新入库”，不制造没有位置的替代父块。
- 过滤只用于子块/模型匹配，首版没有权限能力，不加入空的 user_id/ACL 字段。

示例：同一子块 vector=0.8、keyword=0.4，融合为 0.7；同父块另一个子块融合为 0.5，则父块为 0.6，贡献列表保留两项。

## 实施步骤和文件范围

- 精简 retrieval/hybrid_router.py 为上述主流程，hybrid_fusion.py 保留短纯函数。
- embedding_retriever.py、keyword_retriever.py 的薄调用合并到 router；删除原文件并更新消费者。vectorizer.py、ranking.py、metadata_fields.py 只保留实际使用函数，能内联到对应阶段的短转换就内联。
- infrastructure/elasticsearch_store.py 实现上述两个查询；使用 04 的 mapping 和 profile，不继续搜索已删除的 question_kwd 等字段。
- 删除 infrastructure/rule_reranker.py、cross_encoder_reranker.py；retrieval_config.py 只保留本规格的参数，不保留运行时任意 filters/reranker 白名单体系。
- 更新 QA 调用入口的必要结构适配；回答算法由 06 完成。新增 test_retrieval.py。

## 局部参考

| 当前实现 | RAGFlow 依据 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| 自定义双路融合与父块均值 | search.retrieval / retrieval_by_children | 召回并恢复上下文 | 上游有更多存储与重排分支 | 保留现有权重、均值和子块位置，减少封装 | 简化自有实现 |

[当前路由](D:/code/ragFlow-main/backend/src/retrieval/hybrid_router.py:19)、[上游检索](D:/code/ragFlow-main/ragflow/ragflow-main/rag/nlp/search.py:597)、[上游父块恢复](D:/code/ragFlow-main/ragflow/ragflow-main/rag/nlp/search.py:975)。

## 验收与交接

- 替身分数验证上面的 0.7/0.6 例子、空通道和同父块聚合。
- 真实 ES 检查两路过滤生效，父块不会作为普通候选被召回。
- 语义问题和代码术语问题各测一个；输出与 06 共用的 RetrievedParent 样例。
- 当前：Document 容器迁移已实施；本阶段新增能力未完成，测试未运行。实施完更新状态，进入 06。

</details>
