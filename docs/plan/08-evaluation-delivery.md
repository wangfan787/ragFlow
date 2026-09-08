# 08 评测、清理与交付

v1 / 2026-09-06。**设计已定，可直接实施；代码尚未实施。** 前置 01–07。评测调用正式生产入口，不独立复刻 parser、chunker 或检索。

## 30 秒复习

用小语料证明召回和答案可检查，再用相同命中比较整父块/窗口的证据 tokens 与事实保留。只报告实际结果，不能预写改善百分比。

<details>
<summary>直接实施规格</summary>

## 语料和标注流程已定

1. 使用 [llm-universe/docs](https://github.com/datawhalechina/llm-universe/tree/main/docs)，在实施时记录获取的 commit 和许可证。下载所选 Markdown 及其引用本地图片，保留相对路径；不复制上游业务代码。
2. 选择规则：路径排序，排除 README/LICENSE/导航页，优先取 6 篇包含支持格式本地图片的 Markdown，再从剩余文档取 6 篇。不足的类别用另一类别按序补足；保存实际 12 篇清单到 manifest.json，后续不重新随机选取。
3. 保存到 dataset/demo/corpus，依靠阶段 02 的文档包路径规则导入。实施 AI 阅读原文/原图完成 30 条标注：20 条文字/代码/表格题、5 条只能从图片得出关键信息的题、5 条语料没有答案的题；不要求用户先提供数据或答案。
4. 每条 questions.jsonl = {id,question,relevant_sources,expected_facts,kind}。relevant_sources 使用 corpus 内 source_path；无答案题为 []。expected_facts 是人工核对的简短事实列表，kind 为 text/image/unanswerable。
5. 图片题的 expected_facts 仅放入标注，不作为语料导入，避免答案泄漏。标注由实施者按实际原图核对，模型描述不能作为唯一真值。

## 评测程序已定

- 新建 dataset/evaluate_demo.py，复用 DocumentService.ingest、HybridRouter.retrieve 和 ContextWindowBuilder.build。CLI 提供 --ingest、--mode vector/hybrid/both、--answer；默认 both，不默认调用聊天模型。
- --ingest 把 corpus 打成同结构文档包走正式上传/入库，保存 source_path → doc_id 映射。再次运行优先复用本地 manifest 的 doc_id 和文档登记，不按重新生成 UUID 猜测 qrels。
- 向量和混合检索使用同一索引、同一问题与候选预算。父块分数仍按生产逻辑形成；文档级分数取该文档最高父块分数，去重后计算前 5。
- 仅对 25 条有答案题计算 Recall@5=命中相关文档数/相关文档数，MRR@5=首个相关文档排名倒数，无命中为 0；平均值按题计算。5 条无答案题单独检查是否拒答，不混入检索均值。
- 选固定 5 条有答案题（按问题 id，至少一条 image）做密度对照：只检索一次，把相同父块分别送给 window/full_parent；记录 evidence tokens、事实覆盖与无关内容的简短人工备注。该小样本用于说明取舍，不宣称普遍保证密度。
- --answer 对全部 30 题调用 QA；对上述 5 题额外用完整父块证据执行一次同样生成步骤。复用 QA 内部“给定证据生成回答”的短函数，不复制提示词或增加第二套 QA。

## 固定结果文件

dataset/demo/manifest.json：schema_version=demo-v1、upstream_commit、sources、doc_ids、index、embedding_model/dimensions、生成时间。

dataset/demo/results/{run_id}/results.jsonl：{question_id,mode,retrieved_source_paths,recall_at_5,mrr_at_5,retrieval_ms,evidence_tokens,answer,manual_notes}；没有答案调用时 answer=null，无答案题两个检索指标为 null。

summary.json 保存有效题数、均值、模型/检索/窗口配置和外部失败列表；density.jsonl 单独保存 5 题两种窗口的实际片段、tokens、expected_facts 覆盖备注。所有时间和指标来自实测。

不保存向量 memmap/FAISS 文件、不设计断点续跑。失败保留已输出逐题结果，下一次完整重跑这 30 题即可。

## 实施步骤：清理与交付

1. 汇总并运行 01–07 的核心测试，完成一次真实 ES/模型/网页演示；缺外部条件明确记录，不重新讨论业务设计。
2. 删除 dataset/download_t2retrieval.py、embed_t2retrieval.py、evaluate_t2retrieval.py、apps/services/benchmark_adapter.py，以及 FAISS/pyarrow 等仅为旧基准使用的依赖。原始数据文件不自动删除。
3. 搜索旧 parser、auth、query_rewrite、metadata、reranker、缓存、类型和配置的剩余导入，删除临时适配及过时测试；保留正文、来源、父子和 API 核心验证。
4. requirements 固定真实可用版本，清理旧脚本/说明。后端和网页自有代码争取约为原规模一半，不通过挤成一行来减少统计。
5. 生成仓库 README：环境、模型配置、启动、演示、测试和数据来源。生成 docs/interview.md：一张流程图、3 个核心取舍、真实指标和已知限制；无需再创建面试资料体系。
6. 更新总方案及每阶段实际状态。写明“已实现”“已验证”和“外部联调未完成”的具体项，简历建议只引用已实测能力。

## 参考边界与验收

| 当前实现 | 上游关系 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| T2 大语料和独立向量产物链路 | 不照搬上游评测平台 | 为项目提供效果证据 | 旧脚本不能验证 Markdown 图片/正式混合检索/引用 | 12 篇、30 题复用生产链路 | 替换为小型评测 |

[旧评测](D:/code/ragFlow-main/dataset/evaluate_t2retrieval.py)、[旧适配](D:/code/ragFlow-main/backend/src/apps/services/benchmark_adapter.py)、[核心回归案例](D:/code/ragFlow-main/backend/tests/test_repair_plan.py)。

完成标准：可复现的语料与标注、逐题结果和汇总、真实图文演示、无旧依赖残留、可直接复习的面试说明。当前：代码未实施；测试未运行。

</details>
