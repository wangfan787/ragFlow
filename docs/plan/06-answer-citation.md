# 06 回答与引用

v1 / 2026-09-06。**QA invoke、消息预算及窗口/引用 Document 传递已接入；新引用/响应字段与本阶段验收待完成。** 前置 01–05；输出 JSON 以 [总约定](D:/code/ragFlow-main/docs/plan/plan.md) 为准。

## 30 秒复习

子块负责命中，384-token 窗口控制补充范围，引用回到实际证据。小窗口可能漏事实，大窗口可能带噪声；新窗口尚未实测，不先宣称提升。

<details>
<summary>直接实施规格</summary>

## 窗口算法已定

1. 输入阶段 05 的前 5 个父块。按父块 score 降序处理，父块内按子块 score 降序、parent_start 升序处理。
2. 父块估算 tokens<=384 时保留全部；否则以命中子块 [parent_start,parent_end) 为核心，沿两侧等字符半径扩展，用现有二分计数逻辑找到满足 384 tokens 的最大窗口。
3. 该父块的其他命中若已被窗口完整覆盖，不再生成重复窗口；未覆盖的继续生成。相交窗口能在 384 内合并则合并，否则保留各自片段，最多选择全局前 5 个窗口。
4. Evidence 的正文只保留窗口文本；window_start/end 和 spans 使用父块坐标，matched_children 保留所有与实际窗口相交的命中。
5. full_parent 仅供评测：对同一份父块输入使用完整父块，不重新检索。不实现 LLM 证据压缩、句子重排或回答阶段再次看图。

## 提示词和预算已定

- 使用阶段 01 的 QA ChatOpenAI，单轮、非流式。system 固定要求：“只依据给定证据回答；在事实后用 [编号] 标注来源；证据不足时仅回答‘未找到相关依据’；不要虚构来源。图片描述是模型提取结果，无法确认的细节不要猜测。”
- user 消息包含问题和顺序编号的证据，每项展示文档名、章节、正文；不塞数据库内部 ID、检索 trace 和向量。
- 先计算固定消息 tokens；按 32768 总预算扣 1024 输出和 256 余量，逐个加入证据。超预算时用现有命中锚定方式缩小当前窗口；连命中子块都容纳不了则跳过该窗口，不从父块前缀硬截取。
- 最终对整组消息重新计数；编号在最终证据确定后连续分配 1..N。记录实际 evidence tokens 与供应商 usage（若返回），不把工程估算称为模型精确 token 数。

## 回答和引用已定

- 没有可用证据时不调用聊天模型，返回总约定 no_evidence。
- 模型返回固定句“未找到相关依据”时也设 no_evidence、citations=[]；evidence 保留实际送入的片段，供检查为何拒答。
- 成功回答用正则抽取 [数字]，只保留指向实际 Evidence 的编号，按首次出现顺序去重。未知编号不作为有效 citation；不额外请求模型修复或进行复杂事实核验。
- citations 与 evidence 都使用 {id,doc_id,name,section_path,content,line_ranges,images}。line_ranges 从相交 spans 合并，images 从这些 spans 的 asset_id 查登记；图片 URL 由后端生成。
- 模型调用失败返回 502，不用证据前 120 字伪装答案。不保留 history、query_rewrite 或可调用对象兼容分支。

## 实施步骤和文件范围

- 修改 apps/services/context_window.py，复用现有二分/窗口合并核心并适配 Document；默认从 768 改为 384，不重写成另一个压缩框架。
- 精简 qa_service.py 为 retrieve → build windows → budget → invoke → citations 五步。
- citation/citation_service.py 仅负责有效编号与证据映射；旧 citation/models.py、retrieval/models.py 的冗余类型在消费者迁移后删除。
- 删除 apps/services/query_rewrite.py、test_query_rewrite.py、旧 openai_chat.py；清理 QA 工厂/模型回退和配置残留。
- qa.py 使用总约定的 question 请求与 QAResponse，移除历史和任意检索配置；新增 test_qa.py。

## 局部参考

| 当前实现 | RAGFlow 依据 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| 已有命中窗口，默认常容纳整父块 | dialog_service 与 kb_prompt | 给模型提供证据并引用 | 预算方式与窗口粒度不同 | 保留自有位置算法，使用 384 窗口和短响应 | 自有取舍 |

[当前窗口](D:/code/ragFlow-main/backend/src/apps/services/context_window.py:65)、[当前预算](D:/code/ragFlow-main/backend/src/apps/services/qa_service.py:192)、[上游证据](D:/code/ragFlow-main/ragflow/ragflow-main/rag/prompts/generator.py:140)。

## 验收与交接

- 父块末尾命中的事实必须进入实际模型消息；同父块多处命中仍能各自定位。
- 窗口和整组消息满足预算，引用只对应最终发送片段；图片 URL 能回到原图。
- 测试空候选、模型固定拒答和模型失败，结果遵守上述状态，不虚报生成成功。
- 用相同命中对照 window/full_parent 的 tokens 与必要事实，结果交给 08；没有真实模型时先使用替身。
- 当前：Document 容器迁移已实施；本阶段新增能力未完成，测试未运行。完成后将实际 QAResponse 交给 07。

</details>
