# 03 父子切片

v1 / 2026-09-06。**现有父子切片已输出 Document，重复容器已删除；md-v1 字段、spans/图片映射和本阶段验收待完成。** 前置：[01](D:/code/ragFlow-main/docs/plan/01-foundation.md)、[02](D:/code/ragFlow-main/docs/plan/02-document-parsing.md)。接口与 metadata 使用 [总约定](D:/code/ragFlow-main/docs/plan/plan.md)。

## 30 秒复习

小块用于定位，父块补充解释。块越短不一定信息密度越高；核心是保留正文、位置和必要上下文。已有切片算法可精简复用，不引入新的语义切片模型。

<details>
<summary>直接实施规格</summary>

## 固定算法

1. 输入有序 ParsedBlock。按 section_path 分组，新标题开始前结束当前父块；标题文本保留在新组中，纯标题文档也能形成块。
2. 块正文只统一 LF、移除块首尾空行，以两个换行连接；代码内部缩进和表格内容不改。保留既有长度计数/超长拆分核心，不以 tokenizer 截断正文。
3. 父块目标 512、硬上限 768 tokens：顺序合并普通块，超过目标时在块边界结束；预算内代码/表格整体加入，必要时独占一个父块。
4. 单个块超过 768 时，用现有二分字符前缀方法分成连续片段，尽量在换行或句末结束；每段都重新计数并保证上限，不能只取第一段。
5. 父块内部生成连续、不重叠子块，目标 128、硬上限 192 tokens，同时保证子块 UTF-8 字节数不超过 3072。同样优先句末/换行，再用同时满足两个上限的字符前缀拆分；无摘要、无额外模型、无 overlap。
6. 保证 ''.join(children.page_content) == parent.page_content。即使跨代码或表格拆分也保留每个字符；子块仅用于检索，网页优先展示有上下文的证据。
7. 图片描述按普通可检索文本切分，所有相关片段继续关联原图；不复制原图多次或把 base64 写入文本。

## 位置和 ID

- 组装父块时，按每个源块的加入位置建立 spans：start/end 为父块字符位置，line_start/line_end 为该源块原文行范围；图片 span 额外带 asset_id。
- 子块 parent_start/parent_end 指向父块；spans 选择与子块相交的源块项，保持父块坐标，不减去子块起点。
- 来源只承诺结构块行级准确度；新增的块间换行属于规范化，没有伪造原文字符映射。
- ID 按总约定 md-v1 + doc_id + role + ordinal + content 的 SHA256 生成；相同输入稳定，内容变化使相关 ID 更新。

样例：父块“甲段\n\n乙段”的两个子块可以是“甲段\n\n”和“乙段”，位置分别 [0,4)、[4,6)。行号取各自源块，不能把 4 这个父块偏移当成原文第 4 行。

## 实施步骤和文件范围

- 精简 chunking/markdown_chunker.py 与 block_merge.py，输出总约定的 Document 列表（先 parents 后 children）。
- chunk_config.py 仅保留本阶段四个 token 数；token_counter.py 使用阶段 01 的统一计数。
- 将 chunking/models.py、parsing/models.py 中仍需的数据改为总约定 metadata；消费者迁移后删除冗余类型，保留必要 TypedDict 注释，不重复嵌套同一来源。
- 更新 IngestionPipeline.chunk_stage 的消费者适配，不保留 v2/legacy 双主路径。实际 ES 写入继续交给阶段 04。
- 保留 test_repair_plan.py 中有价值的正文保留/位置案例，迁入 test_chunking.py；删除镜像实现、过时参数和未使用模型字段测试。

## 局部参考

| 当前实现 | RAGFlow 依据 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| 已有硬预算、父子位置 | naive 段落合并与可选 children delimiter | 连接召回粒度和上下文 | 上游并非默认全部启用同一父子算法，超长处理也不同 | 精简现有算法，保持内容连续 | 保留自有实现 |

[当前合并](D:/code/ragFlow-main/backend/src/chunking/block_merge.py)、[上游合并](D:/code/ragFlow-main/ragflow/ragflow-main/rag/app/naive.py:1378)、[上游子块标记](D:/code/ragFlow-main/ragflow/ragflow-main/rag/nlp/__init__.py:458)。

## 验收与交接

- 文本、代码、表格、图片描述四类样例都满足父子长度限制和正文重组等式。
- 一个超过 768 tokens 的代码块尾部内容可在子块中找到；子块位置切片等于实际子块正文。
- 连续运行两次 ID 相同；正文变化影响对应 ID，原文行范围与图片关联保留。
- 输出父子 Document 样例供 04/05 使用。当前：Document 容器迁移已实施；本阶段新增能力未完成，测试未运行。

</details>
