# 02 文档导入与图文解析

v1 / 2026-09-06。**现有解析器已输出 Document；文档包、图片理解、md-v1 解析字段和本阶段验收待完成。** 前置：[01](D:/code/ragFlow-main/docs/plan/01-foundation.md)。字段与 API 以 [总约定](D:/code/ragFlow-main/docs/plan/plan.md) 为准。

## 30 秒复习

Markdown 结构解析保留正文与来源；图片用视觉模型读出文字和关系，再关联原图。描述可能遗漏细节，所以保留原图供检查。当前图片理解还没有完成。

<details>
<summary>直接实施规格</summary>

## 固定决定

- 接收 UTF-8/UTF-8-BOM 的 .md 或 ZIP；ZIP 可以有多篇 Markdown 和本地 PNG/JPEG/WebP。拒绝无 Markdown 的包；不实现远程图片抓取和独立 PDF/TXT/HTML 导入。
- 一次上传生成 batch_id，解包到 data/md-rag/uploads/{batch_id}/original；保留相对目录。每个 Markdown 分配 UUID doc_id，source_path 为包内 POSIX 相对路径。
- ZIP 条目和最终图片解析路径必须落在该 batch 的 original 目录内；Markdown 中 ../images 的引用只要最终仍在包内即允许。这是文件保存边界，不增加文件安全平台。
- 文档引用本地图片但资源缺失/格式不支持时，入库失败并记录原因；远程图片不请求网络，以“[远程图片未解析：alt/url]”保留为普通文本，不使文档整体失败，也不冒充已经理解图片。
- 视觉模型在入库时调用；不在每次提问时重新调用。以图片内容 SHA256 为缓存键，在 batch/vision/{hash}.json 保存模型名和描述，同模型同图片重入库复用。

## 输出与解析规则

1. markdown-it-py 开启 table。按原文顺序产出 Document：heading、text、code、table、image；保留 section_path 和原文行号。
2. 普通正文保留可见文字；列表、引用保留原始 Markdown 结构。代码保留完整 fence、语言标记和缩进；表格保留表头/分隔行。只统一换行符，移除 BOM，块间空行由 03 统一。
3. 图片作为独立 image 块插在其原位置。一个段落有图文时拆为相邻文本/图片块；重复引用同图可以产生多个位置，但共享 asset_id 和描述缓存。
4. Markdown 内常见 <img> 解析为图片引用，其他可见 HTML 文本用标准库 HTMLParser 提取；script/style、HTML 注释和 frontmatter 不作为正文，frontmatter title 可作为 name 的补充。排除项有明文规则，不扩展完整 HTML 导入。
5. 图片描述固定提示词：“提取图中可见文字、数字、节点关系和与文档相关的含义。按【文字】【关系】【说明】输出简短中文；无法辨认写无法辨认，不猜测。”发送 base64 data URL；不使用需要额外解析的 JSON 输出格式。
6. image 块正文为“[图片内容描述，模型生成]”加描述；metadata 带 asset_id、图片引用所在行号和 section_path。assets 字典保存 relative_path、mime_type、description。

## 实施步骤和文件范围

- 修改 document_service.py，实现 upload_file 返回 DocumentRecord 列表；修改 state_store.py 保留 SQLite documents(doc_id PRIMARY KEY,payload JSON) 的简短 CRUD，路径改到新 data root，移除内存 chunk 缓存辅助。
- 精简 parsing/markdown_parser_it.py；新增 parsing/image_describer.py，负责 VLM 调用及文件缓存。parser_factory.py 仅保留 Markdown 主入口，并迁移所有正式调用方。
- 删 parsing/markdown_parser.py、pdf_parser.py、text_parser.py；html_parser.py 的必要文本提取迁到 Markdown 文件后删除独立解析器。删与这些专用导入绑定的测试/依赖，保留原始样本数据。
- documents.py 更新上传/列表，并提供总约定的 source 与 assets GET；所有文件读取先通过 doc_id/asset_id 的登记记录定位，前端不能传任意磁盘路径。
- 入库服务此阶段通过 parse_document(record) 调用，返回约定 Document。后续索引由 04 完成；需要过渡时只做薄返回结构适配，不复制解析算法。

## 局部参考

| 当前实现 | RAGFlow 依据 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| Markdown 有结构、图片尚无理解 | naive 的 Markdown/VisionFigureParser 路径 | 把图像变成可检索信息 | 上游图片理解可选，本项目是首版目标 | 独立图片描述与原图关联 | 借鉴并简化 |

[当前 Markdown](D:/code/ragFlow-main/backend/src/parsing/markdown_parser_it.py:10)、[文档服务](D:/code/ragFlow-main/backend/src/apps/services/document_service.py:46)、[上游视觉路径](D:/code/ragFlow-main/ragflow/ragflow-main/rag/app/naive.py:1269)。

## 验收与交接

- test_parsing.py：一个 ZIP 中两篇 Markdown、共享图片、代码和表格；检查路径、语言、标题、行号和 image 块。
- 本地图片缺失时明确失败；同模型同图重入库不重复请求视觉服务。
- 用一张包含独有数字/流程关系的真实图验证描述，核对原图；没有真实模型时先替身验证并记录外部联调缺失。
- 输出 DocumentRecord + ParsedBlock 样例给 03/04，不让下游重新定义字段。当前：Document 容器迁移已实施；本阶段新增能力未完成，测试未运行。

</details>
