# 07 网页与接口集成

v1 / 2026-09-06。**设计已定，可直接实施；代码尚未实施。** 前置 01–06；API 直接使用 [总约定](D:/code/ragFlow-main/docs/plan/plan.md)，不由前端另定义字段。

## 30 秒复习

演示闭环：上传 → 入库 → 提问 → 看回答 → 打开原文和图片。用轻量单页承载已有 RAG 能力，账户与管理后台延后。

<details>
<summary>直接实施规格</summary>

## 页面与交互已定

- 新建 backend/web/index.html、app.js、style.css，静态资源路径 /static；GET / 返回页面。无 React、构建链、Gradio、登录页或设置后台。
- 浅色页面，280px 左侧文档栏，主内容最大 1100px；正文用 Segoe UI / Microsoft YaHei，代码等宽。窄屏小于 800px 时改上下布局。
- 左栏包含单文件/ZIP 上传、文档名、状态、入库/重新入库按钮。上传成功立即显示列表，浏览器按顺序调用 ingest 自动入库，显示每篇等待/处理中/成功/失败。
- 同步非流式请求：HTTP 等待结果；FastAPI 路由用 async def，阻塞服务统一 await run_in_threadpool 执行。上传文件先 await file.read()，再交给线程中的 DocumentService；不加后台任务队列、轮询任务系统或 SSE。
- 主区提供问题输入、提交、固定加载提示和回答区；提交期间禁用重复按钮，显示后端返回的耗时。只保留当前页面这次会话的内容，刷新无需恢复聊天。
- 点击引用打开右侧面板：文档名、章节、行范围、实际证据文本、原图缩略图。“查看原文”请求 source 并用带行号的纯文本展示；点击图片打开原图。
- no_evidence 直接显示“未找到相关依据”；HTTP 失败显示 detail，保留问题供重试。文档失败显示 error 和重新入库按钮。

## 渲染与接口约定

回答使用 Marked 解析，再经过 DOMPurify；代码/表格使用简单 CSS，不增加高亮框架。证据与原文使用 textContent/pre 展示，避免截断代码 fence 导致布局错乱。

Marked 与 DOMPurify 浏览器发行文件保存在 backend/web/vendor，记录版本和来源，不依赖演示时临时拉取 CDN。保留其许可证，不计入自有代码统计。

只请求总约定中的 /documents、/documents/ingest、/qa/query、source、assets。图片 URL 和引用结构使用后端返回值；不把 key、模型地址、ES 字段或内部 trace 放进默认界面。

## 实施步骤和文件范围

1. 完成三个前端文件和 vendor 资源，通过 FastAPI 挂载。先用固定响应替身检查布局，再接真实接口，不在最终代码保留假数据模式。
2. main.py 注册 documents、qa、health 和静态页面；路由统一返回总约定 JSON，不再套旧 ok 包装。
3. documents.py 调 DocumentService；qa.py 调 QAService。确保阻塞服务在工作线程执行，服务层保持独立可测试。
4. 删除 apps/restful_apis/auth.py、所有 require_authenticated/JWT 调用和专用依赖。common_service.py 只保留仍使用的 ServiceError/错误映射，冗余响应包装删除。
5. 删除 history/retrieval_config 等旧公开请求字段及无消费者兼容代码。同步示例和测试，启动说明固定“从仓库根目录执行 python -m uvicorn backend.src.main:app --host 127.0.0.1 --port 8000”，python 必须是 agent 解释器。

## 参考边界

| 当前实现 | 上游关系 | 解决问题 | 差异 | 首版修改 | 结论 |
|---|---|---|---|---|---|
| 文档接口已有、QA 注册被注释 | 不移植 RAGFlow 完整前端 | 让核心能力可以演示 | 通用平台页面超出需求 | 同源原生单页 | 本项目独立实现 |

[应用入口](D:/code/ragFlow-main/backend/src/main.py:85)、[文档接口](D:/code/ragFlow-main/backend/src/apps/restful_apis/documents.py)、[问答接口](D:/code/ragFlow-main/backend/src/apps/restful_apis/qa.py)。

## 验收与交接

- test_api.py 使用 TestClient 和服务替身核对请求/响应、无登录、错误消息、原图 MIME；GET /health 不访问外部服务。
- 浏览器真实走一遍上传包含图片的包 → 自动入库 → 提问 → 引用 → 原文 → 原图，检查 1280px 与 375px 宽度。
- 刷新后文档列表仍恢复；模型失败不会留下永远转圈的加载状态。
- 记录真实演示截图及未完成联调，不将替身截图当作完整链路验收。
- 当前：代码未实施；测试未运行。完成后进入 08。

</details>
