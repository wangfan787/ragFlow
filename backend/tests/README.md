# 核心回归测试

在仓库根目录激活 Conda `agent` 后，执行 `python -m pytest -q`，默认同时运行 `backend/tests` 与 `evaluation/tests`。

- `parsing/`、`chunking/`：解析、来源定位、Token 上限、文本守恒和父子关系。
- `foundation/`：模型接口、配置、状态和请求参数约束。
- `indexing/`：嵌入输入、非法向量、ES 父子记录与重建预检。
- `services/`、`integration/`：文档生命周期、检索问答、改写回退、图片鉴权及跨层管线契约。
- `evaluation/tests/`：数据确定性、指标计算、结果完整性、T2 缓存恢复和 Dev/Test 隔离。

测试使用小样例、临时目录和 fake 模型/存储，不要求大型数据集或运行中的 ES，也不调用付费模型。
根目录 `conftest.py` 为后端与评测测试统一隔离本机配置；CRUD 评测的小数据集由 `evaluation/tests/conftest.py` 共享构建。
人工演示仅保留 `python -m backend.scripts.demo.walkthrough_demo --source demo`（真实 embedding + ES）。

## 保留与合并的标准

长期回归测试随功能提交；一次性探索、真实模型跑分及生成报告放在被 Git 忽略的产物目录。评测构建器、评分器及其正确性测试继续维护。

测试按当前功能归属组织，不再按开发阶段新增 `repair_plan` / `cleanup_contracts` 文件。B 依赖 A 不等于 B 覆盖 A：只有 B 实际执行 A，且包含相同场景与关键断言，才合并 A 的重复测试。正常链路不能替代错误输入、失败回退和来源定位测试。

本次合并的对应关系：

| 原重复验证 | 保留位置 |
| --- | --- |
| 同一 Markdown/PDF 样例单独解析、再解析分块 | `chunking/test_chunking.py` 同时验证解析来源与分块守恒 |
| 旧父块 mean 聚合 | `integration/test_retrieval.py` 验证 mean/max 排序与所有贡献子块 |
| QA 参数透传与 trace 单独冒烟 | `services/test_qa_query_api.py` 验证 HTTP → 真实 QAService → 替身 Router 的透传与 trace |
| 二次窗口缩小的绝对坐标 | `services/test_qa_service.py` 同时验证原输入不变、命中锚点和原文切片 |
| rerank YAML 默认值、请求覆盖分散验证 | `foundation/test_request_config.py` 集中验证 |
| CRUD 评测重复的数据构造 | `evaluation/tests/conftest.py` 共享 fixture |

HTTP 中的 Router 是替身，因此真实检索、重排、阈值、去重和失败回退测试仍保留在 `integration/test_retrieval.py`；不会因为 HTTP 返回成功就删除它们。
