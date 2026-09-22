# 核心回归测试

在仓库根目录激活 Conda `agent` 后，执行 `python -m pytest -q`，默认同时运行 `backend/tests` 与 `evaluation/tests`。

- `parsing/`、`chunking/`：解析、来源定位、Token 上限、文本守恒和父子关系。
- `embedding/`、`foundation/`：模型接口、配置和状态。
- `services/`、`integration/`：文档生命周期、检索问答、改写回退、图片鉴权及跨层管线契约。
- `evaluation/tests/`：数据确定性、指标计算、结果完整性和 Dev/Test 隔离。

测试使用小样例、临时目录和 fake 模型/存储，不要求大型数据集或运行中的 ES，也不调用付费模型。
人工演示仅保留 `python -m backend.scripts.demo.walkthrough_demo --source demo`（真实 embedding + ES）。
