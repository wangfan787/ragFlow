# 测试目录结构

按 RAG 流水线阶段分类；目录名 = 代码层。从仓库根目录运行 `pytest` 即可收集全部。

```
tests/
├── parsing/      解析层：真实样例回归（md+pdf）、新旧 parser 对照、可视化展示
│   └── data/     样例数据文件（并发编程-锁.md / 简历.pdf / sample_walkthrough.md）
├── chunking/     分块层：守恒不变量测试 + 可视化展示
├── embedding/    向量化适配层
├── foundation/   基础设施：配置 / 状态 / 数据路径
├── services/     应用服务：查询改写
├── integration/  跨层集成：test_repair_plan.py（设计文档 §10 不变量主套件）、文档管线
```

## 两类文件的区分

- **`test_*.py`**：pytest 自动收集的测试。其中 `test_*_showcase.py` 是
  "可视化展示测试"——用真实数据集文档把输入/输出完整打印（`pytest -s` 运行），
  同时内嵌结构断言，兼作回归。
- **`backend/scripts/demo/`**：人工演示脚本（不属于 pytest），手动运行：
  ```bash
  python -m backend.scripts.demo.walkthrough_demo    # 真实样例的父子切片演示
  python -m backend.scripts.demo.show_results [文件]  # 解析结果报告
  ```

## 约定

- 样例数据统一放 `parsing/data/`，测试内用 `Path(__file__)` 相对定位
  （注意目录深度：`tests/<层级>/xxx.py` 到仓库根是 `parents[3]`）。
- `test_repair_plan.py` 是设计文档（designed/repair-plan.md §10）锚定的主套件；
  改动解析/分块/索引/检索任何一层后，先跑它。
