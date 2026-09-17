# RAG 项目

本仓库用于实现和验证一个可运行、可解释、可评测的 RAG 产品。项目实施状态与路线见 [`docs/plan/README.md`](docs/plan/README.md)，算法概念和决策背景见 [`docs/plan/算法层QA.md`](docs/plan/算法层QA.md)，评测数据与执行契约见 [`docs/plan/评测计划.md`](docs/plan/评测计划.md)。

## 开发环境

本项目的 Python 开发环境是 Conda 环境 `agent`。运行 Python、pytest、解析脚本或检查依赖前，先执行：

```bash
conda activate agent
```

不要使用当前 shell 的默认 Python 判断项目依赖是否安装。
