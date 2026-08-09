# AGENTS.md

本文件是给 AI 编码助手（Agent）的项目级指令。会话启动时会自动加载，请务必遵守。

## 运行环境（重要）

本项目的 Python 运行环境是 **conda 的 `agent` 环境**，不是 base 环境。

- conda 环境名：`agent`
- 解释器绝对路径：`/home/xvxing/miniconda3/envs/agent/bin/python`（Python 3.10）
- 激活方式：`conda activate agent`，或直接用上面的绝对路径调用 `python`

### 必须遵守

1. **任何运行、测试、安装依赖，都必须在 `agent` 环境下进行**，不要使用 base 环境的 `python` / `pip`。
2. 运行项目代码时，优先用绝对路径：`/home/xvxing/miniconda3/envs/agent/bin/python ...`，避免依赖 shell 当前的 conda 激活状态（子进程默认可能是 base）。
3. **安装依赖前先检查 `agent` 环境里是否已存在**，避免重复安装：
   ```bash
   /home/xvxing/miniconda3/envs/agent/bin/python -c "import <包名>"
   ```
   已存在则不要重装。
4. 切勿把包装到 base 环境（`/home/xvxing/miniconda3/lib/python3.13/...`），那会污染全局且对项目无效。

> 历史教训：用 base 环境的 `pip install` 装包，会装到 Python 3.13 的 site-packages，而项目实际跑在 agent 环境（Python 3.10），两套环境互不可见，导致“装了等于没装”的重复安装问题。

## RAGFlow 源码对照与设计审批门禁（必须遵守）

涉及架构、功能实现、跨模块修改、评测脚本、参考 RAGFlow/其他上游实现，或影响 parsing、chunking、indexing、retrieval、citation、持久化与指标的修复时，必须先完整读取并执行：

`skills/design-before-code/SKILL.md`

必须先以正式 RAG 链路为主线，确认 `ragflow-main` 的本地版本、运行开关、默认生产路径与 legacy/fallback 路径，再逐模块对照当前实现与 RAGFlow 默认路径的实际源码，然后讨论取舍和目标设计；T2Retrieval/测试脚本只能在正式架构确定后讨论。在取得用户对具体设计决策的明确批准前，不得修改业务代码。批准只覆盖用户明确同意的模块，未批准的依赖不得由 Agent 擅自决定。诊断、分析、比较和讨论请求不构成实现授权。
