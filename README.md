# RAG 项目

本仓库用于实现和验证一个可运行、可解释、可评测的 RAG 产品。项目实施状态与路线见 [`docs/plan/README.md`](docs/plan/README.md)，算法概念和决策背景见 [`docs/plan/算法层QA.md`](docs/plan/算法层QA.md)，评测契约、真实运行记录与参数速查见 [`docs/plan/评测.md`](docs/plan/评测.md)。

## 开发环境

本项目的 Python 开发环境是 Conda 环境 `agent`。运行 Python、pytest、解析脚本或检查依赖前，先执行：

```bash
conda activate agent
```

不要使用当前 shell 的默认 Python 判断项目依赖是否安装。

## 主要入口

- 服务：`python -m uvicorn backend.src.main:app --reload`。
- 核心测试：`python -m pytest -q`（包含后端和评测测试，不依赖大型数据集）。
- 真实链路演示：`python -m backend.scripts.demo.walkthrough_demo --source demo`；准备 T2 子集后可选 `--source t2`。需要本地 ES 和 `config/local.yaml` 模型配置，仅操作 `rag-demo-*` 索引。
- [CRUD-RAG 评测与调参](evaluation/README.md)；[T2Retrieval 下载、向量化与评测](dataset/T2Retrieval-subset/README.md)。

重排默认关闭，保留规则基线与可选 CrossEncoder；后者依赖 `backend/requirements-rerank.txt`。
开启后可通过 `retrieval.rerank_level` 选择 `child` 或 `parent`（仅 CrossEncoder 支持 Parent）；
`retrieval.child_score_aggregation` 支持 `mean` / `max`，用于 Child 重排、关闭重排和模型失败回退时的父块评分。
默认仍为 `child + mean`，可复现已有基线。Parent 重排成功时直接采用模型对父块的分数。
Git 保留源码、小测试样例、manifest 和锁定配置；大型数据、向量及生成结果仅本地保存。取消跟踪不会删除本地文件，也不会移除历史提交中的大文件。

## 配置

运行配置统一放在根目录 `config/`，使用支持 `#` 注释的 YAML：

- [`defaults.yaml`](config/defaults.yaml)：全部默认值与字段说明，提交 Git。
- `local.yaml`：本机密钥、模型地址、数据路径等差异，不提交 Git。
- [`local.example.yaml`](config/local.example.yaml)：新环境的填写模板。

新环境先运行 `cp config/local.example.yaml config/local.yaml`，再填写各模型的 `api_key`。
已有环境已迁移时直接编辑 `local.yaml`，不要用模板覆盖它。
本机 YAML 覆盖公共 YAML；不再读取 `.env`、`MVP_*`、`DEMO_*`、`JWT_*` 或 `T2_DATASET_DIR` 环境变量。
修改配置后重启进程。API 请求与评测 CLI 的显式参数仍可覆盖该次运行，不修改配置文件。
相对数据路径均以仓库根目录解析，与启动目录无关。

Python 中的配置类只声明字段类型和约束，默认值从 YAML 读取。未知字段和错误类型直接报错；
布尔值使用 YAML/JSON 的 `true` / `false`，不接受字符串代替。密钥按服务显式配置，不互相借用。
第三方库的代理、证书、缓存目录等环境变量仍由库自身处理，不参与应用配置优先级。
评测 manifest、锁定结果和 run/score JSON 是实验产物，不是另一套运行配置。

例如，在 `local.yaml` 中选择中文 Parent 重排：

```yaml
retrieval:
  rerank_enabled: true
  rerank_backend: cross-encoder
  rerank_level: parent
reranker:
  model: BAAI/bge-reranker-base
```

API 的 `retrieval_config` 也能逐请求覆盖重排粒度与聚合方式。
两种粒度的 `rerank_top_n` 都表示选取的 Child 数量；Parent 模式先去重回溯再送排，不额外补满 N 个父块。
`similarity_threshold` 在对应重排粒度上过滤最终分数；Parent 模式不会提前用 Child 分数淘汰候选。
Parent 缺失时对主命中 Child 评分，并在 trace 的 `rerank_fallback_child_count` 记录数量；模型失败则回退完整融合池。
trace 同时记录配置/实际粒度、实际聚合方式、Child 池大小和实际送排数，便于复查降级和成本。
实验原文与选型分析见 [父子重排案例](docs/interview/child-parent-rerank-cases.md)。
