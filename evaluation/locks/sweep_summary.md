# Dev 参数扫描与锁定摘要

- Split: `dev`  Git: `84b2bcb1b08f72dd96f7eb3dfde33db76217b3f9`  ES index: `rag-eval-crud_rag_mini_v1`
- 锁定检索配置: `{"candidate_top_k": 10, "similarity_threshold": 0.2, "top_k": 10, "vector_weight": 0.75}`
- Rerank 决策: 保持默认关闭（recall@10 增益 0.0556，mrr@10 增益 -0.0361）

## 基线（Dev）

| 变体 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| vector | 0.9556 | 0.9167 | 0.9126 |
| keyword | 0.9444 | 0.8778 | 0.8678 |
| hybrid | 0.8667 | 0.8861 | 0.8355 |
| hybrid_rerank | 0.9222 | 0.8500 | 0.8600 |

## 单变量扫描选择

### candidate_top_k → `{"candidate_top_k": 10}`

| 配置 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| `{"candidate_top_k": 10}` | 0.9444 | 0.8778 | 0.8711 |
| `{"candidate_top_k": 30}` | 0.8667 | 0.8861 | 0.8355 |
| `{"candidate_top_k": 50}` | 0.8222 | 0.8733 | 0.8118 |
| `{"candidate_top_k": 100}` | 0.8889 | 0.8778 | 0.8562 |

### top_k → `{"top_k": 10}`

| 配置 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| `{"top_k": 3}` | 0.7778 | 0.8778 | 0.7871 |
| `{"top_k": 5}` | 0.8667 | 0.8861 | 0.8355 |
| `{"top_k": 10}` | 0.9444 | 0.8917 | 0.8658 |

### similarity_threshold → `{"similarity_threshold": 0.2}`

| 配置 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| `{"similarity_threshold": 0.0}` | 0.8556 | 0.8861 | 0.8256 |
| `{"similarity_threshold": 0.05}` | 0.8556 | 0.8861 | 0.8256 |
| `{"similarity_threshold": 0.1}` | 0.8667 | 0.8861 | 0.8355 |
| `{"similarity_threshold": 0.2}` | 0.8944 | 0.8528 | 0.8407 |

### vector_weight → `{"vector_weight": 0.75}`

| 配置 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| `{"vector_weight": 0.5}` | 0.8444 | 0.8861 | 0.8178 |
| `{"vector_weight": 0.6}` | 0.8556 | 0.8861 | 0.8283 |
| `{"vector_weight": 0.7}` | 0.8667 | 0.8861 | 0.8344 |
| `{"vector_weight": 0.75}` | 0.8667 | 0.8861 | 0.8355 |
| `{"vector_weight": 0.8}` | 0.8667 | 0.8861 | 0.8355 |
| `{"vector_weight": 0.9}` | 0.8667 | 0.8861 | 0.8355 |

### rerank → `{"rerank_backend": "rule", "rerank_enabled": true, "rerank_top_n": 10}`

| 配置 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| `{"rerank_backend": "rule", "rerank_enabled": true, "rerank_top_n": 10}` | 0.9222 | 0.8500 | 0.8600 |
| `{"rerank_backend": "rule", "rerank_enabled": true, "rerank_top_n": 20}` | 0.9056 | 0.8694 | 0.8510 |
| `{"rerank_backend": "rule", "rerank_enabled": true, "rerank_top_n": 30}` | 0.8944 | 0.8444 | 0.8390 |
