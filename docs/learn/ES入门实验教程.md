# ES 入门实验教程（对着 ragFlow 真实索引学）

目的：搞懂 Elasticsearch 到底怎么用，能看懂面试官问的「父子关系在 ES 里怎么设计」这类问题。
学法：全部命令在你本机 ES（`http://localhost:9200`，8.19.3）上直接执行。只读实验打产品索引
`rag-mvp-chunks`（22 条真实数据），写入实验打沙箱索引 `es-learn-demo`，不污染产品数据。

一键跑完所有实验：`bash docs/learn/es-demo.sh`
单个实验想手动敲：直接从下文复制 curl 命令。

---

## 1. ES 是什么：一个心智模型

ES 就是一个**只讲 HTTP + JSON 的搜索服务**：所有操作都是发一个 JSON 请求、收一个 JSON 响应。
它内部同时维护两种索引结构：

| 结构 | 服务于 | 原理一句话 |
|------|--------|-----------|
| 倒排索引 | BM25 关键词检索 | 「词 → 含这个词的文档列表」，查词直接跳列表 |
| HNSW 图 | 向量检索 | 高维向量之间的近邻图，近似找 top-k 最近邻 |

两种结构放在**同一个 index** 里并存——这就是你的项目能用一个索引同时做关键词召回和向量召回的原因。

和 MySQL 对照：

| MySQL | Elasticsearch | 你的项目里 |
|-------|---------------|-----------|
| database | cluster | 本机单节点 ES |
| table | index | `rag-mvp-chunks` |
| 表结构 schema | mapping | `elasticsearch_store.py` 的 `_base_mapping()` |
| row | document | 一条父块或一条子块 |
| 主键 id | `_id` | 确定性 chunk_id，如 `sample_v2_4e8910b7cc71_2` |

---

## 2. 只读实验：看懂自己的产品索引

### 2.1 库里有哪些索引

```bash
curl -s "http://localhost:9200/_cat/indices?v"
```

`v` 表示带表头。你会看到 `rag-mvp-chunks`（产品索引，22 docs）、`rag-eval-crud_rag_mini_v1`
（评测索引，9430 docs）等。`docs.count` 就是文档条数。

### 2.2 看 mapping（表结构）

```bash
curl -s "http://localhost:9200/rag-mvp-chunks/_mapping" | jq
```

重点看两类字段类型的区别：

- **`keyword`**：不分词，整个值当一个 token，只能精确匹配/过滤/聚合。项目里的
  `doc_id`、`chunk_role`、`parent_id`、`child_ids` 都是 keyword——因为它们是用来
  过滤和按 ID 取的，不是用来模糊搜的。
- **`text`**：写入时先分词，查询时也分词，走倒排索引打 BM25 分。项目里的 `content` 是 text。

还有 **`dense_vector`**：向量字段。项目里的向量字段名是 `q_1024_vec`（命名规则在
`elasticsearch_store.py:32`：`q_{维度}_vec`），1024 维，`index: true` 表示建 HNSW 图。

### 2.3 看一条真实的父块和子块

```bash
# 一条父块（不打印向量和正文）
curl -s "http://localhost:9200/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{"size":1,"query":{"term":{"chunk_role":"parent"}},
       "_source":{"excludes":["content"]}}' | jq '.hits.hits[0]'

# 一条子块
curl -s "http://localhost:9200/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{"size":1,"query":{"term":{"chunk_role":"child"}},
       "_source":{"excludes":["q_1024_vec"]}}' | jq '.hits.hits[0]'
```

对着输出确认三件事（这就是面试题 Q5 的实物）：

1. 父块和子块是**同一个索引里两类平铺的文档**，靠 `chunk_role` 字段区分；
2. 关系是双向冗余的：子块有 `parent_id` 指向父块，父块有 `child_ids` 数组指向所有子块；
3. 只有子块有 `embedding_dim: 1024` 和向量字段，父块没有向量——父块只做召回后的上下文扩展，不参与向量检索。

再看 `_id`：形如 `sample_v2_4e8910b7cc71_2`，由 `doc_id + 版本号 + 切分策略哈希 + 序号`
拼成。同一个文档用同样策略重新入库，`_id` 不变，ES 自动覆盖旧记录——**确定性 ID 让重索引天然幂等**。

### 2.4 分词：为什么中文查询要预分词

```bash
# 看 ES 默认分词器怎么切中文
curl -s "http://localhost:9200/rag-mvp-chunks/_analyze" -H 'Content-Type: application/json' \
  -d '{"analyzer":"standard","text":"分布式锁的过期时间"}' | jq '.tokens[].token'
```

你会发现输出是 `分 布 式 锁 的 过 期 时 间`——standard 分词器对中文只会**逐字切**。
所以查询「过期时间」实际变成「过 OR 期 OR 时 OR 间」，单字命中不算命中词。

这就是项目里 `important_tks`/`question_tks` 这类字段存在的原因：入库前用
`vectorizer.py` 的 `tokenize()` 自己做中文切分——**单字 + 相邻二元组合**
（「过期」会同时产出 `过`、`期`、`过期`），再用空格拼好存进 text 字段，
让 ES 的默认分词器拿到的是已经切好的词。

### 2.5 复现产品里的 BM25 关键词检索

`keyword_search()` 真实发出的查询长这样（`elasticsearch_store.py:200`）：

```bash
curl -s "http://localhost:9200/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{
    "size": 3,
    "query": {
      "bool": {
        "must": [{"multi_match": {
          "query": "锁",
          "fields": ["doc_name^2", "section_path^2", "content"],
          "type": "best_fields"
        }}],
        "filter": [{"term": {"retrieval_eligible": true}}]
      }
    },
    "_source": {"excludes": ["q_*_vec"]}
  }' | jq '.hits.total, [.hits.hits[] | {id:._id, score:._score}]'
```

三个要点：

- `bool.must` 里放打分条件（BM25 算分），`bool.filter` 里放过滤条件（只筛不算分）；
  项目里固定 filter `retrieval_eligible: true`，保证**只有子块能被检索到**。
- `doc_name^2` 表示权重翻倍：文件名命中的块比正文命中的块分数更高。
- `_score` 就是 BM25 分，`keyword_search()` 会拿它归一化成 keyword_score。

### 2.6 term vs match：一个不分词、一个分词

```bash
# term：不分词，整值精确匹配，用于过滤
curl -s "http://localhost:9200/rag-mvp-chunks/_count" -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"chunk_role":"child"}}}'
# → count = 11

# match：查询串先分词再查倒排索引，用于检索打分
curl -s "http://localhost:9200/rag-mvp-chunks/_count" -H 'Content-Type: application/json' \
  -d '{"query":{"match":{"content":"锁"}}}'
```

规则记住一句话：**keyword 字段配 term（过滤），text 字段配 match（搜索）**。

### 2.7 mget：复现「子块命中 → 取父块」链路

产品检索的完整动作是：先向量/BM25 召回子块，再拿子块的 `parent_id` 按主键取父块正文拼上下文。
第二步就是 mget（对应 `query_by_ids()`，`elasticsearch_store.py:251`）：

```bash
# 第一步：找一个子块的 parent_id
PARENT=$(curl -s "http://localhost:9200/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{"size":1,"query":{"term":{"chunk_role":"child"}}}' | jq -r '.hits.hits[0]._source.parent_id')
echo "parent_id = $PARENT"

# 第二步：按 ID 直取父块（走主键，不走倒排索引，等价于 MySQL 主键查询）
curl -s "http://localhost:9200/rag-mvp-chunks/_mget" -H 'Content-Type: application/json' \
  -d "{\"ids\":[\"$PARENT\"]}" | jq '.docs[0] | {found, id:._id, role:._source.chunk_role, child_ids:._source.child_ids}'
```

因为 `parent_id` 是 keyword 且父子同索引，这一步是**一次主键读取**，没有任何关系型
查询——这就是面试答案里「join 结构没必要」的依据。

---

## 3. 写入实验：沙箱索引 es-learn-demo

以下操作都在沙箱索引上，随便折腾。跑一遍 `bash docs/learn/es-demo.sh` 会自动完成全部步骤，
这里解释每步在干什么。

### 3.1 建索引 + mapping

```bash
curl -s -X PUT "http://localhost:9200/es-learn-demo" -H 'Content-Type: application/json' -d '{
  "mappings": {"properties": {
    "doc_id":     {"type": "keyword"},
    "chunk_role": {"type": "keyword"},
    "parent_id":  {"type": "keyword"},
    "content":    {"type": "text"},
    "score_v1":   {"type": "integer"},
    "q_8_vec":    {"type": "dense_vector", "dims": 8, "index": true, "similarity": "cosine"}
  }}
}' | jq
```

对应项目里 `_ensure_index()`：索引不存在才建，mapping 一次定好。项目还多一步——如果索引
已存在但缺当前维度的向量字段，用 `put_mapping` 动态补上（支持换 embedding 模型后维度变化）。

### 3.2 PUT 写文档：同 _id 再写一遍会发生什么

```bash
curl -s -X PUT "http://localhost:9200/es-learn-demo/_doc/doc1_p1" -H 'Content-Type: application/json' \
  -d '{"doc_id":"doc1","chunk_role":"parent","content":"分布式锁第一版：用 SET NX 拿锁"}' | jq
# → result: created, _version: 1

# 同一个 _id 再写一遍
curl -s -X PUT "http://localhost:9200/es-learn-demo/_doc/doc1_p1" -H 'Content-Type: application/json' \
  -d '{"doc_id":"doc1","chunk_role":"parent","content":"分布式锁第二版：SET NX PX 加过期时间"}' | jq
# → result: updated, _version: 2
```

`_version` 从 1 变 2，`result` 从 `created` 变 `updated`——**同 ID 就是覆盖，不是新增**。
这就是项目把 `_id` 设计成确定性 chunk_id 的底气：重跑入库一百遍，索引里也只有一份。

### 3.3 bulk 批量写

```bash
curl -s -X POST "http://localhost:9200/es-learn-demo/_bulk" -H 'Content-Type: application/x-ndjson' --data-binary '
{"index":{"_index":"es-learn-demo","_id":"doc1_c1"}}
{"doc_id":"doc1","chunk_role":"child","parent_id":"doc1_p1","content":"拿锁要保证原子性","score_v1":1,"q_8_vec":[0.1,0.9,0.2,0.3,0.5,0.1,0.2,0.4]}
{"index":{"_index":"es-learn-demo","_id":"doc1_c2"}}
{"doc_id":"doc1","chunk_role":"child","parent_id":"doc1_p1","content":"锁过期会导致并发问题","score_v1":2,"q_8_vec":[0.2,0.8,0.1,0.4,0.4,0.2,0.1,0.3]}
' | jq
```

注意两点：bulk 的格式是 **NDJSON**（两行一组：动作行 + 数据行），必须 `--data-binary`（不能
用 `-d`，否则换行被吞）；项目里 `upsert()` 就是用 bulk 把父子两类记录一次写进去的。

**必踩的坑（也是面试常问点）**：bulk 写完立刻查是查不到的。ES 是**近实时（NRT）**——写入先进
内存缓冲区，默认每 1 秒做一次 refresh，把缓冲区内容变成倒排索引里可搜的段。想立刻可搜就手动
`POST /es-learn-demo/_refresh`。这解释了一个工程事实：项目入库接口返回成功 ≠ 数据已可检索，
只是已落盘；好在真实链路里入库和检索之间隔了人工操作的时间，1 秒 refresh 无感。

### 3.4 组合查询：bool + must + filter

```bash
# 在 doc1 的子块里做关键词检索
curl -s "http://localhost:9200/es-learn-demo/_search" -H 'Content-Type: application/json' \
  -d '{
    "query": {"bool": {
      "must": [{"match": {"content": "锁过期"}}],
      "filter": [{"term": {"chunk_role": "child"}}]
    }}
  }' | jq '[.hits.hits[] | {id:._id, score:._score, text:._source.content}]'
```

### 3.5 knn 向量检索

```bash
curl -s "http://localhost:9200/es-learn-demo/_search" -H 'Content-Type: application/json' \
  -d '{
    "knn": {
      "field": "q_8_vec",
      "query_vector": [0.1, 0.85, 0.2, 0.3, 0.5, 0.1, 0.2, 0.4],
      "k": 2,
      "num_candidates": 10
    }
  }' | jq '[.hits.hits[] | {id:._id, score:._score}]'
```

（这里的 8 维向量是随手编的，只为看 API 形态；真实检索效果验证走 `rag-eval-*` 评测索引。）
`k` 是最终返回条数，`num_candidates` 是 HNSW 图搜索时探查的候选池大小——候选池越大越准越慢，
项目里取 `max(100, top_k*10)`（`elasticsearch_store.py:180`）。

### 3.6 删除：按条件删、按主键删、删索引

```bash
# 按 doc_id 条件删（对应 delete_by_doc_id()）
curl -s -X POST "http://localhost:9200/es-learn-demo/_delete_by_query" -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"doc_id":"doc1"}}}' | jq '{deleted}'

# 删除和写入一样：要等 refresh 才对搜索可见，想立刻确认就手动刷一下
curl -s -X POST "http://localhost:9200/es-learn-demo/_refresh"

# 确认真的删干净了
curl -s "http://localhost:9200/es-learn-demo/_count"

# 整个索删掉（实验做完想清理时用）
curl -s -X DELETE "http://localhost:9200/es-learn-demo"
```

项目里删文档的完整语义是**先写新的、再删旧的**（`upsert()` 之后紧跟
`delete_stale_by_doc_id()`），保证任何时刻文档至少有一份可检索的记录。

---

## 4. 对照项目：HTTP 动作 ↔ 代码方法

| ES HTTP API | `elasticsearch_store.py` | 干什么 |
|-------------|--------------------------|--------|
| `PUT /{index}` + mapping | `_ensure_index()` | 建索引；缺向量字段时 `put_mapping` 补 |
| `POST /_bulk` | `upsert()` | 父子记录一起写入，`_id`=chunk_id 幂等覆盖 |
| `POST /{index}/_delete_by_query` | `delete_by_doc_id()` / `delete_stale_by_doc_id()` | 按文档删、替换后清旧 |
| `search` + `knn` | `vector_search()` | 向量召回子块，filter 只留 eligible |
| `search` + `bool.must.multi_match` | `keyword_search()` | BM25 召回，doc_name/section_path 加权 |
| `GET /{index}/_mget` | `query_by_ids()` | 按子块的 parent_id 取父块正文 |

一次问答的检索链路：`vector_search + keyword_search` 双路召回子块 → 融合去重 →
`query_by_ids` 取父块 → 父块正文进 LLM 上下文。

---

## 5. 回到面试题 Q5：选项对比

| 方案 | 机制 | 为什么不用 |
|------|------|-----------|
| 单索引平铺（项目现状） | 父子两类文档同索引，`chunk_role` 区分，`parent_id`/`child_ids` 双向冗余 | 父块只需按 ID 取（mget 一跳），关系查询为零成本 |
| 双索引（parent/child 各一个） | 同上但拆两个 index | 字段同源、生命周期绑定（整文档替换），拆开管理翻倍还引入跨索引一致性 |
| ES nested | 一个文档内部的数组，每个元素独立查询条件 | 语义不匹配：父子是文档之间的关系，不是文档内部数组 |
| ES join 父子文档 | 同索引内父文档+子文档，按 parent 路由，支持按父查子 | 为关系型查询/打分设计，routing 约束重；这里只用主键取父，杀鸡用牛刀 |
| 数据库存映射 | ES 存 chunk，MySQL 存父子映射 | 同一份关系两个真源，删除/更新要跨存储协调一致性 |

---

## 6. 常用命令速查

```bash
ES=http://localhost:9200
curl -s "$ES/_cat/indices?v"                     # 列索引
curl -s "$ES/{index}/_mapping" | jq              # 看表结构
curl -s "$ES/{index}/_count"                     # 数文档
curl -s "$ES/{index}/_doc/{id}" | jq             # 按主键取一条
curl -s "$ES/{index}/_search" -H 'Content-Type: application/json' -d '{...}' | jq   # 查询
curl -s -X PUT  "$ES/{index}"  -H 'Content-Type: application/json' -d '{...}'       # 建索引
curl -s -X POST "$ES/{index}/_delete_by_query" -H 'Content-Type: application/json' -d '{...}'  # 按条件删
curl -s -X DELETE "$ES/{index}"                  # 删索引
```
