#!/usr/bin/env bash
# ES 入门实验一键脚本，配合 docs/learn/ES入门实验教程.md 使用。
# 只读实验打产品索引 rag-mvp-chunks（不改数据），写入实验打沙箱索引 es-learn-demo。
# 可重复执行：开头会先删掉沙箱索引重建。
set -uo pipefail

ES="${ES:-http://localhost:9200}"
SANDBOX="es-learn-demo"

section() { echo; echo "=================================================="; echo "== $1"; echo "=================================================="; }
run() { echo "\$ $1"; }

# ---------- 0. 环境检查 ----------
section "0. 环境检查"
run 'curl -s $ES/'
curl -s "$ES/" | jq '{cluster_name, version: .version.number}'
curl -s "$ES/_cat/indices?v"

# ---------- 1. 只读：mapping ----------
section "1. 产品索引 mapping（keyword vs text vs dense_vector）"
run 'GET /rag-mvp-chunks/_mapping'
curl -s "$ES/rag-mvp-chunks/_mapping" | jq '.["rag-mvp-chunks"].mappings.properties
  | {doc_id, chunk_role, parent_id, child_ids, content, q_1024_vec}'

# ---------- 2. 只读：一条父块、一条子块 ----------
section "2. 真实父块 vs 子块（chunk_role / parent_id / child_ids 双向冗余）"
run 'search chunk_role=parent（size 1）'
curl -s "$ES/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{"size":1,"query":{"term":{"chunk_role":"parent"}},"_source":{"excludes":["content"]}}' \
  | jq '.hits.hits[0] | {id:._id, role:._source.chunk_role, parent_id:._source.parent_id, child_ids:._source.child_ids, eligible:._source.retrieval_eligible, embedding_dim:._source.embedding_dim}'
run 'search chunk_role=child（size 1，去掉向量列）'
curl -s "$ES/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{"size":1,"query":{"term":{"chunk_role":"child"}},"_source":{"excludes":["q_1024_vec"]}}' \
  | jq '.hits.hits[0] | {id:._id, role:._source.chunk_role, parent_id:._source.parent_id, child_ids:._source.child_ids, eligible:._source.retrieval_eligible, embedding_dim:._source.embedding_dim, content:._source.content}'

# ---------- 3. 只读：中文分词 ----------
section "3. ES 默认分词器怎么切中文（standard = 逐单字）"
run 'POST /rag-mvp-chunks/_analyze  text=分布式锁的过期时间'
curl -s "$ES/rag-mvp-chunks/_analyze" -H 'Content-Type: application/json' \
  -d '{"analyzer":"standard","text":"分布式锁的过期时间"}' | jq '[.tokens[].token]'

# ---------- 4. 只读：term vs match ----------
section "4. term（不分词，过滤） vs match（分词，检索）"
run 'count term chunk_role=child'
curl -s "$ES/rag-mvp-chunks/_count" -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"chunk_role":"child"}}}' | jq
run 'count term chunk_role=parent'
curl -s "$ES/rag-mvp-chunks/_count" -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"chunk_role":"parent"}}}' | jq

# ---------- 5. 只读：复现产品 BM25 查询 ----------
section "5. 复现 keyword_search()：multi_match + filter"
run 'multi_match query=锁, fields=[doc_name^2, section_path^2, content], filter retrieval_eligible=true'
curl -s "$ES/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{
    "size": 3,
    "query": {"bool": {
      "must": [{"multi_match": {"query": "锁", "fields": ["doc_name^2", "section_path^2", "content"], "type": "best_fields"}}],
      "filter": [{"term": {"retrieval_eligible": true}}]
    }},
    "_source": {"excludes": ["q_*_vec"]}
  }' | jq '[.hits.hits[] | {id:._id, score:._score, text:._source.content}]'

# ---------- 6. 只读：mget 取父块（检索链路第二步） ----------
section "6. 子块命中后按 parent_id 取父块（mget，主键直达）"
PARENT=$(curl -s "$ES/rag-mvp-chunks/_search" -H 'Content-Type: application/json' \
  -d '{"size":1,"query":{"term":{"chunk_role":"child"}}}' | jq -r '.hits.hits[0]._source.parent_id')
echo "取到的 parent_id = $PARENT"
run 'GET /rag-mvp-chunks/_mget'
curl -s "$ES/rag-mvp-chunks/_mget" -H 'Content-Type: application/json' \
  -d "{\"ids\":[\"$PARENT\"]}" | jq '.docs[0] | {found, id:._id, role:._source.chunk_role, child_ids:._source.child_ids, content:._source.content}'

# ---------- 7. 写入：建沙箱索引 ----------
section "7. 沙箱实验：建索引 es-learn-demo（先删旧的保证可重跑）"
curl -s -X DELETE "$ES/$SANDBOX" > /dev/null
run "PUT /$SANDBOX + mapping"
curl -s -X PUT "$ES/$SANDBOX" -H 'Content-Type: application/json' -d '{
  "mappings": {"properties": {
    "doc_id":     {"type": "keyword"},
    "chunk_role": {"type": "keyword"},
    "parent_id":  {"type": "keyword"},
    "content":    {"type": "text"},
    "score_v1":   {"type": "integer"},
    "q_8_vec":    {"type": "dense_vector", "dims": 8, "index": true, "similarity": "cosine"}
  }}
}' | jq

# ---------- 8. 写入：PUT 幂等覆盖 ----------
section "8. 同一个 _id 写两遍：created -> updated，_version 1 -> 2"
run "PUT /$SANDBOX/_doc/doc1_p1 （第一遍）"
curl -s -X PUT "$ES/$SANDBOX/_doc/doc1_p1" -H 'Content-Type: application/json' \
  -d '{"doc_id":"doc1","chunk_role":"parent","content":"分布式锁第一版：用 SET NX 拿锁"}' | jq '{result, _version, _id}'
run "PUT /$SANDBOX/_doc/doc1_p1 （第二遍，同 ID 覆盖）"
curl -s -X PUT "$ES/$SANDBOX/_doc/doc1_p1" -H 'Content-Type: application/json' \
  -d '{"doc_id":"doc1","chunk_role":"parent","content":"分布式锁第二版：SET NX PX 加过期时间"}' | jq '{result, _version, _id}'

# ---------- 9. 写入：bulk 父子一起写 ----------
section "9. bulk 写两条子块（NDJSON：动作行 + 数据行）"
run "POST /$SANDBOX/_bulk"
curl -s -X POST "$ES/$SANDBOX/_bulk" -H 'Content-Type: application/x-ndjson' --data-binary '
{"index":{"_index":"es-learn-demo","_id":"doc1_c1"}}
{"doc_id":"doc1","chunk_role":"child","parent_id":"doc1_p1","content":"拿锁要保证原子性","score_v1":1,"q_8_vec":[0.1,0.9,0.2,0.3,0.5,0.1,0.2,0.4]}
{"index":{"_index":"es-learn-demo","_id":"doc1_c2"}}
{"doc_id":"doc1","chunk_role":"child","parent_id":"doc1_p1","content":"锁过期会导致并发问题","score_v1":2,"q_8_vec":[0.2,0.8,0.1,0.4,0.4,0.2,0.1,0.3]}
' | jq '{errors, items: [.items[] | {result:.index.result, id:.index._id}]}'

# 经典坑：ES 是近实时（NRT）的，写入先进内存缓冲，默认每 1 秒 refresh 一次才可搜。
# 脚本连续执行太快，这里手动 refresh，否则下一节的查询会查不到刚写入的数据。
run 'POST /$SANDBOX/_refresh （不 refresh 就查不到，见教程 §3.3 说明）'
curl -s -X POST "$ES/$SANDBOX/_refresh" | jq '._shards.successful'

# ---------- 10. 查询：match + filter ----------
section "10. 组合查询：must(match 打分) + filter(term 过滤)"
run 'match content=锁过期, filter chunk_role=child'
curl -s "$ES/$SANDBOX/_search" -H 'Content-Type: application/json' \
  -d '{"query":{"bool":{"must":[{"match":{"content":"锁过期"}}],"filter":[{"term":{"chunk_role":"child"}}]}}}' \
  | jq '[.hits.hits[] | {id:._id, score:._score, text:._source.content}]'

# ---------- 11. 查询：knn 向量 ----------
section "11. knn 向量检索（8 维演示向量，只看 API 形态）"
run 'knn field=q_8_vec, k=2'
curl -s "$ES/$SANDBOX/_search" -H 'Content-Type: application/json' \
  -d '{"knn":{"field":"q_8_vec","query_vector":[0.1,0.85,0.2,0.3,0.5,0.1,0.2,0.4],"k":2,"num_candidates":10}}' \
  | jq '[.hits.hits[] | {id:._id, score:._score}]'

# ---------- 12. 写入：delete_by_query ----------
section "12. 按 doc_id 条件删除（对应 delete_by_doc_id）"
run 'POST /$SANDBOX/_delete_by_query query=doc_id:doc1'
curl -s -X POST "$ES/$SANDBOX/_delete_by_query" -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"doc_id":"doc1"}}}' | jq '{deleted}'
run 'POST /$SANDBOX/_refresh （删除同样要 refresh 才对搜索可见）'
curl -s -X POST "$ES/$SANDBOX/_refresh" > /dev/null
run 'count 确认删干净'
curl -s "$ES/$SANDBOX/_count" | jq

echo
echo "=================================================="
echo "实验完成。沙箱索引 $SANDBOX 已保留，可自行检查："
echo "  curl -s '$ES/$SANDBOX/_search?pretty'"
echo "清理：curl -s -X DELETE '$ES/$SANDBOX'"
echo "教程：docs/learn/ES入门实验教程.md"
echo "=================================================="
