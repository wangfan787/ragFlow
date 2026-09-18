"""从项目根用 agent 执行 python -m backend.scripts.demo.walkthrough_demo，走通 parse → chunk → index。

前置条件：本地 Elasticsearch（默认 http://localhost:9200）已启动，
.env 已配置 MVP_EMBEDDING_API_KEY / MVP_EMBEDDING_MODEL / MVP_EMBEDDING_BASE_URL。
"""
from pathlib import Path

from backend.src.chunking import ChunkConfig, BlockChunker
from backend.src.indexing.embedding_indexer import EmbeddingIndexer
from backend.src.infrastructure.elasticsearch_store import ElasticsearchStore
from backend.src.parsing.parser_factory import build_parser

MD_FILE = Path(__file__).parents[2] / "tests" / "parsing" / "data" / "sample_walkthrough.md"
DOC_ID = "sample"


def index_stage(chunks):
    print("\n入库：仅子块正文参与嵌入，父块只存上下文与元数据")
    store = ElasticsearchStore()
    indexed = EmbeddingIndexer().index(DOC_ID, chunks, doc_name=MD_FILE.name)
    print(f"写入 ES 完成：{indexed} 条记录（索引 {store.index_name}）")
    stored = store.query_by_ids([chunk.metadata["chunk_id"] for chunk in chunks])
    for record in stored:
        metadata = record.metadata
        role = "子块" if metadata.get("retrieval_eligible") else "父块"
        print(f"  {role} {metadata['chunk_id']} / 向量维度：{metadata.get('embedding_dim')}")


def main():
    print("原始 Markdown")
    for number, line in enumerate(MD_FILE.read_text(encoding="utf-8").splitlines(), 1):
        print(f"{number:3}: {line}")
    blocks = build_parser("md").parse(
        DOC_ID, {"file_type": "md", "file_path": str(MD_FILE), "doc_name": MD_FILE.name},
    )
    print("\n解析结果：Document.page_content 存正文，metadata 存结构和来源")
    for block in blocks:
        metadata = block.metadata
        print(f"\n{metadata['order']} / {metadata['block_type']} / {metadata['source_span']}")
        print(block.page_content)

    config = ChunkConfig()
    chunks = BlockChunker().chunk(blocks, config)
    by_id = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
    print("\n父子切片：父块提供上下文，子块参与召回，长块受 token 上限约束")
    for parent in chunks:
        metadata = parent.metadata
        if metadata["chunk_role"] != "parent":
            continue
        print(f"\n父块 {metadata['chunk_id']} / {metadata['token_count']} tokens")
        print(f"章节：{metadata['section_path']}；来源：{metadata['source_span']}")
        print(parent.page_content)
        family = [by_id[child_id] for child_id in metadata["child_ids"]]
        assert "".join(child.page_content for child in family) == parent.page_content
        for child in family:
            child_meta = child.metadata
            print(f"  子块 {child_meta['chunk_id']} / {child_meta['token_count']} tokens")
            print(f"  父块内范围：[{child_meta['parent_char_start']}, {child_meta['parent_char_end']})")
            print(f"  来源：{child_meta['source_span']}；切分依据：{child_meta['trace']}")
            print(f"  正文：{child.page_content}")
    index_stage(chunks)
    print("\n问答时：命中子块 → 按 parent_id 读父块 → 围绕命中位置取窗口 → 引用窗口来源")


if __name__ == "__main__":
    main()
