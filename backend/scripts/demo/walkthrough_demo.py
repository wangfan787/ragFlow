"""从项目根用 agent 执行 python -m backend.scripts.demo.walkthrough_demo，查看真实父子切片。"""
from pathlib import Path

from backend.src.chunking import ChunkConfig, BlockChunker
from backend.src.parsing.parser_factory import build_parser

MD_FILE = Path(__file__).parents[2] / "tests" / "parsing" / "data" / "sample_walkthrough.md"


def main():
    print("原始 Markdown")
    for number, line in enumerate(MD_FILE.read_text(encoding="utf-8").splitlines(), 1):
        print(f"{number:3}: {line}")
    blocks = build_parser("md").parse(
        "sample", {"file_type": "md", "file_path": str(MD_FILE), "doc_name": MD_FILE.name},
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
    print("\n问答时：命中子块 → 按 parent_id 读父块 → 围绕命中位置取窗口 → 引用窗口来源")


if __name__ == "__main__":
    main()
