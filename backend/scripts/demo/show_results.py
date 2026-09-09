"""查看完整解析内容或生成报告；从项目根用 agent 执行 python -m backend.scripts.demo.show_results。"""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from backend.src.parsing.parser_factory import build_parser


def render(path: Path) -> str:
    blocks = build_parser(path.suffix).parse(
        "preview", {"file_type": path.suffix, "file_path": str(path), "doc_name": path.name},
    )
    parts = [
        f"文件：{path.name}；块数：{len(blocks)}",
        f"类型：{dict(Counter(block.metadata['block_type'] for block in blocks))}",
    ]
    for index, block in enumerate(blocks, start=1):
        metadata = block.metadata
        parts.extend([
            f"\n[{index}] {metadata['block_type']} / {' > '.join(metadata.get('section_path', []))}",
            f"来源：{metadata.get('source_span')}",
            block.page_content,
        ])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="显示 Markdown/PDF 的完整解析结果")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--report", action="store_true", help="同时保存带时间戳的文本报告")
    args = parser.parse_args()
    directory = Path(__file__).parents[2] / "tests" / "parsing" / "data"
    files = args.files or [directory / "并发编程-锁.md", directory / "简历.pdf"]
    report = "\n\n".join(render(path) for path in files)
    print(report)
    if args.report:
        destination = Path(__file__).parent / f"parsing_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        destination.write_text(report, encoding="utf-8")
        print(f"已保存：{destination}")


if __name__ == "__main__":
    main()
