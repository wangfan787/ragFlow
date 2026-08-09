"""切分阶段的数据结构。

ChunkMeta 和 ChunkRecord 是切分模块的输出格式，也是下游检索/Embedding 阶段的输入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.src.parsing.models import SourceSpan


@dataclass
class ChunkMeta:
    """Chunk 的元数据信息。

    包含位置溯源、分块角色、父子关系等字段，下游检索阶段可据此还原上下文。
    """

    # --- 路径与位置 ---
    section_path: list[str]            # 章节路径，如 ["二、锁", "2.1 mutex/自旋锁"]
    field_path: list[str]              # 域路径（当前为 [section_path]）
    chunk_order: int                   # chunk 在文档中的全局顺序号
    mom_id: str | None                 # Mom ID（预留字段，用于图谱/关联 chunk）

    # --- 溯源信息 ---
    block_types: list[str] = field(default_factory=list)
    # 组成此 chunk 的原始 block 类型列表，如 ["heading", "paragraph"]

    source_block_ids: list[str] = field(default_factory=list)
    # 组成此 chunk 的原始 block ID 列表，格式 "doc_id_blk_{order}"

    token_count: int = 0               # 近似 token 数（SimpleTokenCounter）
    page_no: int | None = None         # 所在页码（PDF 有效，Markdown 为 None）
    source_span: SourceSpan | None = None
    # 溯源跨度：最小行号/字符范围（合并多个 source_blocks 后取最小和最大边界）

    trace: dict[str, Any] = field(default_factory=dict)
    # 切分追踪信息：
    #   - parent: {"chunk_role": "parent", "parent_boundary": "heading" | "paragraph"}
    #   - child:  {"chunk_role": "child", "split_by": "heading" | "paragraph" | "sentence"}

    # --- 父子分块关系 ---
    # Small-to-Big 检索策略的核心字段。
    # chunk_role: "parent" = 完整上下文父块（~512 token）
    #             "child"  = 精准召回子块（~128 token）
    # parent_id: 子块指向所属父块；父块为 None
    # child_ids: 父块记录其所有子块 ID 列表；子块为空列表
    # 检索流程：query 命中 child -> child.parent_id 回溯 parent -> 返回 parent 的完整文本
    chunk_role: str = "parent"
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    """一条完整的 Chunk 记录 — 切分模块的最终输出单元。

    每条 ChunkRecord 写入向量库的 payload，下游检索阶段从此处读取。
    """
    chunk_id: str                      # 全局唯一 chunk ID，格式 "doc_id_ck_{order}"
    doc_id: str                        # 所属文档 ID
    text: str                          # chunk 文本内容
    meta: ChunkMeta                    # 元数据（位置/父子关系/溯源/切分追踪）


def validate_chunk_record(record: ChunkRecord) -> None:
    """校验 ChunkRecord 的必填字段非空。

    切分完成后对所有输出做防御式检查，防止空 chunk 进入下游。
    """
    if not record.chunk_id:
        raise ValueError("chunk_id cannot be empty")
    if not record.doc_id:
        raise ValueError("doc_id cannot be empty")
    if not record.text.strip():
        raise ValueError("chunk text cannot be empty")


# ============================================================================
# ChunkMeta 演示 - 展示父子分块关系和元数据结构
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ChunkMeta 数据结构演示 - Small-to-Big 检索策略")
    print("=" * 80)

    # 模拟一个真实的分块场景：基于"并发编程-锁.md"文档
    doc_id = "并发编程_锁_20260802"

    # 示例 1: 创建一个父块
    print("\n📝 示例 1: 父块 (Parent Chunk)")
    print("-" * 80)

    print("🔍 section_path vs field_path 的区别:")

    # 案例 1: 简单情况（当前实现）
    simple_meta = ChunkMeta(
        section_path=["二、锁", "2.1 mutex/自旋锁"],
        field_path=["二、锁", "2.1 mutex/自旋锁"],  # 当前简化为与 section_path 相同
        chunk_order=5,
        mom_id=None,
        block_types=["heading", "paragraph", "table"],
        source_block_ids=[f"{doc_id}_blk_2", f"{doc_id}_blk_3"],
        token_count=487,
    )

    print("  当前实现（简化版）：")
    print(f"    section_path: {simple_meta.section_path}")
    print(f"    field_path:    {simple_meta.field_path}")
    print("    → 两者相同，直接反映文档结构")

    # 案例 2: 复杂情况（未来扩展）
    print("\n  未来扩展（知识图谱增强版）：")
    advanced_example = {
        "section_path": ["一、进程 / 线程 / 协程", "1.0 进程和线程的区别"],
        "field_path": [
            "计算机科学",
            "操作系统",
            "并发编程",
            "进程管理",
            "进程vs线程"
        ]
    }

    print(f"    section_path: {advanced_example['section_path']}")
    print(f"    field_path:    {advanced_example['field_path']}")
    print("    → field_path 提供领域分类，独立于文档结构")

    print("\n  📊 应用场景对比:")
    print("    section_path 用途:")
    print("      - 面包屑导航")
    print("      - 文档结构还原")
    print("      - 上下文重构")
    print("    field_path 用途:")
    print("      - 知识图谱节点分类")
    print("      - 跨文档主题聚类")
    print("      - 领域专家推荐")
    print("      - 智能分类和标签")

    # 实际应用示例
    print("\n  🎯 实际应用示例:")

    # 假设我们有一个技术文档库
    docs_library = {
        "backend": [
            "Java并发编程实战.md",
            "Go语言高级编程.md",
            "Python多线程编程.md"
        ],
        "frontend": [
            "React框架解析.md",
            "Vue.js源码分析.md"
        ],
        "algorithm": [
            "数据结构与算法.md",
            "LeetCode题解.md"
        ]
    }

    print("  假设有技术文档库包含多个领域文档:")
    for category, docs in docs_library.items():
        print(f"    {category}: {', '.join(docs)}")

    print("\n  当用户查询 '线程同步机制' 时:")
    print("    section_path 作用: 帮助定位到具体的章节内容")
    print("    field_path 作用:  帮助推荐相关的 Java/Go/Python 并发编程文档")

    print("\n  section_path = 结构导航 📍")
    print("  field_path    = 语义分类 🏷️")

    print("-" * 80)

    # 继续原来的演示
    parent_meta = ChunkMeta(
        # 路径与位置
        section_path=["二、锁", "2.1 mutex/自旋锁"],
        field_path=["二、锁", "2.1 mutex/自旋锁"],
        chunk_order=5,
        mom_id=None,

        # 溯源信息
        block_types=["heading", "paragraph", "table"],
        source_block_ids=[
            f"{doc_id}_blk_2",   # 对应标题 "## 2.1 mutex / 自旋锁"
            f"{doc_id}_blk_3",   # 对应段落描述
            f"{doc_id}_blk_4",   # 对应表格
        ],
        token_count=487,
        page_no=None,  # Markdown 文档无页码
        source_span=None,  # 简化示例

        # 切分追踪
        trace={
            "chunk_role": "parent",
            "parent_boundary": "heading",
            "split_strategy": "semantic_boundary"
        },

        # 父子关系
        chunk_role="parent",
        parent_id=None,
        child_ids=[]  # 稍后添加子块时会填充
    )

    # 创建完整的父块记录
    parent_text = """| 锁类型 | 底层原理 | 核心开销 | 适用场景 | 字节工程实践 |
| --- | --- | --- | --- | --- |
| std::mutex（互斥锁） | 阻塞时放弃CPU，线程进入内核态等待，解锁时唤醒 | 内核态 / 用户态切换（3~10μs） | 临界区耗时中等 / 较长 | 绝大多数业务场景 |
| 自旋锁 | 不放弃 CPU，循环 CAS 尝试加锁，直到成功 | CPU 空转开销（无内核切换） | 临界区耗时极短（<1μs） | 底层组件 |
| std::shared_mutex | 读共享、写排他：读锁可多线程持有，写锁独占 | 读锁开销≈自旋锁，写锁≈mutex | 读多写少、临界区耗时中等 | 缓存系统、配置中心、日志模块 |"""

    parent_record = ChunkRecord(
        chunk_id=f"{doc_id}_ck_5",
        doc_id=doc_id,
        text=parent_text,
        meta=parent_meta
    )

    print("父块记录结构:")
    print(f"  Chunk ID: {parent_record.chunk_id}")
    print(f"  文档 ID: {parent_record.doc_id}")
    print(f"  文本长度: {len(parent_record.text)} 字符")
    print(f"  Token 数: {parent_record.meta.token_count}")
    print(f"  分块角色: {parent_record.meta.chunk_role}")
    print(f"  章节路径: {' → '.join(parent_record.meta.section_path)}")
    print(f"  源块类型: {', '.join(parent_record.meta.block_types)}")
    print(f"  源块 IDs: {len(parent_record.meta.source_block_ids)} 个")
    print(f"  追踪信息: {parent_record.meta.trace}")

    # 示例 2: 创建子块
    print("\n📝 示例 2: 子块 (Child Chunk)")
    print("-" * 80)

    child_meta_1 = ChunkMeta(
        # 路径与位置（继承父块的路径）
        section_path=["二、锁", "2.1 mutex/自旋锁"],
        field_path=["二、锁", "2.1 mutex/自旋锁"],
        chunk_order=6,
        mom_id=None,

        # 溯源信息（子块包含更细粒度的源块）
        block_types=["table_row"],
        source_block_ids=[
            f"{doc_id}_blk_4_row_1",  # 表格第一行
        ],
        token_count=134,
        page_no=None,
        source_span=None,

        # 切分追踪
        trace={
            "chunk_role": "child",
            "split_by": "table_row",
            "parent_chunk_id": f"{doc_id}_ck_5"
        },

        # 父子关系（关键：指向父块）
        chunk_role="child",
        parent_id=f"{doc_id}_ck_5",  # 指向父块
        child_ids=[]  # 子块不再有子块
    )

    child_text_1 = """| 锁类型 | 底层原理 | 核心开销 | 适用场景 | 字节工程实践 |
| --- | --- | --- | --- | --- |
| std::mutex（互斥锁） | 阻塞时放弃CPU，线程进入内核态等待，解锁时唤醒 | 内核态 / 用户态切换（3~10μs） | 临界区耗时中等 / 较长 | 绝大多数业务场景 |"""

    child_record_1 = ChunkRecord(
        chunk_id=f"{doc_id}_ck_6",
        doc_id=doc_id,
        text=child_text_1,
        meta=child_meta_1
    )

    print("子块记录结构:")
    print(f"  Chunk ID: {child_record_1.chunk_id}")
    print(f"  父块 ID: {child_record_1.meta.parent_id}")
    print(f"  分块角色: {child_record_1.meta.chunk_role}")
    print(f"  Token 数: {child_record_1.meta.token_count}")
    print(f"  拆分方式: {child_record_1.meta.trace.get('split_by')}")

    # 示例 3: 更新父块的子块列表
    print("\n📝 示例 3: 父子关系建立")
    print("-" * 80)

    # 更新父块的子块列表
    parent_record.meta.child_ids.append(child_record_1.chunk_id)

    # 创建更多子块
    child_meta_2 = ChunkMeta(
        section_path=["二、锁", "2.1 mutex/自旋锁"],
        field_path=["二、锁", "2.1 mutex/自旋锁"],
        chunk_order=7,
        mom_id=None,
        block_types=["table_row"],
        source_block_ids=[f"{doc_id}_blk_4_row_2"],
        token_count=98,
        page_no=None,
        source_span=None,
        trace={"chunk_role": "child", "split_by": "table_row"},
        chunk_role="child",
        parent_id=parent_record.chunk_id,
        child_ids=[]
    )

    child_text_2 = """| 自旋锁 | 不放弃 CPU，循环 CAS 尝试加锁，直到成功 | CPU 空转开销（无内核切换） | 临界区耗时极短（<1μs）、竞争频率极低 | 底层组件 |"""

    child_record_2 = ChunkRecord(
        chunk_id=f"{doc_id}_ck_7",
        doc_id=doc_id,
        text=child_text_2,
        meta=child_meta_2
    )

    parent_record.meta.child_ids.append(child_record_2.chunk_id)

    print("父块更新后的状态:")
    print(f"  父块 ID: {parent_record.chunk_id}")
    print(f"  子块数量: {len(parent_record.meta.child_ids)}")
    print(f"  子块 IDs: {', '.join(parent_record.meta.child_ids)}")

    print("\n所有子块信息:")
    for i, child_id in enumerate(parent_record.meta.child_ids, 1):
        print(f"  子块 {i}: {child_id}")

    # 示例 4: 检索流程演示
    print("\n📝 示例 4: Small-to-Big 检索流程演示")
    print("-" * 80)

    print("用户查询: '互斥锁的实现原理'")
    print("检索过程:")
    print(f"  1️⃣  向量检索命中子块: {child_record_1.chunk_id}")
    print(f"  2️⃣  子块内容: {child_record_1.text[:50]}...")
    print(f"  3️⃣  回溯父块 ID: {child_record_1.meta.parent_id}")
    print(f"  4️⃣  获取父块完整上下文: {parent_record.chunk_id}")
    print(f"  5️⃣  父块内容 (完整表格): {parent_record.text[:80]}...")
    print(f"  6️⃣  返回给用户: 父块的完整语义上下文 ({parent_record.meta.token_count} tokens)")

    # 示例 5: 验证功能演示
    print("\n📝 示例 5: 数据验证")
    print("-" * 80)

    try:
        validate_chunk_record(parent_record)
        validate_chunk_record(child_record_1)
        validate_chunk_record(child_record_2)
        print("✅ 所有 chunk 记录验证通过")
    except ValueError as e:
        print(f"❌ 验证失败: {e}")

    # 示例 6: ChunkMeta 结构化展示
    print("\n📝 示例 6: ChunkMeta 完整结构展示")
    print("-" * 80)

    def display_chunkmeta_structure(meta: ChunkMeta, indent: int = 0) -> None:
        """递归展示 ChunkMeta 的层级结构"""
        prefix = "  " * indent
        print(f"{prefix}📦 ChunkMeta ({meta.chunk_role})")
        print(f"{prefix}   ├─ 路径: {' → '.join(meta.section_path)}")
        print(f"{prefix}   ├─ 顺序: #{meta.chunk_order}")
        print(f"{prefix}   ├─ Tokens: {meta.token_count}")
        print(f"{prefix}   ├─ 源块类型: {', '.join(meta.block_types)}")
        print(f"{prefix}   ├─ 源块数量: {len(meta.source_block_ids)}")

        if meta.chunk_role == "parent":
            print(f"{prefix}   ├─ 子块数量: {len(meta.child_ids)}")
            if meta.child_ids:
                print(f"{prefix}   ├─ 子块 IDs: {', '.join(meta.child_ids[:3])}{'...' if len(meta.child_ids) > 3 else ''}")
        else:
            print(f"{prefix}   ├─ 父块 ID: {meta.parent_id}")

        print(f"{prefix}   └─ 追踪: {meta.trace}")

    print("父块结构:")
    display_chunkmeta_structure(parent_record.meta)

    print("\n子块结构:")
    for child_record in [child_record_1, child_record_2]:
        display_chunkmeta_structure(child_record.meta)

    # 示例 7: 数据持久化格式
    print("\n📝 示例 7: 向量库存储格式（JSON）")
    print("-" * 80)

    import json

    # 转换为字典（用于存储到向量库）
    chunk_dict = {
        "chunk_id": parent_record.chunk_id,
        "doc_id": parent_record.doc_id,
        "text": parent_record.text[:100] + "...",  # 截断用于展示
        "meta": {
            "section_path": parent_record.meta.section_path,
            "chunk_role": parent_record.meta.chunk_role,
            "token_count": parent_record.meta.token_count,
            "parent_id": parent_record.meta.parent_id,
            "child_ids": parent_record.meta.child_ids,
            "block_types": parent_record.meta.block_types,
            "trace": parent_record.meta.trace
        }
    }

    print("向量库存储格式示例:")
    print(json.dumps(chunk_dict, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n🔑 关键要点:")
    print("  1. 父子分块关系是 Small-to-Big 检索的核心")
    print("  2. 子块负责精准召回，父块提供完整上下文")
    print("  3. ChunkMeta 包含溯源、位置、追踪等完整元数据")
    print("  4. 所有关系都可以通过 parent_id 和 child_ids 重建")
    print("  5. 适用于 RAG 检索、内容溯源、上下文还原等场景")