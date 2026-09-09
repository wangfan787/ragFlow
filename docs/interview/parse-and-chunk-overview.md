# Parse 与 Chunk 是怎么做的（简版）

> 数据流：原始文档 → **parsing（结构+住址）** → **chunking（合小切大）** → indexing → retrieval

## 一、Parse 层：把任意格式变成统一的结构块

**职责**：只管"这段话是什么结构、在原文哪里"，不管切多碎。

```
输入 ParseSource（file_path 或内存 text，严格 UTF-8）
  → parser_factory 按类型路由
      md     → MarkdownParserIt   （markdown-it-py 语法解析）
      html   → HtmlParser         （BeautifulSoup：删 script/style，白名单标签提取）
      txt    → TextParser         （按空行切段）
      pdf    → PdfParser          （pypdf 文本直读；扫描件/复杂表格明确不支持）
  → 输出统一 block（LangChain Document）
```

**每个 block = page_content（正文）+ metadata（档案）**：

| 字段 | 含义 |
|---|---|
| `block_type` | heading / paragraph / code / table / list |
| `section_path` | 标题层级路径（面包屑） |
| `source_span` | 回原文的行列/字符坐标 |
| `accuracy` | 坐标可信度：exact / line_only / unavailable（宁可降级不伪造） |
| `order` | 文档内顺序号 |
| `normalization_segments` | （仅 HTML）清洗后文字 ↔ 原文的映射表 |
| `page_no` / `bbox` | PDF 页码 / 版面框（预留） |

**特点**：
- 格式差异在 parse 层终结——下游只认 block，不认格式；
- **切分看结构不看大小**：table、code 无论多大都是一个完整 block
  （真实例子：1488 字符的无空行纯文本 = 1 个 block）；
- 标签删除等清洗会记映射表，坐标不可靠时诚实降级精度。

## 二、Chunk 层：对 block 流"合小 + 切大"

**职责**：把大小随机的 block 流，整成固定预算的检索单元。

```
blocks ──合并──▶ Parent（目标 512 / 硬上限 768 token，heading 开新 Parent）
Parent ──切分──▶ Child （目标 128 / 硬上限 192 token，句子边界优先）
```

**关键机制**：

1. **软上限 vs 硬上限**：目标值管"切出来的块好不好看"，硬上限管"输入合不合法"。
   两者之间留 1.5 倍缓冲，让切分器几乎总能落在自然边界（句号/空白），
   只有病态输入（无标点中文、超长 URL）才落到**字符二分硬切**。
2. **硬上限无例外**：code/table 的 preserve 是"优先权"（独占 Parent、不与散文混块），
   不是"豁免权"——超限照样切，并打 continuation 接续标记。
3. **守恒不变量**：Children 依序拼接必须**逐字等于** Parent 正文，
   运行时校验，不满足直接抛异常——保证 Small-to-Big 展开不丢内容。
4. **角色分离**：Child 带 `retrieval_eligible=true` 和向量，唯一负责召回；
   Parent 不生成向量，只存上下文（`parent_id` / `child_ids` 双向链接，
   Child 另带自己在 Parent 内的 `parent_char_start/end` 坐标，供 QA 锚定开窗）。

## 三、一句话口诀

> **Parse 的切分看结构不看大小，chunk 的切分看大小、借结构；**
> **Parent 是 Child 的正文合集（逐字守恒），但不是信息合集——
> Child 是"检索版"（带向量），Parent 是"上下文版"（带全貌）。**

实测佐证：T2 子集 10,000 篇文档（64.8% HTML / 35.2% 纯文本）走同一条链路，
平均每篇切出 8.14 个 Child，全量 81,353 个 Child 无一超限、无一丢内容。
