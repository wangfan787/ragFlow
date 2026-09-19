# 算法层 QA 与可执行工作计划

更新日期：2026-09-19。

本文专门解释 `docs/plan/README.md` 算法层里容易产生歧义的术语、优先级和验收口径，并把讨论结果沉淀成后续可执行清单。

本文不是第二份“项目事实总表”。项目状态仍以 `docs/plan/README.md` 为准；这里形成明确决策后，再把结论回写到 README，避免两份状态互相冲突。

## 0. 先给结论

| 问题 | 结论 |
|---|---|
| 统一解析引擎是不是已经构造完毕？ | 是。Markdown、PDF、HTML、TXT 已统一到 `UnstructuredParser`，当前工作区回归通过；干净环境安装、服务上传联调和冷/热启动耗时属于后续部署验收，不再算算法 P0。 |
| 以前不是已经通过测试了吗？ | 是。2026-09-19 在 Conda `agent` 环境重跑 `backend/tests + evaluation/tests`，结果为 `102 passed`；它能证明当前代码契约，但不能替代真实 ES、真实模型和部署联调。 |
| 图片为什么要分块？ | 不切图片像素。图片保存为资产；`image block` 是图片在统一 Document 契约中的逻辑块。通常一图一块，只有过长的 VLM 文本描述才按文本切分。 |
| VLM 描述是什么？ | 核心确实是把图片交给视觉模型，让它输出可检索的文字；但还需要资产保存、提示词约束、结构化输出、失败处理、索引和引用回传。 |
| 来源坐标为什么是 P0？ | “精度诚实 + 最小回归集”曾是 P0，目前已实现并通过回归；扩大到 20–30 篇并生成比例报告属于 P1。 |
| 父子分块的 profile 是什么？ | 是一组命名的分块参数预设，不是用户画像，也不是模型。当前只有一套默认配置就够了，先保留 hash 和版本，出现明确失败再增加 profile。 |
| 三种 Embedding 输入消融是不是三个模型？ | 不是。固定当前同一个 Embedding 模型，只比较“正文 / 标题+正文 / 章节路径+正文”三种输入模板。个人项目可因成本暂缓，不需要购买或配置多个模型。 |
| 融合排序和 Rerank 为什么影响效果？ | 融合决定向量召回与 BM25 召回如何合并；Rerank 对合并后的少量候选做更精细的二次排序。前者决定候选池，后者主要改善头部顺序。 |
| RRF 和 Rerank 都必须 P0 实现吗？ | 不必。P0 应先测当前 Weighted Sum 和已有 Rerank 的真实基线；Weighted RRF 是低成本候选方案，只有当前融合确有问题时再升为实现任务。 |
| `vector_weight=1` 是不是 Vector-only？ | 不是；不过这个缺口已修复。现在使用 `retrieval_mode=vector`，测试已证明未选中的 BM25 通道调用次数为 0。 |
| 为什么 Evidence Window 还要改配置接口？ | 这个缺口已修复。`QAService.query()` 已接受请求级 `QAConfig`，可在同一 Runner 中逐题比较 `child_only/window/full_parent`。 |
| 为什么总延迟还不够？ | 这个缺口也已修复。QA 已返回请求级分阶段耗时和真实可得的 usage；剩余工作只是由 Runner 原样录制并形成真实 P50/P95、token 与成本基线。 |

---

## 1. 统一解析引擎：已完成什么，还剩什么

### Q1：`UnstructuredParser` 现在算完成了吗？

算“算法实现与当前环境回归已完成”。当前证据包括：

- Markdown、PDF、HTML、TXT 已统一由 `UnstructuredParser` 承接，`parser_factory.py` 已接入统一实现。
- PDF 相关重依赖按需加载，并已规避只读环境中的 Numba 缓存初始化问题。
- 回归覆盖标题、列表、代码块、表格文本、frontmatter、HTML 实体、重复文本回定位、非法 UTF-8 和 PDF 文本提取等关键行为。
- 2026-09-19 在 Conda `agent` 环境运行 `python -m pytest backend/tests evaluation/tests -q`，结果为 `102 passed`。

因此状态应写为：

> **√ 已实现并回归验证**

仍未证明的是“任意干净环境都可一次安装”“真实服务上传链路已联调”“冷/热启动性能有基线”。这些属于部署与性能验收，合并为 P1，不再重复列为算法 P0。

### Q2：后续的部署验收具体是什么？

只保留三个动作：

1. 在干净目标环境验证 Python、Unstructured、PDF 系统库和 NLTK 数据可复现安装，运行时不依赖开发机隐式缓存。
2. 按真实启动方式完成一次“上传 → 解析 → 分块 → 入库”，记录失败阶段和错误语义。
3. 对小、中、大样例记录首次/热启动耗时、block 数和峰值内存，形成 P50/P95 基线。

### Q3：`102 passed` 能证明什么？

| 证据 | 能证明 | 不能证明 |
|---|---|---|
| 当前单元/回归测试通过 | 固定输入下的解析、分块、检索、问答和评测契约符合断言 | 新机器能安装、真实外部服务能用 |
| 使用替身的集成测试通过 | 模块边界和失败语义可组合 | 真实 ES、真实模型、真实网络正常 |
| 后续真实联调 | 指定配置下完整链路可用 | 算法效果一定好、长期不会回归 |
| 后续真实 baseline | 固定数据集上的效果、延迟与成本可比较 | 任意业务分布都同样有效 |

所以测试通过的能力应从 P0 待办中移除；真实部署联调和业务效果评测继续按各自工作包验收。

---

## 2. 图片、VLM、image block 与“图片分块”

### Q4：解析范围不是 Markdown、PDF、HTML、图片吗？图片是不是唯一重要的缺口？

需要先区分“目标范围”和“当前实现”。

当前 `parser_factory.py` 只注册了 Markdown、PDF、HTML、TXT，没有 PNG/JPG 等图片类型。因此：

- 文本格式解析主链已经完成并通过当前环境回归；
- 图片是当前最明显的新能力缺口；
- README 当前 P0 的实际范围是 **Markdown 中引用的本地图片**；
- HTML 远程图片、PDF 内嵌图片、扫描 PDF、独立图片文件应分别定义，不能一句“支持图片”全部覆盖。

个人项目的合理 MVP 是：

> 先支持 Markdown 本地图片 → 保存资产 → VLM 生成描述 → 可检索 → 引用能返回原图。

HTML 远程图片涉及下载安全和网络稳定，PDF 内嵌图片涉及版面与页内关系，扫描 PDF 又属于 OCR；这些不应偷偷塞进同一个 P0。

### Q5：什么是 image block？

`image block` 是解析层输出的一种逻辑 Document，地位类似 heading、paragraph、code、table。它不是图片二进制本身，也不是把图片转成 Base64 塞进 Elasticsearch。

一个建议结构是：

```text
page_content:
  VLM 生成的可检索描述，例如“架构图展示请求依次经过 API、Retriever、Reranker...”

metadata:
  block_type: image
  asset_id: sha256:...
  document_id: ...
  storage_key: 后端内部受控存储键，不是客户端路径
  mime_type: image/png
  alt_text: 原 Markdown alt
  caption: 相邻标题或图注
  section_path: [检索流程, 混合检索]
  source_span: 图片语法在 Markdown 原文中的位置
  vlm_model: ...
  vlm_prompt_version: ...
  description_status: success | failed | skipped
```

这样做的意义是：下游仍然接收统一的 Document，不需要为图片另写一套索引、检索和引用协议。

### Q6：为什么图片也要分块？

这里的“进入分块”容易误导，准确说法应该是：

> image block 进入统一的父子块编排；原始图片本身不切像素。

首版规则建议如下：

1. 原图作为独立 asset 保存，不裁切、不向量化二进制。
2. 一张图片生成一个原子 image block，尽量不与另一张图片合并。
3. VLM 描述、alt、caption、section path 组成可检索文本。
4. 描述在正常 token 上限内时，一图对应一个可检索 Child。
5. 只有描述异常长、超过硬上限时，才切“描述文字”，不是切图片。
6. 命中该 Child 后，Citation 返回 `asset_id`；前端通过需要鉴权的后端接口获取原图或短期签名 URL。

当前 chunker 只特别保护 code/table，没有把 image 当成原子块；而且 chunk 输出只挑选部分 metadata，`asset_id` 也不会自动透传。因此图片 P0 不只是“调用一次 VLM”，还必须改分块、metadata 透传和授权读取三个边界。

### Q7：VLM 描述是不是把图片发给视觉 LLM，让它用自然语言描述一遍？

本质上是，但生产链路多了约束和治理：

```text
发现 Markdown 图片
→ 后端用当前用户和 document_id/asset_id 查询数据库归属
→ 查到授权资源记录
→ 在授权存储边界内校验路径、大小、格式
→ 计算 SHA256，生成 asset_id 并保存原图
→ 将图片、alt、图注、章节上下文发送给 VLM
→ 要求输出 OCR 文字、主体、关键事实、图表趋势和不确定项
→ 校验输出，生成 image block
→ 分块/Embedding/索引
→ 检索命中后返回文字证据与 asset_id
→ 后端再次鉴权并生成原图预览响应或短期签名 URL
```

自然语言描述不能只追求“看起来通顺”。检索更需要稳定、事实密集的内容，例如：

- 截图：界面名称、字段、按钮、错误码；
- 架构图：节点、箭头方向、调用关系；
- 图表：标题、坐标轴、图例、主要数值和趋势；
- 表格图片：表头、关键单元格，必要时保留行列结构；
- 无法辨认的内容：明确输出不确定，不编造。

首版可以让 VLM 返回受约束的 JSON，再由程序拼成 `page_content`。这比完全自由的一段描述更容易回归测试和版本化。

### Q8：未来接入后端后，图片和其他文件的读取权限怎么判断？

个人项目先做简单实现即可，不需要建设完整 IAM、ACL 引擎或独立 AuthorizationService。核心就是：**读取文件前查一次数据库，确认当前用户拥有该文档或资源。**

可以由一条带归属条件的查询完成：

```sql
SELECT a.*
FROM assets a
JOIN documents d ON d.id = a.document_id
WHERE a.asset_id = :asset_id
  AND d.owner_id = :current_user_id;
```

查不到就返回 404 或 403；查到后，后端使用数据库保存的 `storage_key` 读取文件，再交给 parser/VLM。引用预览时复用同一条归属查询。

首版只保留四条底线：

1. 客户端提交 `document_id/asset_id`，不能提交服务器任意本地路径。
2. 数据库查询必须同时带资源 ID 和当前用户的归属条件，不能先按 ID 查出再忘记鉴权。
3. `storage_key` 由后端生成并保存；读取前确认规范化路径仍在项目的文件存储目录内，防止 `..` 等路径穿越。
4. VLM 调用和原图预览都只能使用已经通过上述查询的文件。

SHA256 只是查重标识，不是权限凭证。若将来增加 KnowledgeBase、团队或租户，再把查询条件扩展为成员关系或 tenant 条件即可；当前不提前建设复杂权限系统。

解析器保留 `file_path` 用于受信任的离线测试和内部 CLI；在线 API 主链不直接接受客户端 `file_path`。

---

## 3. 来源坐标与 span 回归集

### Q9：来源坐标解决的是什么背景问题？

用户看到的检索文本通常不是原文：

- Markdown 去掉了 `#`、反引号、表格分隔线；
- HTML 去掉了标签、script/style，并解码了实体；
- PDF 抽取后的阅读顺序可能已经变化；
- 分块又把多个 block 合并或截成 Child。

如果 Citation 要在原文预览里跳转或高亮，就必须知道证据从哪里来。但转换之后，坐标不一定还能精确映射，所以系统使用三档精度：

| 精度 | 含义 | 前端允许行为 |
|---|---|---|
| `exact` | `start_char/end_char` 确实对应原文字符区间 | 可以逐字高亮 |
| `line_only` | 只能保证来源行范围，字符位置不可靠 | 跳到行或高亮整行，不做逐字承诺 |
| `unavailable` | 无法可靠回定位 | 只展示文档、章节或页码，不画假高亮 |

“不得伪装为逐字精确”指的是：不能因为 metadata 里恰好有两个数字，就把它们当成 exact 给用户画精确高亮。

### Q10：什么叫建设 span 回归集？

固定一组容易出错的原文和期望位置，每次换 parser/chunker 都自动验证。例如：

- 同一句话在 Markdown 中重复出现两次，两个 span 必须单调前进；
- 标题、粗体、链接、代码块去掉标记后，至少能定位到正确行；
- HTML 实体 `&amp;` 和嵌套标签不能让后续块全部错位；
- Child 只覆盖 Parent 中间一段时，exact 坐标要投影到对应子区间；
- PDF 没有可靠字符映射时必须标 unavailable，不能猜。

### Q11：`exact/line_only/unavailable` 比例是什么？

它是对解析产出的 block 或最终 Child 按精度分类计数，例如：

```text
Markdown: exact 0%, line_only 97%, unavailable 3%
TXT:      exact 100%, line_only 0%, unavailable 0%
PDF:      exact 0%, line_only 0%, unavailable 100%
```

这些数字不是越高越好。PDF 全部 unavailable 但诚实，优于伪造 100% exact。比例的用途是发现回归：某次升级后 Markdown unavailable 从 3% 突然变成 40%，说明回定位算法坏了。

### Q12：为什么这是 P0？是不是优先级过高？

拆开看：

- **P0：坐标契约诚实 + 最小回归集。** 因为项目把“来源可追溯”当作核心卖点，错误高亮会直接破坏可信度。
- **P1：扩大到 20–30 篇文档、按格式输出完整比例报告和可视化。** 这是质量分析，不应阻塞最小闭环。

这个最小 P0 已完成：现有回归覆盖重复文本、HTML 实体和 Child 坐标投影，并约束不把 `line_only/unavailable` 冒充 `exact`。后续只保留 P1：扩大到 20–30 篇文档并生成按格式统计报告。

---

## 4. 父子分块里的 profile

### Q13：profile 到底是什么？

profile 在这里是“命名配置预设”。例如：

```text
default:
  parent_target_tokens=512
  child_target_tokens=128
  preserve_code_block=true
  preserve_table_block=true

code-heavy:
  更强调代码块原子性，可能使用不同 token 上限

table-heavy:
  更强调表格完整性，超长表格使用专门降级策略
```

它不是：

- 用户画像；
- Embedding 模型；
- 新的插件系统；
- 自动判断文档类型的分类模型。

当前代码已经有 `ChunkConfig`、`chunk_profile_version` 和 `chunk_profile_hash`，但只有一套默认参数，没有真正的 `default/code-heavy/table-heavy` 命名注册表。

### Q14：现在需要增加多个 profile 吗？

不需要立刻增加。

多个 profile 会带来新的索引版本、重入库、评测矩阵和解释成本。如果当前业务文档尚未证明默认配置在代码或表格上明显失败，那么一个默认 profile 更适合个人项目。

建议把 README 的优先级改成 `P2（条件触发）`：

> 保留 profile version/hash；只有 Golden Set 显示 code/table 分片存在稳定失败模式时，才新增一个针对性 profile，并与 default 做同集 A/B。

原因不是 profile 不重要，而是 README 对 P2 的定义本来就是“只有业务评测证明需要时才实施”。当前没有失败证据时，给它固定 P1 会让“P0 后必须做”与“有条件才做”互相矛盾。

面试回答可以是：

> 我设计了 profile/version/hash 以支持不同文档类型，但个人项目当前只启用 default，避免没有数据支撑的参数组合爆炸。等失败样例证明代码或表格需要不同策略，再增加一个 profile，并用固定评测集决定是否上线。

---

## 5. Embedding 输入消融与单模型成本

### Q15：三种输入消融是不是让我配多个词嵌入模型？

不是。这里固定当前的 `embedding-3`，只改变同一个 Child 送入模型的文字。

| 方案 | 送入同一 Embedding 模型的文本 | 可能优点 | 可能缺点 |
|---|---|---|---|
| body-only | `子块正文` | 最省 token，正文语义纯净 | 短块可能缺少主题 |
| title+body | `文档标题 + 子块正文` | 文档主题更明确 | 标题可能对所有块造成同质化噪声 |
| section+body | `章节路径 + 子块正文` | 局部上下文更明确 | token 增加，错误标题会污染向量 |

“消融”的含义是：其余条件不变，只改一个变量，观察 Recall、MRR、nDCG 和 token 成本变化。这与“比较多个 Embedding 模型”是两种独立实验。

当前代码和 artifact 已明确采用 `child-body-v1`，模型是 `embedding-3`。所以当前事实不是“方案未定”，而是“已经有一个可用且付过成本的 baseline”。

### Q16：个人项目没有预算，应该怎么做？

你的判断是对的：现在继续使用单模型、body-only 完全合理。

建议当前决策：

1. 不配置多个付费 Embedding 模型。
2. 不为了完成表格而全量重算三套向量。
3. 保留 `embedding_profile=child-body-v1` 和模型名，保证以后可比较。
4. 如果以后确实要试，只在小型业务 Golden Set 或 100–500 篇代表文档上先筛选；有明显收益再决定是否全量重嵌入。
5. 将“三种输入消融”从无条件 P1 改成“预算允许或当前召回失败指向上下文不足时再做”。

面试时可以这样回答：

> 标准工程里我会先固定 Golden Set，区分两类实验：第一类固定 Embedding 模型，只比较 body-only、title+body、section+body 的输入模板；第二类才是不同模型的横向比较。选择时同时看 Recall@K、MRR/nDCG、向量维度、最大输入、中文效果、延迟和每百万 token 成本。这个个人项目受预算限制，目前只使用 embedding-3 和 child-body-v1，并保存 model/profile/hash，保证结论诚实、结果可复现；没有把未做的多模型对比包装成已完成。

---

## 6. 融合排序、RRF 与 Rerank

### Q17：完整检索链路里的几个名词是什么关系？

```text
用户 Query
├─ 向量召回：找“语义相似”的 Child
└─ BM25 召回：找“关键词、型号、数字精确匹配”的 Child
        ↓
融合排序 Fusion：把两路结果合成一个候选列表
        ↓
候选池 Candidate Pool：例如前 30 或前 100 个 Child
        ↓
Rerank：用更贵但更精细的方法重新判断 Query 与每个候选的相关性
        ↓
最终 TopK：例如 5 个 Child
        ↓
恢复 Parent → 构造 Evidence Window → 回答与引用
```

#### 向量召回

Query 和 Child 各自编码成向量，通过距离找语义接近内容。擅长同义词、口语表达，可能漏掉型号和精确数字。

#### BM25

传统关键词相关性算法。擅长术语、错误码、型号和原文词匹配，但不理解“互斥量”和“锁”可能语义相关。

#### Fusion / 融合排序

同一个 Child 可能只在向量结果里、只在 BM25 结果里，或两路都命中。Fusion 负责合并、去重并给出统一顺序。

#### Rerank / 重排

召回阶段通常分别编码 Query 和文档，便宜但交互较弱。Cross Encoder Reranker 把 `Query + 候选文本` 成对输入模型，能更细地判断相关性，但必须对每个候选做推理，所以只用于较小候选池。

### Q18：当前 Weighted Sum 是什么？有什么问题？

当前代码使用：

```text
fused_score = vector_weight × vector_score
            + keyword_weight × keyword_score
```

默认权重是向量 0.75、关键词 0.25。代码把向量分数限制在 0–1，把本次 BM25 命中的最高分归一为 1，再做加权。

优点：简单、直观、可以解释。

风险：

- 两路分数虽然都在 0–1，含义仍不完全相同；
- BM25 按“本次结果的最高分”归一，相同文档在不同 Query 下的刻度会变化；
- 权重 0.75/0.25 目前不是业务 Golden Set 选出来的；
- 分数阈值会同时受到归一化和权重影响。

所以“已实现”不等于“当前权重已证明最优”。

### Q19：Weighted RRF 是什么？

RRF（Reciprocal Rank Fusion）主要看每个候选在各通道的名次，不直接比较原始分数：

```text
rrf_score(d) = w_vector / (k + rank_vector(d))
             + w_bm25   / (k + rank_bm25(d))
```

例如某个 Child 在向量通道排第 1、BM25 排第 5，它会同时得到两路名次贡献。没有出现在某一路时，该路贡献为 0。

优点是对“向量分数和 BM25 分数刻度不一致”更稳健；缺点是丢掉了分数间距信息，而且 `k`、通道权重、候选深度仍要选择。RRF 不是天然一定优于 Weighted Sum，只是一个值得低成本 A/B 的候选。

### Q20：Rerank 能不能弥补召回失败？

不能。

如果正确文档没有进入候选池，Rerank 看不到它，就不可能把它排上来。可以记成：

> Recall 决定天花板，Rerank 改善候选池里的头部顺序。

因此实验必须固定三个数字：

- `candidate_top_k`：两路召回和融合后保留多少候选；
- `rerank_top_n`：送入 Reranker 的候选数；
- `final_top_k`：最终送给父块恢复/回答的数量。

### Q21：当前项目的 Fusion/Rerank 到底做到哪了？

当前代码事实：

- Weighted Sum 已实现；
- Weighted RRF 尚未实现；
- 规则 Reranker 已实现；
- Cross Encoder Reranker 已实现并可失败回退；
- `rerank_enabled` 默认是 false；
- Cross Encoder 默认模型仍是英文 `ms-marco-MiniLM`，中文场景若启用应显式配置合适模型；
- 既有 T2Retrieval 记录已经给出一次 `BAAI/bge-reranker-base` 的离线增益：Recall@10 从 79.99% 到 84.08%，MRR@10 从 0.8507 到 0.9097。

所以 README 中“没有真实增益报告”的说法不够准确。更准确的是：

> 已有 T2 10k 子集上的向量召回 + Rerank 历史报告，但缺少机器可复算的正式 report 文件，也没有证明当前生产 Hybrid Weighted Sum + 业务 Golden Set 上仍有同样增益。

### Q22：几个常见指标分别是什么？

| 指标 | 回答的问题 |
|---|---|
| Hit@K | 前 K 个结果里是否至少出现一个正确文档？ |
| Recall@K | 所有正确文档中，有多少比例进入了前 K？ |
| MRR | 第一个正确结果排得有多靠前？第一名最好。 |
| nDCG@K | 多个相关结果的整体顺序是否合理，并可考虑不同相关等级？ |
| Evidence Recall | 回答所需的关键事实有多少真正进入 Prompt 证据？ |
| P95 latency | 95% 请求能在多长时间内完成，用于观察长尾延迟。 |

### Q23：Fusion 和 Rerank 应该怎么重新定优先级？

建议拆成“必须测”和“是否实现”：

#### P0：必须测

1. 固定同一 Golden Set、同一索引、同一 Embedding 模型。
2. 保存当前 Vector-only 与 Hybrid Weighted Sum baseline。
3. 固定候选漏斗，比较 Rerank off/on 的 nDCG、MRR、Evidence Recall、P95。
4. 中文场景显式使用中文/多语 Reranker，不能拿英文默认模型下结论。
5. 报告失败样例，而不只报告总平均分。

#### P2：失败证据触发后实现

1. 如果 Weighted Sum 在不同 Query 上明显受分数刻度影响，再实现 Weighted RRF。
2. 如果 RRF 在同集显著提升且没有不可接受的副作用，再替换默认融合。
3. 如果 Cross Encoder 的业务增益覆盖延迟和部署成本，再在生产默认开启；否则保留为可选能力。

换句话说，**P0 是建立可信选择依据，不是强制把所有候选算法都上线。**

### Q24：为什么 `vector_weight=1, keyword_weight=0` 不能当 Vector-only？

因为“最终分数不给 BM25 权重”和“根本不执行 BM25”是两件事。旧版 `HybridRouter.retrieve()` 会依次调用向量与关键词 Retriever，再做融合。即使关键词权重为 0：

- BM25 仍产生调用耗时，Vector-only 的延迟不真实；
- 关键词候选仍可能进入融合数据结构，阈值为 0 时尤其容易污染候选集合；
- 单通道异常仍可能让所谓 Vector-only 实验失败。

因此生产配置需要显式模式：

```text
retrieval_mode = vector  → 只调用 EmbeddingRetriever
retrieval_mode = keyword → 只调用 KeywordRetriever
retrieval_mode = hybrid  → 两路召回后 Weighted Sum
```

该缺口已于 2026-09-18 修复：`retrieval_mode` 已进入生产配置，验收测试使用可计数假 Retriever 证明未选通道调用次数为 0。Runner 只需传配置，不允许自己绕开 `HybridRouter` 拼一套实验实现。

### Q25：Rerank 已经有开关，为什么还要改？

`rerank_enabled` 和 `rerank_backend` 原本已存在，规则/Cross Encoder 的核心实现不需要重写；当时的缺口是没有独立 `rerank_top_n`，启用后会对整个融合池重排，无法固定下面的实验漏斗：

```text
candidate_top_k（召回池）
→ rerank_top_n（送排池，且不得大于召回池）
→ top_k（最终结果）
```

该缺口已于 2026-09-18 修复：生产配置已有 `rerank_top_n` 和范围校验，Trace 已保存 backend、模型名、送排数量和 fallback。剩余工作只有用真实业务集判断提升是否值得部署成本。

### Q26：为什么 Evidence Window 必须改成请求级 `QAConfig`？

旧版窗口算法已经实现，但 `context_top_k` 和 `evidence_window_tokens` 在 `QAService` 构造时从全局配置读取。用修改环境变量、重启进程的方式扫描参数会带来三个问题：容易残留上一次配置、难以并行、run 文件无法证明某一题实际用了哪个值。

建议引入与 `RetrievalConfig` 分离的不可变请求配置：

```text
QAConfig:
  context_top_k
  evidence_mode = child_only | window | full_parent
  evidence_window_tokens
```

该缺口已于 2026-09-18 修复：`QAService.query(question, retrieval_config, qa_config)` 已把最终生效值写入请求级 Trace，并支持 `child_only/window/full_parent`。Dev 集只需通过生产接口比较候选配置，不在 Runner 内复制 `ContextWindowBuilder`。

### Q27：结构化 usage 和分阶段耗时具体记录什么？

旧版只有总请求延迟和 `prompt_used_tokens` 估算，供应商响应里的真实 usage 没有进入 QA payload，总耗时也无法定位退化发生在哪一段。

每次 QA 应返回请求级、不可共享的 Trace：

```text
trace.config:
  retrieval_config / qa_config / model / index fingerprint

trace.timings_ms:
  retrieval / window / generation / citation / total

trace.usage:
  model / input_tokens / output_tokens / total_tokens / cost
```

该缺口已于 2026-09-18 修复：计时使用单调时钟，QA payload 已返回请求级 config/timings/usage；供应商未返回的字段记 `unavailable`，不补零。`last_trace` 仅保留兼容调试，正式评测使用 `retrieve_detailed()` 返回的请求局部 Trace。

---

## 7. 合并后的算法层工作计划

旧计划把解析验收、来源坐标、检索接口、多个参数实验分别列成 P0，导致同一条生产链被重复计数。2026-09-19 重新核对当前代码并在 Conda `agent` 环境运行 `backend/tests + evaluation/tests`，结果为 `102 passed`。当前状态合并如下。

### 7.1 已完成，不再列入 P0 待办

| 能力 | 完成证据 | 后续非阻塞事项 |
|---|---|---|
| 统一解析与来源追踪 | md/pdf/html/txt 统一走 `UnstructuredParser`；Numba 缓存问题已规避；重复文本、HTML 实体、非法 UTF-8、嵌套代码、表格文本和来源精度分级有回归 | `P1` 补部署 bootstrap、冷/热耗时和 20–30 篇精度分布报告 |
| 父子分块与 Embedding | 硬 token 上限、Parent/Child 内容守恒、坐标投影、profile hash、只嵌 Child 及向量合法性有回归 | 失败数据触发后再比较 profile 或 Embedding 输入模板 |
| 离线评测基础设施 | CRUD-RAG Mini 1,000 文档/150 问题、确定性 validator/scorer/report 已完成并验证可复现 | 真实 run 由剩余 P0-B 生成 |
| 生产评测接口 | `retrieval_mode`、`rerank_top_n`、请求级 `QAConfig`、请求级 config/timings/usage Trace 已实现并测试 | Runner 只负责编排与录制，不再复制算法 |

### 7.2 两个 P0 工作包：P0-A 已实现（2026-09-19），P0-B 待实现

#### P0-A：Markdown 图片/VLM 最小闭环 —— √ 已实现并回归验证

要求与落实情况：

- 只支持 Markdown 相对路径本地 PNG/JPEG/WebP，不顺带实现 HTML/PDF 图片或 OCR；
  → `backend/src/parsing/markdown_images.py`：URL/绝对路径/`..` 穿越/不支持类型/缺文件全部显式 skipped 并带原因，围栏代码块中的图片语法不计。
- 资产读取必须基于 `document_id/asset_id/current_user_id` 鉴权，拒绝任意路径和 `..` 穿越；
  → `backend/src/assets/asset_store.py`：`AssetRegistry.authorized_asset()` 单条 JOIN 同时校验资源、文档归属与 owner；`AssetFileStore` 的 storage_key 读取前确认仍在资产根目录内。
- 原图按 SHA256 保存，VLM 只生成受 schema 约束的描述文字；
  → `AssetFileStore.save()` 幂等去重；`vlm_describer.py` 要求 JSON（ocr_text/subjects/key_facts/chart_trends/uncertainties）并程序拼装 `page_content`。
- image 作为原子 block，描述进入分块/Embedding，`asset_id/document_id` 透传到 Citation；
  → `BlockMergeStrategy` 无条件 preserve image；`BlockChunker` 透传 `asset_id` 等键；`CitationService` 与 `HybridRouter.matched_children` 携带 `asset_id`。
- 预览再次鉴权；VLM 失败保留资产和失败状态，不伪造描述；
  → `GET /assets/preview`（`AssetService.preview`）；失败路径在 assets 表记 `failed` 并保留占位正文。
- 验收有权限、无权限、路径穿越、重复图片和失效资源；
  → `backend/tests/{parsing/test_markdown_images, chunking/test_image_blocks, services/test_image_pipeline, services/test_asset_preview}.py` 共 21 项。

剩余归 P1：真实 VLM 模型（`MVP_VISION_LLM_*`）联调、线上伴随图片上传通道；HTML/PDF 图片与 OCR 维持不做。

#### P0-B：生产 Runner、baseline 与一次参数锁定

- `evaluation/runner` 只调用生产 `IngestionPipeline/HybridRouter/QAService`；
- 建独立评测索引和 index manifest，保存 dataset/Git/config/model/profile 指纹；
- 录制 Vector、Hybrid Weighted Sum、Hybrid+Rerank，以及 child/window/full-parent 的真实 run；
- 只在 Dev 扫 `candidate_top_k/top_k/threshold/vector_weight/rerank_top_n/evidence_window_tokens`，避免无边界组合；
- 锁定后在 Test 各运行一次，生成机器可读 score、Markdown 报告和失败案例；
- Rerank 无稳定收益则保持默认关闭；Weighted Sum 未暴露刻度问题则不实现 RRF。

### 7.3 P1：P0 稳定后再做

- 清理无消费者 metadata，并扩充解析/span 质量报告；
- 接入 `question_kwd/question_tks` 等索引侧增强并评测口语问题切片；
- 接入指代消解与口语规范化查询改写，建设多轮/歧义评测；
- 多检索通道并行与失败隔离；
- 经人工校准的 Faithfulness、Answer Correctness、Context Precision 等 Judge 指标。

### 7.4 P2：失败证据触发后再做

- 针对 code/table 的新 chunk profile；
- Weighted RRF、Embedding 输入模板消融；
- HyDE、Step-back、查询拆分；
- OCR/hi_res、HTML/PDF 图片、知识图谱、联网搜索和 Agent/MCP。

### 当前明确不做

- 不为了面试展示同时购买或配置多个 Embedding 模型；
- 不切割图片像素来适配文本 chunk；
- 不在没有失败数据时维护多套 profile 或替换 Weighted Sum；
- 不把 `102 passed` 写成真实 ES、模型或业务指标已经联调。

---

## 8. 下一轮讨论入口

后续每次讨论只需要在本文继续补三类内容：

1. **问题**：某个术语、设计或优先级为什么这样定；
2. **决定**：当前项目做什么、不做什么，以及原因；
3. **执行项**：文件、测试、指标、命令和验收标准。

当决定稳定后，将状态和最终路线回写 `docs/plan/README.md`；本文保留解释与决策背景。
