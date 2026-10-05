# 算法层 QA 与可执行工作计划

更新日期：2026-09-28。

本文专门解释 `docs/plan/README.md` 算法层里容易产生歧义的术语、优先级和验收口径，并把讨论结果沉淀成后续可执行清单。

本文不是第二份“项目事实总表”。项目状态仍以 `docs/plan/README.md` 为准；这里形成明确决策后，再把结论回写到 README，避免两份状态互相冲突。

## 0. 先给结论

| 问题 | 结论 |
|---|---|
| 统一解析引擎是不是已经构造完毕？ | 是。Markdown、PDF、HTML、TXT 已统一到 `UnstructuredParser`，当前工作区回归通过；干净环境安装、服务上传联调和冷/热启动耗时属于后续部署验收，不再算算法 P0。 |
| 以前不是已经通过测试了吗？ | 是。2026-09-23 在 Conda `agent` 环境重跑 `backend/tests + evaluation/tests`，结果为 `195 passed`（2026-09-21 历史记录为 164）；它能证明受测代码契约，但不能替代真实 ES、真实模型和部署联调。 |
| 图片为什么要分块？ | 不切图片像素。图片保存为资产；`image block` 是图片在统一 Document 契约中的逻辑块。通常一图一块，只有过长的 VLM 文本描述才按文本切分。 |
| VLM 描述是什么？ | 核心确实是把图片交给视觉模型，让它输出可检索的文字；但还需要资产保存、提示词约束、结构化输出、失败处理、索引和引用回传。 |
| 来源坐标为什么是 P0？ | “精度诚实 + 最小回归集”曾是 P0，目前已实现并通过回归；扩大到 20–30 篇并生成比例报告属于 P1。 |
| 父子分块的 profile 是什么？ | 是一组命名的分块参数预设，不是用户画像，也不是模型。当前只有一套默认配置就够了，先保留 hash 和版本，出现明确失败再增加 profile。 |
| 三种 Embedding 输入消融是不是三个模型？ | 不是。固定当前同一个 Embedding 模型，只比较“正文 / 标题+正文 / 章节路径+正文”三种输入模板。个人项目可因成本暂缓，不需要购买或配置多个模型。 |
| 融合排序和 Rerank 为什么影响效果？ | 融合决定向量召回与 BM25 召回如何合并；Rerank 对合并后的少量候选做更精细的二次排序。前者决定候选池，后者主要改善头部顺序。 |
| 当前使用哪种融合？ | 等权 RRF（k=60），固定缩放到 0～1；旧 Weighted Sum 已删除。历史调参收益仍属于旧实现，RRF 质量待重新评测。 |
| `vector_weight=1` 是不是 Vector-only？ | 不是；不过这个缺口已修复。现在使用 `retrieval_mode=vector`，测试已证明未选中的 BM25 通道调用次数为 0。 |
| 为什么 Evidence Window 还要改配置接口？ | 这个缺口已修复。`QAService.query()` 已接受请求级 `QAConfig`，可在同一 Runner 中逐题比较 `child_only/window/full_parent`。 |
| 为什么总延迟还不够？ | 这个缺口也已修复。QA 已返回请求级分阶段耗时和真实可得的 usage；Runner 已完成历史真实录制与 P50/P95、token 基线；当前配置变化后的性能需要另行实测。 |
| 为什么 2026-09-22 消融里三种 Rerank 都没有净收益？ | 历史实验表现为头部指标下降、尾部指标上升，未满足默认开启条件。饱和程度、规则信号重叠、CE 整分替换和 Child 聚合都是分析线索，未隔离唯一原因；详见 Q28–Q30。 |
| CRUD-RAG-mini 还能不能测出 Rerank？ | 历史实验检测到较大差异，但对小差异的把握受样本量、饱和度和逐题方差影响，不能给出统一的 ±5pt 检测边界。见 Q30–Q31。 |
| 当前支持 Parent 重排和 max 聚合吗？ | 支持。新增请求级 rerank_level 与 child_score_aggregation，默认仍为关闭重排、child、mean；历史 T2 实验与当前生产接口的边界见 Q32。 |

---

## 1. 统一解析引擎：已完成什么，还剩什么

### Q1：`UnstructuredParser` 现在算完成了吗？

算“算法实现与当前环境回归已完成”。当前证据包括：

- Markdown、PDF、HTML、TXT 已统一由 `UnstructuredParser` 承接，`parser_factory.py` 已接入统一实现。
- PDF 相关重依赖按需加载，并已规避只读环境中的 Numba 缓存初始化问题。
- 回归覆盖标题、列表、代码块、表格文本、frontmatter、HTML 实体、重复文本回定位、非法 UTF-8 和 PDF 文本提取等关键行为。
- 2026-09-23 在 Conda `agent` 环境运行 `python -m pytest backend/tests evaluation/tests -q`，结果为 `195 passed`；2026-09-21 的 `164 passed` 保留为历史里程碑。

因此状态应写为：

> **√ 已实现并回归验证**

仍未证明的是“任意干净环境都可一次安装”“真实服务上传链路已联调”“冷/热启动性能有基线”。这些属于部署与性能验收，合并为 P1，不再重复列为算法 P0。

### Q2：后续的部署验收具体是什么？

只保留三个动作：

1. 在干净目标环境验证 Python、Unstructured、PDF 系统库和 NLTK 数据可复现安装，运行时不依赖开发机隐式缓存。
2. 按真实启动方式完成一次“上传 → 解析 → 分块 → 入库”，记录失败阶段和错误语义。
3. 对小、中、大样例记录首次/热启动耗时、block 数和峰值内存，形成 P50/P95 基线。

### Q3：回归测试通过能证明什么？

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
- **Markdown 中引用的本地图片**已通过 `ImagePipeline` 接入入库、资产登记、VLM 描述、分块和引用预览；
- 伴随图片上传通道和真实 VLM 联调仍待完成，不能把内部文件链路可用等同于线上完整上传体验；
- HTML 远程图片、PDF 内嵌图片、扫描 PDF、独立图片文件应分别定义，不能一句“支持图片”全部覆盖。

当前已编码并有替身测试的 MVP 是：

> Markdown 本地图片 → 保存资产 → VLM 生成描述 → 可检索 → 引用能返回原图。

HTML 远程图片涉及下载安全和网络稳定，PDF 内嵌图片涉及版面与页内关系，扫描 PDF 又属于 OCR；这些不应偷偷塞进同一个 P0。

### Q5：什么是 image block？

`image block` 是解析层输出的一种逻辑 Document，地位类似 heading、paragraph、code、table。它不是图片二进制本身，也不是把图片转成 Base64 塞进 Elasticsearch。

当前成功描述后的关键字段示意如下；失败/跳过时保留占位文本，资产相关字段可能缺失：

```text
page_content:
  VLM 生成的可检索描述，例如“架构图展示请求依次经过 API、Retriever、Reranker...”

metadata:
  block_type: image
  asset_id: sha256:...
  doc_id: ...（入库分块阶段补齐；资产表对应字段为 document_id）
  storage_key: 后端内部受控存储键，不是客户端路径
  mime_type: image/png
  alt_text: 原 Markdown alt
  caption: 交错定位后按最近标题回填，缺失时使用 alt
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

当前规则如下：

1. 原图作为独立 asset 保存，不裁切、不向量化二进制。
2. 一张图片生成一个 image block，不与相邻文本或另一张图片合并。
3. VLM 结构化描述与 alt 拼成正文；caption、section_path 作为元数据回填，当前不拼入嵌入正文。
4. 描述在正常 token 上限内时，一图对应一个可检索 Child。
5. 只有描述异常长、超过硬上限时，才切“描述文字”，不是切图片。
6. 命中该 Child 后，Citation 返回 `asset_id`；客户端通过需要鉴权的 `/assets/preview` 获取原图；当前未实现短期签名 URL。

当前 `BlockMergeStrategy` 已隔离 image，`BlockChunker` 已显式透传 `asset_id` 等图片元数据，检索与 Citation 保留资产引用。与完整 code/table 不同，过长图片描述仍按普通文本预算切分；“原子”指不与相邻块混合，不承诺任意长度的一图只有一个 Child。

### Q7：VLM 描述是不是把图片发给视觉 LLM，让它用自然语言描述一遍？

本质上是，但生产链路多了约束和治理：

```text
从服务端登记的 Markdown 文件发现本地相对路径图片
→ 校验引用路径边界、文件类型和大小
→ 计算 SHA256，生成 asset_id、保存原图，并关联 document_id/owner
→ 将图片和 alt 发送给 VLM，校验结构化 JSON
→ 拼成 image block；失败保留资产和 failed 状态，非法引用保留 skipped 占位
→ 与正文交错排列并回填章节元数据
→ 分块/Embedding/索引
→ 检索命中后返回文字证据与 asset_id
→ /assets/preview 按 asset_id + document_id + 当前用户 JOIN 鉴权
→ 校验 storage_key 路径边界并返回原图
```

自然语言描述不能只追求“看起来通顺”。检索更需要稳定、事实密集的内容，例如：

- 截图：界面名称、字段、按钮、错误码；
- 架构图：节点、箭头方向、调用关系；
- 图表：标题、坐标轴、图例、主要数值和趋势；
- 表格图片：表头、关键单元格，必要时保留行列结构；
- 无法辨认的内容：明确输出不确定，不编造。

当前 VLM 已返回受约束的 JSON，再由程序拼成 `page_content`。这比完全自由的一段描述更容易回归测试和版本化。

### Q8：当前图片预览的读取权限怎么判断？

当前图片预览采用简单归属校验：**读取资产文件前查数据库，确认当前用户拥有所关联文档。** 文档列表、源文件、入库和删除端点目前仅要求登录，完整文档归属隔离仍是 P1，不能把图片预览鉴权扩大描述为全站权限隔离。

`AssetRegistry.authorized_asset()` 用一条查询完成：

```sql
SELECT a.*
FROM assets a
JOIN documents d ON d.doc_id = a.document_id
WHERE a.asset_id = :asset_id
  AND a.document_id = :document_id
  AND json_extract(d.payload, '$.owner_id') = :current_user_id;
```

预览查不到授权记录时不返回文件；查到后，后端使用数据库保存的 `storage_key` 受控读取。入库 VLM 使用解析管线校验过的本地图片，并不执行这条预览 JOIN；其触发端点的文档 owner 校验仍待补齐。

图片预览接口的约束：

1. 客户端提交 `document_id/asset_id`，不能提交服务器任意本地路径。
2. 数据库查询必须同时带资源 ID 和当前用户的归属条件，不能先按 ID 查出再忘记鉴权。
3. `storage_key` 由后端生成并保存；读取前确认规范化路径仍在项目的文件存储目录内，防止 `..` 等路径穿越。
4. 原图预览只返回通过归属查询和存储边界检查的文件；不得把 asset_id 的可猜测性当作权限校验。

SHA256 只是查重标识，不是权限凭证。若将来增加 KnowledgeBase、团队或租户，再把查询条件扩展为成员关系或 tenant 条件即可；当前不提前建设复杂权限系统。

解析器保留 `file_path` 用于受信任的离线测试和内部 CLI；在线 API 主链不直接接受客户端 `file_path`。

---

## 3. 来源坐标与 span 回归集

### Q9：来源坐标解决的是什么背景问题？

用户看到的检索文本通常不是原文：

- 普通 Markdown 文本可能去掉标记；当前受保护的代码与表格则保留原文；
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
├─ 向量召回：语义相似 Child
└─ BM25 召回：关键词匹配 Child
        ↓
Fusion：按 Child 合并去重、等权 RRF 评分
        ↓
关闭重排：融合分过滤 → Parent mean/max 聚合
Child 重排：取前 N 个 Child → 重排 → 分数过滤 → Parent mean/max 聚合
Parent 重排：取同样 N 个 Child → 恢复并去重 Parent → CE 重排 → 分数过滤
        ↓
取最终 top_k 个结果（正常为 Parent，父块缺失可回落 Child）
        ↓
按 context_top_k 选取 → 构造 Evidence → 回答与引用
```

#### 向量召回

Query 和 Child 各自编码成向量，通过距离找语义接近内容。擅长同义词、口语表达，可能漏掉型号和精确数字。

#### BM25

传统关键词相关性算法。擅长术语、错误码、型号和原文词匹配，但不理解“互斥量”和“锁”可能语义相关。

#### Fusion / 融合排序

同一个 Child 可能只在向量结果里、只在 BM25 结果里，或两路都命中。Fusion 负责合并、去重并给出统一顺序。

#### Rerank / 重排

召回阶段通常分别编码 Query 和文档，便宜但交互较弱。Cross Encoder Reranker 把 `Query + 候选文本` 成对输入模型，能更细地判断相关性，但必须对每个候选做推理，所以只用于较小候选池。

### Q18：旧版 Weighted Sum 是什么？有什么问题？

2026-09-28 之前的代码使用（现已删除）：

```text
fused_score = vector_weight × vector_score
            + keyword_weight × keyword_score
```

旧版默认权重是向量 0.75、关键词 0.25。代码把 ES 转换后的向量分数限制在 0–1，把本次 BM25 命中的最高分归一为 1，再做加权。

优点：简单、直观、可以解释。

风险：

- 两路分数虽然都在 0–1，含义仍不完全相同；
- BM25 按“本次结果的最高分”归一，相同文档在不同 Query 下的刻度会变化；
- 权重 0.75/0.25 不是业务 Golden Set 选出来的；
- 分数阈值会同时受到归一化和权重影响。

历史调参过程仍可讲述，但成绩属于旧版 Weighted Sum，不能归因于后来替换的 RRF。

### Q19：当前 RRF 怎么计算？

RRF（Reciprocal Rank Fusion）主要看每个候选在各通道的名次，不直接比较原始分数：

```text
rrf_score(d) = 1 / (60 + rank_vector(d))
             + 1 / (60 + rank_bm25(d))
fused_score(d) = rrf_score(d) / (2 / 61)
```

例如某个 Child 在向量通道排第 1、BM25 排第 5，它会同时得到两路名次贡献。没有出现在某一路时，该路贡献为 0。

当前使用等权 RRF，排名从 1 开始；分母 2/61 是两路都排第一的理论上限，即使某一路为空也保持不变。固定缩放不改变 RRF 排序，保留原始 `rrf_score` 供追踪；向量/BM25 分数仅保留作诊断，混合排序不使用它们，同分按块顺序及 ID 排。纯向量/纯关键词模式保留通道原分数。

优点是避开两路分数刻度不一致；缺点是丢掉分差，仍受候选深度、通道质量和 k 影响。0～1 的 RRF 分数不是相关概率；旧阈值和规则重排加分尚需重新评测，不能宣称替换后质量已提高。

### Q20：Rerank 能不能弥补召回失败？

不能。

如果正确文档没有进入候选池，Rerank 看不到它，就不可能把它排上来。可以记成：

> Recall 决定天花板，Rerank 改善候选池里的头部顺序。

因此实验必须固定三个数字：

- `candidate_top_k`：每个启用通道召回的 Child 数，双路融合池可达其两倍；
- `rerank_top_n`：从融合池选取的 Child 数，None 使用 candidate_top_k；Parent 模式去重后实际送排数可能更少；
- `top_k`（Trace 中的 `final_top_k`）：聚合后最终返回的数量，正常为 Parent。

### Q21：当前项目的 Fusion/Rerank 到底做到哪了？

当前代码事实：

- 等权 RRF 已替换 Weighted Sum，旧权重配置与扫描已删除；
- 真实 RRF 质量评测尚未完成；历史 Weighted Sum run 保留供离线复算；
- 规则 Reranker 已实现；
- Cross Encoder Reranker 已实现，支持 Child/Parent 粒度并可失败回退；Child 分数支持 mean/max 聚合；
- `rerank_enabled` 默认是 false；
- Cross Encoder 默认模型仍是英文 `ms-marco-MiniLM`，中文场景若启用应显式配置合适模型；
- 既有 T2Retrieval 记录已经给出一次 `BAAI/bge-reranker-base` 的离线增益：Recall@10 从 79.99% 到 84.08%，MRR@10 从 0.8507 到 0.9097。

这些数字属于历史实验。后续还有 2026-09-22 T2 冻结候选池的 Parent 重排报告，见 [案例与产物链接](../interview/child-parent-rerank-cases.md)。不同实验不能混为同一轮收益，也不能直接外推为当前 v3 全链路效果。

### Q22：几个常见指标分别是什么？

| 指标 | 回答的问题 |
|---|---|
| Hit@K | 前 K 个结果里是否至少出现一个正确文档？ |
| Recall@K | 所有正确文档中，有多少比例进入了前 K？ |
| MRR | 第一个正确结果排得有多靠前？第一名最好。 |
| nDCG@K | 多个相关结果的整体顺序是否合理，并可考虑不同相关等级？ |
| Evidence Recall | 所需正例文档有多少进入 Prompt 证据集合？当前不测关键事实覆盖。 |
| P95 latency | 95% 请求能在多长时间内完成，用于观察长尾延迟。 |

### Q23：Fusion 和 Rerank 的评测准入原则是什么？

建议拆成“必须测”和“是否实现”：

#### 基线原则（历史 P0 已完成，新策略按同样原则比较）

1. 固定同一 Golden Set、同一索引、同一 Embedding 模型。
2. 保存当前 Vector-only 与 Hybrid RRF baseline；旧 Weighted Sum 的产物独立保留，不覆盖。
3. 固定候选漏斗，比较 Rerank off/on 的 nDCG、MRR、Evidence Recall、P95。
4. 中文场景显式使用中文/多语 Reranker，不能拿英文默认模型下结论。
5. 报告失败样例，而不只报告总平均分。

#### 新融合的验收

1. 当前已按需求直接替换为 RRF；替换实现不等于已证明收益。
2. 固定索引、模型和候选池，对比旧版与 RRF；在 Dev 重选阈值，再做独立验收。
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

该缺口已于 2026-09-18 修复：计时使用单调时钟，QA payload 已返回请求级 config/timings/usage；供应商未返回的字段记 `unavailable`，不补零。当前 `HybridRouter` 已无 `last_trace` 共享状态；正式评测使用 `retrieve_detailed()` 返回的请求局部 Trace。

### Q28：2026-09-22 受控 Rerank 消融测了什么，结果如何？

> 历史实验记录：以下四组结果对应当时的代码。项目收尾已移除 `rule-jieba` 实验后端，保留原始本地 run/score/report；一次性消融脚本已移除，当前运行时保留 `rule` 与 `cross-encoder`。Q29–Q31 是当时的分析与候选方案，不代表已验证的因果结论或必须实施的功能。

四组对照共用同一候选池与请求配置（`candidate_top_k=30`、`rerank_top_n=30`、hybrid 0.75/0.25，Dev 30 题；CrossEncoder 档显式配置 `BAAI/bge-reranker-base`），产物在 `evaluation/reports/generated/dev_rerank_ablation.json` 与 `evaluation/scores/dev_retrieval_rerank_*.score.json`：

| 组 | MRR@10 | Hit@1 | Recall@3 | Recall@10 | nDCG@10 | P50 延迟 |
|---|---|---|---|---|---|---|
| none（不重排） | **0.892** | **0.833** | 0.767 | 0.933 | 0.856 | 44ms |
| rule_char_bigram | 0.857 | 0.767 | 0.828 | 0.972 | 0.873 | 40ms |
| rule_jieba | 0.857 | 0.767 | 0.839 | 0.972 | 0.874 | 42ms |
| cross_encoder | 0.846 | 0.733 | **0.861** | **0.978** | **0.876** | 416ms |

统一模式是"头部变差、尾部变好"。以 qrels 为金标准的逐题金文档名次对比：cross_encoder 使 4 题变差（1→2、1→2、1→3、2→5）、2 题变好（6→2、4→2）、24 题不变；rule 两档各为 4 差 / 1 好 / 25 不变。四组 `error_count`、`fallback_count` 均为 0，排除"模型失败退回原排序"造成的假差异。

### Q29：为什么三种 Rerank 看起来都不起作用？

需要区分实测现象和解释假设，不能排除模型、输入粒度或聚合方式本身的影响：

1. **可提升空间有限。** 历史分析中 Dev 30 题有 26 题不开重排已达到 Recall@10 满分，因此该指标的正向空间集中在剩余 4 题；头部排序仍有改进和退化空间。
2. **规则信号与 BM25 有重叠。** rule 使用字面覆盖率，并在 Child 融合分上最多加 0.1。这个上限约束同一 Child 池中的名次变化，但不能把它直接套到经过阈值过滤、Parent 聚合和文档去重后的最终名次。两种分词的 MRR 相同，Recall@3 与 nDCG 却不同，因此不能说“没有任何指标相关变化”。
3. **CE 整分替换是一个待验证因素。** 实现确实以 CE 分替换融合分；历史结果同时出现尾部召回提升与头部退化。没有固定输入比较分数混合、重排粒度和聚合方式前，不能断言退化由整分替换导致。历史分析给出的 Recall@10 差值配对区间为 [0.000, 0.100]，包含零，仅说明该次估计未排除零收益。

面试表述应是：

> 这轮重排提高了尾部召回，但未满足 MRR 不下降的开启条件。我检查了基准饱和度、规则信号重叠、CE 整分替换与 Child 聚合，保留这些解释作为下一步消融方向，没有把相关现象包装成唯一因果结论。

### Q30：CRUD-RAG-mini 基准的分辨力边界在哪里？

语料构成（`evaluation/build_dataset.py`）：1000 篇 = 300 篇金文档 + 150 篇困难负例（按词重叠挑选）+ 550 篇随机填充。

**能测的**：检索模式级差异。Test 120 题上 hybrid 0.882 < vector 0.924 < locked 0.975（Recall@10）；同首排配置、仅差 rerank 开关的旧实验对（rule 后排、`rerank_top_n=10`）测出 +7.5pt Recall@10，配对 bootstrap 95% CI [+0.039, +0.117]，不含零，显著。

**需要谨慎的**：强配置之间的小差异。检测能力取决于逐题方差和样本量，不能统一断言 ±5pt 都测不了。当前有两个限制：池子小且随机干扰占多数，dense 首排近乎饱和（Dev 26/30 满分；locked 已到 0.975，headline 指标仅剩 2.5pt 理论空间）；每题金文档只有 1~3 个，Recall@10 每题只能取 0、1/3、1/2、2/3、1 等离散值，30 题平均后颗粒极粗。

因此在该基准上的敏感指标是 **Hit@1（0.833）、Recall@1（0.506）、MRR@10（0.892）**——它们仍有头寸；Q28 消融的退化只在头部指标上显现，与此一致。配套纪律：比较两个配置必须报配对 bootstrap CI，**CI 跨零不改默认配置**。

### Q31：历史候选动作（2026-09-22，未实施，不构成当前待办承诺）

1. **分数混合替代整分替换**：`final = α × rerank分 + (1−α) × 原融合分`，α 做成请求级配置，与 `rerank_top_n` 同路数；`cross_encoder_reranker.py` 与 `rule_reranker.py` 同改。验收标准：Dev 扫 α ∈ {0.3, 0.5, 0.7}，**MRR 不低于 none 且 Recall@10 不低于 none** 才换默认值，赢的 α 上 Test 确认。
2. **`score_run.py` 补配对 bootstrap CI 输出**，零成本落实 Q30 的 CI 纪律。
3. **扩池重建基准**：复用 query-first 抽样（金文档强制入池）与现有困难负例逻辑，`corpus_size` 扩到 5000~10000，目标把基线 Recall@10 压回 0.6~0.75 工作区间，重新开放强配置区的分辨力。成本为重建索引与重跑冻结流程。
4. **rule 保持可选**：本轮未满足默认开启条件，不能据此否定它在其他场景的排序价值；当前保留代码且默认关闭，作为消融对照组。

以上是当时的候选路径，分数混合和扩池均未实施。若启动新实验，一次只改变一个因素，保留独立验收集；不得依据已查看过的 Test 反复选择参数。当前新增能力是 Q32 的重排粒度和聚合开关。

### Q32：当前 Child/Parent 重排和 mean/max 聚合如何使用？

`RetrievalConfig` 已支持 `rerank_level=child/parent` 与 `child_score_aggregation=mean/max`。默认仍为重排关闭、child、mean；开启 parent 重排要求 `rerank_backend=cross-encoder`。

两种重排粒度先取同一批前 N 个 Child。Child 模式逐子块评分、过滤，再按 Parent 聚合；Parent 模式先取回父块正文并去重，由 CE 直接评分，成功时不使用 Child 聚合分作为最终分。关闭重排或重排失败时，以 Child 融合分过滤后按所选 mean/max 聚合；失败回退使用完整融合池。

`top_k` 在父块聚合后截取；`rerank_top_n=None` 使用 `candidate_top_k`，不是整个双路融合池。完整参数表见 [评测速查](评测.md#31-检索漏斗参数retrievalconfig)，实现见 `backend/src/retrieval/hybrid_router.py`。

独立 T2 冻结候选池实验比较了 Child-mean、Child-max 与 Parent CE，案例和结果见 [重排案例](../interview/child-parent-rerank-cases.md)。这些历史缓存实验及 Router 回放不等于重新执行当前解析、嵌入、实时检索与回答链路，不据此替换默认配置。

---

## 7. 算法层工作计划（统一指向 README）

工作包状态、P0/P1/P2 待办与"明确暂不做"清单统一维护在 [`README.md`](README.md)：§4 能力表与近期目标、§7 实施路线、§8 明确暂不做。本文不再另行维护计划清单，避免双份清单漂移（2026-09-19 文档收敛时移除原 §7.1-7.4）；评测的执行契约、真实 run 记录与参数速查见 [`评测.md`](评测.md)。本文只保留问答本身——回答"为什么这样设计、怎么验证"。

## 8. 下一轮讨论入口

后续每次讨论只需要在本文继续补三类内容：

1. **问题**：某个术语、设计或优先级为什么这样定；
2. **决定**：当前项目做什么、不做什么，以及原因；
3. **执行项**：文件、测试、指标、命令和验收标准。

当决定稳定后，将状态和最终路线回写 `docs/plan/README.md`；本文保留解释与决策背景。
