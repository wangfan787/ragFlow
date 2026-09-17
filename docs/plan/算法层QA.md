# 算法层 QA 与可执行工作计划

更新日期：2026-09-17。

本文专门解释 `docs/plan/README.md` 算法层里容易产生歧义的术语、优先级和验收口径，并把讨论结果沉淀成后续可执行清单。

本文不是第二份“项目事实总表”。项目状态仍以 `docs/plan/README.md` 为准；这里形成明确决策后，再把结论回写到 README，避免两份状态互相冲突。

## 0. 先给结论

| 问题 | 结论 |
|---|---|
| 统一解析引擎是不是已经构造完毕？ | 新实现已经基本写出，但当前更准确的状态是“迁移已编码，待环境验收”，还不能写“已完成”。 |
| 以前不是已经通过测试了吗？ | 历史测试通过能证明当时环境里的代码行为，不等于新环境可安装、首次启动可用、容器可部署，也不等于当前未提交迁移仍然通过。 |
| 图片为什么要分块？ | 不切图片像素。图片保存为资产；`image block` 是图片在统一 Document 契约中的逻辑块。通常一图一块，只有过长的 VLM 文本描述才按文本切分。 |
| VLM 描述是什么？ | 核心确实是把图片交给视觉模型，让它输出可检索的文字；但还需要资产保存、提示词约束、结构化输出、失败处理、索引和引用回传。 |
| 来源坐标为什么是 P0？ | P0 不是要求所有坐标都逐字精确，而是要求系统诚实标记精度，不能让错误坐标驱动错误高亮。最小回归集是 P0，大规模比例统计可以后置。 |
| 父子分块的 profile 是什么？ | 是一组命名的分块参数预设，不是用户画像，也不是模型。当前只有一套默认配置就够了，先保留 hash 和版本，出现明确失败再增加 profile。 |
| 三种 Embedding 输入消融是不是三个模型？ | 不是。固定当前同一个 Embedding 模型，只比较“正文 / 标题+正文 / 章节路径+正文”三种输入模板。个人项目可因成本暂缓，不需要购买或配置多个模型。 |
| 融合排序和 Rerank 为什么影响效果？ | 融合决定向量召回与 BM25 召回如何合并；Rerank 对合并后的少量候选做更精细的二次排序。前者决定候选池，后者主要改善头部顺序。 |
| RRF 和 Rerank 都必须 P0 实现吗？ | 不必。P0 应先测当前 Weighted Sum 和已有 Rerank 的真实基线；Weighted RRF 是低成本候选方案，只有当前融合确有问题时再升为实现任务。 |

---

## 1. 统一解析引擎：为什么仍然写“重构中”

### Q1：已经有 `UnstructuredParser` 了，为什么不算完成？

“代码已经写出来”和“工程能力已经验收”是两件事。

截至本文更新时，仓库证据是：

- `backend/src/parsing/unstructured_parser.py` 已经统一承接 Markdown、PDF、HTML、TXT。
- `parser_factory.py` 已切到统一实现。
- 旧的四类手写解析器处于删除状态，新解析器仍是未提交文件；这说明迁移尚未形成稳定版本边界。
- `backend/requirements.txt` 新增了较重的 `unstructured[md,pdf]`，并注明需要 NLTK 数据。
- 项目的正确开发环境是 Conda `agent`；此前直接使用默认 Python 得出的“依赖缺失”结论无效。
- 在 `agent` 环境中已确认 `pytest 8.4.2`、`unstructured`、`langchain_core 1.6.2`、`nltk 3.10.3` 均已安装。
- 本次在 `agent` 中运行解析、分块和 document pipeline 测试时，5 个测试文件仍在收集阶段报错：导入 Unstructured PDF 模块时，Numba 报 `cannot cache function 'projection_by_bboxes': no locator available`。因此当前没有得到 passed/failed 用例结果，依赖组合或导入方式仍需修复。

所以“△ 重构中”表达的是迁移和验收状态，不是在说设计还没写。更准确的文字应改成：

> **△ 迁移已编码，待环境验收**

只有下面四层都满足，才改成“√ 已完成”：

1. 代码验收：统一 parser 已接入主链，旧路径删除，没有双实现。
2. 行为验收：Markdown、PDF、HTML、TXT 回归样例全部通过。
3. 环境验收：全新环境按照文档一次安装成功，NLTK/系统依赖不靠开发者机器的隐式缓存。
4. 部署验收：实际启动方式或容器中首次解析成功，并记录耗时与失败原因。

### Q2：`完成依赖、冷启动和部署验证；保留格式能力测试；记录真实解析耗时` 分别是什么意思？

#### 依赖验证

确认 `requirements.txt` 能在目标 Python 版本安装，且 PDF 相关系统库、NLTK 数据、Unstructured extras 都齐全。重点不是“我电脑上能 import”，而是“干净环境也能装”。

#### 冷启动验证

冷启动指第一次启动时，本机没有已经下载好的 NLTK 数据、模型缓存和临时文件。要验证：

- 程序是否会临时联网下载；
- 无网络部署是否会直接失败；
- 首次解析耗时是否远高于后续解析；
- 缺依赖时错误信息是否能指出解决办法。

理想做法是把所需数据放进部署构建阶段，运行阶段不再偷偷下载。

#### 部署验证

使用项目真正的启动方式，而不是只在 IDE 或测试进程里调用 parser。至少跑一次：启动服务 → 上传样例 → 解析 → 分块 → 返回成功状态。

#### 格式能力测试

迁移到统一引擎不能只检查“有文本输出”，还要固定每种格式的重要能力：

- Markdown：标题层级、列表、代码块、表格、frontmatter、重复文本回定位；
- HTML：删除 script/style、实体解码、标题层级和保守坐标；
- TXT：严格 UTF-8、段落与 exact 坐标；
- PDF：页码、文本层 PDF 可读、扫描件明确不支持。

#### 真实解析耗时

不是在单元测试里 mock 一个时间，而是对真实小、中、大样例记录：文件大小、页数或字符数、block 数、首次耗时、热启动耗时、峰值内存。首版记录每类格式的 P50/P95 已经足够。

### Q3：以前的 79 个测试通过不算数吗？

算数，但它证明的范围有限。

| 证据 | 能证明 | 不能证明 |
|---|---|---|
| 单元/回归测试通过 | 固定输入下的代码行为符合断言 | 新机器能安装、真实外部服务能用 |
| 集成测试使用替身通过 | 各模块契约能连起来 | 真实 ES、真实模型、真实网络正常 |
| 真实联调通过 | 当时配置下的完整链路可用 | 以后不会回归、性能可接受 |
| 冷启动/部署验证 | 从零部署可复现 | 算法效果一定好 |
| Golden Set 指标 | 给定数据集上的检索或回答效果 | 部署一定稳定 |

因此正确表述是：

> 历史记录显示迁移阶段曾有 79 个测试通过；当前还需在固定目标环境重跑，并补齐冷启动、部署和性能证据。

---

## 2. 图片、VLM、image block 与“图片分块”

### Q4：解析范围不是 Markdown、PDF、HTML、图片吗？图片是不是唯一重要的缺口？

需要先区分“目标范围”和“当前实现”。

当前 `parser_factory.py` 只注册了 Markdown、PDF、HTML、TXT，没有 PNG/JPG 等图片类型。因此：

- 文本格式的解析主链已经存在，仍需要迁移验收；
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

因此原 README 把整件事都写成 P0 偏粗。更合理的验收是：P0 先覆盖 8–12 个高风险样例，保证不会把 line_only/unavailable 当 exact；完整统计随后补。

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

#### P1：视证据实现

1. 如果 Weighted Sum 在不同 Query 上明显受分数刻度影响，再实现 Weighted RRF。
2. 如果 RRF 在同集显著提升且没有不可接受的副作用，再替换默认融合。
3. 如果 Cross Encoder 的业务增益覆盖延迟和部署成本，再在生产默认开启；否则保留为可选能力。

换句话说，**P0 是建立可信选择依据，不是强制把所有候选算法都上线。**

---

## 7. 修订后的算法层工作计划

下面的顺序按“先恢复可验证性，再补图片闭环，再做效果实验”排列。

### P0-A：统一解析迁移验收

目标：把状态从“迁移已编码”推进为“目标环境已验证”。

任务：

- [ ] 固定 Python 版本和依赖安装方式。
- [ ] 将 NLTK 数据下载放进明确的 bootstrap/镜像构建步骤。
- [ ] 在干净环境安装 `backend/requirements.txt`。
- [ ] 修复 Unstructured PDF 导入时的 Numba `no locator available` 缓存错误。
- [ ] 运行 parsing、chunking 和 document pipeline 测试。
- [ ] 为 md/html/txt/pdf 各保留至少 2 个代表样例。
- [ ] 记录 collected/passed/failed/skipped、总耗时和失败日志。
- [ ] 记录每种格式冷/热解析耗时、输入规模和 block 数。
- [ ] 全部通过后提交统一 parser 与旧 parser 删除，README 改成“√ 已验证”。

建议验证命令（先按项目根目录 README 进入开发环境）：

```bash
PYTHONPATH=. python -m pytest \
  backend/tests/parsing \
  backend/tests/chunking \
  backend/tests/integration/test_document_pipeline.py -q
```

验收标准：

- 新环境不依赖开发机已有缓存；
- 四种文本格式测试通过；
- 文本层 PDF 能解析，扫描 PDF 明确报“不支持/无可解析内容”；
- 没有恢复旧 parser 或双轨兼容层；
- 产生一份带日期、Git commit、环境、耗时的验证记录。

### P0-B：Markdown 图片最小闭环

目标：Markdown 本地图片能够被保存、理解、检索和引用。

任务：

- [ ] 只定义首版范围：Markdown 相对路径本地 PNG/JPEG/WebP；暂不抓远程 URL。
- [ ] 在线 API 只接收 `document_id/asset_id` 等受控 ID，不接受客户端任意文件路径。
- [ ] assets 关联 document；读取时通过 `asset_id + current_user_id` 查询数据库并检查 document owner。
- [ ] 提取图片语法与原文 span；使用数据库保存的 storage_key 读取文件。
- [ ] 防止绝对路径和 `..` 逃逸出项目文件存储目录。
- [ ] SHA256 去重，生成 `asset_id`，原图保存到文档资产目录。
- [ ] 定义 VLM JSON 输出 schema 和 prompt version。
- [ ] 只有数据库归属检查通过后才能读取图片并调用 VLM。
- [ ] 生成 `block_type=image` 的 Document。
- [ ] chunker 将 image 视为原子块，并透传 `asset_id/document_id`。
- [ ] Embedding 只处理描述文字，不处理图片二进制。
- [ ] Citation 返回 `asset_id`；预览接口再次查询 owner 后返回原图。
- [ ] VLM 超时、拒绝、空描述时保留资产并标记失败，不伪造描述。
- [ ] 增加有权限、无权限、路径穿越和失效资源四类测试。

验收标准：

- 同一图片重复出现时资产不重复保存；
- 图片问题能召回 image Child；
- 有权限用户的引用能打开正确原图，无权限用户得到 403/404；
- 任何客户端输入都不能令服务读取授权 storage root 之外的文件；
- 删除文档时资产按明确规则清理；
- 无 Vision 配置时给出明确状态，不能静默生成假描述。

### P0-C：来源坐标可信度

目标：引用展示不会把不可靠坐标伪装成逐字精确。

任务：

- [ ] 建立 8–12 个高风险 span 样例。
- [ ] exact 必须验证 `raw[start:end]` 与期望文本一致。
- [ ] line_only 只允许行级展示。
- [ ] unavailable 不允许逐字高亮。
- [ ] 输出按格式的三档计数；完整 20–30 篇统计放 P1。
- [ ] image block 的 source span 指向 Markdown 图片语法，asset 指向原图。

验收标准：重复文本、Markdown 标记、HTML 实体、Child 坐标投影均不产生假 exact。

### P0-D：当前检索基线与 Rerank 决策

目标：在不更换 Embedding 模型的前提下，证明当前默认检索是否合理。

任务：

- [ ] 保持 `embedding-3 + child-body-v1`，不做多模型采购。
- [ ] 固定 Golden Set、索引指纹和检索参数。
- [ ] 保存 Vector-only baseline。
- [ ] 保存当前 Hybrid Weighted Sum baseline。
- [ ] 固定 `candidate_top_k/rerank_top_n/final_top_k`。
- [ ] 使用合适的中文/多语模型比较 Rerank off/on。
- [ ] 报告 Hit/Recall/MRR/nDCG、Evidence Recall、P50/P95 和候选数。
- [ ] 将已有 T2 历史结果整理成机器可读 JSON，再生成 Markdown 报告。

验收决策：

- Rerank 有稳定业务增益且延迟可接受：进入生产候选；
- 只在 T2 提升、业务集不提升：保持默认关闭；
- 当前 Weighted Sum 出现明显刻度或权重问题：新增 Weighted RRF 实验；
- 没有证据表明 Weighted Sum 有问题：RRF 留在 P2 条件任务，不为“算法完整”而实现。

### P1：P0 稳定后完善质量报告

- [ ] 将 span 集扩到 20–30 篇并形成比例趋势报告。

### P2：业务评测触发后再做

- [ ] 当 code/table 失败形成稳定类别时，增加一个针对性 chunk profile。
- [ ] 当 Weighted Sum 失败时，实现 Weighted RRF 并做同集 A/B。
- [ ] 当召回失败指向上下文不足且预算允许时，在小样本上比较三种 Embedding 输入模板。
- [ ] 再评估 HTML 图片、独立图片文件和 PDF 内嵌图片。

### 当前明确不做

- 不为了面试展示同时购买或配置多个 Embedding 模型。
- 不切割图片像素来适配文本 chunk。
- 不在没有失败数据时同时维护 default/code-heavy/table-heavy 多套索引。
- 不因为 RRF 名字更“高级”就默认替换 Weighted Sum。
- 不把历史 pytest 通过写成当前新环境、真实 ES 或真实模型已经联调。

---

## 8. 建议回写 `docs/plan/README.md` 的文字

等本文结论确认后，建议对 README 做以下精确调整：

1. “统一解析引擎：△ 重构中”改成“△ 迁移已编码，待环境验收”。
2. 图片链路的“进入分块”改成“后端先校验资源归属和读取权限，再生成原子 image block；描述文字进入父子块与索引，原图不切分，预览再次鉴权”。
3. 来源坐标拆成：P0 最小 span 回归和精度诚实；P1 扩大样本与完整比例报告。
4. 父子分块 profile 改成 `P2（条件触发）`，不预先建设三套配置。
5. Embedding 输入消融注明“固定同一模型，仅改变输入模板”，并改成预算/失败驱动任务。
6. 融合排序改成：P0 保存当前 Weighted Sum baseline；RRF 作为失败驱动的 P1 候选。
7. Rerank 改成：已有 T2 历史增益，P0 补当前业务 Hybrid 链路、延迟和可复现 report；是否默认开启由结果决定。

## 9. 下一轮讨论入口

后续每次讨论只需要在本文继续补三类内容：

1. **问题**：某个术语、设计或优先级为什么这样定；
2. **决定**：当前项目做什么、不做什么，以及原因；
3. **执行项**：文件、测试、指标、命令和验收标准。

当决定稳定后，将状态和最终路线回写 `docs/plan/README.md`；本文保留解释与决策背景。
