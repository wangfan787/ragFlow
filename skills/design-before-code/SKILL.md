---
name: design-before-code
description: RAGFlow-source-first architecture review and approval gate for this repository. Use before changing the production RAG design or code in parsing, chunking, metadata, indexing, embedding, storage, retrieval, reranking, citations, ingestion, or dataset benchmarks. Require a module-by-module comparison of the current production implementation with local ragflow-main source before proposing changes, and require explicit user approval before implementation.
---

# RAGFlow Source-First Design Review

Prioritize the production RAG architecture. Treat dataset and benchmark scripts only after the production design is settled.

## Mandatory order

1. Read the current production call chain and relevant consumers.
2. Establish the RAGFlow source authority before comparison:
   - record the local version/commit when available;
   - inspect runtime switches, default values, factories, and call sites;
   - identify the default production path, optional paths, and legacy/fallback paths;
   - use the default production path as primary evidence and label legacy code only as comparison evidence.
3. Read the corresponding implementation under `ragflow-main`; cite exact files and lines.
4. For every affected module, write this table before proposing code:

   | Current implementation | RAGFlow authoritative implementation | Problem each solves | Difference/risk | Proposed modification | Adopt, adapt, or reject |

5. State where RAGFlow has no equivalent and where RAGFlow itself has limitations. Never treat it as automatically correct.
6. Only after the comparison, propose changes to the production project, organized by the repository's actual modules and files.
7. Discuss dataset/T2Retrieval adapters after the production architecture, showing how they reuse production code.
8. Put tests and migration after the design, not before it.
9. Stop for explicit user approval before editing source, tests, configuration, schemas, dataset scripts, or artifacts.

Approval must name the visible design or concrete choices. “Directly start” does not approve undisclosed decisions. User-requested design documents may be edited before approval and must be marked as drafts.

Approval is decision-scoped. Implement only the modules and choices the user explicitly approved. If an approved module depends on an unresolved design choice, stop at that dependency instead of silently choosing it.

## Required reasoning

- Map `document → parsed block → chunk/mother/child → embedding text → vector record → retrieved evidence → actual QA prompt window → citation`.
- Distinguish token count, model context, vector dimension, chunk count, document count, and score aggregation.
- Distinguish an answer model's total context limit from its completion `max_tokens`; derive evidence budget only after subtracting completion reserve, fixed prompt overhead, and safety margin using an explicit token counter.
- Identify every place content can be lost, duplicated, replaced, truncated, or made unavailable.
- Require content conservation across parsing and chunking: apart from explicitly documented normalization, every source span must be represented by one or more retrievable children or by a deliberately excluded block with a stated reason.
- Require every final embedding input to fit the selected model's measured input limit. Model-adapter truncation is only a last safety guard with observable warning/error semantics, never the normal long-document algorithm.
- Separate retrieval features from provenance fields.
- Trace provenance field-by-field through parser offsets, child spans, stored payload, retrieval models, the exact evidence sent to the answer model, and citations. Do not claim a field is supported merely because it exists in an intermediate nested object.
- Separate raw-source coordinates from normalized block/parent coordinates. Parent-window anchoring requires child offsets inside the normalized parent text; source citation requires a mapping back to raw input with declared accuracy.
- Inspect QA context assembly for top-k and length limits. Parent expansion must anchor the prompt window around the matched child; taking only the parent prefix is not a valid small-to-big implementation.
- Verify whether Parent and Child both enter the candidate pool and how family scores are aggregated.
- When a family score aggregates multiple child hits, preserve the contributing children as a list with scores/spans; do not collapse evidence to one child unless that loss is an explicitly approved design choice.
- Check whether benchmark code exercises the production parser/chunker/text builder or duplicates them.
- For benchmark proposals, specify stable chunk IDs, chunk-to-document mapping, qrels mapping, document-score aggregation, artifact schema/version, vector shape/dtype, and resume/checkpoint semantics before implementation.
- For storage/schema changes, state how old records behave under new mandatory filters and compare rebuild, backfill, and read-compatibility. Never introduce a filter that silently hides all legacy records.
- Explain why each RAGFlow mechanism is adopted, simplified, or rejected for this personal project.

## Forbidden responses

- Do not start with a self-invented repair plan and add RAGFlow references afterward.
- Do not focus on T2/test scripts while the production architecture is unresolved.
- Do not describe only high-level RAG concepts; connect claims to both codebases.
- Do not call model-limit truncation a long-document strategy.
- Do not preserve old outputs or compatibility unless the user chooses that priority.
- Do not claim provenance requires embedding all metadata; inspect stored payload and citation flow.
- Do not copy RAGFlow enterprise machinery that does not solve the current project's problem.

## Deliverable structure

1. Scope and production call chains.
2. RAGFlow source authority: version, runtime switch, default path, legacy paths.
3. Executive current-vs-RAGFlow comparison.
4. Per-module source comparison in production order.
5. Decisions: retain, modify, remove, or defer.
6. Proposed production target architecture organized by actual directories/files.
7. Only then: dataset/T2 adaptation and benchmark artifact contract.
8. Implementation phases, tests, migration, and acceptance criteria.
9. Explicit unresolved decisions requiring user approval.
