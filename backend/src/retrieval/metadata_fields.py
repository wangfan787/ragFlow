from __future__ import annotations

import re
from collections.abc import Iterable

_TEXT_LIST_SPLIT_RE = re.compile(r"[,，;；、\n\r\t]+")


def as_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = _TEXT_LIST_SPLIT_RE.split(value)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        values = value
    else:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def retrieval_metadata_fields(source: dict) -> dict:
    fields = "title_tks important_kwd important_tks question_kwd question_tks section_path".split()
    return {
        **{field: as_text_list(source.get(field, [])) for field in fields},
        "retrieval_metadata_trace": dict(source.get("retrieval_metadata_trace", {})),
        "page_no": source.get("page_no"),
        # 父子分块字段：让 chunk_role 进 payload，便于检索时按角色过滤或反查父子关系。
        "chunk_role": source.get("chunk_role", "parent"),
        "parent_id": source.get("parent_id"),
        "child_ids": list(source.get("child_ids", []) or []),
        "chunk_order": source.get("chunk_order"),
        "retrieval_eligible": bool(source.get("retrieval_eligible", False)),
        "source_span": dict(source.get("source_span") or {}),
        "source_block_ids": as_text_list(source.get("source_block_ids", [])),
        "parent_char_start": source.get("parent_char_start"),
        "parent_char_end": source.get("parent_char_end"),
        "chunk_profile_version": source.get("chunk_profile_version"),
        "chunk_profile_hash": source.get("chunk_profile_hash"),
    }


def retrieved_chunk_payload(chunk) -> dict:
    fields = (
        "chunk_id doc_id content score vector_score keyword_score rerank_score fused_score "
        "section_path page_no doc_name important_kwd question_kwd embedding_backend "
        "embedding_model embedding_dim chunk_role parent_id child_ids chunk_order matched_child_id "
        "retrieval_eligible source_span source_block_ids parent_char_start parent_char_end "
        "matched_children family_contributors primary_matched_child_id context_span prompt_span "
        "context_source_block_ids"
    ).split()
    return {field: getattr(chunk, field) for field in fields}


def dedupe_text_list(values, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in as_text_list(values):
        clean = re.sub(r"\s+", " ", value).strip(" .,;:，。；：")
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if limit is not None and len(out) >= limit:
            break
    return out
