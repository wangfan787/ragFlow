from __future__ import annotations

from dataclasses import replace

from backend.src.chunking.token_counter import SimpleTokenCounter
from backend.src.retrieval.models import RetrievedChunk


class ContextWindowBuilder:
    """Create token-bounded Parent windows anchored on matched Children."""

    def __init__(self, counter: SimpleTokenCounter | None = None) -> None:
        self.counter = counter or SimpleTokenCounter()

    def _largest_radius(self, text: str, start: int, end: int, budget: int) -> tuple[int, int]:
        if self.counter.count(text[start:end]) > budget:
            # This indicates a broken Child contract. Do not silently trim the
            # matched evidence, because it would make provenance dishonest.
            raise ValueError("matched Child exceeds QA evidence window budget")
        low, high = 0, max(start, len(text) - end)
        best = (start, end)
        while low <= high:
            radius = (low + high) // 2
            left = max(0, start - radius)
            right = min(len(text), end + radius)
            if self.counter.count(text[left:right]) <= budget:
                best = (left, right)
                low = radius + 1
            else:
                high = radius - 1
        return best

    def anchored(self, chunk: RetrievedChunk, budget: int) -> RetrievedChunk:
        text = chunk.content
        base_start = int(chunk.prompt_span.get("parent_char_start", 0) or 0)
        base_end = int(
            chunk.prompt_span.get("parent_char_end", base_start + len(text))
            or (base_start + len(text))
        )
        matched = chunk.matched_children or []
        primary_id = chunk.primary_matched_child_id or chunk.matched_child_id
        primary = next(
            (item for item in matched if str(item.get("chunk_id")) == str(primary_id)),
            matched[0] if matched else None,
        )
        if (
            chunk.chunk_role == "parent"
            and primary is not None
            and (primary.get("parent_char_start") is None or primary.get("parent_char_end") is None)
        ):
            fallback = str(primary.get("snippet", ""))
            if self.counter.count(fallback) > budget:
                raise ValueError("matched Child fallback exceeds QA evidence window budget")
            return replace(
                chunk,
                content=fallback,
                chunk_role="child",
                chunk_id=str(primary.get("chunk_id") or chunk.chunk_id),
                matched_children=[primary],
                source_span=dict(primary.get("source_span") or {}),
                source_block_ids=list(primary.get("source_block_ids", []) or []),
                context_span={"fallback": "matched_child"},
                prompt_span={"fallback": "matched_child"},
            )
        if self.counter.count(text) <= budget:
            return replace(
                chunk,
                context_span={"parent_char_start": base_start, "parent_char_end": base_end},
                prompt_span={"parent_char_start": base_start, "parent_char_end": base_end},
            )
        if primary is None:
            # A missing Parent coordinate must fall back to the actual Child,
            # never to the Parent prefix.
            raise ValueError("expanded Parent is missing matched Child provenance")
        start = primary.get("parent_char_start")
        end = primary.get("parent_char_end")
        if start is None or end is None:
            raise ValueError("matched Child coordinates are missing")
        local_start = int(start) - base_start
        local_end = int(end) - base_start
        if local_start < 0 or local_end > len(text) or local_start >= local_end:
            raise ValueError("matched Child coordinates are outside the current Parent window")
        left, right = self._largest_radius(text, local_start, local_end, budget)
        absolute_left = base_start + left
        absolute_right = base_start + right
        covered = [
            item
            for item in matched
            if item.get("parent_char_start") is not None
            and item.get("parent_char_end") is not None
            and int(item["parent_char_start"]) < absolute_right
            and int(item["parent_char_end"]) > absolute_left
        ]
        return replace(
            chunk,
            content=text[left:right],
            matched_children=covered or [primary],
            context_span={"parent_char_start": absolute_left, "parent_char_end": absolute_right},
            prompt_span={"parent_char_start": absolute_left, "parent_char_end": absolute_right},
        )

    def candidates(
        self,
        chunks: list[RetrievedChunk],
        *,
        top_k: int,
        max_window_tokens: int,
    ) -> list[RetrievedChunk]:
        windows: list[RetrievedChunk] = []
        for chunk in chunks[:top_k]:
            if self.counter.count(chunk.content) <= max_window_tokens or not chunk.matched_children:
                windows.append(self.anchored(chunk, max_window_tokens))
                continue
            family_windows: list[RetrievedChunk] = []
            ordered_children = sorted(
                chunk.matched_children,
                key=lambda item: (
                    item.get("parent_char_start") is None,
                    int(item.get("parent_char_start") or 0),
                ),
            )
            for child in ordered_children:
                child_id = str(child.get("chunk_id") or "")
                focused = replace(
                    chunk,
                    primary_matched_child_id=child_id,
                    matched_child_id=child_id,
                )
                window = self.anchored(focused, max_window_tokens)
                if family_windows:
                    previous = family_windows[-1]
                    prev_start = previous.prompt_span.get("parent_char_start")
                    prev_end = previous.prompt_span.get("parent_char_end")
                    win_start = window.prompt_span.get("parent_char_start")
                    win_end = window.prompt_span.get("parent_char_end")
                    if None not in (prev_start, prev_end, win_start, win_end) and int(win_start) <= int(prev_end):
                        union_start = min(int(prev_start), int(win_start))
                        union_end = max(int(prev_end), int(win_end))
                        union_text = chunk.content[union_start:union_end]
                        if self.counter.count(union_text) <= max_window_tokens:
                            covered = [
                                item
                                for item in chunk.matched_children
                                if item.get("parent_char_start") is not None
                                and item.get("parent_char_end") is not None
                                and int(item["parent_char_start"]) < union_end
                                and int(item["parent_char_end"]) > union_start
                            ]
                            family_windows[-1] = replace(
                                previous,
                                content=union_text,
                                matched_children=covered,
                                context_span={"parent_char_start": union_start, "parent_char_end": union_end},
                                prompt_span={"parent_char_start": union_start, "parent_char_end": union_end},
                            )
                            continue
                family_windows.append(window)
            windows.extend(family_windows)
        return windows
