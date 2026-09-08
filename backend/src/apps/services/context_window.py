"""按命中子块截取父块窗口，所有坐标始终相对原父块。"""

from langchain_core.documents import Document

from backend.src.chunking.token_counter import SimpleTokenCounter


class ContextWindowBuilder:
    def __init__(self, counter: SimpleTokenCounter | None = None) -> None:
        self.counter = counter or SimpleTokenCounter()

    def _largest_radius(self, text: str, start: int, end: int, budget: int) -> tuple[int, int]:
        if self.counter.count(text[start:end]) > budget:
            raise ValueError("matched Child exceeds QA evidence window budget")
        low, high = 0, max(start, len(text) - end)
        best = (start, end)
        while low <= high:
            radius = (low + high) // 2
            left, right = max(0, start - radius), min(len(text), end + radius)
            if self.counter.count(text[left:right]) <= budget:
                best = (left, right)
                low = radius + 1
            else:
                high = radius - 1
        return best

    @staticmethod
    def _window(chunk: Document, text: str, **updates) -> Document:
        return Document(page_content=text, metadata={**chunk.metadata, **updates})

    def anchored(self, chunk: Document, budget: int) -> Document:
        text, metadata = chunk.page_content, chunk.metadata
        prompt_span = metadata.get("prompt_span") or {}
        base_start = int(prompt_span.get("parent_char_start", 0) or 0)
        base_end = int(prompt_span.get("parent_char_end", base_start + len(text)) or base_start + len(text))
        matched = metadata.get("matched_children") or []
        primary_id = metadata.get("primary_matched_child_id") or metadata.get("matched_child_id")
        primary = next(
            (item for item in matched if str(item.get("chunk_id")) == str(primary_id)),
            matched[0] if matched else None,
        )
        if (
            metadata.get("chunk_role", "parent") == "parent" and primary is not None
            and (primary.get("parent_char_start") is None or primary.get("parent_char_end") is None)
        ):
            fallback = str(primary.get("snippet", ""))
            if self.counter.count(fallback) > budget:
                raise ValueError("matched Child fallback exceeds QA evidence window budget")
            return self._window(
                chunk, fallback, chunk_role="child",
                chunk_id=str(primary.get("chunk_id") or metadata["chunk_id"]),
                matched_children=[primary],
                source_span=dict(primary.get("source_span") or {}),
                source_block_ids=list(primary.get("source_block_ids") or []),
                context_span={"fallback": "matched_child"}, prompt_span={"fallback": "matched_child"},
            )
        if self.counter.count(text) <= budget:
            span = {"parent_char_start": base_start, "parent_char_end": base_end}
            return self._window(chunk, text, context_span=span, prompt_span=span)
        if primary is None:
            raise ValueError("expanded Parent is missing matched Child provenance")
        start, end = primary.get("parent_char_start"), primary.get("parent_char_end")
        if start is None or end is None:
            raise ValueError("matched Child coordinates are missing")
        local_start, local_end = int(start) - base_start, int(end) - base_start
        if local_start < 0 or local_end > len(text) or local_start >= local_end:
            raise ValueError("matched Child coordinates are outside the current Parent window")
        left, right = self._largest_radius(text, local_start, local_end, budget)
        absolute_left, absolute_right = base_start + left, base_start + right
        covered = [
            item for item in matched
            if item.get("parent_char_start") is not None and item.get("parent_char_end") is not None
            and int(item["parent_char_start"]) < absolute_right and int(item["parent_char_end"]) > absolute_left
        ]
        span = {"parent_char_start": absolute_left, "parent_char_end": absolute_right}
        return self._window(chunk, text[left:right], matched_children=covered or [primary], context_span=span, prompt_span=span)

    def candidates(
        self, chunks: list[Document], *, top_k: int, max_window_tokens: int,
    ) -> list[Document]:
        windows = []
        for chunk in chunks[:top_k]:
            matched = chunk.metadata.get("matched_children") or []
            if self.counter.count(chunk.page_content) <= max_window_tokens or not matched:
                windows.append(self.anchored(chunk, max_window_tokens))
                continue
            family_windows = []
            ordered = sorted(matched, key=lambda item: (
                item.get("parent_char_start") is None, int(item.get("parent_char_start") or 0),
            ))
            for child in ordered:
                child_id = str(child.get("chunk_id") or "")
                focused = self._window(
                    chunk, chunk.page_content, primary_matched_child_id=child_id, matched_child_id=child_id,
                )
                window = self.anchored(focused, max_window_tokens)
                if family_windows:
                    previous = family_windows[-1]
                    prior_span, current_span = previous.metadata["prompt_span"], window.metadata["prompt_span"]
                    prev_start, prev_end = prior_span.get("parent_char_start"), prior_span.get("parent_char_end")
                    win_start, win_end = current_span.get("parent_char_start"), current_span.get("parent_char_end")
                    if None not in (prev_start, prev_end, win_start, win_end) and int(win_start) <= int(prev_end):
                        union_start, union_end = min(int(prev_start), int(win_start)), max(int(prev_end), int(win_end))
                        base_start = int((chunk.metadata.get("prompt_span") or {}).get("parent_char_start", 0) or 0)
                        union_text = chunk.page_content[union_start - base_start:union_end - base_start]
                        if self.counter.count(union_text) <= max_window_tokens:
                            covered = [
                                item for item in matched
                                if item.get("parent_char_start") is not None and item.get("parent_char_end") is not None
                                and int(item["parent_char_start"]) < union_end and int(item["parent_char_end"]) > union_start
                            ]
                            span = {"parent_char_start": union_start, "parent_char_end": union_end}
                            family_windows[-1] = self._window(previous, union_text, matched_children=covered, context_span=span, prompt_span=span)
                            continue
                family_windows.append(window)
            windows.extend(family_windows)
        return windows
