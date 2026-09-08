"""Strict parent/child chunk construction with content-conserving ranges."""

from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document

from backend.src.chunking.chunk_config import ChunkConfig
from backend.src.chunking.token_counter import SimpleTokenCounter

_PARENT_BOUNDARY_TYPES = {"heading", "frontmatter", "hr"}
_PREFERRED_BOUNDARY_RE = re.compile(r"[。！？!?；;\n]|\s")


@dataclass(frozen=True)
class TextFragment:
    text: str
    start: int
    end: int
    split_by: str
    continuation: bool = False


class OversizedSplitter:
    """Split every output under a hard token budget without dropping text."""

    def __init__(self, counter: SimpleTokenCounter | None = None) -> None:
        self.counter = counter or SimpleTokenCounter()

    def _largest_end(self, text: str, start: int, budget: int) -> int:
        low, high = start + 1, len(text)
        best = start
        while low <= high:
            mid = (low + high) // 2
            if self.counter.count(text[start:mid]) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best == start:
            raise ValueError("token budget cannot fit a single source character")
        return best

    def _preferred_end(self, text: str, start: int, hard_end: int, target: int) -> int:
        target_end = self._largest_end(text[:hard_end], start, target)
        minimum = start + max(1, (target_end - start) // 2)
        preferred = [m.end() for m in _PREFERRED_BOUNDARY_RE.finditer(text, minimum, hard_end)]
        if not preferred:
            return hard_end
        before_target = [end for end in preferred if end <= target_end]
        return before_target[-1] if before_target else preferred[0]

    def split(
        self,
        text: str,
        *,
        target_tokens: int,
        max_tokens: int,
        structural_type: str = "paragraph",
    ) -> list[TextFragment]:
        if target_tokens <= 0 or max_tokens <= 0 or target_tokens > max_tokens:
            raise ValueError("chunk token budgets must satisfy 0 < target <= max")
        if not text:
            return []
        fragments: list[TextFragment] = []
        cursor = 0
        while cursor < len(text):
            if self.counter.count(text[cursor:]) <= max_tokens:
                end = len(text)
            else:
                hard_end = self._largest_end(text, cursor, max_tokens)
                end = self._preferred_end(text, cursor, hard_end, target_tokens)
            if end <= cursor:
                raise RuntimeError("strict splitter made no progress")
            piece = text[cursor:end]
            if self.counter.count(piece) > max_tokens:
                raise RuntimeError("strict splitter emitted an oversized fragment")
            fragments.append(
                TextFragment(
                    text=piece,
                    start=cursor,
                    end=end,
                    split_by=structural_type if len(fragments) == 0 and end == len(text) else "token_budget",
                    continuation=cursor > 0 or end < len(text),
                )
            )
            cursor = end
        return fragments


class BlockMergeStrategy:
    """Build bounded parents, then bounded children anchored inside each parent."""

    def __init__(self) -> None:
        self._counter = SimpleTokenCounter()
        self._splitter = OversizedSplitter(self._counter)

    def _block_type(self, block) -> str:
        return str(block.metadata.get("block_type", "paragraph"))

    def _atomic_fragments(self, blocks: list[Document], config: ChunkConfig) -> list[dict]:
        atoms: list[dict] = []
        for block in blocks:
            original = block.page_content
            # Leading/trailing whitespace normalization is explicit. Offsets
            # retain where the normalized text came from in the block.
            leading = len(original) - len(original.lstrip())
            normalized = original.strip()
            if not normalized:
                continue
            block_type = self._block_type(block)
            pieces = self._splitter.split(
                normalized,
                target_tokens=config.parent_target_tokens,
                max_tokens=config.parent_max_tokens,
                structural_type=block_type,
            )
            for piece in pieces:
                atoms.append(
                    {
                        "text": piece.text,
                        "block": block,
                        "block_type": block_type,
                        "block_char_start": leading + piece.start,
                        "block_char_end": leading + piece.end,
                        "continuation": piece.continuation,
                        "split_by": piece.split_by,
                    }
                )
        return atoms

    def _make_parent(self, atoms: list[dict], boundary: str | None = None) -> dict:
        texts: list[str] = []
        segment_map: list[dict] = []
        cursor = 0
        for atom in atoms:
            if texts:
                texts.append("\n\n")
                cursor += 2
            start = cursor
            texts.append(atom["text"])
            cursor += len(atom["text"])
            segment_map.append(
                {
                    "parent_start": start,
                    "parent_end": cursor,
                    "block": atom["block"],
                    "block_char_start": atom["block_char_start"],
                    "block_char_end": atom["block_char_end"],
                }
            )
        trace = {"chunk_role": "parent"}
        if boundary:
            trace["parent_boundary"] = boundary
        if any(atom["continuation"] for atom in atoms):
            trace["continuation"] = True
        return {
            "text": "".join(texts),
            "source_blocks": [atom["block"] for atom in atoms],
            "segment_map": segment_map,
            "trace": trace,
            "chunk_role": "parent",
        }

    def _merge_into_parents(self, blocks: list[Document], config: ChunkConfig) -> list[dict]:
        atoms = self._atomic_fragments(blocks, config)
        parents: list[dict] = []
        buffer: list[dict] = []

        def flush(boundary: str | None = None) -> None:
            nonlocal buffer
            if buffer:
                parents.append(self._make_parent(buffer, boundary))
                buffer = []

        for atom in atoms:
            block_type = atom["block_type"]
            preserve = (block_type == "code" and config.preserve_code_block) or (
                block_type == "table" and config.preserve_table_block
            )
            if (
                (config.align_to_boundary and block_type in _PARENT_BOUNDARY_TYPES) or preserve
            ) and buffer:
                flush(block_type)

            candidate_atoms = [*buffer, atom]
            candidate = self._make_parent(candidate_atoms)["text"]
            if buffer and self._counter.count(candidate) > config.parent_target_tokens:
                flush()
                candidate_atoms = [atom]
                candidate = atom["text"]
            if self._counter.count(candidate) > config.parent_max_tokens:
                raise RuntimeError("parent construction violated hard token budget")
            buffer = candidate_atoms
            if preserve or self._counter.count(candidate) >= config.parent_target_tokens:
                flush()
        flush()
        return parents

    def _overlapping_blocks(self, parent: dict, start: int, end: int) -> list:
        blocks = []
        seen: set[int] = set()
        for segment in parent["segment_map"]:
            if segment["parent_end"] <= start or segment["parent_start"] >= end:
                continue
            block = segment["block"]
            if id(block) not in seen:
                seen.add(id(block))
                blocks.append(block)
        return blocks or list(parent["source_blocks"][:1])

    def _split_into_children(self, parent: dict, config: ChunkConfig) -> list[dict]:
        pieces = self._splitter.split(
            parent["text"],
            target_tokens=config.child_target_tokens,
            max_tokens=config.child_max_tokens,
            structural_type="paragraph",
        )
        children: list[dict] = []
        for piece in pieces:
            children.append(
                {
                    "text": piece.text,
                    "source_blocks": self._overlapping_blocks(parent, piece.start, piece.end),
                    "parent_char_start": piece.start,
                    "parent_char_end": piece.end,
                    "parent_segment_map": parent["segment_map"],
                    "trace": {
                        "chunk_role": "child",
                        "split_by": piece.split_by,
                        "continuation": piece.continuation,
                    },
                    "chunk_role": "child",
                }
            )
        # Exact normalized-parent conservation is a production invariant.
        if "".join(child["text"] for child in children) != parent["text"]:
            raise RuntimeError("child chunks do not conserve normalized parent content")
        return children

    def merge(self, blocks: list[Document], config: ChunkConfig) -> list[dict]:
        merged: list[dict] = []
        for parent in self._merge_into_parents(blocks, config):
            merged.append(parent)
            for child in self._split_into_children(parent, config):
                child["parent_ref"] = parent
                merged.append(child)
        return merged
