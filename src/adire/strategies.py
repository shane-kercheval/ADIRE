"""Re-embedding strategies: naive, paragraph-reuse, chunk-hash, ADIRE, ADIRE wide-window."""

from __future__ import annotations

import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from adire.chunking import (
    Chunk,
    Paragraph,
    document_hash,
    greedy_chunk,
    hash_text,
    split_paragraphs,
)


@dataclass
class UpdateResult:
    """Result of processing a document update."""

    chunks: list[Chunk]
    chunks_reembedded: int
    tokens_reembedded: int
    chunks_reused: int

    preservation_rate: float
    reembedding_rate: float
    token_savings_rate: float

    algorithm_time_ms: float

    fragment_count: int
    fragment_ratio: float
    oversized_count: int
    oversized_ratio: float


def _build_result(
    chunks: list[Chunk],
    chunks_reused: int,
    tokens_reused: int,
    max_tokens: int,
    start_time: float,
) -> UpdateResult:
    """Build an UpdateResult from final chunks and reuse counts."""
    total = len(chunks)
    total_tokens = sum(c.token_count for c in chunks)
    reembedded = total - chunks_reused
    tokens_reembedded = total_tokens - tokens_reused
    min_threshold = max_tokens * 0.25

    fragment_count = sum(1 for c in chunks if c.token_count < min_threshold)
    oversized_count = sum(1 for c in chunks if c.token_count > max_tokens)

    return UpdateResult(
        chunks=chunks,
        chunks_reembedded=reembedded,
        tokens_reembedded=tokens_reembedded,
        chunks_reused=chunks_reused,
        preservation_rate=chunks_reused / total if total else 0.0,
        reembedding_rate=reembedded / total if total else 0.0,
        token_savings_rate=1.0 - (tokens_reembedded / total_tokens) if total_tokens else 0.0,
        algorithm_time_ms=(time.perf_counter() - start_time) * 1000,
        fragment_count=fragment_count,
        fragment_ratio=fragment_count / total if total else 0.0,
        oversized_count=oversized_count,
        oversized_ratio=oversized_count / total if total else 0.0,
    )


# ---------------------------------------------------------------------------
# Strategy 1: Naive
# ---------------------------------------------------------------------------

def naive_rechunk(
    new_text: str,
    old_chunks: list[Chunk],  # noqa: ARG001
    max_tokens: int = 512,
) -> UpdateResult:
    """Re-chunk and re-embed everything. Baseline strategy."""
    start = time.perf_counter()
    paras = split_paragraphs(new_text)
    chunks = greedy_chunk(paras, max_tokens)
    return _build_result(chunks, chunks_reused=0, tokens_reused=0,
                         max_tokens=max_tokens, start_time=start)


# ---------------------------------------------------------------------------
# Strategy 2: Paragraph-Level Reuse
# ---------------------------------------------------------------------------

def paragraph_reuse_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
) -> UpdateResult:
    """Each paragraph is its own chunk. Reuse by hash set lookup."""
    start = time.perf_counter()
    old_hashes: set[str] = set()
    for chunk in old_chunks:
        old_hashes.update(chunk.paragraph_hashes)

    paras = split_paragraphs(new_text)
    chunks: list[Chunk] = []
    reused = 0
    tokens_reused = 0

    for i, para in enumerate(paras):
        chunk = Chunk(
            index=i,
            text=para.text,
            paragraph_hashes=[para.hash],
            token_count=para.token_count,
        )
        chunks.append(chunk)
        if para.hash in old_hashes:
            reused += 1
            tokens_reused += para.token_count

    return _build_result(chunks, reused, tokens_reused,
                         max_tokens=max_tokens, start_time=start)


# ---------------------------------------------------------------------------
# Strategy 3: Chunk-Hash Match
# ---------------------------------------------------------------------------

def chunk_hash_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
) -> UpdateResult:
    """Re-chunk from scratch, reuse chunks whose full text hash matches."""
    start = time.perf_counter()
    old_text_hashes: set[str] = {hash_text(c.text) for c in old_chunks}

    paras = split_paragraphs(new_text)
    chunks = greedy_chunk(paras, max_tokens)

    reused = 0
    tokens_reused = 0
    for chunk in chunks:
        if hash_text(chunk.text) in old_text_hashes:
            reused += 1
            tokens_reused += chunk.token_count

    return _build_result(chunks, reused, tokens_reused,
                         max_tokens=max_tokens, start_time=start)


# ---------------------------------------------------------------------------
# ADIRE internals (independently testable sub-functions)
# ---------------------------------------------------------------------------

def build_position_to_chunk(old_chunks: list[Chunk]) -> dict[int, int]:
    """Map each old paragraph position to the chunk's list position (not chunk.index)."""
    pos_to_chunk: dict[int, int] = {}
    pos = 0
    for list_idx, chunk in enumerate(old_chunks):
        for _ in chunk.paragraph_hashes:
            pos_to_chunk[pos] = list_idx
            pos += 1
    return pos_to_chunk


Opcode = tuple[str, int, int, int, int]


def find_dirty_chunks(
    opcodes: list[Opcode],
    pos_to_chunk: dict[int, int],
    total_old_positions: int,
) -> set[int]:
    """Identify chunk indices that contain changed, inserted, or deleted paragraphs.

    Assumes total_old_positions > 0. The caller must handle the empty-old-chunks
    case separately (there are no chunks to mark dirty).
    """  # noqa: D213
    dirty: set[int] = set()
    for tag, i1, i2, _j1, _j2 in opcodes:
        if tag == "equal":
            continue
        for old_pos in range(i1, i2):
            dirty.add(pos_to_chunk[old_pos])
        if tag == "insert" and i1 < total_old_positions:
            dirty.add(pos_to_chunk[i1])
    return dirty


def _expand_dirty_set(dirty: set[int], num_chunks: int) -> set[int]:
    """Expand dirty set by adding immediate neighbors (for wide-window variant)."""
    expanded: set[int] = set()
    for idx in dirty:
        if idx > 0:
            expanded.add(idx - 1)
        expanded.add(idx)
        if idx < num_chunks - 1:
            expanded.add(idx + 1)
    return expanded


def collect_new_paragraphs_for_dirty_region(
    opcodes: list[Opcode],
    region_start: int,
    region_end: int,
    new_paragraphs: list[Paragraph],
) -> list[Paragraph]:
    """Collect new-side paragraphs that correspond to a dirty old-side region.

    The dirty region spans old positions [region_start, region_end). We walk
    opcodes and collect new paragraphs for every opcode that overlaps this range.
    """  # noqa: D213
    result: list[Paragraph] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "insert":
            # Insertion at old position i1 (i1 == i2). Include if the insertion
            # point falls within [region_start, region_end). Inserts at region_end
            # are excluded — they belong to the next dirty region or are trailing.
            # This is safe because the dirty-region invariant guarantees that any
            # non-equal opcode overlapping a region is fully contained by it.
            if region_start <= i1 < region_end:
                result.extend(new_paragraphs[j1:j2])
            continue

        if tag == "delete":
            continue

        # For 'equal' and 'replace', the opcode covers old[i1:i2].
        if i2 <= region_start:
            continue
        if i1 >= region_end:
            continue

        if tag == "equal":
            clip_start = max(i1, region_start)
            clip_end = min(i2, region_end)
            offset = clip_start - i1
            length = clip_end - clip_start
            result.extend(new_paragraphs[j1 + offset: j1 + offset + length])
        elif tag == "replace":
            result.extend(new_paragraphs[j1:j2])

    return result


def _get_trailing_paragraphs(
    opcodes: list[Opcode],
    total_old_positions: int,
    new_paragraphs: list[Paragraph],
) -> list[Paragraph]:
    """Get new paragraphs appended after the last old paragraph."""
    trailing: list[Paragraph] = []
    for tag, i1, _i2, j1, j2 in opcodes:
        if tag == "insert" and i1 >= total_old_positions:
            trailing.extend(new_paragraphs[j1:j2])
    return trailing


def _build_prefix_positions(old_chunks: list[Chunk]) -> list[int]:
    """Build prefix sum of paragraph counts: prefix[i] = total paragraphs before chunk i."""
    prefix = [0]
    for chunk in old_chunks:
        prefix.append(prefix[-1] + len(chunk.paragraph_hashes))
    return prefix


# ---------------------------------------------------------------------------
# Strategy 4: ADIRE
# ---------------------------------------------------------------------------

def _adire_core(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int,
    expand_window: bool,
    old_doc_hash: str | None = None,
) -> UpdateResult:
    """Core ADIRE implementation shared by standard and wide-window variants."""
    start = time.perf_counter()

    # Fast path: no old chunks means this is a new document.
    if not old_chunks:
        paras = split_paragraphs(new_text)
        chunks = greedy_chunk(paras, max_tokens)
        return _build_result(chunks, chunks_reused=0, tokens_reused=0,
                             max_tokens=max_tokens, start_time=start)

    # Step 1: Document-level fast path.
    if old_doc_hash is None:
        old_doc_hash = document_hash("\n\n".join(c.text for c in old_chunks))
    if document_hash(new_text) == old_doc_hash:
        total_tokens = sum(c.token_count for c in old_chunks)
        reindexed = [
            Chunk(index=i, text=c.text, paragraph_hashes=list(c.paragraph_hashes),
                  token_count=c.token_count)
            for i, c in enumerate(old_chunks)
        ]
        return _build_result(reindexed, chunks_reused=len(old_chunks),
                             tokens_reused=total_tokens,
                             max_tokens=max_tokens, start_time=start)

    # Step 2: Split new text into paragraphs.
    new_paras = split_paragraphs(new_text)
    new_hashes = [p.hash for p in new_paras]

    # Step 3: Reconstruct old paragraph hash sequence.
    old_hashes: list[str] = []
    for chunk in old_chunks:
        old_hashes.extend(chunk.paragraph_hashes)

    # Step 4: Diff.
    opcodes: list[Opcode] = list(
        SequenceMatcher(None, old_hashes, new_hashes, autojunk=False).get_opcodes(),
    )

    # Step 5: Find dirty chunks.
    pos_to_chunk = build_position_to_chunk(old_chunks)
    dirty = find_dirty_chunks(opcodes, pos_to_chunk, len(old_hashes))

    if expand_window:
        dirty = _expand_dirty_set(dirty, len(old_chunks))

    # Step 6: Walk old chunks, preserve clean ones, re-chunk dirty regions.
    prefix = _build_prefix_positions(old_chunks)
    final_chunks: list[Chunk] = []
    reused_count = 0
    tokens_reused = 0
    i = 0

    while i < len(old_chunks):
        if i not in dirty:
            # Keep this chunk — copy to avoid mutating the caller's objects.
            final_chunks.append(
                Chunk(index=0, text=old_chunks[i].text,
                      paragraph_hashes=list(old_chunks[i].paragraph_hashes),
                      token_count=old_chunks[i].token_count),
            )
            reused_count += 1
            tokens_reused += old_chunks[i].token_count
            i += 1
        else:
            # Dirty region: collect consecutive dirty chunks.
            region_start_pos = prefix[i]
            while i < len(old_chunks) and i in dirty:
                i += 1
            region_end_pos = prefix[i]

            region_paras = collect_new_paragraphs_for_dirty_region(
                opcodes, region_start_pos, region_end_pos, new_paras,
            )
            rechunked = greedy_chunk(region_paras, max_tokens)
            final_chunks.extend(rechunked)

    # Step 7: Handle trailing insertions.
    trailing = _get_trailing_paragraphs(opcodes, len(old_hashes), new_paras)
    if trailing:
        rechunked = greedy_chunk(trailing, max_tokens)
        final_chunks.extend(rechunked)

    # Step 8: Reindex.
    for idx, chunk in enumerate(final_chunks):
        chunk.index = idx

    return _build_result(final_chunks, reused_count, tokens_reused,
                         max_tokens=max_tokens, start_time=start)


def adire_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
    old_doc_hash: str | None = None,
) -> UpdateResult:
    """ADIRE: diff paragraph hashes, re-chunk only dirty regions."""
    return _adire_core(new_text, old_chunks, max_tokens, expand_window=False,
                       old_doc_hash=old_doc_hash)


# ---------------------------------------------------------------------------
# Strategy 5: ADIRE Wide Window
# ---------------------------------------------------------------------------

def adire_wide_window_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
    old_doc_hash: str | None = None,
) -> UpdateResult:
    """ADIRE with expanded dirty regions (one neighbor on each side)."""
    return _adire_core(new_text, old_chunks, max_tokens, expand_window=True,
                       old_doc_hash=old_doc_hash)
