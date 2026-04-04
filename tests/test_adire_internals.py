"""Tests for ADIRE internal sub-functions."""

from adire.chunking import Chunk, Paragraph
from adire.strategies import (
    build_position_to_chunk,
    collect_new_paragraphs_for_dirty_region,
    find_dirty_chunks,
    _expand_dirty_set,
    _get_trailing_paragraphs,
)


def _chunk(index: int, num_paras: int, start_hash: int = 0) -> Chunk:
    """Create a Chunk with num_paras synthetic paragraph hashes."""
    hashes = [f"h{start_hash + j}" for j in range(num_paras)]
    return Chunk(index=index, text="", paragraph_hashes=hashes, token_count=100 * num_paras)


def _para(name: str) -> Paragraph:
    """Create a Paragraph with a synthetic hash."""
    return Paragraph(text=name, hash=f"hash_{name}", token_count=100)


# ---------------------------------------------------------------------------
# build_position_to_chunk
# ---------------------------------------------------------------------------

class TestBuildPositionToChunk:
    def test_three_chunks_varying_sizes(self):
        chunks = [_chunk(0, 2), _chunk(1, 3), _chunk(2, 1)]
        ptc = build_position_to_chunk(chunks)
        assert ptc == {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}

    def test_single_chunk(self):
        chunks = [_chunk(0, 4)]
        ptc = build_position_to_chunk(chunks)
        assert ptc == {0: 0, 1: 0, 2: 0, 3: 0}

    def test_empty_chunk_list(self):
        assert build_position_to_chunk([]) == {}


# ---------------------------------------------------------------------------
# find_dirty_chunks
# ---------------------------------------------------------------------------

class TestFindDirtyChunks:
    def _setup(self, chunk_sizes: list[int]):
        """Create chunks and position mapping from a list of paragraph counts."""
        chunks = []
        start = 0
        for i, size in enumerate(chunk_sizes):
            chunks.append(_chunk(i, size, start_hash=start))
            start += size
        ptc = build_position_to_chunk(chunks)
        total = sum(chunk_sizes)
        return chunks, ptc, total

    def test_replace_dirties_correct_chunk(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        # Replace old[2:3] (first para of chunk 1)
        opcodes = [("equal", 0, 2, 0, 2), ("replace", 2, 3, 2, 3), ("equal", 3, 6, 3, 6)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == {1}

    def test_delete_dirties_correct_chunk(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        # Delete old[5:6] (the single para in chunk 2)
        opcodes = [("equal", 0, 5, 0, 5), ("delete", 5, 6, 5, 5)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == {2}

    def test_insert_in_middle(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        # Insert at old position 2 (start of chunk 1)
        opcodes = [("equal", 0, 2, 0, 2), ("insert", 2, 2, 2, 3), ("equal", 2, 6, 3, 7)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == {1}

    def test_insert_at_position_0(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        opcodes = [("insert", 0, 0, 0, 1), ("equal", 0, 6, 1, 7)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == {0}

    def test_insert_past_end_no_dirty(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        # Insert at old position 6 (past the end — trailing insertion)
        opcodes = [("equal", 0, 6, 0, 6), ("insert", 6, 6, 6, 8)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == set()

    def test_change_spanning_chunk_boundary(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        # Replace old[1:3] (spans chunk 0 and chunk 1)
        opcodes = [("equal", 0, 1, 0, 1), ("replace", 1, 3, 1, 3), ("equal", 3, 6, 3, 6)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == {0, 1}

    def test_multiple_non_adjacent_changes(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        # Replace in chunk 0 and chunk 2
        opcodes = [
            ("replace", 0, 1, 0, 1),
            ("equal", 1, 5, 1, 5),
            ("replace", 5, 6, 5, 6),
        ]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == {0, 2}

    def test_all_equal_no_dirty(self):
        _chunks, ptc, total = self._setup([2, 3, 1])
        opcodes = [("equal", 0, 6, 0, 6)]
        dirty = find_dirty_chunks(opcodes, ptc, total)
        assert dirty == set()

    def test_duplicate_paragraphs_positional_correctness(self):
        # Two chunks, each with hash "dup": chunk 0 has [dup, a], chunk 1 has [dup, b]
        c0 = Chunk(index=0, text="", paragraph_hashes=["dup", "a"], token_count=200)
        c1 = Chunk(index=1, text="", paragraph_hashes=["dup", "b"], token_count=200)
        ptc = build_position_to_chunk([c0, c1])
        # Replace old[2:3] — position 2 is "dup" in chunk 1, not position 0's "dup"
        opcodes = [("equal", 0, 2, 0, 2), ("replace", 2, 3, 2, 3), ("equal", 3, 4, 3, 4)]
        dirty = find_dirty_chunks(opcodes, ptc, 4)
        assert dirty == {1}


# ---------------------------------------------------------------------------
# _expand_dirty_set
# ---------------------------------------------------------------------------

class TestExpandDirtySet:
    def test_expands_neighbors(self):
        assert _expand_dirty_set({2}, 5) == {1, 2, 3}

    def test_clamps_at_zero(self):
        assert _expand_dirty_set({0}, 5) == {0, 1}

    def test_clamps_at_end(self):
        assert _expand_dirty_set({4}, 5) == {3, 4}

    def test_overlapping_expansions_merge(self):
        assert _expand_dirty_set({1, 3}, 5) == {0, 1, 2, 3, 4}

    def test_empty_input(self):
        assert _expand_dirty_set(set(), 5) == set()


# ---------------------------------------------------------------------------
# collect_new_paragraphs_for_dirty_region
# ---------------------------------------------------------------------------

class TestCollectNewParagraphs:
    def test_equal_fully_inside_region(self):
        new_paras = [_para("A"), _para("B"), _para("C")]
        opcodes = [("equal", 0, 3, 0, 3)]
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 3, new_paras)
        assert [p.text for p in result] == ["A", "B", "C"]

    def test_equal_partially_overlapping(self):
        new_paras = [_para("A"), _para("B"), _para("C"), _para("D")]
        opcodes = [("equal", 0, 4, 0, 4)]
        # Region covers old[1:3] — should get B and C
        result = collect_new_paragraphs_for_dirty_region(opcodes, 1, 3, new_paras)
        assert [p.text for p in result] == ["B", "C"]

    def test_replace_collects_all_new_side(self):
        new_paras = [_para("X"), _para("Y"), _para("Z")]
        # 1 old replaced by 3 new
        opcodes = [("replace", 0, 1, 0, 3)]
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 1, new_paras)
        assert [p.text for p in result] == ["X", "Y", "Z"]

    def test_insert_inside_region(self):
        new_paras = [_para("A"), _para("NEW"), _para("B")]
        opcodes = [
            ("equal", 0, 1, 0, 1),
            ("insert", 1, 1, 1, 2),
            ("equal", 1, 2, 2, 3),
        ]
        # Region covers old[0:2]
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 2, new_paras)
        assert [p.text for p in result] == ["A", "NEW", "B"]

    def test_insert_at_region_start(self):
        new_paras = [_para("NEW"), _para("A"), _para("B")]
        opcodes = [
            ("insert", 0, 0, 0, 1),
            ("equal", 0, 2, 1, 3),
        ]
        # Region covers old[0:2]
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 2, new_paras)
        assert [p.text for p in result] == ["NEW", "A", "B"]

    def test_insert_outside_region_not_collected(self):
        new_paras = [_para("A"), _para("NEW"), _para("B"), _para("C")]
        opcodes = [
            ("equal", 0, 1, 0, 1),
            ("insert", 1, 1, 1, 2),
            ("equal", 1, 3, 2, 4),
        ]
        # Region covers old[2:3] only — insert is at old pos 1, outside
        result = collect_new_paragraphs_for_dirty_region(opcodes, 2, 3, new_paras)
        assert [p.text for p in result] == ["C"]

    def test_delete_contributes_nothing(self):
        new_paras = [_para("A"), _para("C")]
        opcodes = [
            ("equal", 0, 1, 0, 1),
            ("delete", 1, 2, 1, 1),
            ("equal", 2, 3, 1, 2),
        ]
        # Region covers old[0:3] (everything)
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 3, new_paras)
        assert [p.text for p in result] == ["A", "C"]

    def test_mixed_opcodes_correct_order(self):
        new_paras = [_para("A"), _para("NEW"), _para("X"), _para("Y"), _para("D")]
        opcodes = [
            ("equal", 0, 1, 0, 1),       # A=A
            ("insert", 1, 1, 1, 2),       # insert NEW
            ("replace", 1, 3, 2, 4),      # B,C -> X,Y
            ("equal", 3, 4, 4, 5),        # D=D
        ]
        # Dirty region: old[0:4]
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 4, new_paras)
        assert [p.text for p in result] == ["A", "NEW", "X", "Y", "D"]

    def test_replace_spanning_3_chunks_invariant(self):
        # 3 chunks of 1 paragraph each. Replace spans all 3 old positions.
        new_paras = [_para("X"), _para("Y")]
        opcodes = [("replace", 0, 3, 0, 2)]
        # All 3 chunks dirty -> one region [0, 3)
        result = collect_new_paragraphs_for_dirty_region(opcodes, 0, 3, new_paras)
        assert [p.text for p in result] == ["X", "Y"]

    def test_dirty_region_spanning_chunks_with_insertions(self):
        # Old: [A, B, C, D] in 2 chunks of 2. Edit: insert X between B and C.
        new_paras = [_para("A"), _para("B"), _para("X"), _para("C"), _para("D")]
        opcodes = [
            ("equal", 0, 2, 0, 2),
            ("insert", 2, 2, 2, 3),
            ("equal", 2, 4, 3, 5),
        ]
        # Insert at old pos 2 dirties chunk 1 (positions 2-3).
        # Region is old[2:4].
        result = collect_new_paragraphs_for_dirty_region(opcodes, 2, 4, new_paras)
        assert [p.text for p in result] == ["X", "C", "D"]


# ---------------------------------------------------------------------------
# _get_trailing_paragraphs
# ---------------------------------------------------------------------------

class TestGetTrailingParagraphs:
    def test_trailing_insert(self):
        new_paras = [_para("A"), _para("B"), _para("NEW")]
        opcodes = [("equal", 0, 2, 0, 2), ("insert", 2, 2, 2, 3)]
        result = _get_trailing_paragraphs(opcodes, 2, new_paras)
        assert [p.text for p in result] == ["NEW"]

    def test_no_trailing(self):
        new_paras = [_para("A"), _para("B")]
        opcodes = [("equal", 0, 2, 0, 2)]
        result = _get_trailing_paragraphs(opcodes, 2, new_paras)
        assert result == []

    def test_non_trailing_insert_not_included(self):
        new_paras = [_para("NEW"), _para("A"), _para("B")]
        opcodes = [("insert", 0, 0, 0, 1), ("equal", 0, 2, 1, 3)]
        result = _get_trailing_paragraphs(opcodes, 2, new_paras)
        assert result == []


# ---------------------------------------------------------------------------
# Dirty region boundary detection
# ---------------------------------------------------------------------------

class TestDirtyRegionBoundaries:
    """Test that the dirty chunk walk produces correct region boundaries.

    These test the logic that _adire_core uses to determine region_start_pos
    and region_end_pos for each dirty region.
    """

    def _get_dirty_regions(self, chunk_sizes: list[int], dirty: set[int]):
        """Simulate the dirty region walk to extract [start, end) position ranges."""
        chunks = []
        start = 0
        for i, size in enumerate(chunk_sizes):
            chunks.append(_chunk(i, size, start_hash=start))
            start += size

        regions = []
        i = 0
        while i < len(chunks):
            if chunks[i].index not in dirty:
                i += 1
            else:
                region_start = sum(len(chunks[k].paragraph_hashes) for k in range(i))
                while i < len(chunks) and chunks[i].index in dirty:
                    i += 1
                region_end = sum(len(chunks[k].paragraph_hashes) for k in range(i))
                regions.append((region_start, region_end))
        return regions

    def test_single_dirty_chunk(self):
        # 3 chunks: [2, 3, 1], chunk 1 dirty
        regions = self._get_dirty_regions([2, 3, 1], {1})
        assert regions == [(2, 5)]

    def test_two_consecutive_dirty_chunks(self):
        regions = self._get_dirty_regions([2, 3, 1], {1, 2})
        assert regions == [(2, 6)]

    def test_two_separated_dirty_chunks(self):
        regions = self._get_dirty_regions([2, 3, 1], {0, 2})
        assert regions == [(0, 2), (5, 6)]

    def test_dirty_at_start(self):
        regions = self._get_dirty_regions([2, 3, 1], {0})
        assert regions == [(0, 2)]

    def test_dirty_at_end(self):
        regions = self._get_dirty_regions([2, 3, 1], {2})
        assert regions == [(5, 6)]
