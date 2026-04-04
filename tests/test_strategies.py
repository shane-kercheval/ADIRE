"""Tests for re-embedding strategies."""

from adire.chunking import Chunk, greedy_chunk, split_paragraphs
from adire.strategies import (
    UpdateResult,
    adire_rechunk,
    adire_wide_window_rechunk,
    chunk_hash_rechunk,
    naive_rechunk,
    paragraph_reuse_rechunk,
)


def _make_doc(*paragraphs: str) -> str:
    """Join paragraph texts with double-newline."""
    return "\n\n".join(paragraphs)


def _initial_chunks(text: str, max_tokens: int = 512) -> list[Chunk]:
    """Split and chunk a document (simulates the first-time embedding)."""
    return greedy_chunk(split_paragraphs(text), max_tokens)


def _reconstruct(result: UpdateResult) -> str:
    """Reconstruct document text from result chunks."""
    return "\n\n".join(c.text for c in result.chunks)


# Design doc walkthrough: 6 paras -> 3 chunks at 512 tokens.
# Para A: 180 tokens, B: 200, C: 250, D: 190, E: 300, F: 200
# Chunk 0: [A+B]=380, Chunk 1: [C+D]=440, Chunk 2: [E+F]=500
DOC_A = "a" * 720   # 180 tokens
DOC_B = "b" * 800   # 200 tokens
DOC_C = "c" * 1000  # 250 tokens
DOC_D = "d" * 760   # 190 tokens
DOC_E = "e" * 1200  # 300 tokens
DOC_F = "f" * 800   # 200 tokens

WALKTHROUGH_DOC = _make_doc(DOC_A, DOC_B, DOC_C, DOC_D, DOC_E, DOC_F)
WALKTHROUGH_CHUNKS = _initial_chunks(WALKTHROUGH_DOC)


def test_walkthrough_setup():
    assert len(WALKTHROUGH_CHUNKS) == 3
    assert WALKTHROUGH_CHUNKS[0].token_count == 380
    assert WALKTHROUGH_CHUNKS[1].token_count == 440
    assert WALKTHROUGH_CHUNKS[2].token_count == 500
    assert len(WALKTHROUGH_CHUNKS[0].paragraph_hashes) == 2
    assert len(WALKTHROUGH_CHUNKS[1].paragraph_hashes) == 2
    assert len(WALKTHROUGH_CHUNKS[2].paragraph_hashes) == 2


# ---------------------------------------------------------------------------
# Naive strategy
# ---------------------------------------------------------------------------

class TestNaive:
    def test_always_reembeds_everything(self):
        result = naive_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 0
        assert result.preservation_rate == 0.0
        assert result.chunks_reembedded == 3
        assert result.reembedding_rate == 1.0

    def test_correct_chunk_count_and_tokens(self):
        result = naive_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert len(result.chunks) == 3
        total_tokens = sum(c.token_count for c in result.chunks)
        assert result.tokens_reembedded == total_tokens

    def test_reconstructs_document(self):
        result = naive_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert _reconstruct(result) == WALKTHROUGH_DOC


# ---------------------------------------------------------------------------
# Paragraph-level reuse strategy
# ---------------------------------------------------------------------------

class TestParagraphReuse:
    def test_unchanged_preserves_all(self):
        result = paragraph_reuse_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 6  # 6 paragraphs
        assert result.chunks_reembedded == 0
        assert result.preservation_rate == 1.0

    def test_typo_fix_one_paragraph(self):
        edited = _make_doc(DOC_A, DOC_B + "x", DOC_C, DOC_D, DOC_E, DOC_F)
        result = paragraph_reuse_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reembedded == 1
        assert result.chunks_reused == 5

    def test_paragraph_insertion(self):
        new_para = "x" * 400  # 100 tokens
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, DOC_D, DOC_E, DOC_F)
        result = paragraph_reuse_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reembedded == 1  # only the new paragraph
        assert result.chunks_reused == 6

    def test_paragraph_deletion(self):
        edited = _make_doc(DOC_A, DOC_C, DOC_D, DOC_E, DOC_F)
        result = paragraph_reuse_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reembedded == 0  # everything remaining is reused
        assert result.chunks_reused == 5

    def test_each_chunk_is_one_paragraph(self):
        result = paragraph_reuse_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert len(result.chunks) == 6
        for chunk in result.chunks:
            assert len(chunk.paragraph_hashes) == 1

    def test_duplicate_paragraphs(self):
        dup = "z" * 400
        doc = _make_doc(dup, DOC_A, dup)
        initial = _initial_chunks(doc)
        edited = _make_doc(dup + "x", DOC_A, dup)
        result = paragraph_reuse_rechunk(edited, initial)
        assert result.chunks_reembedded == 1
        assert result.chunks_reused == 2

    def test_reconstructs_document(self):
        result = paragraph_reuse_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert _reconstruct(result) == WALKTHROUGH_DOC


# ---------------------------------------------------------------------------
# Chunk-hash match strategy
# ---------------------------------------------------------------------------

class TestChunkHash:
    def test_unchanged_preserves_all(self):
        result = chunk_hash_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 3
        assert result.preservation_rate == 1.0

    def test_edit_at_bottom_preserves_earlier(self):
        edited = _make_doc(DOC_A, DOC_B, DOC_C, DOC_D, DOC_E, DOC_F + "x")
        result = chunk_hash_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused >= 2

    def test_edit_at_top_cascade(self):
        edited = _make_doc(DOC_A + "x", DOC_B, DOC_C, DOC_D, DOC_E, DOC_F)
        result = chunk_hash_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reembedded >= 1

    def test_insertion_in_middle_cascade(self):
        new_para = "x" * 400
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, DOC_D, DOC_E, DOC_F)
        result = chunk_hash_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused <= 2

    def test_uses_greedy_combining(self):
        result = chunk_hash_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert len(result.chunks) == 3

    def test_reconstructs_document(self):
        result = chunk_hash_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert _reconstruct(result) == WALKTHROUGH_DOC


# ---------------------------------------------------------------------------
# ADIRE — walkthrough example from design doc
# ---------------------------------------------------------------------------

class TestAdireWalkthrough:
    def test_design_doc_example(self):
        """6 paragraphs, insert after B, edit D.

        Expected: chunks 0 and 2 reused, dirty region re-chunked into 2 new chunks.
        Final: 4 chunks total, 2 reused, 2 re-embedded.
        """
        new_para = "x" * 600  # 150 tokens
        doc_d_edited = "d" * 760 + "EDITED"
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, doc_d_edited, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)

        assert result.chunks_reused == 2
        assert result.chunks_reembedded == 2
        assert len(result.chunks) == 4

        # Chunk 0 reused: [A + B]
        assert result.chunks[0].token_count == 380
        assert len(result.chunks[0].paragraph_hashes) == 2

        # Chunk 3 (old chunk 2, reindexed): [E + F] reused
        assert result.chunks[3].token_count == 500
        assert len(result.chunks[3].paragraph_hashes) == 2

        assert _reconstruct(result) == edited

    def test_fragment_metrics(self):
        new_para = "x" * 600
        doc_d_edited = "d" * 760 + "EDITED"
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, doc_d_edited, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.fragment_count >= 0
        assert 0.0 <= result.fragment_ratio <= 1.0


# ---------------------------------------------------------------------------
# ADIRE — edit types
# ---------------------------------------------------------------------------

class TestAdireEditTypes:
    def test_typo_fix(self):
        edited = _make_doc(DOC_A, DOC_B, DOC_C + "x", DOC_D, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 2
        assert _reconstruct(result) == edited

    def test_paragraph_insertion(self):
        new_para = "x" * 400
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, DOC_D, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused >= 2
        assert _reconstruct(result) == edited

    def test_paragraph_deletion(self):
        edited = _make_doc(DOC_A, DOC_B, DOC_D, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused >= 1
        assert _reconstruct(result) == edited

    def test_section_rewrite(self):
        new_c = "c" * 900
        new_d = "d" * 700
        edited = _make_doc(DOC_A, DOC_B, new_c, new_d, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 2
        assert _reconstruct(result) == edited

    def test_append(self):
        new_para = "x" * 400
        edited = WALKTHROUGH_DOC + "\n\n" + new_para
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 3
        assert _reconstruct(result) == edited

    def test_no_change_fast_path(self):
        result = adire_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 3
        assert result.chunks_reembedded == 0
        assert result.preservation_rate == 1.0

    def test_prepend(self):
        new_para = "x" * 400
        edited = new_para + "\n\n" + WALKTHROUGH_DOC
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused >= 2
        assert _reconstruct(result) == edited

    def test_all_paragraphs_changed(self):
        edited = _make_doc("x" * 720, "y" * 800, "z" * 1000, "w" * 760, "v" * 1200, "u" * 800)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 0
        assert _reconstruct(result) == edited


# ---------------------------------------------------------------------------
# ADIRE — edge cases
# ---------------------------------------------------------------------------

class TestAdireEdgeCases:
    def test_no_old_chunks(self):
        result = adire_rechunk(WALKTHROUGH_DOC, [])
        assert result.chunks_reused == 0
        assert result.chunks_reembedded == 3
        assert _reconstruct(result) == WALKTHROUGH_DOC

    def test_single_paragraph_document(self):
        doc = "a" * 400
        chunks = _initial_chunks(doc)
        assert len(chunks) == 1
        edited = "a" * 400 + "x"
        result = adire_rechunk(edited, chunks)
        assert result.chunks_reembedded == 1
        assert _reconstruct(result) == edited

    def test_duplicate_paragraphs_correct_chunk_dirty(self):
        dup = "z" * 400
        doc = _make_doc(dup, DOC_A, DOC_B, dup, DOC_C)
        chunks = _initial_chunks(doc)
        edited = _make_doc(dup, DOC_A, DOC_B, dup + "x", DOC_C)
        result = adire_rechunk(edited, chunks)
        assert result.chunks_reused >= 1
        assert _reconstruct(result) == edited

    def test_sequential_indices(self):
        new_para = "x" * 400
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, DOC_D, DOC_E, DOC_F)
        result = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    def test_does_not_mutate_input_chunks(self):
        chunks = _initial_chunks(WALKTHROUGH_DOC)
        original_indices = [c.index for c in chunks]
        original_hashes = [list(c.paragraph_hashes) for c in chunks]
        edited = _make_doc(DOC_A, DOC_B, "x" * 400, DOC_C, DOC_D, DOC_E, DOC_F)
        adire_rechunk(edited, chunks)
        assert [c.index for c in chunks] == original_indices
        assert [c.paragraph_hashes for c in chunks] == original_hashes

    def test_stale_chunk_indices_do_not_affect_correctness(self):
        # Simulate chunks loaded from storage with non-sequential indices
        chunks = _initial_chunks(WALKTHROUGH_DOC)
        chunks[0].index = 99
        chunks[1].index = 5
        chunks[2].index = 42
        edited = _make_doc(DOC_A, DOC_B, DOC_C + "x", DOC_D, DOC_E, DOC_F)
        result = adire_rechunk(edited, chunks)
        assert result.chunks_reused == 2
        assert _reconstruct(result) == edited
        # Output indices are sequential regardless of input
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    def test_old_doc_hash_parameter_skips_reconstruction(self):
        from adire.chunking import document_hash
        doc_hash = document_hash(WALKTHROUGH_DOC)
        result = adire_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS, old_doc_hash=doc_hash)
        assert result.chunks_reused == 3
        assert result.chunks_reembedded == 0


# ---------------------------------------------------------------------------
# ADIRE Wide Window
# ---------------------------------------------------------------------------

class TestAdireWideWindow:
    def test_expands_dirty_region(self):
        edited = _make_doc(DOC_A, DOC_B, DOC_C + "x", DOC_D, DOC_E, DOC_F)
        standard = adire_rechunk(edited, WALKTHROUGH_CHUNKS)
        wide = adire_wide_window_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert wide.chunks_reembedded >= standard.chunks_reembedded
        assert _reconstruct(wide) == edited

    def test_no_change_fast_path(self):
        result = adire_wide_window_rechunk(WALKTHROUGH_DOC, WALKTHROUGH_CHUNKS)
        assert result.chunks_reused == 3
        assert result.chunks_reembedded == 0

    def test_reconstructs_document(self):
        edited = _make_doc(DOC_A, DOC_B + "x", DOC_C, DOC_D, DOC_E, DOC_F)
        result = adire_wide_window_rechunk(edited, WALKTHROUGH_CHUNKS)
        assert _reconstruct(result) == edited


# ---------------------------------------------------------------------------
# Cross-strategy consistency
# ---------------------------------------------------------------------------

class TestCrossStrategy:
    def test_all_strategies_reconstruct_same_document(self):
        new_para = "x" * 400
        edited = _make_doc(DOC_A, DOC_B, new_para, DOC_C, DOC_D, DOC_E, DOC_F)

        results = [
            naive_rechunk(edited, WALKTHROUGH_CHUNKS),
            paragraph_reuse_rechunk(edited, WALKTHROUGH_CHUNKS),
            chunk_hash_rechunk(edited, WALKTHROUGH_CHUNKS),
            adire_rechunk(edited, WALKTHROUGH_CHUNKS),
            adire_wide_window_rechunk(edited, WALKTHROUGH_CHUNKS),
        ]

        for result in results:
            assert _reconstruct(result) == edited

    def test_metrics_are_valid_ranges(self):
        edited = _make_doc(DOC_A, DOC_B + "x", DOC_C, DOC_D, DOC_E, DOC_F)
        for strategy in [naive_rechunk, paragraph_reuse_rechunk, chunk_hash_rechunk,
                         adire_rechunk, adire_wide_window_rechunk]:
            result = strategy(edited, WALKTHROUGH_CHUNKS)
            assert 0.0 <= result.preservation_rate <= 1.0
            assert 0.0 <= result.reembedding_rate <= 1.0
            assert 0.0 <= result.token_savings_rate <= 1.0
            assert result.chunks_reused + result.chunks_reembedded == len(result.chunks)
            assert result.algorithm_time_ms >= 0.0
            assert 0.0 <= result.fragment_ratio <= 1.0
            assert 0.0 <= result.oversized_ratio <= 1.0
