"""Tests for adire.document_generator module."""

from adire.chunking import greedy_chunk, split_paragraphs
from adire.document_generator import (
    BIMODAL,
    LARGE,
    LONG_PARAGRAPHS,
    MAX_SIZE,
    MEDIUM,
    MIXED,
    OVERSIZED_PARAGRAPHS,
    SHORT_PARAGRAPHS,
    SMALL,
    STRUCTURELESS_BLOB,
    TINY,
    generate_document,
)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestDocumentSize:
    """Generated document character count is within 10% of target_chars."""

    def _assert_within_10_pct(self, profile, target_chars):
        doc = generate_document(profile, target_chars, seed=42)
        low = target_chars * 0.9
        high = target_chars * 1.1
        assert low <= len(doc) <= high, (
            f"{profile.name} at {target_chars}: got {len(doc)} chars"
        )

    def test_short_paragraphs_sizes(self):
        for size in [TINY, SMALL, MEDIUM, LARGE, MAX_SIZE]:
            self._assert_within_10_pct(SHORT_PARAGRAPHS, size)

    def test_mixed_sizes(self):
        for size in [TINY, SMALL, MEDIUM, LARGE, MAX_SIZE]:
            self._assert_within_10_pct(MIXED, size)

    def test_long_paragraphs_sizes(self):
        for size in [TINY, SMALL, MEDIUM, LARGE, MAX_SIZE]:
            self._assert_within_10_pct(LONG_PARAGRAPHS, size)

    def test_oversized_paragraphs_sizes(self):
        for size in [TINY, SMALL, MEDIUM, LARGE, MAX_SIZE]:
            self._assert_within_10_pct(OVERSIZED_PARAGRAPHS, size)

    def test_structureless_blob_sizes(self):
        for size in [TINY, SMALL, MEDIUM, LARGE, MAX_SIZE]:
            doc = generate_document(STRUCTURELESS_BLOB, size, seed=42)
            assert abs(len(doc) - size) <= size * 0.1

    def test_bimodal_sizes(self):
        for size in [TINY, SMALL, MEDIUM, LARGE, MAX_SIZE]:
            self._assert_within_10_pct(BIMODAL, size)


class TestDocumentStructure:
    def test_short_paragraphs_count(self):
        doc = generate_document(SHORT_PARAGRAPHS, MEDIUM, seed=42)
        paras = split_paragraphs(doc)
        # 25K chars / ~150 chars per para -> ~125-200 paragraphs
        assert 100 <= len(paras) <= 250

    def test_oversized_paragraphs_exceed_token_budget(self):
        doc = generate_document(OVERSIZED_PARAGRAPHS, LARGE, seed=42)
        paras = split_paragraphs(doc)
        oversized = sum(1 for p in paras if p.token_count > 512)
        assert oversized / len(paras) > 0.5

    def test_structureless_blob_single_paragraph(self):
        doc = generate_document(STRUCTURELESS_BLOB, MEDIUM, seed=42)
        assert "\n\n" not in doc
        paras = split_paragraphs(doc)
        assert len(paras) == 1

    def test_bimodal_alternates_short_and_long(self):
        doc = generate_document(BIMODAL, MEDIUM, seed=42)
        paras = split_paragraphs(doc)
        assert len(paras) >= 10
        short_threshold = 120
        long_threshold = 400
        for i in range(0, min(len(paras) - 1, 10), 2):
            assert len(paras[i].text) < short_threshold, f"Para {i} should be short"
            assert len(paras[i + 1].text) > long_threshold, f"Para {i+1} should be long"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        doc_a = generate_document(MIXED, MEDIUM, seed=123)
        doc_b = generate_document(MIXED, MEDIUM, seed=123)
        assert doc_a == doc_b

    def test_different_seeds_different_output(self):
        doc_a = generate_document(MIXED, MEDIUM, seed=1)
        doc_b = generate_document(MIXED, MEDIUM, seed=2)
        assert doc_a != doc_b

    def test_determinism_across_profiles(self):
        for profile in [SHORT_PARAGRAPHS, LONG_PARAGRAPHS, STRUCTURELESS_BLOB, BIMODAL]:
            a = generate_document(profile, SMALL, seed=99)
            b = generate_document(profile, SMALL, seed=99)
            assert a == b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_small_target(self):
        doc = generate_document(MIXED, 100, seed=42)
        assert len(doc) > 0
        paras = split_paragraphs(doc)
        assert len(paras) >= 1

    def test_mean_larger_than_target(self):
        doc = generate_document(LONG_PARAGRAPHS, 200, seed=42)
        assert len(doc) > 0
        paras = split_paragraphs(doc)
        assert len(paras) >= 1

    def test_tiny_bimodal(self):
        doc = generate_document(BIMODAL, 100, seed=42)
        assert len(doc) > 0

    def test_tiny_structureless_blob(self):
        doc = generate_document(STRUCTURELESS_BLOB, 100, seed=42)
        assert len(doc) > 0
        assert "\n\n" not in doc


# ---------------------------------------------------------------------------
# Integration with chunking
# ---------------------------------------------------------------------------

class TestChunkingIntegration:
    def test_generated_documents_can_be_chunked(self):
        for profile in [SHORT_PARAGRAPHS, MIXED, LONG_PARAGRAPHS,
                        OVERSIZED_PARAGRAPHS, STRUCTURELESS_BLOB, BIMODAL]:
            doc = generate_document(profile, MEDIUM, seed=42)
            paras = split_paragraphs(doc)
            chunks = greedy_chunk(paras)
            assert len(chunks) >= 1
            # Reconstruct to verify no data loss
            reconstructed = "\n\n".join(c.text for c in chunks)
            assert reconstructed == doc

    def test_paragraph_counts_match_profile(self):
        doc = generate_document(MIXED, MEDIUM, seed=42)
        paras = split_paragraphs(doc)
        # 25K chars / ~400 chars per para -> ~50-80 paragraphs
        assert 30 <= len(paras) <= 120
