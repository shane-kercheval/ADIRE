"""Tests for adire.chunking module."""

from adire.chunking import Paragraph, document_hash, greedy_chunk, split_paragraphs


def _make_para(text: str, token_count: int) -> Paragraph:
    """Create a Paragraph with a synthetic hash for testing greedy_chunk."""
    return Paragraph(text=text, hash=f"hash_{text}", token_count=token_count)


# ---------------------------------------------------------------------------
# split_paragraphs
# ---------------------------------------------------------------------------

class TestSplitParagraphs:
    def test_multiple_paragraphs(self):
        text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
        paras = split_paragraphs(text)
        assert len(paras) == 3
        assert paras[0].text == "Alpha paragraph."
        assert paras[1].text == "Beta paragraph."
        assert paras[2].text == "Gamma paragraph."

    def test_hashes_are_strings(self):
        paras = split_paragraphs("Hello world.\n\nGoodbye world.")
        for p in paras:
            assert isinstance(p.hash, str)
            assert len(p.hash) == 16

    def test_token_count_approximation(self):
        text = "a" * 100  # 100 chars -> 25 tokens
        paras = split_paragraphs(text)
        assert paras[0].token_count == 25

    def test_short_paragraph_has_minimum_one_token(self):
        paras = split_paragraphs("FAQ")
        assert paras[0].token_count == 1

    def test_custom_separator(self):
        text = "Line one.\nLine two.\nLine three."
        paras = split_paragraphs(text, separator="\n")
        assert len(paras) == 3
        assert paras[0].text == "Line one."

    def test_empty_text(self):
        assert split_paragraphs("") == []

    def test_whitespace_only_text(self):
        assert split_paragraphs("   \n\n   \n\n   ") == []

    def test_no_separators(self):
        text = "Single paragraph with no double newlines."
        paras = split_paragraphs(text)
        assert len(paras) == 1
        assert paras[0].text == text

    def test_consecutive_separators_no_empty_paragraphs(self):
        text = "First.\n\n\n\nSecond."
        paras = split_paragraphs(text)
        assert len(paras) == 2

    def test_whitespace_normalization_same_hash(self):
        text_a = "  Hello   world  \n\nFoo."
        text_b = "Hello world\n\nFoo."
        paras_a = split_paragraphs(text_a)
        paras_b = split_paragraphs(text_b)
        assert paras_a[0].hash == paras_b[0].hash

    def test_whitespace_normalization_preserves_original_text(self):
        text = "  Hello   world  \n\nFoo."
        paras = split_paragraphs(text)
        assert paras[0].text == "Hello   world"

    def test_duplicate_paragraphs_produce_same_hash(self):
        paras = split_paragraphs("TODO\n\nTODO\n\nTODO")
        assert len(paras) == 3
        assert paras[0].hash == paras[1].hash == paras[2].hash

    def test_determinism(self):
        text = "Paragraph one.\n\nParagraph two."
        paras_a = split_paragraphs(text)
        paras_b = split_paragraphs(text)
        assert [p.hash for p in paras_a] == [p.hash for p in paras_b]


# ---------------------------------------------------------------------------
# greedy_chunk
# ---------------------------------------------------------------------------

class TestGreedyChunk:
    def test_combines_small_paragraphs(self):
        paras = [_make_para(f"Para {i}", 100) for i in range(4)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].token_count == 400

    def test_starts_new_chunk_when_exceeding_budget(self):
        paras = [_make_para("A", 200), _make_para("B", 200), _make_para("C", 200)]
        chunks = greedy_chunk(paras, max_tokens=450)
        assert len(chunks) == 2
        assert chunks[0].token_count == 400
        assert chunks[1].token_count == 200

    def test_budget_boundary(self):
        paras = [_make_para("A", 256), _make_para("B", 256)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].token_count == 512

    def test_single_oversized_paragraph(self):
        paras = [_make_para("Big", 1000)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].token_count == 1000

    def test_oversized_then_normal(self):
        paras = [_make_para("Big", 1000), _make_para("Small", 100)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 2
        assert chunks[0].token_count == 1000
        assert chunks[1].token_count == 100

    def test_empty_list(self):
        assert greedy_chunk([], max_tokens=512) == []

    def test_indices_are_sequential(self):
        paras = [_make_para(f"P{i}", 200) for i in range(5)]
        chunks = greedy_chunk(paras, max_tokens=512)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_paragraph_hashes_match_constituents(self):
        paras = [_make_para(f"Para {i}", 100) for i in range(4)]
        chunks = greedy_chunk(paras, max_tokens=250)
        all_hashes = []
        for chunk in chunks:
            all_hashes.extend(chunk.paragraph_hashes)
        assert all_hashes == [p.hash for p in paras]

    def test_chunk_text_joins_with_separator(self):
        paras = [_make_para("Alpha", 100), _make_para("Beta", 100)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert chunks[0].text == "Alpha\n\nBeta"

    def test_single_paragraph_exactly_at_max(self):
        paras = [_make_para("Exact", 512)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].token_count == 512

    def test_two_paragraphs_exactly_at_max(self):
        paras = [_make_para("A", 256), _make_para("B", 256)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 1

    def test_three_300_token_paragraphs(self):
        paras = [_make_para(f"Para {i}", 300) for i in range(3)]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 3
        assert all(c.token_count == 300 for c in chunks)

    def test_mixed_sizes(self):
        paras = [
            _make_para("A", 200),
            _make_para("B", 200),
            _make_para("C", 200),
            _make_para("D", 100),
        ]
        chunks = greedy_chunk(paras, max_tokens=512)
        assert len(chunks) == 2
        assert chunks[0].token_count == 400
        assert chunks[1].token_count == 300

    def test_determinism(self):
        paras = [_make_para(f"P{i}", 150) for i in range(6)]
        chunks_a = greedy_chunk(paras, max_tokens=512)
        chunks_b = greedy_chunk(paras, max_tokens=512)
        assert len(chunks_a) == len(chunks_b)
        for a, b in zip(chunks_a, chunks_b):
            assert a.paragraph_hashes == b.paragraph_hashes
            assert a.token_count == b.token_count


# ---------------------------------------------------------------------------
# document_hash
# ---------------------------------------------------------------------------

class TestDocumentHash:
    def test_returns_16_hex_chars(self):
        h = document_hash("Some document text")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert document_hash("Hello") == document_hash("Hello")

    def test_different_texts_different_hashes(self):
        assert document_hash("A") != document_hash("B")

    def test_canonicalizes_whitespace(self):
        raw = "  Para one.  \n\n  Para two.  "
        clean = "Para one.\n\nPara two."
        assert document_hash(raw) == document_hash(clean)

    def test_consistent_with_chunk_reconstruction(self):
        raw = "  Para one.  \n\n  Para two.  "
        paras = split_paragraphs(raw)
        chunks = greedy_chunk(paras, max_tokens=5000)
        reconstructed = "\n\n".join(c.text for c in chunks)
        assert document_hash(raw) == document_hash(reconstructed)


# ---------------------------------------------------------------------------
# Integration: split_paragraphs -> greedy_chunk round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_reconstruct_document(self):
        original = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        paras = split_paragraphs(original)
        chunks = greedy_chunk(paras, max_tokens=5000)
        reconstructed = "\n\n".join(c.text for c in chunks)
        assert reconstructed == original

    def test_reconstruct_with_multiple_chunks(self):
        original = "Alpha.\n\nBeta.\n\nGamma.\n\nDelta."
        paras = split_paragraphs(original)
        chunks = greedy_chunk(paras, max_tokens=1)
        reconstructed = "\n\n".join(c.text for c in chunks)
        assert reconstructed == original

    def test_stripping_means_messy_input_does_not_round_trip(self):
        original = "  Para one.  \n\n  Para two.  "
        paras = split_paragraphs(original)
        chunks = greedy_chunk(paras, max_tokens=5000)
        reconstructed = "\n\n".join(c.text for c in chunks)
        assert reconstructed != original
        assert reconstructed == "Para one.\n\nPara two."
