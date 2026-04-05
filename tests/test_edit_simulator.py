"""Tests for adire.edit_simulator module."""

from adire.chunking import split_paragraphs
from adire.document_generator import MIXED, STRUCTURELESS_BLOB, generate_document
from adire.edit_simulator import (
    POSITION_INDEPENDENT_TYPES,
    EditPosition,
    EditSpec,
    EditType,
    apply_edit,
)


def _make_doc(n_paras: int = 20, seed: int = 42) -> str:
    """Generate a test document with the given number of paragraphs."""
    doc = generate_document(MIXED, n_paras * 400, seed=seed)
    paras = split_paragraphs(doc)
    if len(paras) > n_paras:
        return "\n\n".join(p.text for p in paras[:n_paras])
    return doc


def _para_hashes(text: str) -> list[str]:
    """Get paragraph hashes for a document."""
    return [p.hash for p in split_paragraphs(text)]


DOC = _make_doc(20)
DOC_PARAS = split_paragraphs(DOC)
DOC_PARA_COUNT = len(DOC_PARAS)


# ---------------------------------------------------------------------------
# Core behavior — each edit type
# ---------------------------------------------------------------------------

class TestTypoFix:
    def test_same_paragraph_count(self):
        edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.MIDDLE, 1), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT

    def test_exactly_one_paragraph_differs(self):
        edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.MIDDLE, 1), seed=1)
        old_hashes = _para_hashes(DOC)
        new_hashes = _para_hashes(edited)
        assert len(old_hashes) == len(new_hashes)
        diffs = sum(1 for a, b in zip(old_hashes, new_hashes) if a != b)
        assert diffs == 1


class TestSentenceAddition:
    def test_same_paragraph_count(self):
        edited = apply_edit(DOC, EditSpec(EditType.SENTENCE_ADDITION, EditPosition.MIDDLE, 1), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT

    def test_one_paragraph_is_longer(self):
        edited = apply_edit(DOC, EditSpec(EditType.SENTENCE_ADDITION, EditPosition.MIDDLE, 1), seed=1)
        old_paras = split_paragraphs(DOC)
        new_paras = split_paragraphs(edited)
        longer_count = sum(1 for a, b in zip(old_paras, new_paras) if len(b.text) > len(a.text))
        assert longer_count == 1


class TestParagraphInsert:
    def test_count_increases_by_magnitude(self):
        edited = apply_edit(DOC, EditSpec(EditType.PARAGRAPH_INSERT, EditPosition.MIDDLE, 2), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT + 2

    def test_magnitude_3(self):
        edited = apply_edit(DOC, EditSpec(EditType.PARAGRAPH_INSERT, EditPosition.TOP, 3), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT + 3


class TestParagraphDelete:
    def test_count_decreases_by_magnitude(self):
        edited = apply_edit(DOC, EditSpec(EditType.PARAGRAPH_DELETE, EditPosition.MIDDLE, 2), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT - 2

    def test_magnitude_3(self):
        edited = apply_edit(DOC, EditSpec(EditType.PARAGRAPH_DELETE, EditPosition.TOP, 3), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT - 3


class TestSectionRewrite:
    def test_paragraph_count_same(self):
        edited = apply_edit(DOC, EditSpec(EditType.SECTION_REWRITE, EditPosition.MIDDLE, 3), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT

    def test_affected_paragraphs_have_new_hashes(self):
        edited = apply_edit(DOC, EditSpec(EditType.SECTION_REWRITE, EditPosition.MIDDLE, 3), seed=1)
        old_hashes = _para_hashes(DOC)
        new_hashes = _para_hashes(edited)
        diffs = sum(1 for a, b in zip(old_hashes, new_hashes) if a != b)
        assert diffs == 3


class TestSectionInsert:
    def test_count_increases_by_magnitude(self):
        edited = apply_edit(DOC, EditSpec(EditType.SECTION_INSERT, EditPosition.MIDDLE, 3), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT + 3

    def test_inserted_paragraphs_are_section_sized(self):
        edited = apply_edit(DOC, EditSpec(EditType.SECTION_INSERT, EditPosition.MIDDLE, 3), seed=1)
        old_hashes = set(_para_hashes(DOC))
        new_paras = split_paragraphs(edited)
        inserted = [p for p in new_paras if p.hash not in old_hashes]
        assert len(inserted) == 3
        for p in inserted:
            assert len(p.text) >= 450  # section-sized (500-800 target, truncation may shorten)

    def test_longer_than_paragraph_insert(self):
        pi_edited = apply_edit(DOC, EditSpec(EditType.PARAGRAPH_INSERT, EditPosition.MIDDLE, 3), seed=1)
        si_edited = apply_edit(DOC, EditSpec(EditType.SECTION_INSERT, EditPosition.MIDDLE, 3), seed=1)
        pi_hashes = set(_para_hashes(DOC))
        si_hashes = set(_para_hashes(DOC))
        pi_inserted = [p for p in split_paragraphs(pi_edited) if p.hash not in pi_hashes]
        si_inserted = [p for p in split_paragraphs(si_edited) if p.hash not in si_hashes]
        avg_pi = sum(len(p.text) for p in pi_inserted) / len(pi_inserted)
        avg_si = sum(len(p.text) for p in si_inserted) / len(si_inserted)
        assert avg_si > avg_pi  # section inserts are meaningfully larger


class TestAppend:
    def test_count_increases_by_magnitude(self):
        edited = apply_edit(DOC, EditSpec(EditType.APPEND, EditPosition.BOTTOM, 2), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT + 2

    def test_last_paragraphs_are_new(self):
        edited = apply_edit(DOC, EditSpec(EditType.APPEND, EditPosition.BOTTOM, 2), seed=1)
        old_hashes = set(_para_hashes(DOC))
        new_paras = split_paragraphs(edited)
        for para in new_paras[-2:]:
            assert para.hash not in old_hashes


class TestScatteredEdits:
    def test_correct_number_of_changed_paragraphs(self):
        edited = apply_edit(DOC, EditSpec(EditType.SCATTERED_EDITS, EditPosition.MIDDLE, 3), seed=1)
        old_hashes = _para_hashes(DOC)
        new_hashes = _para_hashes(edited)
        diffs = sum(1 for a, b in zip(old_hashes, new_hashes) if a != b)
        assert diffs == 3

    def test_same_paragraph_count(self):
        edited = apply_edit(DOC, EditSpec(EditType.SCATTERED_EDITS, EditPosition.MIDDLE, 3), seed=1)
        assert len(split_paragraphs(edited)) == DOC_PARA_COUNT

    def test_changed_indices_are_non_adjacent(self):
        doc = _make_doc(30, seed=99)
        edited = apply_edit(doc, EditSpec(EditType.SCATTERED_EDITS, EditPosition.MIDDLE, 4), seed=1)
        old_hashes = _para_hashes(doc)
        new_hashes = _para_hashes(edited)
        changed = [i for i, (a, b) in enumerate(zip(old_hashes, new_hashes)) if a != b]
        assert len(changed) == 4
        for i in range(len(changed)):
            for j in range(i + 1, len(changed)):
                assert abs(changed[i] - changed[j]) > 1


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_top_affects_beginning(self):
        for seed in range(5):
            edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.TOP, 1), seed=seed)
            old_hashes = _para_hashes(DOC)
            new_hashes = _para_hashes(edited)
            changed = [i for i, (a, b) in enumerate(zip(old_hashes, new_hashes)) if a != b]
            assert len(changed) == 1
            assert changed[0] < DOC_PARA_COUNT * 0.2

    def test_bottom_affects_end(self):
        for seed in range(5):
            edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.BOTTOM, 1), seed=seed)
            old_hashes = _para_hashes(DOC)
            new_hashes = _para_hashes(edited)
            changed = [i for i, (a, b) in enumerate(zip(old_hashes, new_hashes)) if a != b]
            assert len(changed) == 1
            assert changed[0] >= DOC_PARA_COUNT * 0.8

    def test_middle_affects_center(self):
        for seed in range(5):
            edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.MIDDLE, 1), seed=seed)
            old_hashes = _para_hashes(DOC)
            new_hashes = _para_hashes(edited)
            changed = [i for i, (a, b) in enumerate(zip(old_hashes, new_hashes)) if a != b]
            assert len(changed) == 1
            assert DOC_PARA_COUNT * 0.3 <= changed[0] <= DOC_PARA_COUNT * 0.7

    def test_different_seeds_can_produce_different_positions(self):
        positions = set()
        for seed in range(20):
            edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.MIDDLE, 1), seed=seed)
            old_hashes = _para_hashes(DOC)
            new_hashes = _para_hashes(edited)
            changed = [i for i, (a, b) in enumerate(zip(old_hashes, new_hashes)) if a != b]
            positions.add(changed[0])
        assert len(positions) > 1  # not always the same index


# ---------------------------------------------------------------------------
# Position-independent types
# ---------------------------------------------------------------------------

class TestPositionIndependence:
    def test_append_and_scattered_are_position_independent(self):
        assert EditType.APPEND in POSITION_INDEPENDENT_TYPES
        assert EditType.SCATTERED_EDITS in POSITION_INDEPENDENT_TYPES

    def test_other_types_are_position_dependent(self):
        for et in EditType:
            if et not in (EditType.APPEND, EditType.SCATTERED_EDITS):
                assert et not in POSITION_INDEPENDENT_TYPES


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_delete_more_than_exist(self):
        short_doc = "Para one.\n\nPara two."
        edited = apply_edit(short_doc, EditSpec(EditType.PARAGRAPH_DELETE, EditPosition.TOP, 10), seed=1)
        paras = split_paragraphs(edited)
        assert len(paras) >= 1

    def test_single_paragraph_insert(self):
        blob = generate_document(STRUCTURELESS_BLOB, 5000, seed=42)
        assert "\n\n" not in blob
        edited = apply_edit(blob, EditSpec(EditType.PARAGRAPH_INSERT, EditPosition.MIDDLE, 1), seed=1)
        paras = split_paragraphs(edited)
        assert len(paras) == 2

    def test_magnitude_zero_returns_unchanged(self):
        edited = apply_edit(DOC, EditSpec(EditType.TYPO_FIX, EditPosition.MIDDLE, 0), seed=1)
        assert edited == DOC

    def test_scattered_magnitude_larger_than_para_count(self):
        short_doc = "A.\n\nB.\n\nC."
        edited = apply_edit(short_doc, EditSpec(EditType.SCATTERED_EDITS, EditPosition.MIDDLE, 100), seed=1)
        paras = split_paragraphs(edited)
        assert len(paras) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        spec = EditSpec(EditType.SECTION_REWRITE, EditPosition.MIDDLE, 3)
        a = apply_edit(DOC, spec, seed=42)
        b = apply_edit(DOC, spec, seed=42)
        assert a == b

    def test_different_seeds_different_output(self):
        spec = EditSpec(EditType.SECTION_REWRITE, EditPosition.MIDDLE, 3)
        a = apply_edit(DOC, spec, seed=1)
        b = apply_edit(DOC, spec, seed=2)
        assert a != b
