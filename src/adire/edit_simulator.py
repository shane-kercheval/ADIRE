"""Edit simulator: apply controlled edits to documents for simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from adire.document_generator import VOCABULARY


class EditType(Enum):
    """Types of edits that can be applied to a document."""

    TYPO_FIX = "typo_fix"
    SENTENCE_ADDITION = "sentence_addition"
    PARAGRAPH_INSERT = "paragraph_insert"
    PARAGRAPH_DELETE = "paragraph_delete"
    SECTION_REWRITE = "section_rewrite"
    SECTION_INSERT = "section_insert"
    APPEND = "append"
    SCATTERED_EDITS = "scattered_edits"


# Edit types where EditPosition is meaningless — the operation is inherently
# position-independent. The experiment runner should not vary position for these.
POSITION_INDEPENDENT_TYPES: frozenset[EditType] = frozenset({
    EditType.APPEND,
    EditType.SCATTERED_EDITS,
})


class EditPosition(Enum):
    """Where in the document the edit is applied."""

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


@dataclass
class EditSpec:
    """Specification for a single edit operation."""

    edit_type: EditType
    position: EditPosition
    magnitude: int


def _split(text: str) -> list[str]:
    """Split document into paragraphs, preserving non-empty ones."""
    return [p for p in text.split("\n\n") if p.strip()]


def _join(paragraphs: list[str]) -> str:
    """Join paragraphs back into a document."""
    return "\n\n".join(paragraphs)


def _resolve_position(rng: random.Random, position: EditPosition, para_count: int) -> int:
    """Sample a paragraph index within the position's band."""
    if para_count == 0:
        return 0
    if position == EditPosition.TOP:
        band_end = max(1, para_count // 10)
        return rng.randint(0, min(band_end - 1, para_count - 1))
    if position == EditPosition.BOTTOM:
        band_start = para_count - max(1, para_count // 10)
        return rng.randint(max(0, band_start), para_count - 1)
    # MIDDLE: 40%-60% of the document
    mid_start = max(0, int(para_count * 0.4))
    mid_end = min(para_count - 1, int(para_count * 0.6))
    return rng.randint(mid_start, max(mid_start, mid_end))


def _generate_text(rng: random.Random, target_chars: int) -> str:
    """Generate random text of approximately target_chars length."""
    words: list[str] = []
    length = 0
    while length < target_chars:
        words.append(rng.choice(VOCABULARY))
        length += len(words[-1]) + 1
    return " ".join(words)[:target_chars]


def _generate_paragraph(rng: random.Random, length: int = 300) -> str:
    """Generate a new paragraph of approximately the given character length."""
    return _generate_text(rng, length)


def apply_edit(
    text: str,
    edit_spec: EditSpec,
    seed: int | None = None,
) -> str:
    """Apply a specified edit to the document text. Returns the modified text."""
    if edit_spec.magnitude == 0:
        return text

    rng = random.Random(seed)
    paras = _split(text)

    handlers = {
        EditType.TYPO_FIX: _apply_typo_fix,
        EditType.SENTENCE_ADDITION: _apply_sentence_addition,
        EditType.PARAGRAPH_INSERT: _apply_paragraph_insert,
        EditType.PARAGRAPH_DELETE: _apply_paragraph_delete,
        EditType.SECTION_REWRITE: _apply_section_rewrite,
        EditType.SECTION_INSERT: _apply_section_insert,
        EditType.APPEND: _apply_append,
        EditType.SCATTERED_EDITS: _apply_scattered_edits,
    }

    handler = handlers[edit_spec.edit_type]
    result_paras = handler(rng, paras, edit_spec)
    return _join(result_paras)


def _apply_typo_fix(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Replace a random word in one paragraph with a different word."""
    if not paras:
        return paras
    idx = _resolve_position(rng, spec.position, len(paras))
    words = paras[idx].split()
    if words:
        word_idx = rng.randrange(len(words))
        new_word = rng.choice(VOCABULARY)
        while new_word == words[word_idx] and len(VOCABULARY) > 1:
            new_word = rng.choice(VOCABULARY)
        words[word_idx] = new_word
        paras[idx] = " ".join(words)
    return paras


def _apply_sentence_addition(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Append 1-2 random sentences (50-100 chars) to a paragraph."""
    if not paras:
        return paras
    idx = _resolve_position(rng, spec.position, len(paras))
    addition_len = rng.randint(50, 100)
    paras[idx] = paras[idx] + " " + _generate_text(rng, addition_len)
    return paras


def _apply_paragraph_insert(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Insert magnitude new paragraphs (~300 chars each) at the specified position."""
    idx = _resolve_position(rng, spec.position, len(paras))
    new_paras = [_generate_paragraph(rng) for _ in range(spec.magnitude)]
    return paras[:idx] + new_paras + paras[idx:]


def _apply_paragraph_delete(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Remove magnitude paragraphs at the specified position."""
    if not paras:
        return paras
    idx = _resolve_position(rng, spec.position, len(paras))
    count = min(spec.magnitude, len(paras) - idx)
    # Keep at least one paragraph if possible
    if count >= len(paras):
        count = max(0, len(paras) - 1)
    return paras[:idx] + paras[idx + count:]


def _apply_section_rewrite(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Replace magnitude consecutive paragraphs with new ones of similar individual lengths."""
    if not paras:
        return paras
    idx = _resolve_position(rng, spec.position, len(paras))
    count = min(spec.magnitude, len(paras) - idx)
    old_section = paras[idx:idx + count]
    # Sample each replacement from original lengths with modest variation
    new_section = []
    for old_para in old_section:
        base_len = max(50, len(old_para))
        variation = int(base_len * 0.2)
        para_len = rng.randint(base_len - variation, base_len + variation)
        new_section.append(_generate_paragraph(rng, para_len))
    return paras[:idx] + new_section + paras[idx + count:]


def _apply_section_insert(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Insert magnitude new section-sized paragraphs (500-800 chars) as a block."""
    idx = _resolve_position(rng, spec.position, len(paras))
    new_paras = [
        _generate_paragraph(rng, rng.randint(500, 800))
        for _ in range(spec.magnitude)
    ]
    return paras[:idx] + new_paras + paras[idx:]


def _apply_append(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Add magnitude new paragraphs at the end. Position is ignored."""
    new_paras = [_generate_paragraph(rng) for _ in range(spec.magnitude)]
    return paras + new_paras


def _apply_scattered_edits(
    rng: random.Random,
    paras: list[str],
    spec: EditSpec,
) -> list[str]:
    """Apply magnitude typo-fix-style changes at non-adjacent paragraphs. Position is ignored."""
    if not paras:
        return paras
    count = min(spec.magnitude, len(paras))
    # Choose non-adjacent indices
    available = list(range(len(paras)))
    rng.shuffle(available)
    chosen: list[int] = []
    for idx in available:
        if not any(abs(idx - c) <= 1 for c in chosen):
            chosen.append(idx)
        if len(chosen) >= count:
            break
    # If we can't find enough non-adjacent, fill with remaining
    if len(chosen) < count:
        for idx in available:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= count:
                break

    for idx in chosen:
        words = paras[idx].split()
        if words:
            word_idx = rng.randrange(len(words))
            words[word_idx] = rng.choice(VOCABULARY)
            paras[idx] = " ".join(words)
    return paras
