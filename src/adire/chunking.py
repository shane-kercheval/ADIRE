"""Core chunking primitives: paragraph splitting, greedy combining, and hashing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class Paragraph:
    """A single paragraph extracted from a document."""

    text: str
    hash: str  # SHA-256 truncated to 16 hex chars (computed from normalized text)
    token_count: int


@dataclass
class Chunk:
    """A chunk composed of one or more paragraphs."""

    index: int
    text: str
    paragraph_hashes: list[str]
    token_count: int


def _normalize(text: str) -> str:
    """Normalize whitespace: strip and collapse internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", text.strip())


def hash_text(text: str) -> str:
    """SHA-256 hash truncated to 16 hex chars."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def split_paragraphs(text: str, separator: str = "\n\n") -> list[Paragraph]:
    """Split text into paragraphs, compute hash and token count for each.

    Hash uses normalized text (collapsed whitespace) for stable identity across
    formatting differences. Token count uses original stripped text because that
    reflects actual content size for embedding cost estimation.

    Custom separators control paragraph identification, but greedy_chunk always
    joins with the default double-newline. Faithful round-trip reconstruction is
    only guaranteed for the default separator.
    """  # noqa: D213
    raw_parts = text.split(separator)
    paragraphs: list[Paragraph] = []
    for part in raw_parts:
        stripped = part.strip()
        if not stripped:
            continue
        normalized = _normalize(stripped)
        paragraphs.append(
            Paragraph(
                text=stripped,
                hash=hash_text(normalized),
                token_count=max(1, len(stripped) // 4),
            ),
        )
    return paragraphs


def greedy_chunk(paragraphs: list[Paragraph], max_tokens: int = 512) -> list[Chunk]:
    """Combine paragraphs greedily into chunks up to max_tokens.

    Chunk text is always joined with double-newline regardless of how paragraphs
    were originally split. Round-trip reconstruction (joining all chunk texts)
    is only faithful when the default paragraph separator was used.
    """  # noqa: D213
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_hashes: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        if current_texts and current_tokens + para.token_count > max_tokens:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text="\n\n".join(current_texts),
                    paragraph_hashes=current_hashes,
                    token_count=current_tokens,
                ),
            )
            current_texts = []
            current_hashes = []
            current_tokens = 0

        current_texts.append(para.text)
        current_hashes.append(para.hash)
        current_tokens += para.token_count

    if current_texts:
        chunks.append(
            Chunk(
                index=len(chunks),
                text="\n\n".join(current_texts),
                paragraph_hashes=current_hashes,
                token_count=current_tokens,
            ),
        )

    return chunks


def document_hash(text: str, separator: str = "\n\n") -> str:
    """Hash a canonicalized form of the document for fast no-change detection.

    Canonicalizes by splitting into paragraphs, stripping each, filtering
    empties, and rejoining with the separator. This ensures the hash is
    consistent with chunk reconstruction — whitespace-only differences
    between the raw input and reconstructed chunk text won't cause mismatches.
    """  # noqa: D213
    parts = [p.strip() for p in text.split(separator)]
    canonical = separator.join(p for p in parts if p)
    return hash_text(canonical)
