"""Synthetic document generator with controllable size and structure."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

# ~200 common English words for lorem-ipsum-style paragraph text.
VOCABULARY = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make",
    "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
    "most", "us", "great", "between", "need", "large", "often", "should", "never", "each",
    "much", "where", "right", "still", "world", "long", "before", "must", "through", "very",
    "find", "here", "thing", "many", "system", "process", "part", "high", "last", "small",
    "keep", "start", "point", "read", "hand", "turn", "move", "change", "help", "show", "home",
    "side", "life", "night", "write", "next", "end", "both", "group", "begin", "seem", "while",
    "head", "run", "place", "state", "own", "line", "open", "same", "tell", "call", "try",
    "ask", "close", "follow", "learn", "stop", "watch", "plan", "name", "word", "form", "might",
    "case", "set", "old", "study", "build", "note", "number", "data", "step", "level", "order",
    "few", "under", "report", "result", "field", "table", "test", "model", "record", "above",
    "below", "support", "early", "area", "along", "during", "already", "against", "always",
    "able", "enough", "across", "create", "simple", "clear", "around",
]


class GeneratorKind(Enum):
    """How paragraphs are generated for a document profile."""

    GAUSSIAN = "gaussian"
    BLOB = "blob"
    BIMODAL = "bimodal"


@dataclass
class DocumentProfile:
    """Configuration for generating a synthetic document.

    For GAUSSIAN profiles, paragraph lengths are sampled from a normal distribution
    clipped to [paragraph_length_min, paragraph_length_max]. For BLOB and BIMODAL
    profiles, the numeric fields are unused — generation follows fixed logic.
    """  # noqa: D213

    name: str
    kind: GeneratorKind
    paragraph_length_mean: int = 0
    paragraph_length_std: int = 0
    paragraph_length_min: int = 0
    paragraph_length_max: int = 0


# --- Structural profile presets ---

SHORT_PARAGRAPHS = DocumentProfile(
    name="short_paragraphs", kind=GeneratorKind.GAUSSIAN,
    paragraph_length_mean=150, paragraph_length_std=50,
    paragraph_length_min=50, paragraph_length_max=250,
)

MIXED = DocumentProfile(
    name="mixed", kind=GeneratorKind.GAUSSIAN,
    paragraph_length_mean=400, paragraph_length_std=150,
    paragraph_length_min=80, paragraph_length_max=800,
)

LONG_PARAGRAPHS = DocumentProfile(
    name="long_paragraphs", kind=GeneratorKind.GAUSSIAN,
    paragraph_length_mean=800, paragraph_length_std=200,
    paragraph_length_min=400, paragraph_length_max=1200,
)

OVERSIZED_PARAGRAPHS = DocumentProfile(
    name="oversized_paragraphs", kind=GeneratorKind.GAUSSIAN,
    paragraph_length_mean=2500, paragraph_length_std=800,
    paragraph_length_min=1500, paragraph_length_max=4000,
)

STRUCTURELESS_BLOB = DocumentProfile(name="structureless_blob", kind=GeneratorKind.BLOB)

BIMODAL = DocumentProfile(name="bimodal", kind=GeneratorKind.BIMODAL)

ALL_PROFILES = [SHORT_PARAGRAPHS, MIXED, LONG_PARAGRAPHS, OVERSIZED_PARAGRAPHS,
                STRUCTURELESS_BLOB, BIMODAL]

# --- Size presets (target character counts) ---

TINY = 2_000
SMALL = 5_000
MEDIUM = 25_000
LARGE = 50_000
MAX_SIZE = 100_000


def _generate_text(rng: random.Random, target_chars: int) -> str:
    """Generate random text of approximately target_chars length."""
    words: list[str] = []
    length = 0
    while length < target_chars:
        word = rng.choice(VOCABULARY)
        words.append(word)
        length += len(word) + 1  # +1 for the space
    return " ".join(words)[:target_chars]


def _sample_paragraph_length(rng: random.Random, profile: DocumentProfile) -> int:
    """Sample a paragraph length from the profile's distribution, clipped to [min, max]."""
    length = int(rng.gauss(profile.paragraph_length_mean, profile.paragraph_length_std))
    return max(profile.paragraph_length_min, min(profile.paragraph_length_max, length))


def generate_document(
    profile: DocumentProfile,
    target_chars: int,
    seed: int | None = None,
) -> str:
    """Generate a synthetic document matching the given profile and target size."""
    rng = random.Random(seed)

    if profile.kind == GeneratorKind.BLOB:
        return _generate_text(rng, target_chars)

    if profile.kind == GeneratorKind.BIMODAL:
        return _generate_bimodal(rng, target_chars)

    return _generate_standard(rng, profile, target_chars)


def _generate_bimodal(rng: random.Random, target_chars: int) -> str:
    """Generate alternating short (~80 char) and long (~700 char) paragraphs."""
    short_len = 80
    long_len = 700
    paragraphs: list[str] = []
    total = 0
    is_short = True

    while total < target_chars:
        target = short_len if is_short else long_len
        remaining = target_chars - total
        if remaining < short_len // 2:
            break
        sep_cost = 2 if paragraphs else 0
        para_len = min(target, remaining - sep_cost)
        if para_len <= 0:
            break
        paragraphs.append(_generate_text(rng, para_len))
        total += para_len + sep_cost
        is_short = not is_short

    if not paragraphs:
        paragraphs.append(_generate_text(rng, min(short_len, target_chars)))

    return "\n\n".join(paragraphs)


def _generate_standard(
    rng: random.Random,
    profile: DocumentProfile,
    target_chars: int,
) -> str:
    """Generate a document with paragraphs sampled from the profile's distribution."""
    paragraphs: list[str] = []
    total = 0

    while total < target_chars:
        para_len = _sample_paragraph_length(rng, profile)
        sep_cost = 2 if paragraphs else 0
        remaining = target_chars - total - sep_cost
        if remaining <= 0:
            break
        para_len = min(para_len, remaining)
        paragraphs.append(_generate_text(rng, para_len))
        total += para_len + sep_cost

    if not paragraphs:
        paragraphs.append(_generate_text(rng, max(1, min(profile.paragraph_length_min,
                                                         target_chars))))

    return "\n\n".join(paragraphs)
