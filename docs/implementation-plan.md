# ADIRE Research Implementation Plan

## Overview

This plan implements a simulation framework to evaluate the ADIRE (Anchor-Diffed Incremental Re-Embedding) algorithm against baseline chunking strategies. The goal is to answer: **how much does ADIRE save in re-embedding costs, and under what conditions?**

The simulation runs entirely locally (no embedding API calls for the efficiency experiments). It compares five strategies across a matrix of document sizes, document structures, edit types, and other dimensions.

**Strategy hierarchy:**
- **Primary comparison** (same chunking granularity — greedy combining to ~512 tokens): naive → chunk-hash match → ADIRE. This is the core experiment — same chunk sizes, increasing sophistication of reuse.
- **Secondary comparisons**: paragraph-level reuse (tests whether combining matters for reuse) and ADIRE wide-window (tests whether expanding dirty regions reduces fragmentation). These answer auxiliary questions and are not directly comparable to the primary strategies on preservation_rate due to different chunking granularity (paragraph reuse) or different dirty-region scope (wide window).

**Before implementing any milestone**, read the full design document at `docs/adire-anchor-diffed-incremental-re-embedding.md`. The algorithm pseudocode (lines 74-201), walkthrough example (lines 203-305), and simulation design (Appendix C, lines 491-622) are the primary references.

---

## Milestone 1: Core Chunking Primitives

### Goal & Outcome

Build the foundational text processing layer that all five strategies share.

After this milestone:
- Paragraphs can be split from raw text using configurable anchor unit separators
- Paragraph hashes can be computed deterministically
- Paragraphs can be greedily combined into chunks up to a token budget
- Token counting uses a simple approximation (chars / 4) — no external tokenizer dependency needed for the simulation

### Implementation Outline

**First**, update `pyproject.toml` to point the wheel build at `src/adire` instead of `src/package`. Create `src/adire/__init__.py`. Remove the template files `src/app.py`, `src/sandbox.ipynb`, `tests/test_app.py`, and clean up `tests/conftest.py` (remove the `fake_dataset` fixture). This must happen before creating any modules to avoid import and test discovery issues.

Create module `src/adire/chunking.py` with the following components.

**Paragraph splitting:**

```python
@dataclass
class Paragraph:
    text: str
    hash: str  # SHA-256 truncated to 16 hex chars
    token_count: int

def split_paragraphs(text: str, separator: str = "\n\n") -> list[Paragraph]:
    """Split text into paragraphs, compute hash and token count for each."""
```

- Normalize whitespace within each paragraph (strip leading/trailing, collapse internal whitespace) before hashing. This ensures minor formatting changes don't produce false diffs.
- Hash: `hashlib.sha256(normalized_text.encode()).hexdigest()[:16]`
- Token count: `len(text) // 4` (approximate, sufficient for simulation)
- Filter out empty paragraphs after splitting.

**Greedy chunk combining:**

```python
@dataclass
class Chunk:
    index: int
    text: str
    paragraph_hashes: list[str]
    token_count: int

def greedy_chunk(paragraphs: list[Paragraph], max_tokens: int = 512) -> list[Chunk]:
    """Combine paragraphs greedily into chunks up to max_tokens."""
```

- Walk paragraphs in order. Accumulate into current chunk until adding the next paragraph would exceed `max_tokens`. Then start a new chunk.
- A single paragraph larger than `max_tokens` becomes its own chunk (don't split paragraphs).
- Each chunk stores the ordered list of paragraph hashes it contains.

**Document-level hash:**

```python
def document_hash(text: str) -> str:
    """Hash the full document text for fast no-change detection."""
```

Also create `src/adire/__init__.py` (empty).

### Testing Strategy

Create `tests/test_chunking.py`. Remove the template `tests/test_app.py` and clean up `tests/conftest.py` (remove the fake_dataset fixture).

**Core behavior:**
- `split_paragraphs` on text with multiple `\n\n`-separated paragraphs returns correct count, text, and hashes
- `split_paragraphs` with custom separator (e.g., `\n`) works
- `greedy_chunk` combines small paragraphs into chunks up to the token budget
- `greedy_chunk` starts a new chunk when adding the next paragraph would exceed budget
- `greedy_chunk` with a single paragraph larger than `max_tokens` produces a one-paragraph chunk
- `greedy_chunk` on an empty list returns an empty list
- Chunk indices are sequential starting from 0
- Each chunk's `paragraph_hashes` matches the hashes of its constituent paragraphs

**Edge cases:**
- Empty text produces no paragraphs
- Text with no separators produces one paragraph
- Text with consecutive separators (e.g., `\n\n\n\n`) doesn't produce empty paragraphs
- Whitespace normalization: same content with different leading/trailing whitespace produces the same hash
- Single paragraph that exactly equals `max_tokens` fits in one chunk (boundary condition)

**Determinism:**
- Same input text always produces the same paragraph hashes
- Same paragraphs with same `max_tokens` always produce the same chunks

---

## Milestone 2: Re-Embedding Strategies (including ADIRE algorithm)

### Goal & Outcome

Implement the five re-embedding strategies as composable functions with a shared interface.

After this milestone:
- All five strategies (naive, paragraph-level reuse, chunk-hash match, ADIRE, ADIRE wide-window) can process a document update and return which chunks need re-embedding vs. which are preserved
- Each strategy returns a standardized metrics/results object for comparison
- The ADIRE algorithm handles insertions, deletions, replacements, and mixed edits correctly

### Implementation Outline

Create `src/adire/strategies.py`.

**Shared data structures:**

```python
@dataclass
class UpdateResult:
    """Result of processing a document update."""
    chunks: list[Chunk]               # final chunk list after update
    chunks_reembedded: int            # count of chunks that need new embeddings
    tokens_reembedded: int            # total tokens in re-embedded chunks
    chunks_preserved: int             # count of chunks kept from old version

    # --- Primary metrics (the headline numbers) ---
    preservation_rate: float          # chunks_preserved / total_chunks — what % of chunks we reused
    reembedding_rate: float           # chunks_reembedded / total_chunks — what % we had to pay for
    token_savings_rate: float         # 1 - (tokens_reembedded / total_tokens) — what % of token cost we avoided

    # --- Chunk quality metrics ---
    fragment_count: int               # chunks below 25% of target size
    fragment_ratio: float             # fragment_count / total_chunks
    oversized_count: int              # chunks exceeding max_tokens
    oversized_ratio: float            # oversized_count / total_chunks
```

The three primary metrics are what the experiments are designed to compare:
- **Preservation rate**: the core "hit rate" — what fraction of chunks survived the edit unchanged
- **Re-embedding rate**: the inverse, framed as cost — what fraction required an API call
- **Token savings rate**: cost-weighted version — accounts for the fact that preserving a large chunk saves more than preserving a small one

**Cross-strategy comparability:** `token_savings_rate` is the headline metric for comparing across all five strategies because it's denominated in tokens, not chunks — comparable regardless of chunking granularity. `preservation_rate` is only meaningful when comparing strategies with the same chunk granularity (naive, chunk-hash, ADIRE, ADIRE wide-window). Paragraph-level reuse has a different chunk definition (1 paragraph = 1 chunk), so its preservation_rate is not directly comparable to the others.

**Strategy 1 — Naive:**

```python
def naive_rechunk(
    new_text: str,
    old_chunks: list[Chunk],  # unused, but kept for interface consistency
    max_tokens: int = 512,
) -> UpdateResult:
```

Always re-chunks and "re-embeds" everything. This is the baseline.

**Strategy 2 — Paragraph-Level Reuse:**

```python
def paragraph_reuse_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
) -> UpdateResult:
```

Each paragraph becomes its own chunk. On edit, split the new text into paragraphs, hash each one, and do a set lookup against all paragraph hashes from the old version. If a hash exists in the old set, reuse the embedding — order and position don't matter. Only new or changed paragraphs need embedding.

This is the simplest possible reuse strategy: no diffing, no dirty regions, no combining. It's fully cascade-resistant (a paragraph inserted at the top doesn't affect any other paragraph's hash).

The tradeoff it highlights is **chunk size variability** — a 20-character heading and a 1500-character prose block both become individual chunks. Small chunks produce low-quality embeddings (insufficient semantic content), and large chunks dilute relevance. This is what ADIRE's greedy combining solves, so comparing these two strategies isolates the value of that combining step.

Implementation: build a `set` of old paragraph hashes from `old_chunks[*].paragraph_hashes`. For each new paragraph, check membership. Chunks are just individual paragraphs (no greedy combining).

**Strategy 3 — Chunk-Hash Match:**

```python
def chunk_hash_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
) -> UpdateResult:
```

Re-chunk the new text from scratch using greedy paragraph combining (same as naive), then hash each resulting chunk's full text and check if it matches any old chunk's text hash. If so, reuse the embedding.

This is the "common optimization" that the design doc claims breaks down due to cascade. The simulation will prove this empirically: inserting a paragraph near the top shifts all downstream chunk boundaries, so most chunk text hashes change even though the underlying paragraphs are the same. Edits near the bottom may preserve earlier chunks by coincidence.

Implementation: run `greedy_chunk` on the new text (identical to naive), then build a `set` of old chunk text hashes. For each new chunk, check if `hash(chunk.text)` is in the old set. If so, mark as preserved.

**Strategy 4 — ADIRE:**

```python
def adire_rechunk(
    new_text: str,
    old_chunks: list[Chunk],
    max_tokens: int = 512,
) -> UpdateResult:
```

This is the core algorithm from the design doc. Implementation steps:

1. **Document-level fast path**: hash `new_text`, compare to hash of reconstructed old text. If equal, return early with all chunks preserved.
2. **Split new text into paragraphs and hash each one.**
3. **Reconstruct old paragraph hash sequence** from `old_chunks[*].paragraph_hashes`.
4. **Diff old vs. new paragraph hash sequences** using `difflib.SequenceMatcher(a, b, autojunk=False)`. This returns opcodes as positional ranges: `(tag, i1, i2, j1, j2)` where `old[i1:i2]` maps to `new[j1:j2]`.
5. **Map changed positions to dirty chunk indices.** Build a `position -> chunk_index` lookup from old chunks (position 0 is the first paragraph of chunk 0, etc.). **Do NOT use a hash-based lookup** (`hash -> chunk_index`) — it breaks on duplicate paragraphs (repeated bullets, boilerplate) because later occurrences overwrite earlier ones. For each non-equal opcode, use the old-side positions (`i1:i2`) to find dirty chunks. For insertions (`i1 == i2`), dirty the chunk owning old position `i1` (the following chunk); if `i1` is past the end, it's a trailing insertion handled in step 7.
6. **Walk old chunks in order.** Unchanged chunks are kept. Consecutive dirty chunks form a "dirty region." For each dirty region, collect the corresponding new paragraphs (step 6a) and re-chunk them with `greedy_chunk`.
7. **Handle trailing insertions** (new paragraphs appended after the last old chunk).
8. **Assemble the final chunk list** with sequential indices. **Kept chunks must retain all fields from the old chunk** (text, paragraph_hashes, token_count) so they can serve as `old_chunks` input to the next edit in an edit chain.
9. **Compute metrics** (re-embedded count, preserved count, fragment count, etc.).

Use `difflib.SequenceMatcher` for the sequence diff — it's in the standard library and returns opcodes (`equal`, `replace`, `insert`, `delete`) that map directly to the algorithm's needs.

The `MIN_CHUNK_THRESHOLD` for fragment counting should be `max_tokens * 0.25` (i.e., 128 tokens for a 512 target), as suggested in the design doc.

**Step 6a — Collecting new paragraphs for a dirty region** (the hardest sub-function):

A dirty region spans old positions `[region_start, region_end)`. To find the new paragraphs for that region, walk the opcodes and collect the new-side paragraphs for every opcode that overlaps the old-side range:

- **`equal`**: 1:1 mapping between old and new positions. Clip to the overlap and collect the corresponding new paragraphs.
- **`replace`**: Old and new ranges may differ in length (1 old paragraph replaced by 3 new ones). If **any** part of the old range overlaps the dirty region, include **all** new-side paragraphs for that opcode. Don't try to clip proportionally.
- **`insert`**: No old positions consumed (`i1 == i2`). Include if the insertion point `i1` falls within `[region_start, region_end)`.
- **`delete`**: Old positions consumed but nothing on the new side. Skip (contributes no new paragraphs).

Skip opcodes entirely before the region (`i2 <= region_start` and not an insert at `region_start`). Stop when opcodes are entirely past the region (`i1 >= region_end`).

**Critical invariant:** A dirty region always fully contains every non-equal opcode that overlaps it. This is guaranteed because `find_dirty_chunks` marks **all** chunks touched by any non-equal opcode as dirty, and consecutive dirty chunks merge into a single region. This means the "include all new-side paragraphs" rule for `replace` is safe — a replace opcode can never partially overlap a dirty region. If a replace spans chunks 1-4, all four are marked dirty, forming one contiguous region. The collection rule depends on this invariant; changing `find_dirty_chunks` to mark fewer chunks would break it.

Each of the sub-functions in steps 4-6a (positional mapping, dirty chunk identification, dirty region collection) should be implemented as separate, independently testable functions — not inlined into one monolithic `adire_rechunk`.

### Testing Strategy

Create `tests/test_strategies.py`.

**Naive strategy:**
- Always re-embeds all chunks, preservation rate is 0%
- Returns correct chunk count and token counts

**Paragraph-level reuse strategy:**
- Unchanged document preserves all paragraphs (all hashes match)
- Typo fix: only 1 paragraph re-embedded, all others preserved regardless of position
- Paragraph insertion: only the new paragraph is embedded, all existing preserved
- Paragraph deletion: nothing to embed, all remaining paragraphs preserved
- Each chunk contains exactly one paragraph (no greedy combining)
- Chunk count equals paragraph count
- Chunk sizes are highly variable (matches individual paragraph sizes)
- Duplicate paragraphs: if two paragraphs have the same text, editing one still preserves the other (set-based lookup, both share the same hash)

**Chunk-hash match strategy:**
- Unchanged document preserves all chunks (text hashes match exactly)
- Edit at bottom of document: chunks before the edit are preserved (boundaries unaffected)
- Edit at top of document: most/all chunks re-embedded due to cascade (boundaries shift downstream). This is the key test — it demonstrates the cascade problem empirically.
- Paragraph insertion in the middle: chunks before the insertion may be preserved, chunks after are re-embedded
- Uses greedy combining (same chunk sizes as naive), unlike paragraph-level reuse

**ADIRE sub-function tests** — each internal function should be independently testable. Create `tests/test_adire_internals.py` for these.

*Positional mapping (`build_position_to_chunk`):*
- 3 chunks with 2, 3, 1 paragraphs → positions 0-1 map to chunk 0, 2-4 to chunk 1, 5 to chunk 2
- Single chunk → all positions map to chunk 0
- Empty chunk list → empty mapping

*Dirty chunk identification (`find_dirty_chunks`):*
- `replace` opcode: old positions map to correct chunk(s)
- `delete` opcode: old positions map to correct chunk
- `insert` in middle: dirties the chunk owning the insertion point (the following chunk)
- `insert` at position 0 (prepend): dirties chunk 0
- `insert` past the end (append): returns no dirty chunks (trailing insertion)
- Change spanning a chunk boundary (e.g., `replace` covering positions in two chunks): both chunks dirty
- Multiple non-adjacent changes: multiple non-adjacent chunks dirty
- Duplicate paragraphs: editing one instance dirties the correct chunk, not the other instance
- All-equal opcodes: no dirty chunks

*Dirty region paragraph collection (`collect_new_paragraphs_for_dirty_region`):*
- `equal` opcode fully inside region: collects new-side paragraphs
- `equal` opcode partially overlapping region: collects only the overlapping portion
- `replace` opcode overlapping region: collects **all** new-side paragraphs (not clipped), even if old and new ranges differ in length
- `replace` with 1 old → 3 new, region covers the 1 old position: all 3 new paragraphs collected
- `insert` inside region: inserted paragraphs collected
- `insert` at region boundary (start): included
- `insert` outside region: not collected
- `delete` inside region: contributes nothing (no new paragraphs)
- Mixed opcodes across a dirty region: paragraphs collected in correct order
- Dirty region spanning multiple consecutive chunks with insertions between them
- **Invariant test**: `replace` opcode spanning 3 chunks → all 3 marked dirty by `find_dirty_chunks` → one contiguous dirty region → all new paragraphs collected correctly. Validates that a replace can never partially overlap a dirty region.

*Dirty region boundary detection:*
- Single dirty chunk: region is that chunk's positions
- Two consecutive dirty chunks: merged into one region
- Two dirty chunks separated by a clean chunk: two separate regions (the clean chunk is preserved, not merged)
- Dirty chunk at start of document: region starts at position 0
- Dirty chunk at end of document: region ends at last position

**ADIRE integration tests** — test the full `adire_rechunk` function end-to-end. Keep in `tests/test_strategies.py`.

*Walkthrough example from the design doc:*
- 6 paragraphs, insert after B, edit D. Verify chunks 0 and 2 (renumbered to 3) are preserved, chunks 1 and 2 are re-embedded.
- This is a critical test — it validates the algorithm against the worked example in the design doc.

*Edit types:*
- Typo fix (change text within one paragraph): only 1 chunk dirty
- Paragraph insertion: adjacent chunk(s) dirty, others preserved
- Paragraph deletion: chunk containing deleted paragraph is dirty
- Section rewrite (multiple consecutive paragraphs replaced): consecutive chunks dirty
- Append at end: only new/last chunk affected
- No change: document hash matches, all chunks preserved (fast path)

*Edge cases:*
- First document version (no old chunks): equivalent to naive, all chunks are new
- Single-paragraph document: any edit re-embeds the one chunk
- Edit that makes a paragraph empty (effectively deleting it)
- Insert at the very beginning of the document (prepend)
- Insert at the very end of the document (append/trailing)
- All paragraphs changed (worst case — should degrade to naive-equivalent cost)
- Old chunks is empty (new document)
- Duplicate paragraphs: edit one instance, verify the correct chunk is dirty

**Cross-strategy consistency:**
- For each strategy, concatenating all chunk texts (joined by paragraph separator) should reconstruct the full new document text. Note: chunk boundaries may differ between strategies (ADIRE re-chunks only dirty regions), but the full document content must be identical.

**Fragmentation metrics:**
- After an edit that produces a small chunk (like the walkthrough where chunk 2 becomes 190 tokens), verify `fragment_count` and `fragment_ratio` are computed correctly

---

## Milestone 3: Document Generator

### Goal & Outcome

Build a synthetic document generator that produces documents with controllable size and structure.

After this milestone:
- Documents can be generated with specific target sizes (character count)
- Documents can be generated with specific structural profiles (paragraph length distributions)
- Generation is seeded/deterministic for reproducibility

### Implementation Outline

Create `src/adire/document_generator.py`.

```python
@dataclass
class DocumentProfile:
    """Configuration for generating a synthetic document."""
    name: str
    target_chars: int
    paragraph_length_mean: int    # in characters
    paragraph_length_std: int     # in characters
    paragraph_length_min: int     # floor (clips distribution)
    paragraph_length_max: int     # ceiling (clips distribution)

def generate_document(profile: DocumentProfile, seed: int | None = None) -> str:
    """Generate a synthetic document matching the given profile."""
```

**Structural profiles to define as constants/presets:**

| Profile Name | Mean Para Length | Std | Min | Max | Description |
|---|---|---|---|---|---|
| `short_paragraphs` | 150 | 50 | 50 | 250 | Bullet-heavy, short notes |
| `mixed` | 400 | 150 | 80 | 800 | Realistic note-taking |
| `long_paragraphs` | 800 | 200 | 400 | 1200 | Technical/prose |
| `oversized_paragraphs` | 2500 | 800 | 1500 | 4000 | Paragraphs that exceed the 512-token (~2048 char) chunk budget |
| `structureless_blob` | N/A | N/A | N/A | N/A | Single giant paragraph (special case) |
| `bimodal` | N/A | N/A | N/A | N/A | Alternating short (~80 char) and long (~700 char) paragraphs |

**Size presets:**

| Name | Target Chars |
|---|---|
| `tiny` | 2_000 |
| `small` | 5_000 |
| `medium` | 25_000 |
| `large` | 50_000 |
| `max_size` | 100_000 |

For paragraph text content: use `random` module to generate lorem-ipsum-style text (random words from a fixed vocabulary). The actual text content doesn't matter for the efficiency simulation — only the structure (paragraph count, sizes) matters. A simple approach: pick random words from a list of ~200 common English words, concatenated with spaces, until the target paragraph length is reached.

For the `structureless_blob` profile: generate one giant paragraph with no `\n\n` separators.

For the `bimodal` profile: alternate between short and long paragraphs (e.g., a heading-like line followed by a prose block).

The generator should accept a `seed` parameter for reproducibility. Use `random.Random(seed)` instance (not global state) so tests are deterministic.

### Testing Strategy

Create `tests/test_document_generator.py`.

**Core behavior:**
- Generated document character count is within 10% of `target_chars` for each size preset
- Paragraph count is reasonable for the profile (e.g., `short_paragraphs` at 25K chars should produce ~125-200 paragraphs)
- `oversized_paragraphs` produces paragraphs that mostly exceed `max_tokens` (each paragraph becomes its own chunk)
- `structureless_blob` produces exactly 1 paragraph (no `\n\n` in output)
- `bimodal` alternates between short and long paragraphs

**Determinism:**
- Same `seed` produces identical documents
- Different seeds produce different documents

**Edge cases:**
- Very small target (e.g., 100 chars) still produces at least one paragraph
- Profile with `paragraph_length_mean` larger than `target_chars` produces one paragraph

**Integration with chunking:**
- Generated documents can be split into paragraphs and chunked without errors
- Paragraph counts from split match expected counts for the profile

---

## Milestone 4: Edit Simulator

### Goal & Outcome

Build an edit simulator that applies controlled, realistic edits to documents.

After this milestone:
- Edits of specific types can be applied to documents at specific positions
- Edit magnitude (how much text is affected) is controllable
- Random edits can be generated for simulation runs
- Edit chains (sequential edits on the same document) can be produced

### Implementation Outline

Create `src/adire/edit_simulator.py`.

**Edit types** (from the design doc Appendix C, with magnitude as a parameter):

```python
class EditType(Enum):
    TYPO_FIX = "typo_fix"                    # change chars within a paragraph
    SENTENCE_ADDITION = "sentence_addition"    # add text to existing paragraph
    PARAGRAPH_INSERT = "paragraph_insert"      # insert new paragraph(s)
    PARAGRAPH_DELETE = "paragraph_delete"      # remove existing paragraph(s)
    SECTION_REWRITE = "section_rewrite"        # replace consecutive paragraphs
    SECTION_INSERT = "section_insert"          # insert block of new paragraphs
    APPEND = "append"                          # add paragraphs at end
    SCATTERED_EDITS = "scattered_edits"        # changes in multiple non-adjacent locations

class EditPosition(Enum):
    TOP = "top"          # first 10% of paragraphs
    MIDDLE = "middle"    # middle of document
    BOTTOM = "bottom"    # last 10% of paragraphs

@dataclass
class EditSpec:
    edit_type: EditType
    position: EditPosition
    magnitude: int  # number of paragraphs affected (1=small, 3-5=medium, 10+=large)

def apply_edit(
    text: str,
    edit_spec: EditSpec,
    seed: int | None = None,
) -> str:
    """Apply a specified edit to the document text. Returns the modified text."""
```

**Implementation details for each edit type:**

- **Typo fix**: Pick a paragraph at the specified position. Replace a random word with a different random word. Paragraph count unchanged, 1 paragraph hash changes.
- **Sentence addition**: Pick a paragraph at the specified position. Append 1-2 random sentences (50-100 chars). Paragraph count unchanged, 1 paragraph hash changes.
- **Paragraph insert**: Insert `magnitude` new paragraphs at the specified position. Paragraph count increases.
- **Paragraph delete**: Remove `magnitude` paragraphs at the specified position. Paragraph count decreases.
- **Section rewrite**: Replace `magnitude` consecutive paragraphs with new paragraphs of similar total length. Paragraph count roughly unchanged, but all replaced paragraph hashes are new.
- **Section insert**: Insert `magnitude` new paragraphs as a block at the specified position.
- **Append**: Add `magnitude` new paragraphs at the end of the document.
- **Scattered edits**: Apply `magnitude` typo-fix-style changes at randomly chosen non-adjacent paragraphs throughout the document.

Position mapping: `TOP` = paragraph index in `[0, 10% of paragraph count)`, `MIDDLE` = around the 50% mark, `BOTTOM` = last 10%.

Use a `random.Random(seed)` instance for reproducibility.

### Testing Strategy

Create `tests/test_edit_simulator.py`.

**Core behavior (for each edit type):**
- `TYPO_FIX`: document has same paragraph count, exactly 1 paragraph differs
- `SENTENCE_ADDITION`: same paragraph count, 1 paragraph is longer
- `PARAGRAPH_INSERT` with magnitude=2: paragraph count increases by 2
- `PARAGRAPH_DELETE` with magnitude=2: paragraph count decreases by 2
- `SECTION_REWRITE` with magnitude=3: paragraph count roughly the same, 3 paragraphs have new hashes
- `SECTION_INSERT` with magnitude=3: paragraph count increases by 3
- `APPEND` with magnitude=2: paragraph count increases by 2, last 2 paragraphs are new
- `SCATTERED_EDITS` with magnitude=3: 3 non-adjacent paragraphs have changed hashes

**Position:**
- `TOP` edits affect paragraphs near the beginning
- `BOTTOM` edits affect paragraphs near the end
- `MIDDLE` edits affect paragraphs near the center

**Edge cases:**
- Edit on a very short document (e.g., 2 paragraphs) — deleting more paragraphs than exist should be handled gracefully (delete what's available, or clamp magnitude)
- Edit on a structureless blob (1 paragraph) — paragraph insert should create paragraph breaks
- Magnitude of 0 returns document unchanged
- Scattered edits with magnitude larger than paragraph count is handled gracefully

**Determinism:**
- Same seed produces identical edits
- Different seeds produce different edits

---

## Milestone 5: Experiment Runner & Results Collection

### Goal & Outcome

Build the experiment runner that executes the simulation matrix and collects results into a structured format for analysis.

After this milestone:
- The full simulation matrix can be run with a single command
- Results are collected into a structured format (dataframe/CSV)
- Edit chain experiments (sequential edits measuring fragmentation over time) run alongside single-edit experiments
- Progress is reported during long runs

### Implementation Outline

Create `src/adire/experiment.py`.

**Experiment configuration:**

```python
@dataclass
class ExperimentConfig:
    """Configuration for a simulation run."""
    document_sizes: list[int]              # target char counts
    document_profiles: list[DocumentProfile]
    edit_types: list[EditType]
    edit_positions: list[EditPosition]
    edit_magnitudes: list[int]
    max_tokens: int                         # chunk target size
    trials_per_combo: int                   # statistical trials per combination
    chain_length: int                       # number of sequential edits for chain experiments
    seed: int                               # base seed for reproducibility
```

**Smoke config** (fast validation — use during development and CI):

```python
SMOKE_CONFIG = ExperimentConfig(
    document_sizes=[5_000, 50_000],
    document_profiles=[mixed, long_paragraphs],
    edit_types=[TYPO_FIX, PARAGRAPH_INSERT, SECTION_REWRITE],
    edit_positions=[MIDDLE],
    edit_magnitudes=[1],
    max_tokens=512,
    trials_per_combo=5,
    chain_length=5,
    seed=42,
)
```

**Full config** (the complete sweep — run explicitly with `--full`):

```python
FULL_CONFIG = ExperimentConfig(
    document_sizes=[2_000, 5_000, 25_000, 50_000, 100_000],
    document_profiles=[short_paragraphs, mixed, long_paragraphs, oversized_paragraphs, structureless_blob, bimodal],
    edit_types=[all 8 types],
    edit_positions=[TOP, MIDDLE, BOTTOM],
    edit_magnitudes=[1, 3, 10],
    max_tokens=512,
    trials_per_combo=100,
    chain_length=20,
    seed=42,
)
```

The CLI default is the smoke config. Use `--full` for the complete sweep.

**Result collection:**

Each trial produces one row per strategy:

```python
@dataclass
class TrialResult:
    # Experiment parameters
    document_size: int
    document_profile: str
    edit_type: str
    edit_position: str
    edit_magnitude: int
    trial_number: int
    strategy: str  # "naive", "paragraph_reuse", "chunk_hash", "adire", "adire_wide_window"
    max_tokens: int

    # Document stats
    paragraph_count_before: int
    paragraph_count_after: int
    chunk_count_before: int
    chunk_count_after: int

    # Strategy metrics
    chunks_reembedded: int
    tokens_reembedded: int
    chunks_preserved: int
    preservation_rate: float
    fragment_count: int
    fragment_ratio: float
    oversized_count: int
    oversized_ratio: float
```

**Runner logic:**

```python
def run_experiments(config: ExperimentConfig) -> list[TrialResult]:
    """Run the full simulation matrix."""
```

The runner should:
1. For each `(document_size, document_profile)` combination, generate one base document per trial (using `seed + trial_number` for reproducibility).
2. For each `(edit_type, edit_position, edit_magnitude)` combination, apply the edit to the base document.
3. Run all five strategies on the `(old_chunks, edited_document)` pair.
4. Record `TrialResult` for each strategy.
5. Print progress (e.g., every 1000 trials or every combination).

**Strategy 5 — ADIRE with wide window:**

In addition to the four strategies above, run a variant of ADIRE that expands the dirty region to include one neighbor chunk on each side. This means when a chunk is dirty, its immediate left and right neighbors are also marked dirty (if they exist), producing a wider re-chunk region.

This trades higher re-embedding cost for better chunk quality (fewer fragments, closer to from-scratch boundary alignment). The simulation will show whether the tradeoff is worth it — if the wide window produces significantly better chunk sizes without sacrificing too much preservation rate, it may be the better default.

Implementation: this is a small wrapper around the existing ADIRE logic. After `find_dirty_chunks` returns the dirty set, expand it by adding `chunk_index - 1` and `chunk_index + 1` for each dirty chunk (clamped to valid range). Everything else is identical.

**Edit chain runner:**

```python
def run_chain_experiments(config: ExperimentConfig) -> list[TrialResult]:
    """Run sequential edit chains to measure fragmentation accumulation."""
```

For each `(document_size, document_profile)` combination:
1. Generate a base document.
2. Apply `chain_length` sequential edits (randomly chosen edit types, weighted by realistic frequency).
3. After each edit, run ADIRE and record metrics (including fragment_ratio).
4. This tests whether fragmentation accumulates to the defrag threshold.

Suggested realistic edit weights: typo_fix=30%, sentence_addition=25%, paragraph_insert=15%, paragraph_delete=5%, section_rewrite=5%, append=15%, scattered_edits=5%.

**Output:** Save results as CSV using the standard library `csv` module (avoid adding pandas as a dependency just for this). The analysis/visualization step can use pandas if needed.

**CLI entry point:** Update `src/app.py` (or replace it) with a click command to run experiments:

```
uv run python src/app.py run --output results.csv
uv run python src/app.py run --config custom_config.json --output results.csv
```

### Testing Strategy

Create `tests/test_experiment.py`.

**Core behavior:**
- Running a minimal config (1 size, 1 profile, 1 edit type, 1 position, 1 magnitude, 5 trials) produces the expected number of result rows (5 trials x 5 strategies = 25 rows)
- Results contain all expected fields populated
- Naive strategy always shows 0% preservation rate in results
- ADIRE shows non-zero preservation rate for small edits on multi-chunk documents

**Edit chains:**
- Chain of 5 edits produces 5 result rows per strategy
- Fragment ratio is tracked across the chain
- Chunks from the previous edit are used as input to the next edit (not re-chunked from scratch)

**Reproducibility:**
- Same config + seed produces identical results
- Results CSV can be written and read back correctly

**Integration:**
- Full pipeline test: generate document -> apply edit -> run all strategies -> verify results are sane (preservation rates are in [0, 1], token counts are positive, etc.)

---

## Milestone 6: Analysis & Visualization

### Goal & Outcome

Build analysis and visualization for experiment results in a Jupyter notebook.

After this milestone:
- Key findings are visualized as charts (preservation rate by dimension, fragmentation over edit chains, etc.)
- Summary statistics are computed for each dimension
- Charts are saved as image files and/or displayed in a notebook

### Implementation Outline

Add `pandas` and `matplotlib` as dev dependencies (`uv add pandas matplotlib --group dev`).

Create a Jupyter notebook `notebooks/analysis.ipynb`. Helper functions for reusable analysis logic can go in `src/adire/analysis.py`, but keep it lightweight — the notebook is the primary interface, not a CLI tool.

**Key visualizations:**

1. **Token savings rate by document size** (grouped bar chart, one bar per strategy): The headline chart. Shows where ADIRE starts to matter. Use token_savings_rate as the primary metric since it's comparable across all strategies regardless of chunking granularity.

2. **Preservation rate by document size** (grouped bar chart, primary strategies only — naive, chunk-hash, ADIRE): Apples-to-apples comparison of the same-granularity strategies.

3. **Preservation rate by document structure** (grouped bar chart): Shows the structureless blob degradation and which structures benefit most.

4. **Preservation rate by edit type** (grouped bar chart): Shows which edits benefit most from ADIRE (hypothesis: typo fix and append are near 100%, section rewrite is lower).

5. **Token savings heatmap** (document size x edit type): The primary result visualization — shows the full interaction between size and edit type.

6. **Cascade effect comparison** (chunk-hash vs. ADIRE, by edit position): Demonstrates the cascade problem — chunk-hash preservation should drop significantly for top-of-document edits while ADIRE remains stable.

7. **Fragmentation over edit chains** (line chart, fragment_ratio over edit number): Shows whether fragmentation accumulates and when the defrag threshold would trigger.

8. **Chunk size distribution comparison** (ADIRE vs. from-scratch): For each trial, the naive strategy already produces the "optimal" from-scratch chunking. Compare ADIRE's chunk size distribution (mean, median, std of token counts) against naive's for the same document. This quantifies how much ADIRE's boundary preservation diverges from optimal without needing actual embeddings.

9. **Cost savings estimate** (derived chart): Using approximate embedding costs (e.g., $0.00002 per 1K tokens for a typical embedding model), compute estimated dollar savings per edit across document sizes.

**The notebook should:**
- Load the CSV results
- Produce each visualization
- Include brief narrative interpretation of each chart
- Summarize the key findings (does ADIRE meet the hypothesis from the design doc?)

### Testing Strategy

Create `tests/test_analysis.py`.

**Core behavior:**
- Analysis helper functions accept a dataframe and return expected output without errors
- Summary statistics computation produces expected columns and no NaN values for required fields
- Cost estimation calculation is correct (spot-check with known inputs)

Keep visualization tests lightweight — test that functions run without errors and produce the expected output types, not pixel-level image comparison.

---

## Notes for the Implementing Agent

- **No external embedding API calls** are needed for Milestones 1-5. The simulation counts which chunks *would* be re-embedded. Only a future search-quality validation (not in this plan) would need actual embeddings.
- **Keep it simple.** The value is in the experimental results, not in a polished library. Prioritize correctness and clear code over abstraction.
- **The design doc is the spec.** When in doubt about algorithm behavior, refer to the pseudocode and walkthrough example in `docs/adire-anchor-diffed-incremental-re-embedding.md`.
- **Ask clarifying questions** rather than guessing, especially around: edge cases in the ADIRE dirty-region mapping, how to handle edits that span chunk boundaries, and what "adjacent chunk" means for insertions at document boundaries.
