# Anchor-Diffed Incremental Re-Embedding (ADIRE)

A technique for minimizing re-embedding in vector search systems. Instead of re-chunking and re-embedding entire documents on every edit, ADIRE tracks content at a configurable structural granularity (paragraphs, lines, blocks), diffs anchor sequences between versions, and re-embeds only the affected chunks.

Designed for edit-heavy use cases (note-taking apps, wikis, collaborative docs) where the standard "delete all chunks and re-ingest" approach is wasteful.

## The Problem

When a document is edited, the naive approach is to delete all chunks and re-embed from scratch. A common optimization — hash each chunk and skip unchanged ones — breaks down because inserting a paragraph near the top shifts all subsequent chunk boundaries (the **cascade problem**), making every downstream chunk hash different even though the content is the same.

## How ADIRE Solves It

ADIRE separates **identity tracking** (paragraph-level) from **chunk grouping** (greedy combining up to a configurable token budget, e.g., 512 tokens). The key steps:

1. **Split** document content into structural anchor units (e.g., paragraphs)
2. **Hash** each anchor unit independently
3. **Combine** paragraphs greedily into chunks up to the token budget — small paragraphs are grouped together, large paragraphs may fill a chunk on their own
4. **On edit — Diff** the old and new anchor hash sequences (using positional mapping, not hash-based lookup, to handle duplicate paragraphs correctly)
5. **Map** changed positions back to their containing chunks to identify "dirty" chunks
6. **Re-chunk and re-embed** only the dirty regions (greedily re-combine their paragraphs into chunks) — unchanged chunks keep their existing embeddings

A document-level hash check short-circuits the entire process when nothing changed at all.

## Example

A document with 7 paragraphs. Each paragraph is hashed independently, then paragraphs are greedily combined into chunks up to a token budget (e.g., 512 tokens). Small paragraphs get grouped together; a large paragraph may fill a chunk on its own.

```
                                          hash    tokens   chunk
                                          ────    ──────   ─────
How to deploy to Railway...               aa11     180 ─┐
                                                         ├─ Chunk 0  → embedded
First, create a new project...            bb22     200 ─┘

Configure your environment variables      cc33     480 ── Chunk 1  → embedded
by setting the following values...

Next, connect your GitHub repo...         dd44     190 ─┐
                                                         ├─ Chunk 2  → embedded
Finally, set up your domain...            ee55     300 ─┘

For monitoring, Railway provides...       ff66     200 ─┐
                                                         ├─ Chunk 3  → embedded
Troubleshooting: check the deploy logs... gg77     250 ─┘
```

Chunk 0 combines two small paragraphs (380 tokens). Chunk 1 is a single large paragraph (480 tokens) that nearly fills the budget on its own. Chunks 2 and 3 each combine two paragraphs.

The user inserts a new paragraph after "First, create a new project..." and edits "Next, connect your GitHub repo...":

```
                                          hash    tokens   chunk
                                          ────    ──────   ─────
How to deploy to Railway...               aa11     180 ─┐
                                                         ├─ Chunk 0  → KEPT ✓
First, create a new project...            bb22     200 ─┘

Before deploying, run your tests...       xx99     150 ─┐ ← INSERTED
                                                         ├─ Chunk 1  → re-embedded
Configure your environment variables      cc33     480 ─┘
by setting the following values...

Connect your repo via the dashboard...    7f3a     190 ─┐ ← EDITED (was dd44)
                                                         ├─ Chunk 2  → re-embedded
Finally, set up your domain...            ee55     300 ─┘

For monitoring, Railway provides...       ff66     200 ─┐
                                                         ├─ Chunk 3  → KEPT ✓
Troubleshooting: check the deploy logs... gg77     250 ─┘
```

**2 of 4 chunks preserved** — their paragraph hashes didn't change, so their embeddings are reused. Only the dirty region (chunks 1-2) is re-chunked and re-embedded. With larger documents (50+ chunks), a typical single-paragraph edit preserves 95%+ of chunks.

## This Repo

This is a **research/simulation project** that validates ADIRE empirically. The simulation compares re-embedding strategies across different document sizes, structures, edit types, and edit patterns — all without actual embedding API calls (it counts which chunks *would* be re-embedded).

### Strategies Compared

| Strategy | Reuse Mechanism | Chunk Sizes | Cascade Resistant? |
|---|---|---|---|
| **Naive** | None — re-embed everything | Optimal (greedy combining) | N/A |
| **Paragraph-level reuse** | Set lookup by paragraph hash | Uncontrolled (1 paragraph = 1 chunk) | Yes |
| **Chunk-hash match** | Re-chunk from scratch, hash-match chunks | Optimal (greedy combining) | No (cascade) |
| **ADIRE** | Positional diff + dirty regions | Optimal (greedy combining) | Yes |
| **ADIRE (wide window)** | Same as ADIRE, but includes neighbor chunks in dirty region | Optimal (fewer fragments) | Yes |

The simulation answers several questions: Does chunk-hash matching actually fail due to cascade? Does ADIRE's complexity over paragraph-level reuse justify itself (both achieve high reuse, but ADIRE produces well-sized chunks)? Does the wide window reduce fragmentation enough to matter?

### Experiment Dimensions

- **Document size**: 2K to 100K characters
- **Document structure**: short paragraphs, mixed, long paragraphs, structureless blob, bimodal
- **Edit type**: typo fix, paragraph insert/delete, section rewrite, append, scattered edits
- **Edit position**: top, middle, bottom
- **Edit magnitude**: 1, 3, or 10 paragraphs affected
- **Edit chains**: 20 sequential edits to measure fragmentation accumulation

### Primary Metrics

- **Preservation rate**: what % of chunks were reused without re-embedding
- **Re-embedding rate**: what % required an API call
- **Token savings rate**: what % of token cost was avoided (cost-weighted)
- **Fragment ratio**: chunk quality degradation over time

## Key Properties

- **Unchanged chunks are preserved by design**, not by coincidence — the diff determines which chunks are affected
- **Anchor unit is configurable** — paragraphs (`\n\n`), lines, sections, or application-defined content blocks
- **No external dependencies** for the core algorithm — just hashing and sequence diffing
- **Defrag on threshold** — when incremental edits accumulate too many undersized fragments, falls back to a clean full re-chunk

## Documentation

- [`docs/adire-anchor-diffed-incremental-re-embedding.md`](docs/adire-anchor-diffed-incremental-re-embedding.md) — full design document with algorithm pseudocode, walkthrough examples, and research context
- [`docs/implementation-plan.md`](docs/implementation-plan.md) — implementation plan for the simulation framework

## Development

- `make tests` — run linting and tests
- `uv add <package>` — add a dependency
- `uv add <package> --group dev` — add a dev dependency
