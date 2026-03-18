# Anchor-Diffed Incremental Re-Embedding (ADIRE)

A technique for minimizing re-embedding in vector search systems. Instead of re-chunking and re-embedding entire documents on every edit, ADIRE tracks content at a configurable structural granularity (paragraphs, lines, blocks), diffs anchor sequences between versions, and re-embeds only the affected chunks.

Designed for edit-heavy use cases (note-taking apps, wikis, collaborative docs) where the standard "delete all chunks and re-ingest" approach is wasteful.

## How it works

1. **Split** document content into structural anchor units (e.g., paragraphs)
2. **Hash** each anchor unit independently
3. **Diff** the old and new anchor hash sequences to identify insertions, deletions, and changes
4. **Map** changed anchors back to their containing chunks to find "dirty" chunks
5. **Re-chunk and re-embed** only the dirty regions — unchanged chunks keep their existing embeddings

A document-level hash check (step 0) short-circuits the entire process when nothing changed at all.

## Key properties

- **Unchanged chunks are preserved by design**, not by coincidence — the diff determines which chunks are affected, and untouched chunks are never re-processed
- **Anchor unit is configurable** — paragraphs (`\n\n`), lines, sections, or application-defined content blocks
- **No external dependencies** for the core algorithm — just hashing and sequence diffing
- **Defrag on threshold** — when incremental edits accumulate too many undersized fragments, falls back to a clean full re-chunk

## Status

Draft / proof of concept. See [`docs/adire-anchor-diffed-incremental-re-embedding.md`](docs/adire-anchor-diffed-incremental-re-embedding.md) for the full design document, algorithm pseudocode, walkthrough examples, simulation design, and research context.

## Development

- `make tests` — run linting and tests
- `uv add <package>` — add a dependency
- `uv add <package> --group dev` — add a dev dependency
