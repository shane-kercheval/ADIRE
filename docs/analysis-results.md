# ADIRE Simulation Results & Recommendations

Analysis of 903,000 simulation trials comparing five re-embedding strategies across document sizes (2K-100K chars), document structures (6 profiles), edit types (8 types), edit positions, magnitudes, and 20-step edit chains.

Full visualizations are in `notebooks/analysis.ipynb`.

## Strategy Overview

| Strategy | Cascade Resistant | Chunk Size Control | Complexity |
|----------|:-:|:-:|---|
| Naive | N/A (re-embeds everything) | Optimal | None |
| Paragraph Reuse | Yes | Uncontrolled (1 para = 1 chunk) | Low |
| Chunk-Hash Match | No | Optimal (~512 tokens) | Low |
| ADIRE | Yes | Optimal (~512 tokens) | Medium |
| ADIRE (Wide Window) | Yes | Optimal (~512 tokens) | Medium |

## Token Savings by Document Size

Mean token savings rate across all edit types and profiles:

| Size | Paragraph Reuse | ADIRE | Chunk-Hash | ADIRE (Wide) | Naive |
|------|:-:|:-:|:-:|:-:|:-:|
| 2K | 60.5% | 7.6% | 13.3% | 7.6% | 0% |
| 5K | 73.8% | 51.9% | 50.8% | 20.0% | 0% |
| 25K | 86.4% | 77.4% | 73.1% | 68.2% | 0% |
| 50K | 88.8% | 81.7% | 78.2% | 75.8% | 0% |
| 100K | 90.1% | 84.1% | 82.0% | 81.0% | 0% |

**Document size is the dominant factor.** At 2K characters, documents have only a few chunks, so any edit touches a large fraction. Incremental strategies become worthwhile at 5K+ and deliver strong savings at 25K+.

## Realistic Scenarios

Filtered to specific (profile, size, edit type) combinations that represent real usage patterns:

### Typical Note (mixed paragraphs, 25K-50K, common edits)

| Strategy | Token Savings |
|----------|:-:|
| Paragraph Reuse | 97.8% |
| ADIRE | 93.4% |
| Chunk-Hash Match | 82.9% |
| ADIRE (Wide) | 84.9% |

### Power User Note (mixed paragraphs, 100K, all edit types)

| Strategy | Token Savings |
|----------|:-:|
| Paragraph Reuse | 98.8% |
| ADIRE | 97.1% |
| Chunk-Hash Match | 90.0% |
| ADIRE (Wide) | 93.3% |

### Bullet-Heavy Note (short paragraphs, 5K-25K, common edits)

| Strategy | Token Savings |
|----------|:-:|
| Paragraph Reuse | 94.4% |
| ADIRE | 76.4% |
| Chunk-Hash Match | 60.4% |
| ADIRE (Wide) | 48.9% |

### Pasted Content (blob + oversized paragraphs, all sizes)

| Strategy | Token Savings |
|----------|:-:|
| Paragraph Reuse | 60.9% |
| Chunk-Hash Match | 59.3% |
| ADIRE | 44.1% |
| ADIRE (Wide) | 36.7% |

## Key Findings

### 1. Paragraph Reuse wins on token savings in every scenario

Paragraph Reuse saves the most tokens across all document sizes, profiles, and edit types. It is also cascade resistant — paragraph hashes are independent of each other, so inserting content at the top doesn't invalidate downstream hashes.

### 2. The cascade problem is real

Chunk-hash match drops to ~50% preservation when edits happen near the top of a document (vs ~63% for edits at the bottom). ADIRE stays stable at ~58-60% regardless of edit position. This confirms that greedy combining creates a positional dependency that hash-based matching can't overcome.

### 3. ADIRE's value is chunk size control, not cascade resistance alone

Both Paragraph Reuse and ADIRE are cascade resistant. ADIRE's differentiator is that it produces well-sized ~512-token chunks via greedy combining, while Paragraph Reuse creates one chunk per paragraph with uncontrolled sizes. A 20-character heading and a 1500-character prose block both become individual chunks under Paragraph Reuse.

### 4. ADIRE (Wide Window) is not worth it

The wide window variant consistently underperforms standard ADIRE on token savings (it re-embeds neighboring chunks for better boundary alignment). The payoff — reduced fragmentation — doesn't materialize: fragment ratios for standard ADIRE are nearly identical to naive (~0.9% at 100K), and neither variant crosses the 30% defrag threshold during 20-step chains.

### 5. Structureless content defeats paragraph-level strategies

For blob and oversized-paragraph documents, all paragraph-based strategies (including ADIRE) degrade because there are no paragraph boundaries to anchor on. Chunk-hash match slightly outperforms ADIRE in this case (59% vs 44%). Paragraph Reuse still edges ahead (61%) because its single-paragraph chunks happen to be large enough to remain stable.

### 6. Cost savings scale with document size and edit frequency

Per-edit savings are small in absolute terms (~$0.0004 per edit at 100K chars with `text-embedding-3-small` pricing). The value compounds with many users and frequent edits.

### 7. Latency differences are marginal

Algorithm CPU time is 1-3ms for all strategies. Total latency (algorithm + estimated API latency) differs by ~10ms between naive and ADIRE at 100K characters, since embedding API batch overhead dominates.

## Recommendations

### For search/retrieval where embeddings point to parent documents

**Use Paragraph Reuse.** When the embedding's job is to match a query to the right parent document (with deduplication on the parent), chunk size uniformity matters much less. Smaller paragraph-level chunks are arguably better for this — a short heading like "Kubernetes Pod Networking" is a highly specific search signal. You get the highest token savings, cascade resistance, and a simpler implementation than ADIRE.

### For RAG where chunks are sent directly to the LLM

**Use ADIRE.** When chunks are fed directly into an LLM's context window as retrieved passages, uniform ~512-token chunks provide more consistent context. A chunk containing just a heading isn't useful as a standalone passage. ADIRE gives you cascade resistance and strong savings (77-97% at 25K+) while maintaining chunk quality.

### For applications with mostly unstructured/pasted content

**Use Chunk-Hash Match.** If documents lack paragraph structure (e.g., OCR output, code blocks, pasted prose without line breaks), paragraph-level diffing has nothing to anchor on. Chunk-hash is simpler and performs as well or better than ADIRE in this regime. Accept that edits near the top of documents will cause more re-embedding.

### For small documents (<5K chars)

**Use Naive.** All strategies save less than 15% at 2K characters. The complexity isn't justified — just re-embed everything.

### Strategy decision tree

```
Document size < 5K chars?
  → Naive (re-embed everything)

Documents are mostly unstructured blobs?
  → Chunk-Hash Match

Embeddings used for parent-doc retrieval / search?
  → Paragraph Reuse

Embeddings used as standalone passages (RAG)?
  → ADIRE
```
