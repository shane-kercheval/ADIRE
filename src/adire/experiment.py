"""Experiment runner: execute the simulation matrix and collect results."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from adire.chunking import Chunk, document_hash, greedy_chunk, split_paragraphs
from adire.document_generator import (
    ALL_PROFILES,
    LARGE,
    LONG_PARAGRAPHS,
    MAX_SIZE,
    MEDIUM,
    MIXED,
    SMALL,
    TINY,
    DocumentProfile,
    generate_document,
)
from adire.edit_simulator import (
    POSITION_INDEPENDENT_TYPES,
    EditPosition,
    EditSpec,
    EditType,
    apply_edit,
)
from adire.strategies import (
    UpdateResult,
    adire_rechunk,
    adire_wide_window_rechunk,
    chunk_hash_rechunk,
    naive_rechunk,
    paragraph_reuse_rechunk,
)

STRATEGIES: list[tuple[str, type]] = [
    ("naive", naive_rechunk),
    ("paragraph_reuse", paragraph_reuse_rechunk),
    ("chunk_hash", chunk_hash_rechunk),
    ("adire", adire_rechunk),
    ("adire_wide_window", adire_wide_window_rechunk),
]

# Realistic edit weights for chain experiments.
CHAIN_EDIT_WEIGHTS: list[tuple[EditType, float]] = [
    (EditType.TYPO_FIX, 0.30),
    (EditType.SENTENCE_ADDITION, 0.25),
    (EditType.PARAGRAPH_INSERT, 0.15),
    (EditType.PARAGRAPH_DELETE, 0.05),
    (EditType.SECTION_REWRITE, 0.05),
    (EditType.APPEND, 0.15),
    (EditType.SCATTERED_EDITS, 0.05),
]


@dataclass
class ExperimentConfig:
    """Configuration for a simulation run."""

    document_sizes: list[int]
    document_profiles: list[DocumentProfile]
    edit_types: list[EditType]
    edit_positions: list[EditPosition]
    edit_magnitudes: list[int]
    max_tokens: int
    trials_per_combo: int
    chain_length: int
    seed: int
    embedding_batch_size: int
    per_batch_latency_ms: float


SMOKE_CONFIG = ExperimentConfig(
    document_sizes=[SMALL, LARGE],
    document_profiles=[MIXED, LONG_PARAGRAPHS],
    edit_types=[EditType.TYPO_FIX, EditType.PARAGRAPH_INSERT, EditType.SECTION_REWRITE],
    edit_positions=[EditPosition.MIDDLE],
    edit_magnitudes=[1],
    max_tokens=512,
    trials_per_combo=5,
    chain_length=5,
    seed=42,
    embedding_batch_size=100,
    per_batch_latency_ms=200.0,
)

FULL_CONFIG = ExperimentConfig(
    document_sizes=[TINY, SMALL, MEDIUM, LARGE, MAX_SIZE],
    document_profiles=ALL_PROFILES,
    edit_types=list(EditType),
    edit_positions=list(EditPosition),
    edit_magnitudes=[1, 3, 10],
    max_tokens=512,
    trials_per_combo=100,
    chain_length=20,
    seed=42,
    embedding_batch_size=100,
    per_batch_latency_ms=200.0,
)


@dataclass
class TrialResult:
    """Result of one strategy on one trial."""

    experiment_type: str  # "single" or "chain"
    document_size: int
    document_profile: str
    edit_type: str
    edit_position: str
    edit_magnitude: int
    trial_number: int
    strategy: str
    max_tokens: int

    paragraph_count_before: int
    paragraph_count_after: int
    chunk_count_before: int
    chunk_count_after: int

    chunks_reembedded: int
    tokens_reembedded: int
    chunks_reused: int
    preservation_rate: float
    reembedding_rate: float
    token_savings_rate: float
    fragment_count: int
    fragment_ratio: float
    oversized_count: int
    oversized_ratio: float

    algorithm_time_ms: float
    estimated_api_batches: int
    estimated_api_latency_ms: float
    estimated_total_latency_ms: float


def _make_trial_result(
    strategy_name: str,
    result: UpdateResult,
    config: ExperimentConfig,
    *,
    experiment_type: str,
    document_size: int,
    document_profile: str,
    edit_type: str,
    edit_position: str,
    edit_magnitude: int,
    trial_number: int,
    paragraph_count_before: int,
    paragraph_count_after: int,
    chunk_count_before: int,
) -> TrialResult:
    """Build a TrialResult from a strategy's UpdateResult plus experiment metadata."""
    batches = (
        math.ceil(result.chunks_reembedded / config.embedding_batch_size)
        if result.chunks_reembedded > 0
        else 0
    )
    api_latency = batches * config.per_batch_latency_ms

    return TrialResult(
        experiment_type=experiment_type,
        document_size=document_size,
        document_profile=document_profile,
        edit_type=edit_type,
        edit_position=edit_position,
        edit_magnitude=edit_magnitude,
        trial_number=trial_number,
        strategy=strategy_name,
        max_tokens=config.max_tokens,
        paragraph_count_before=paragraph_count_before,
        paragraph_count_after=paragraph_count_after,
        chunk_count_before=chunk_count_before,
        chunk_count_after=len(result.chunks),
        chunks_reembedded=result.chunks_reembedded,
        tokens_reembedded=result.tokens_reembedded,
        chunks_reused=result.chunks_reused,
        preservation_rate=result.preservation_rate,
        reembedding_rate=result.reembedding_rate,
        token_savings_rate=result.token_savings_rate,
        fragment_count=result.fragment_count,
        fragment_ratio=result.fragment_ratio,
        oversized_count=result.oversized_count,
        oversized_ratio=result.oversized_ratio,
        algorithm_time_ms=result.algorithm_time_ms,
        estimated_api_batches=batches,
        estimated_api_latency_ms=api_latency,
        estimated_total_latency_ms=result.algorithm_time_ms + api_latency,
    )


def _run_strategies(
    edited: str,
    old_chunks: list[Chunk],
    old_doc_hash: str,
    config: ExperimentConfig,
    *,
    experiment_type: str,
    document_size: int,
    document_profile: str,
    edit_type: str,
    edit_position: str,
    edit_magnitude: int,
    trial_number: int,
    paragraph_count_before: int,
    paragraph_count_after: int,
    chunk_count_before: int,
) -> list[TrialResult]:
    """Run all 5 strategies on one (old_chunks, edited) pair and return results."""
    trial_results: list[TrialResult] = []
    for strategy_name, strategy_fn in STRATEGIES:
        kwargs = {
            "new_text": edited,
            "old_chunks": old_chunks,
            "max_tokens": config.max_tokens,
        }
        if strategy_name in ("adire", "adire_wide_window"):
            kwargs["old_doc_hash"] = old_doc_hash
        update_result = strategy_fn(**kwargs)
        trial_results.append(
            _make_trial_result(
                strategy_name,
                update_result,
                config,
                experiment_type=experiment_type,
                document_size=document_size,
                document_profile=document_profile,
                edit_type=edit_type,
                edit_position=edit_position,
                edit_magnitude=edit_magnitude,
                trial_number=trial_number,
                paragraph_count_before=paragraph_count_before,
                paragraph_count_after=paragraph_count_after,
                chunk_count_before=chunk_count_before,
            ),
        )
    return trial_results


def run_experiments(config: ExperimentConfig) -> list[TrialResult]:
    """Run the full simulation matrix."""
    results: list[TrialResult] = []
    combo_count = 0

    for doc_size in config.document_sizes:
        for profile in config.document_profiles:
            for edit_type in config.edit_types:
                positions = (
                    [EditPosition.MIDDLE]
                    if edit_type in POSITION_INDEPENDENT_TYPES
                    else config.edit_positions
                )
                for edit_pos in positions:
                    for magnitude in config.edit_magnitudes:
                        combo_count += 1

                        for trial in range(config.trials_per_combo):
                            trial_seed = config.seed + trial
                            doc = generate_document(
                                profile, doc_size, seed=trial_seed,
                            )
                            old_paras = split_paragraphs(doc)
                            old_chunks = greedy_chunk(
                                old_paras, config.max_tokens,
                            )
                            old_doc_hash = document_hash(doc)

                            edit_spec = EditSpec(
                                edit_type, edit_pos, magnitude,
                            )
                            edited = apply_edit(
                                doc, edit_spec, seed=trial_seed + 1000,
                            )
                            new_paras = split_paragraphs(edited)

                            results.extend(
                                _run_strategies(
                                    edited, old_chunks, old_doc_hash,
                                    config,
                                    experiment_type="single",
                                    document_size=doc_size,
                                    document_profile=profile.name,
                                    edit_type=edit_type.value,
                                    edit_position=edit_pos.value,
                                    edit_magnitude=magnitude,
                                    trial_number=trial,
                                    paragraph_count_before=len(old_paras),
                                    paragraph_count_after=len(new_paras),
                                    chunk_count_before=len(old_chunks),
                                ),
                            )

            print(
                f"  {profile.name} {doc_size}: "
                f"{combo_count} combos done",
            )

    return results


def run_chain_experiments(config: ExperimentConfig) -> list[TrialResult]:
    """Run sequential edit chains to measure fragmentation accumulation."""
    results: list[TrialResult] = []

    # Intentionally shared across (doc_size, profile) pairs: using the same edit
    # sequence makes results comparable across document configurations by isolating
    # the effect of document properties from the effect of edit sequence variation.
    rng = random.Random(config.seed)

    edit_types = [et for et, _ in CHAIN_EDIT_WEIGHTS]
    edit_weights = [w for _, w in CHAIN_EDIT_WEIGHTS]

    for doc_size in config.document_sizes:
        for profile in config.document_profiles:
            doc = generate_document(profile, doc_size, seed=config.seed)
            old_paras = split_paragraphs(doc)
            old_chunks = greedy_chunk(old_paras, config.max_tokens)
            current_doc = doc

            strategy_chunks: dict[str, list[Chunk]] = {
                name: list(old_chunks) for name, _ in STRATEGIES
            }

            for step in range(config.chain_length):
                edit_type = rng.choices(
                    edit_types, weights=edit_weights, k=1,
                )[0]
                position = rng.choice(list(EditPosition))
                magnitude = rng.choice([1, 1, 1, 2, 3])

                # Normalize position for position-independent edit types
                if edit_type in POSITION_INDEPENDENT_TYPES:
                    position = EditPosition.MIDDLE

                edit_spec = EditSpec(edit_type, position, magnitude)
                edited = apply_edit(
                    current_doc, edit_spec, seed=config.seed + step,
                )
                new_paras = split_paragraphs(edited)
                para_count_before = len(split_paragraphs(current_doc))
                step_doc_hash = document_hash(current_doc)

                for strategy_name, strategy_fn in STRATEGIES:
                    s_old_chunks = strategy_chunks[strategy_name]
                    kwargs = {
                        "new_text": edited,
                        "old_chunks": s_old_chunks,
                        "max_tokens": config.max_tokens,
                    }
                    if strategy_name in ("adire", "adire_wide_window"):
                        kwargs["old_doc_hash"] = step_doc_hash
                    update_result = strategy_fn(**kwargs)
                    strategy_chunks[strategy_name] = update_result.chunks

                    results.append(
                        _make_trial_result(
                            strategy_name,
                            update_result,
                            config,
                            experiment_type="chain",
                            document_size=doc_size,
                            document_profile=profile.name,
                            edit_type=edit_type.value,
                            edit_position=position.value,
                            edit_magnitude=magnitude,
                            trial_number=step,
                            paragraph_count_before=para_count_before,
                            paragraph_count_after=len(new_paras),
                            chunk_count_before=len(s_old_chunks),
                        ),
                    )

                current_doc = edited

            print(
                f"  chain {profile.name} {doc_size}: "
                f"{config.chain_length} steps done",
            )

    return results


def write_results_csv(results: list[TrialResult], path: str | Path) -> None:
    """Write trial results to a CSV file."""
    if not results:
        return
    fieldnames = [f.name for f in fields(TrialResult)]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def read_results_csv(path: str | Path) -> list[dict[str, str]]:
    """Read trial results from a CSV file. Returns list of dicts (all values as strings)."""
    with Path(path).open() as f:
        return list(csv.DictReader(f))
