"""Tests for adire.experiment module."""

from adire.document_generator import LONG_PARAGRAPHS, MIXED
from adire.edit_simulator import EditPosition, EditType
from adire.experiment import (
    DEFAULT_CHAIN_EDIT_WEIGHTS,
    DEFAULT_CHAIN_MAGNITUDES,
    ExperimentConfig,
    read_results_csv,
    run_chain_experiments,
    run_experiments,
    write_results_csv,
)


def _minimal_config(**overrides: object) -> ExperimentConfig:
    """Create a minimal experiment config for fast testing."""
    defaults: dict[str, object] = {
        "document_sizes": [5_000],
        "document_profiles": [MIXED],
        "max_tokens": 512,
        "seed": 42,
        "embedding_batch_size": 100,
        "per_batch_latency_ms": 200.0,
        "edit_types": [EditType.TYPO_FIX],
        "edit_positions": [EditPosition.MIDDLE],
        "edit_magnitudes": [1],
        "trials_per_combo": 5,
        "chain_length": 5,
        "chain_edit_weights": DEFAULT_CHAIN_EDIT_WEIGHTS,
        "chain_magnitudes": DEFAULT_CHAIN_MAGNITUDES,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestRunExperiments:
    def test_expected_row_count(self):
        config = _minimal_config(trials_per_combo=5)
        results = run_experiments(config)
        # 1 size x 1 profile x 1 edit_type x 1 position x 1 magnitude x 5 trials x 5 strategies
        assert len(results) == 25

    def test_multiple_dimensions(self):
        config = _minimal_config(
            edit_types=[EditType.TYPO_FIX, EditType.PARAGRAPH_INSERT],
            edit_magnitudes=[1, 3],
            trials_per_combo=2,
        )
        results = run_experiments(config)
        # 1 size x 1 profile x 2 edit_types x 1 position x 2 magnitudes x 2 trials x 5 strategies
        assert len(results) == 40

    def test_all_fields_populated(self):
        config = _minimal_config(trials_per_combo=1)
        results = run_experiments(config)
        for r in results:
            assert r.experiment_type == "single"
            assert r.document_size == 5000
            assert r.document_profile == "mixed"
            assert r.edit_type == "typo_fix"
            assert r.edit_position == "middle"
            assert r.edit_magnitude == 1
            assert r.trial_number == 0
            assert r.strategy in {"naive", "paragraph_reuse", "chunk_hash", "adire", "adire_wide_window"}
            assert r.max_tokens == 512
            assert r.paragraph_count_before > 0
            assert r.paragraph_count_after > 0
            assert r.chunk_count_before > 0
            assert r.chunk_count_after > 0
            assert 0.0 <= r.reembedding_rate <= 1.0
            assert r.algorithm_time_ms >= 0
            assert r.estimated_api_batches >= 0
            assert r.estimated_api_latency_ms >= 0
            assert r.estimated_total_latency_ms >= 0

    def test_naive_zero_preservation(self):
        config = _minimal_config(trials_per_combo=3)
        results = run_experiments(config)
        naive_results = [r for r in results if r.strategy == "naive"]
        assert len(naive_results) == 3
        for r in naive_results:
            assert r.preservation_rate == 0.0
            assert r.chunks_reused == 0

    def test_adire_nonzero_preservation(self):
        config = _minimal_config(
            document_sizes=[50_000],
            trials_per_combo=3,
        )
        results = run_experiments(config)
        adire_results = [r for r in results if r.strategy == "adire"]
        assert len(adire_results) == 3
        assert all(r.preservation_rate > 0 for r in adire_results)

    def test_position_independent_types_skip_position_variation(self):
        config = _minimal_config(
            edit_types=[EditType.APPEND],
            edit_positions=[EditPosition.TOP, EditPosition.MIDDLE, EditPosition.BOTTOM],
            trials_per_combo=1,
        )
        results = run_experiments(config)
        # APPEND is position-independent, only MIDDLE is used
        assert len(results) == 5  # 1 position x 1 trial x 5 strategies
        assert all(r.edit_position == "middle" for r in results)


# ---------------------------------------------------------------------------
# Edit chains
# ---------------------------------------------------------------------------

class TestChainExperiments:
    def test_chain_produces_expected_rows(self):
        config = _minimal_config(chain_length=5)
        results = run_chain_experiments(config)
        # 1 size x 1 profile x 5 steps x 5 strategies
        assert len(results) == 25

    def test_chain_results_have_correct_experiment_type(self):
        config = _minimal_config(chain_length=3)
        results = run_chain_experiments(config)
        assert all(r.experiment_type == "chain" for r in results)

    def test_fragment_ratio_tracked(self):
        config = _minimal_config(chain_length=5)
        results = run_chain_experiments(config)
        for r in results:
            assert 0.0 <= r.fragment_ratio <= 1.0

    def test_chain_uses_previous_output(self):
        config = _minimal_config(document_sizes=[50_000], chain_length=5)
        results = run_chain_experiments(config)
        adire_results = [r for r in results if r.strategy == "adire"]
        reused_any = any(r.chunks_reused > 0 for r in adire_results)
        assert reused_any

    def test_chain_normalizes_position_independent_types(self):
        config = _minimal_config(chain_length=20)
        results = run_chain_experiments(config)
        for r in results:
            if r.edit_type in {"append", "scattered_edits"}:
                assert r.edit_position == "middle"

    def test_chain_same_edit_sequence_across_pairs(self):
        config = _minimal_config(
            document_sizes=[5_000, 25_000],
            chain_length=5,
        )
        results = run_chain_experiments(config)
        # Group by (doc_size, step) and check edit_type is the same
        size_5k = [r for r in results if r.document_size == 5000 and r.strategy == "naive"]
        size_25k = [r for r in results if r.document_size == 25000 and r.strategy == "naive"]
        assert len(size_5k) == len(size_25k) == 5
        for a, b in zip(size_5k, size_25k):
            assert a.edit_type == b.edit_type
            assert a.edit_magnitude == b.edit_magnitude


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_results(self):
        config = _minimal_config(trials_per_combo=3)
        results_a = run_experiments(config)
        results_b = run_experiments(config)
        assert len(results_a) == len(results_b)
        for a, b in zip(results_a, results_b):
            assert a.chunks_reembedded == b.chunks_reembedded
            assert a.chunks_reused == b.chunks_reused
            assert a.tokens_reembedded == b.tokens_reembedded

    def test_csv_round_trip(self, tmp_path):
        config = _minimal_config(trials_per_combo=2)
        results = run_experiments(config)

        path = tmp_path / "results.csv"
        write_results_csv(results, path)
        rows = read_results_csv(path)
        assert len(rows) == len(results)
        assert rows[0]["strategy"] in {"naive", "paragraph_reuse", "chunk_hash", "adire", "adire_wide_window"}
        assert rows[0]["document_size"] == "5000"
        assert rows[0]["experiment_type"] == "single"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_sanity(self):
        config = _minimal_config(
            document_sizes=[5_000, 25_000],
            document_profiles=[MIXED, LONG_PARAGRAPHS],
            edit_types=[EditType.TYPO_FIX, EditType.PARAGRAPH_INSERT],
            trials_per_combo=2,
        )
        results = run_experiments(config)
        for r in results:
            assert 0.0 <= r.preservation_rate <= 1.0
            assert 0.0 <= r.reembedding_rate <= 1.0
            assert 0.0 <= r.token_savings_rate <= 1.0
            assert r.tokens_reembedded >= 0
            assert r.chunks_reembedded + r.chunks_reused == r.chunk_count_after
            assert r.estimated_total_latency_ms >= r.algorithm_time_ms
