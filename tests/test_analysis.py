"""Tests for adire.analysis module."""

import pandas as pd

from adire.analysis import (
    COST_PER_1K_TOKENS,
    STRATEGY_ORDER,
    compute_cost_savings,
    format_size_label,
    heatmap_data,
    mean_by,
    pivot_for_grouped_bar,
)


def _sample_df() -> pd.DataFrame:
    """Create a small sample dataframe mimicking experiment results."""
    rows = []
    for size in [5000, 50000]:
        for strategy in STRATEGY_ORDER:
            for trial in range(3):
                is_naive = strategy == "naive"
                rows.append({
                    "experiment_type": "single",
                    "document_size": size,
                    "document_profile": "mixed",
                    "edit_type": "typo_fix",
                    "edit_position": "middle",
                    "edit_magnitude": 1,
                    "trial_number": trial,
                    "strategy": strategy,
                    "max_tokens": 512,
                    "paragraph_count_before": 10,
                    "paragraph_count_after": 10,
                    "chunk_count_before": 3,
                    "chunk_count_after": 3,
                    "chunks_reembedded": 3 if is_naive else 1,
                    "tokens_reembedded": 1500 if is_naive else 500,
                    "chunks_reused": 0 if is_naive else 2,
                    "preservation_rate": 0.0 if is_naive else 0.67,
                    "reembedding_rate": 1.0 if is_naive else 0.33,
                    "token_savings_rate": 0.0 if is_naive else 0.67,
                    "fragment_count": 0,
                    "fragment_ratio": 0.0,
                    "oversized_count": 0,
                    "oversized_ratio": 0.0,
                    "algorithm_time_ms": 0.1,
                    "estimated_api_batches": 1,
                    "estimated_api_latency_ms": 200.0,
                    "estimated_total_latency_ms": 200.1,
                })
    return pd.DataFrame(rows)


class TestMeanBy:
    def test_groups_correctly(self):
        df = _sample_df()
        result = mean_by(df, ["document_size", "strategy"], "token_savings_rate")
        assert len(result) == 10  # 2 sizes x 5 strategies
        assert "token_savings_rate" in result.columns

    def test_no_nan_values(self):
        df = _sample_df()
        result = mean_by(df, ["document_size", "strategy"], "token_savings_rate")
        assert result["token_savings_rate"].isna().sum() == 0


class TestPivotForGroupedBar:
    def test_produces_strategy_columns(self):
        df = _sample_df()
        pivoted = pivot_for_grouped_bar(df, "document_size", "token_savings_rate")
        assert list(pivoted.columns) == STRATEGY_ORDER

    def test_filters_strategies(self):
        df = _sample_df()
        pivoted = pivot_for_grouped_bar(
            df, "document_size", "token_savings_rate",
            strategies=["naive", "adire"],
        )
        assert list(pivoted.columns) == ["naive", "adire"]


class TestHeatmapData:
    def test_produces_pivot_table(self):
        df = _sample_df()
        hm = heatmap_data(df, "document_size", "edit_type", "token_savings_rate", "adire")
        assert hm.shape[0] == 2  # 2 sizes
        assert hm.shape[1] == 1  # 1 edit type
        assert not hm.isna().any().any()


class TestComputeCostSavings:
    def test_adds_cost_columns(self):
        df = _sample_df()
        result = compute_cost_savings(df)
        assert "cost_per_edit_usd" in result.columns
        assert "naive_cost_per_edit_usd" in result.columns
        assert "cost_savings_usd" in result.columns

    def test_naive_has_zero_savings(self):
        df = _sample_df()
        result = compute_cost_savings(df)
        naive_savings = result[result["strategy"] == "naive"]["cost_savings_usd"]
        assert (naive_savings == 0.0).all()

    def test_cost_calculation_correct(self):
        df = _sample_df()
        result = compute_cost_savings(df)
        adire_row = result[(result["strategy"] == "adire") & (result["trial_number"] == 0)].iloc[0]
        expected_cost = 500 / 1000 * COST_PER_1K_TOKENS
        assert abs(adire_row["cost_per_edit_usd"] - expected_cost) < 1e-12


class TestFormatSizeLabel:
    def test_thousands(self):
        assert format_size_label(5000) == "5K"
        assert format_size_label(100000) == "100K"

    def test_small(self):
        assert format_size_label(500) == "500"
