"""Analysis helpers for ADIRE experiment results."""

from __future__ import annotations

import pandas as pd

# Strategy display order and labels
STRATEGY_ORDER = ["naive", "paragraph_reuse", "chunk_hash", "adire", "adire_wide_window"]
STRATEGY_LABELS = {
    "naive": "Naive",
    "paragraph_reuse": "Paragraph Reuse",
    "chunk_hash": "Chunk-Hash Match",
    "adire": "ADIRE",
    "adire_wide_window": "ADIRE (Wide)",
}

# Strategies with the same chunking granularity (greedy combining to ~512 tokens)
SAME_GRANULARITY_STRATEGIES = ["naive", "chunk_hash", "adire", "adire_wide_window"]

# Approximate embedding cost per 1K tokens (typical embedding model, e.g. text-embedding-3-small)
COST_PER_1K_TOKENS = 0.00002


def load_results(path: str = "results.parquet") -> pd.DataFrame:
    """Load experiment results from Parquet."""
    return pd.read_parquet(path)


def single_edit_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to single-edit experiment rows only."""
    return df[df["experiment_type"] == "single"].copy()


def chain_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to chain experiment rows only."""
    return df[df["experiment_type"] == "chain"].copy()


def mean_by(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str = "token_savings_rate",
) -> pd.DataFrame:
    """Compute mean of a metric grouped by the given columns."""
    return df.groupby(group_cols, observed=True)[metric].mean().reset_index()


def pivot_for_grouped_bar(
    data: pd.DataFrame,
    index_col: str,
    metric: str = "token_savings_rate",
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot mean metrics into a strategy-column DataFrame for grouped bar charts."""
    filtered = data[data["strategy"].isin(strategies)] if strategies else data
    agg = mean_by(filtered, [index_col, "strategy"], metric)
    pivoted = agg.pivot_table(index=index_col, columns="strategy", values=metric)
    cols = [s for s in STRATEGY_ORDER if s in pivoted.columns]
    return pivoted[cols]


def heatmap_data(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric: str = "token_savings_rate",
    strategy: str = "adire",
) -> pd.DataFrame:
    """Pivot data for a heatmap: rows=row_col, columns=col_col, values=mean metric."""
    filtered = df[df["strategy"] == strategy]
    agg = mean_by(filtered, [row_col, col_col], metric)
    return agg.pivot_table(index=row_col, columns=col_col, values=metric)


def compute_cost_savings(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimated dollar cost columns to the dataframe."""
    out = df.copy()
    out["cost_naive"] = out["tokens_reembedded"].where(
        out["strategy"] == "naive", other=0,
    )
    # For cost savings, compare each strategy's tokens_reembedded against naive
    naive_tokens = df[df["strategy"] == "naive"].set_index(
        ["document_size", "document_profile", "edit_type", "edit_position",
         "edit_magnitude", "trial_number"],
    )["tokens_reembedded"]

    merge_cols = ["document_size", "document_profile", "edit_type", "edit_position",
                  "edit_magnitude", "trial_number"]
    out = out.merge(
        naive_tokens.rename("naive_tokens_reembedded"),
        left_on=merge_cols,
        right_index=True,
        how="left",
    )
    out["cost_per_edit_usd"] = out["tokens_reembedded"] / 1000 * COST_PER_1K_TOKENS
    out["naive_cost_per_edit_usd"] = out["naive_tokens_reembedded"] / 1000 * COST_PER_1K_TOKENS
    out["cost_savings_usd"] = out["naive_cost_per_edit_usd"] - out["cost_per_edit_usd"]
    return out


def format_size_label(chars: int) -> str:
    """Format character count as a readable size label."""
    if chars >= 1000:
        return f"{chars // 1000}K"
    return str(chars)
