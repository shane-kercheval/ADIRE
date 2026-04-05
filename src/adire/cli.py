"""CLI entry point for running ADIRE experiments."""

from __future__ import annotations

import click

from adire.experiment import (
    FULL_CONFIG,
    SMOKE_CONFIG,
    run_chain_experiments,
    run_experiments,
    write_results,
)


@click.group()
def main() -> None:
    """ADIRE simulation framework."""


@main.command()
@click.option("--full", is_flag=True, help="Run the full experiment matrix (slow).")
@click.option("--output", "-o", default="results_smoke.parquet", help="Output Parquet path.")
def run(full: bool, output: str) -> None:
    """Run the simulation experiments."""
    config = FULL_CONFIG if full else SMOKE_CONFIG
    mode = "full" if full else "smoke"
    click.echo(f"Running {mode} experiments...")

    click.echo("Running single-edit experiments...")
    results = run_experiments(config)
    click.echo(f"  {len(results)} single-edit results collected.")

    click.echo("Running edit chain experiments...")
    chain_results = run_chain_experiments(config)
    click.echo(f"  {len(chain_results)} chain results collected.")

    all_results = results + chain_results
    write_results(all_results, output)
    click.echo(f"Results written to {output} ({len(all_results)} rows).")


if __name__ == "__main__":
    main()
