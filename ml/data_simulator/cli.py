"""
ReelMind Data Simulator CLI — Command-line interface for synthetic data generation.

Usage:
    python -m ml.data_simulator.cli --num-users 50000 --num-videos 100000 --num-interactions 10000000
    python -m ml.data_simulator.cli --preset small
    python -m ml.data_simulator.cli --preset large --output-dir ./data/large
"""

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from .generator import DataGenerator, GeneratorConfig

console = Console()

PRESETS = {
    "tiny": GeneratorConfig(
        num_users=1_000, num_videos=2_000, num_interactions=20_000,
        num_creators=200, simulation_days=7,
    ),
    "small": GeneratorConfig(
        num_users=5_000, num_videos=10_000, num_interactions=100_000,
        num_creators=500, simulation_days=14,
    ),
    "medium": GeneratorConfig(
        num_users=20_000, num_videos=50_000, num_interactions=1_000_000,
        num_creators=2_000, simulation_days=30,
    ),
    "large": GeneratorConfig(
        num_users=50_000, num_videos=100_000, num_interactions=10_000_000,
        num_creators=5_000, simulation_days=30,
    ),
}


@click.command()
@click.option("--preset", type=click.Choice(["tiny", "small", "medium", "large"]), default=None)
@click.option("--num-users", type=int, default=None)
@click.option("--num-videos", type=int, default=None)
@click.option("--num-interactions", type=int, default=None)
@click.option("--num-creators", type=int, default=None)
@click.option("--simulation-days", type=int, default=30)
@click.option("--seed", type=int, default=42)
@click.option("--output-dir", type=str, default="./data")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    preset: str | None,
    num_users: int | None,
    num_videos: int | None,
    num_interactions: int | None,
    num_creators: int | None,
    simulation_days: int,
    seed: int,
    output_dir: str,
    verbose: bool,
) -> None:
    """Generate synthetic recommendation system training data."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    if preset:
        config = PRESETS[preset]
        config = GeneratorConfig(
            num_users=num_users or config.num_users,
            num_videos=num_videos or config.num_videos,
            num_interactions=num_interactions or config.num_interactions,
            num_creators=num_creators or config.num_creators,
            simulation_days=simulation_days,
            seed=seed,
            output_dir=output_dir,
        )
    else:
        config = GeneratorConfig(
            num_users=num_users or 5_000,
            num_videos=num_videos or 10_000,
            num_interactions=num_interactions or 100_000,
            num_creators=num_creators or 500,
            simulation_days=simulation_days,
            seed=seed,
            output_dir=output_dir,
        )

    # Display config
    console.print("\n[bold cyan]ReelMind Data Generator[/bold cyan]")
    console.print("=" * 50)

    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Users", f"{config.num_users:,}")
    config_table.add_row("Creators", f"{config.num_creators:,}")
    config_table.add_row("Videos", f"{config.num_videos:,}")
    config_table.add_row("Interactions", f"{config.num_interactions:,}")
    config_table.add_row("Simulation Days", str(config.simulation_days))
    config_table.add_row("Seed", str(config.seed))
    config_table.add_row("Output", config.output_dir)
    console.print(config_table)
    console.print()

    # Generate
    generator = DataGenerator(config)
    generator.generate_all()

    # Save
    paths = generator.save_parquet()

    # Summary
    console.print("\n[bold green]Generation Complete![/bold green]")
    summary_table = Table(title="Output Files")
    summary_table.add_column("Entity", style="cyan")
    summary_table.add_column("Path", style="white")
    for entity, path in paths.items():
        summary_table.add_row(entity, path)
    console.print(summary_table)

    # Stats
    if generator.stats.get("event_type_distribution"):
        stats_table = Table(title="Event Distribution")
        stats_table.add_column("Event Type", style="cyan")
        stats_table.add_column("Count", style="green")
        stats_table.add_column("Percentage", style="yellow")
        total = generator.stats["total_interactions"]
        for et, count in sorted(
            generator.stats["event_type_distribution"].items(),
            key=lambda x: -x[1],
        ):
            pct = count / total * 100
            stats_table.add_row(et, f"{count:,}", f"{pct:.1f}%")
        console.print(stats_table)


if __name__ == "__main__":
    main()
