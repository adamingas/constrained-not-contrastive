"""Run a small sample unless you ask for more."""

from datetime import datetime, timezone
import json
from pathlib import Path

import click

from .data import ROOT, FIXTURES, read, sha, write


@click.group()
def cli():
    """CLM versus constrained Qwen on an Apple Silicon Mac."""


@cli.command()
def download():
    """Download the two exact model revisions (about 16.5 GB)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt",
                    revision="e939398d4556fcd9400c76fa8c5a513202f42b0a", local_dir=ROOT / "models/clm")
    snapshot_download("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], local_dir=ROOT / "models/qwen3-8b")


def start(output, benchmark, options):
    if output.exists():
        raise click.ClickException(f"{output} already exists. Use a new output directory; runs never overwrite or resume caches.")
    if not (ROOT / "upstream/src/clm/engine.py").exists():
        raise click.ClickException("Initialize upstream first: git submodule update --init --recursive")
    if not (ROOT / "models/qwen3-8b/config.json").exists() or not (ROOT / "models/clm/CLM_v0.1-8B.pt").exists():
        raise click.ClickException("Run `uv run compare download` first.")
    from .runtime import Runtime
    output.mkdir(parents=True)
    rt = Runtime()
    write(output / "run.json", {"benchmark": benchmark, "options": options,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime": rt.metadata(),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")},
        "fixture_sha256": {p.name: sha(p) for p in FIXTURES.iterdir() if p.is_file()}})
    return rt


@cli.command()
@click.option("--limit", type=click.IntRange(1, 500), default=5, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default="runs/bfcl", show_default=True)
def bfcl(limit, output):
    """Compare tool choices on BFCL. Both caches start empty."""
    from .bfcl import run
    rt = start(output, "bfcl", {"limit": limit})
    click.echo(json.dumps(run(rt, output, limit), indent=2))


@cli.command()
@click.option("--seed", "seeds", type=int, multiple=True, default=(42,), show_default=True)
@click.option("--seconds", type=click.FloatRange(.1, 30), default=1., show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default="runs/trex", show_default=True)
def trex(seeds, seconds, output):
    """Play T-Rex with each model’s prompt cached before the game."""
    if abs(seconds * 10 - round(seconds * 10)) > 1e-6:
        raise click.BadParameter("Use a multiple of 0.1 seconds (six frames).", param_hint="--seconds")
    if len(set(seeds)) != len(seeds):
        raise click.BadParameter("Seeds must be distinct.", param_hint="--seed")
    from .trex import run
    rt = start(output, "trex", {"seeds": seeds, "seconds": seconds})
    click.echo(json.dumps(run(rt, output, seeds, seconds), indent=2))


@cli.command()
@click.option("--routes", type=click.IntRange(1, 5), default=1, show_default=True)
@click.option("--clicks", type=click.IntRange(1, 8), default=1, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default="runs/wiki", show_default=True)
def wiki(routes, clicks, output):
    """Follow Wikipedia links to reach a target article."""
    from .wiki import run
    rt = start(output, "wiki", {"routes": routes, "clicks": clicks})
    click.echo(json.dumps(run(rt, output, routes, clicks), indent=2))


@cli.command()
def results():
    """Calculate the website’s results from the saved decisions without running models."""
    from .bfcl import summarize
    published = ROOT / "results/published"
    click.echo("BFCL\n" + json.dumps(summarize(read(published / "bfcl_predictions.json.gz")), indent=2))
    episodes = read(published / "wiki.json.gz")
    click.echo("Wikipedia\n" + json.dumps({m: sum(r["success"] for r in episodes if r["variant"] == v)
        for m, v in (("clm", "clm_base"), ("qwen", "qwen_raw_base"))}, indent=2))
    from .data import timing
    for method in ("clm", "qwen"):
        traces = read(published / f"trex_{method}_trace.json.gz")["episodes"]
        click.echo(f"T-Rex {method}\n" + json.dumps(timing([r["ms"] for e in traces for r in e["records"]]), indent=2))


if __name__ == "__main__":
    cli()
