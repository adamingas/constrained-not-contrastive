"""Read and save experiment data, and calculate timing summaries."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"


def read(path):
    path = Path(path)
    text = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
    return [json.loads(line) for line in text.splitlines()] if path.suffix == ".jsonl" else json.loads(text)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(values):
    values = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values), "p95_ms": values[math.ceil(.95 * len(values)) - 1]} if values else None
