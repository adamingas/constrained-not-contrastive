"""Ask each model to choose a tool for each saved BFCL request."""

import string
import time

from .data import FIXTURES, append, read, timing, write
from clm.schema import build_pairs

RULE = ("Choose the available function that best fulfills the request. "
        "Reply with exactly one option letter, without explanation.\n\nAvailable functions:\n\n")


def prompt(row, rt):
    state, keys, actions = build_pairs(row["state"], row["questions"])["function"]
    labels = string.ascii_uppercase[:len(keys)]
    assert len(labels) == len(keys)
    system = RULE + "\n\n".join(f"{label}:\n{text}" for label, text in zip(labels, actions))
    return rt.chat(system, state), rt.preamble(system), keys, rt.labels(labels), actions


def run(rt, output, limit):
    rows = read(FIXTURES / "bfcl.jsonl")[:limit]
    rt.warm_kernels()  # Unrelated text only; both benchmark caches start empty.
    seen = set()
    for i, row in enumerate(rows):
        full, prefix, keys, labels, actions = prompt(row, rt)
        result = {"id": row["id"], "category": row["category"], "sequence": i}
        order = ("qwen", "clm") if i % 2 == 0 else ("clm", "qwen")
        result["first_method"] = order[0]
        for method in order:
            if method == "qwen":
                result[method] = rt.qwen_reuse(full, prefix, keys, labels, row["id"])
            else:
                rt.embed_calls = []
                start = time.perf_counter()
                answer = rt.engine.answer(row["state"], row["questions"])
                elapsed = (time.perf_counter() - start) * 1000
                new = [s for s in dict.fromkeys(actions) if s not in seen]
                assert len(rt.embed_calls) == 1 + len(new)
                result[method] = {**answer["answers"]["function"], "ms": elapsed,
                                  "processed_tokens": answer["usage"]["input_tokens"],
                                  "new_actions": len(new), "all_actions_cached": not new}
                seen.update(actions)
        append(output / "predictions.jsonl", result)
        print(f"BFCL {i + 1}/{len(rows)}", flush=True)
    summary = summarize(read(output / "predictions.jsonl"))
    write(output / "summary.json", summary)
    return summary


def summarize(rows):
    gold = read(FIXTURES / "bfcl_answers.json")  # Labels are used only after inference.
    assert len({r["id"] for r in rows}) == len(rows)
    cached = [r for r in rows if r["qwen"]["full_preamble_hit"] and r["clm"]["all_actions_cached"]]
    return {name: {"n": len(group), **{m: {
                "correct": sum(r[m]["choice"] == gold[r["id"]] for r in group),
                "latency": timing([r[m]["ms"] for r in group])} for m in ("clm", "qwen")}}
            for name, group in (("all", rows), ("both_cached", cached))}
