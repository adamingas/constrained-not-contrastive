"""Try to reach a target Wikipedia article by following links."""

from collections import defaultdict
import copy
import gzip
import hashlib
import itertools
import random
from urllib.parse import unquote

from .data import FIXTURES, append, read, write

RULE = "Choose a linked Wikipedia article that helps reach the target article."
QUESTION = "Which linked article should we visit next to reach the target?"


def graph():
    links = defaultdict(set)
    with gzip.open(FIXTURES / "wiki_links.tsv.gz", "rt") as stream:
        for line in stream:
            if line.strip() and not line.startswith("#"):
                source, target = line.strip().split("\t")
                links[source].add(target)
    return {source: sorted(targets) for source, targets in links.items()}


def texts(path, target, links, labels):
    title = lambda s: unquote(s).replace("_", " ")
    keys = list(links.get(path[-1], []))
    random.Random(int(hashlib.sha256(path[-1].encode()).hexdigest()[:8], 16)).shuffle(keys)
    state = (f"Current Wikipedia article: {title(path[-1])}.\nTarget article: {title(target)}.\nVisited articles: "
             + " -> ".join(title(x) for x in path) + ".\n\n" + QUESTION)
    actions = [title(k) for k in keys]
    prefix = (RULE + "\nReply with exactly one option label, without explanation.\nAvailable options:\n"
              + "\n".join(f"{label}: {text}" for label, text in zip(labels, actions)) + "\n\nState:\n")
    return state, prefix, keys, actions


def run(rt, output, routes, clicks):
    import mlx.core as mx
    import mlx.nn as nn
    links = graph()
    labels = ["".join(cs) for n in (1, 2) for cs in itertools.product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=n)]
    labels = [s for s in labels if len(rt.tok.encode(s, add_special_tokens=False)) == 1]
    weights = {kind: {k: mx.array(v.detach().float().numpy()) for k, v in getattr(rt.head, kind + "_head").state_dict().items()}
               for kind in ("state", "action")}
    mx.eval(weights)

    def norm(x):
        x = x.astype(mx.float32)
        return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    def project(x, kind):
        # Only this older experiment used an MLX port of the released heads.
        w, cfg = weights[kind], rt.head.cfg
        assert cfg.get("activation", "gelu") == "gelu"
        x = nn.gelu(x @ w["inp.weight"].T + w["inp.bias"])
        for i in range(cfg["depth"] - 2):
            h = x @ w[f"hidden.{i}.weight"].T + w[f"hidden.{i}.bias"]
            if cfg.get("layernorm"):
                h = mx.fast.layer_norm(h, w[f"norms.{i}.weight"], w[f"norms.{i}.bias"], 1e-5)
            h = nn.gelu(h)
            x = x + h if cfg.get("residual") else h
        return norm(x @ w["out.weight"].T + w["out.bias"])

    records = []
    for start in read(FIXTURES / "wiki_routes.json")[:routes]:
        for method in ("clm", "qwen"):
            path, steps, menus, vectors = [start["src"]], [], {}, {}
            for _ in range(clicks):
                state, preamble, keys, actions = texts(path, start["target"], links, labels)
                if not keys:
                    break
                assert len(keys) <= len(labels)
                prefix_text = preamble if method == "qwen" else ""
                prompt = prefix_text + state + ("\nAnswer:" if method == "qwen" else "")
                ids = rt.tok.encode(prompt, add_special_tokens=True)
                prefix = rt.tok.encode(prefix_text, add_special_tokens=True)[:-1] if prefix_text else []
                assert ids[:len(prefix)] == prefix
                if path[-1] not in menus:
                    cache = rt.prepare(prefix, chunk=1024) if prefix else None
                    if method == "clm":
                        for action in actions:
                            if action not in vectors:
                                h = rt.forward(rt.tok.encode(action, add_special_tokens=True), chunk=1024)
                                vectors[action] = project(norm(h), "action")
                                mx.eval(vectors[action])
                        rows = mx.concatenate([vectors[a] for a in actions])
                    else:
                        rows = rt.model.lm_head.weight[mx.array(rt.labels(labels[:len(keys)]))]
                    mx.eval(rows)
                    menus[path[-1]] = (cache, rows)
                cache, rows = menus[path[-1]]
                h = rt.forward(ids[len(prefix):], copy.deepcopy(cache), chunk=1024)
                if method == "clm":
                    h = project(norm(h), "state")
                answer = rt.distribution(h, keys, rows, scale=rt.head.scale if method == "clm" else 1)
                steps.append({"src": path[-1], "exact_input": prompt, "actions": dict(zip(keys, actions)), **answer})
                path.append(answer["choice"])
                print(f"Wiki {method}: {path[-2]} -> {path[-1]}", flush=True)
                if path[-1] == start["target"]:
                    break
            row = {"id": start["id"], "method": method, "success": path[-1] == start["target"], "path": path, "steps": steps}
            append(output / "routes.jsonl", row)
            records.append(row)
    summary = {m: {"routes": routes, "successes": sum(r["success"] for r in records if r["method"] == m)} for m in ("clm", "qwen")}
    write(output / "summary.json", summary)
    return summary
