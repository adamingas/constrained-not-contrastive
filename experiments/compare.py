"""Same-backbone CLM vs one-token Qwen3 decisions on Apple Silicon.

Run from workspace root: .venv/bin/python -u experiments/compare.py
No generation loop, network inference, quantization, or hidden reasoning tokens.
"""
import copy
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/src"))
from clm.heads import HeadPair
from clm.schema import build_pairs


def norm(x):
    x = x.astype(mx.float32)
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Comparison:
    def __init__(self):
        self.model, self.tok = load(str(ROOT / "models/qwen3-8b"))
        self.model.eval()
        self.head = HeadPair("reference", str(ROOT / "models/clm/CLM_v0.1-8B.pt"), "cpu").ensure()
        self.weights = {}
        for kind in ("state", "action"):
            self.weights[kind] = {k: mx.array(v.detach().float().numpy()) for k, v in getattr(self.head, kind + "_head").state_dict().items()}
        assert self.head.cfg.get("activation", "gelu") == "gelu"
        self.action_cache = {}
        self.prefix_cache = {}
        self.rows_cache = {}
        mx.eval(self.model.parameters(), self.weights)
        # Validate the port against the repository's PyTorch head on normalized inputs.
        rng = np.random.default_rng(42)
        x = rng.normal(size=(4, 4096)).astype(np.float32)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        errors = {}
        for kind in ("state", "action"):
            actual = np.asarray(self.project(mx.array(x), kind))
            expected = getattr(self.head, "project_" + ("states" if kind == "state" else "actions"))(x).numpy()
            errors[kind] = float(np.max(np.abs(actual - expected)))
            assert errors[kind] < 1e-5, errors
        self.meta = {"backbone": "Qwen/Qwen3-8B", "dtype": str(self.model.model.embed_tokens.weight.dtype),
                     "head_parameters_total": self.head.n_params, "head_config": self.head.cfg,
                     "head_scale": self.head.scale, "head_port_max_abs_error": errors,
                     "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip(),
                     "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "torch", "transformers")}}
        print(json.dumps(self.meta), flush=True)

    def project(self, x, kind):
        w = self.weights[kind]
        x = nn.gelu(x @ w["inp.weight"].T + w["inp.bias"])
        for i in range(self.head.cfg["depth"] - 2):
            h = x @ w[f"hidden.{i}.weight"].T + w[f"hidden.{i}.bias"]
            if self.head.cfg.get("layernorm"):
                h = mx.fast.layer_norm(h, w[f"norms.{i}.weight"], w[f"norms.{i}.bias"], 1e-5)
            h = nn.gelu(h)
            x = x + h if self.head.cfg.get("residual") else h
        return norm(x @ w["out.weight"].T + w["out.bias"])

    def embed(self, text):
        ids = self.tok.encode(text, add_special_tokens=True)
        h = self.model.model(mx.array([ids]))[:, -1, :]
        return norm(h), len(ids)

    def actions(self, texts):
        for text in texts:
            if text not in self.action_cache:
                emb, _ = self.embed(text)
                z = self.project(emb, "action")
                mx.eval(z)
                self.action_cache[text] = z
        return mx.concatenate([self.action_cache[t] for t in texts])

    def clm(self, state, question):
        start = time.perf_counter()
        text, keys, texts = build_pairs(state, {"q": question})["q"]
        actions = self.actions(texts)
        h, n = self.embed(text)  # Always recompute the state: no full-state cache hits.
        logits = self.head.scale * (self.project(h, "state") @ actions.T)[0]
        probs = mx.softmax(logits)
        mx.eval(probs)
        return {"choice": keys[int(mx.argmax(probs).item())], "probabilities": dict(zip(keys, probs.tolist())),
                "ms": (time.perf_counter() - start) * 1000, "input_tokens": n}

    def prompt(self, state, q):
        _, keys, texts = build_pairs(state, {"q": q})["q"]
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(keys)])
        assert len(labels) == len(keys)
        label_ids = [self.tok.encode(x, add_special_tokens=False) for x in labels]
        assert all(len(x) == 1 for x in label_ids), label_ids
        system = "Choose the best option for the user's state and question. Reply with exactly one option letter.\nOptions:\n"
        system += "\n".join(f"{a}: {key}: {text}" for a, key, text in zip(labels, keys, texts))
        # Options precede the varying state, allowing genuine prefix-KV reuse.
        messages = [{"role": "system", "content": system}, {"role": "user", "content": state + "\n\n" + q["instructions"]}]
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        blank = self.tok.apply_chat_template([messages[0], {"role": "user", "content": ""}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        common = 0
        while common < min(len(ids), len(blank)) and ids[common] == blank[common]:
            common += 1
        return ids, tuple(ids[:common]), keys, [x[0] for x in label_ids]

    def qwen(self, state, q, cached=False, selected=True):
        start = time.perf_counter()
        ids, prefix, keys, labels = self.prompt(state, q)
        cache = None
        n = len(ids)
        if cached:
            if prefix not in self.prefix_cache:
                cache = make_prompt_cache(self.model)
                h = self.model.model(mx.array([prefix]), cache=cache)
                mx.eval(h, [c.state for c in cache])
                self.prefix_cache[prefix] = cache
            cache = copy.deepcopy(self.prefix_cache[prefix])
            ids = ids[len(prefix):]
        h = self.model.model(mx.array([ids]), cache=cache)[:, -1, :]
        if selected:
            k = tuple(labels)
            if k not in self.rows_cache:
                self.rows_cache[k] = self.model.lm_head.weight[mx.array(labels)]
                mx.eval(self.rows_cache[k])
            logits = (h @ self.rows_cache[k].T)[0].astype(mx.float32)
        else:
            logits = self.model.lm_head(h)[0, mx.array(labels)].astype(mx.float32)
        probs = mx.softmax(logits)
        mx.eval(probs)
        return {"choice": keys[int(mx.argmax(probs).item())], "probabilities": dict(zip(keys, probs.tolist())),
                "ms": (time.perf_counter() - start) * 1000, "input_tokens": n,
                "prefix_tokens": len(prefix), "processed_tokens": len(ids)}


def question(criteria, instructions="Which team should handle this?"):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


BILLING = question({"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"})
MARIO = question({"jump": "Jump straight up", "right": "Walk right", "left": "Walk left", "jump_right": "Jump up and move right", "jump_left": "Jump up and move left"}, "Which action should Mario take to reach the goal safely?")


def cases():
    billing = [
        ("Customer: my invoice was charged twice and nobody answers the phone!", "billing"),
        ("The app crashes immediately after I log in.", "technical"),
        ("Please refund the subscription I cancelled last month.", "billing"),
        ("The server is unreachable and all API requests time out.", "technical"),
        ("Why was I charged tax twice on the same purchase?", "billing"),
        ("The upload button does nothing when clicked.", "technical"),
        ("I need a corrected invoice with my company's address.", "billing"),
        ("I cannot reset my password because the reset link errors.", "technical"),
        ("Payment succeeded, but the invoice page crashes with error 500. I need the page fixed, not a refund.", "technical"),
        ("The technical outage is over. I am requesting the promised service credit.", "billing"),
        ("No charge is wrong. The billing dashboard renders a blank screen.", "technical"),
        ("The application works perfectly. The subscription price on my bill is wrong.", "billing"),
        ("Please stop charging my expired subscription.", "billing"),
        ("The video freezes every few seconds on all devices.", "technical"),
        ("I want an explanation of this unexpected card charge.", "billing"),
        ("My data export always fails with a software exception.", "technical"),
        ("I was promised a refund but have not received it.", "billing"),
        ("My account is paid up; however the login service is down.", "technical"),
        ("The phone app works, but I paid for the same order twice.", "billing"),
        ("The credit card input widget is broken and refuses all typing.", "technical"),
    ]
    out = [{"group": "routing", "state": s, "q": BILLING, "gold": g} for s, g in billing]
    mario = [
        ("Mario is on flat safe ground. The goal is to his right with no obstacles. Walking is preferred to jumping when safe.", "right"),
        ("Mario is on flat safe ground. The goal is to his left with no obstacles. Walking is preferred to jumping when safe.", "left"),
        ("Mario stands at the left edge of a pit. The goal platform is across the pit to the right. Walking falls in; a jump right reaches it.", "jump_right"),
        ("Mario stands at the right edge of a pit. The goal platform is across the pit to the left. Walking falls in; a jump left reaches it.", "jump_left"),
        ("Mario must hit a block directly above his head. There are pits immediately on both sides, so he must keep his horizontal position.", "jump"),
        ("A low obstacle blocks the path to the goal on the right. Mario can clear it only by jumping forward to the right.", "jump_right"),
        ("A low obstacle blocks the path to the goal on the left. Mario can clear it only by jumping forward to the left.", "jump_left"),
        ("Mario is under a low ceiling with a safe corridor to his right leading to the exit. Jumping hits spikes on the ceiling.", "right"),
        ("Mario is under a low ceiling with a safe corridor to his left leading to the exit. Jumping hits spikes on the ceiling.", "left"),
        ("Mario must jump vertically to collect a coin directly above him without moving left or right.", "jump"),
    ]
    out += [{"group": "mario_text", "state": s, "q": MARIO, "gold": g} for s, g in mario]
    # Candidate-relative reasoning: state alone cannot know which two numbers are compared.
    for options, gold in [(["7", "9"], "9"), (["12", "4"], "12"), (["21", "19"], "21"), (["-3", "-8"], "-3"), (["0.9", "0.11"], "0.9")]:
        q = question({x: x for x in options}, "Which of the offered numbers is larger?")
        out.append({"group": "relative_numbers", "state": "Compare the candidate numbers numerically.", "q": q, "gold": gold})
    return out


def summary(samples):
    a = np.array(samples)
    return {"n": len(a), "p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min())}


def main():
    c = Comparison()
    result = {"metadata": c.meta, "quality": [], "latency": {}}
    data = cases()
    (ROOT / "results/cases.json").write_text(json.dumps(data, indent=2))
    for i, case in enumerate(data):
        row = {**case, "clm": c.clm(case["state"], case["q"]), "qwen": c.qwen(case["state"], case["q"])}
        # Reverse options to detect arbitrary label / position sensitivity.
        qrev = {**case["q"], "criteria": dict(reversed(list(case["q"]["criteria"].items())))}
        row["qwen_reversed"] = c.qwen(case["state"], qrev)
        result["quality"].append(row)
        print("quality", i, case["group"], "gold", case["gold"], "clm", row["clm"]["choice"], "qwen", row["qwen"]["choice"], "reverse", row["qwen_reversed"]["choice"], flush=True)
    scenarios = {
        "billing_2": (data[0]["state"], BILLING),
        "mario_5": (data[22]["state"], MARIO),
        "routing_26_long_options": ("A customer asks for a refund of a duplicate payment.", question({chr(65 + i): ("Handle duplicate payments, invoice disputes, and refunds." if i == 0 else f"Handle requests for product category {i}: investigate configuration, installation, feature setup and operational support for this product category.") for i in range(26)})),
        "billing_long_state": (("Earlier support note: the customer has used the application for several years and the account is active. " * 24) + data[0]["state"], BILLING),
    }
    rng = random.Random(42)
    for name, (state, q) in scenarios.items():
        methods = {"clm_cached_actions": lambda: c.clm(state, q),
                   "qwen_uncached_selected": lambda: c.qwen(state, q),
                   "qwen_prefix_cached_selected": lambda: c.qwen(state, q, cached=True),
                   "qwen_uncached_full_vocab": lambda: c.qwen(state, q, selected=False)}
        for f in methods.values():
            for _ in range(3):
                f()
        samples = {k: [] for k in methods}
        details = {}
        for _ in range(25):
            order = list(methods)
            rng.shuffle(order)
            for method in order:
                r = methods[method]()
                samples[method].append(r["ms"])
                details[method] = r
        result["latency"][name] = {k: {**summary(v), "samples_ms": v, "example": details[k]} for k, v in samples.items()}
        print("latency", name, json.dumps({k: summary(v) for k, v in samples.items()}), flush=True)
        (ROOT / "results/comparison.json").write_text(json.dumps(result, indent=2))
    # Explicitly measure head cost independently from backbone work.
    x = norm(mx.ones((1, 4096)))
    a = mx.ones((5, 512))
    samples = []
    for _ in range(103):
        start = time.perf_counter()
        p = mx.softmax(c.head.scale * (c.project(x, "state") @ a.T))
        mx.eval(p)
        samples.append((time.perf_counter() - start) * 1000)
    result["head_only"] = summary(samples[3:])
    result["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    (ROOT / "results/comparison.json").write_text(json.dumps(result, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
