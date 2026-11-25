"""Load both models on a Mac and reuse their cached inputs."""

from collections import OrderedDict
import copy
import importlib.metadata
import platform
import subprocess
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import numpy as np
import torch

from .data import ROOT, sha
from clm.embedder import l2
from clm.engine import Engine


class ActionCacheEngine(Engine):
    """Keep cached actions, but calculate the state again for every decision."""

    def _cached(self, namespace, dim, texts, tokens, project=None):
        if namespace.endswith("/state"):
            embeddings, spent = self.embedder.embed(texts)
            tokens.append(spent)
            return project(embeddings) if project else self._to_device(embeddings)
        return super()._cached(namespace, dim, texts, tokens, project)


def compact(cache, length):
    for layer in cache:
        layer.trim(layer.size() - length)
        layer.keys = layer.keys[..., :length, :]
        layer.values = layer.values[..., :length, :]
    mx.eval([layer.state for layer in cache])


class Runtime:
    def __init__(self):
        torch.set_num_threads(4)
        mx.set_wired_limit(min(24 * 1024**3, mx.device_info()["max_recommended_working_set_size"]))
        self.model, self.tok = load(str(ROOT / "models/qwen3-8b"))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.prefixes = OrderedDict()  # Keep up to 16 BFCL instruction prefixes, without the questions.
        self.state_prefix = None       # Cache the CLM instructions before playing T-Rex.
        self.embed_calls = []
        self.engine = ActionCacheEngine(embedder=self, checkpoint=str(ROOT / "models/clm/CLM_v0.1-8B.pt"),
                                        device="cpu", action_cache="32MiB")
        self.head = self.engine.heads["clm-latest"]

    def forward(self, ids, cache=None, chunk=512):
        assert ids, "A decision must process a changing suffix."
        if cache is None:
            cache = make_prompt_cache(self.model)
        for offset in range(0, len(ids), chunk):
            h = self.model.model(mx.array([ids[offset:offset + chunk]]), cache=cache)[:, -1, :]
            mx.eval(h, [layer.state for layer in cache])
        return h

    def prepare(self, ids, chunk=512):
        cache = make_prompt_cache(self.model)
        self.forward(ids, cache, chunk=chunk)
        compact(cache, len(ids))
        return cache

    def embed(self, texts):
        vectors, spent = [], 0
        for text in texts:
            ids = self.tok.encode(text, add_special_tokens=True)
            n, cache = 0, None
            if self.state_prefix and text.startswith(self.state_prefix[0]):
                _, prefix, fixed = self.state_prefix
                assert ids[:len(prefix)] == prefix
                n, cache = len(prefix), copy.deepcopy(fixed)
            h = self.forward(ids[n:], cache)
            vectors.append(l2(np.asarray(h.astype(mx.float32))[0]))
            spent += len(ids) - n
            self.embed_calls.append({"total_tokens": len(ids), "cached_tokens": n,
                                     "processed_tokens": len(ids) - n})
        return np.stack(vectors), spent

    def labels(self, labels):
        ids = [self.tok.encode(label, add_special_tokens=False) for label in labels]
        assert all(len(x) == 1 for x in ids), "Every allowed label must be one token."
        return [x[0] for x in ids]

    def chat(self, system, state):
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": state}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def preamble(self, system):
        return self.tok.apply_chat_template([{"role": "system", "content": system}],
            tokenize=False, add_generation_prompt=False) + "<|im_start|>user\n"

    def distribution(self, h, keys, rows, scale=1):
        # Score only the letters the model is allowed to choose.
        logits = (h @ rows.T)[0]
        if scale != 1:
            logits = scale * logits
        probabilities = mx.softmax(logits.astype(mx.float32))
        mx.eval(probabilities)
        values = probabilities.tolist()
        return {"choice": keys[int(np.argmax(values))], "probabilities": dict(zip(keys, values))}

    def qwen_reuse(self, prompt, preamble, keys, label_ids, request_id):
        start = time.perf_counter()
        ids = self.tok.encode(prompt, add_special_tokens=False)
        prefix = tuple(self.tok.encode(preamble, add_special_tokens=False))
        assert ids[:len(prefix)] == list(prefix)
        best, matched = None, 0
        for key in self.prefixes:
            n = next((i for i, (a, b) in enumerate(zip(prefix, key)) if a != b), min(len(prefix), len(key)))
            if n > matched:
                best, matched = key, n
        cache, source = make_prompt_cache(self.model), None
        if best is not None:
            old, source = self.prefixes[best]
            cache = copy.deepcopy(old)
            self.prefixes.move_to_end(best)
            for layer in cache:
                layer.trim(layer.size() - matched)
        h = self.forward(ids[matched:], cache)
        answer = self.distribution(h, keys, self.model.lm_head.weight[mx.array(label_ids)])
        compact(cache, len(prefix))
        self.prefixes[prefix] = (cache, request_id)
        self.prefixes.move_to_end(prefix)
        while len(self.prefixes) > 16:
            self.prefixes.popitem(last=False)
        return {**answer, "ms": (time.perf_counter() - start) * 1000, "total_tokens": len(ids),
                "prefix_tokens": len(prefix), "cached_tokens": matched, "processed_tokens": len(ids) - matched,
                "full_preamble_hit": matched == len(prefix), "cache_source_request": source}

    def warm_kernels(self):
        h = self.forward(self.tok.encode("The sky is blue.", add_special_tokens=True))
        mx.eval(h @ self.model.lm_head.weight[mx.array([32, 33])].T)
        zeros = np.zeros((1, 4096), dtype=np.float32)
        self.head.project(zeros, zeros)

    def metadata(self):
        return {"platform": platform.platform(), "device": mx.device_info(),
                "dtype": str(self.model.model.embed_tokens.weight.dtype), "clm_heads_device": "cpu",
                "torch_threads": torch.get_num_threads(), "head_sha256": sha(ROOT / "models/clm/CLM_v0.1-8B.pt"),
                "upstream_commit": subprocess.run(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip() or None,
                "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "torch", "transformers")}}
