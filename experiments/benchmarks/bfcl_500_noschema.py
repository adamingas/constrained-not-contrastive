"""One sequential pass over 500 requests, with caches populated only by past requests."""
import argparse
import collections
import copy
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/bfcl_author_reproduction/reconstruction'
OUT = SOURCE / 'noschema_500'
sys.path.insert(0, str(ROOT / 'upstream/src'))
SEED = 20261004
CAPACITY = 16
RULE = ('Choose the available function that best fulfills the request. '
        'Reply with exactly one option letter, without explanation.\n\nAvailable functions:\n\n')


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def common(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def prepare():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-8b')
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists()
    exclude = {r['id'] for r in rows(SOURCE / 'qwen_small_sample/inputs.jsonl')}
    eligible = [r for r in rows(SOURCE / 'inputs.jsonl')
                if r['variant'] == 'name_description' and r['id'] not in exclude]
    chosen = set(random.Random(SEED).sample(range(len(eligible)), 500))
    inputs = []
    for i, r in enumerate(eligible):
        if i not in chosen:
            continue
        labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(r['action_keys'])]
        assert len(labels) == len(r['action_keys'])
        label_ids = [tok.encode(x, add_special_tokens=False) for x in labels]
        assert all(len(x) == 1 for x in label_ids)
        system = RULE + '\n\n'.join(f'{label}:\n{text}' for label, text in zip(labels, r['exact_action_texts']))
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': r['exact_state_text']}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        preamble = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False) + '<|im_start|>user\n'
        ids = tok.encode(prompt, add_special_tokens=False)
        prefix_ids = tok.encode(preamble, add_special_tokens=False)
        assert ids[:len(prefix_ids)] == prefix_ids and len(ids) > len(prefix_ids)
        inputs.append({**r, 'qwen_messages': messages, 'qwen_prompt': prompt,
                       'qwen_preamble': preamble, 'qwen_input_ids': ids,
                       'qwen_prefix_ids': prefix_ids, 'allowed_token_ids': [x[0] for x in label_ids], 'labels': labels})
    assert len(inputs) == len({r['id'] for r in inputs}) == 500
    with (OUT / 'inputs.jsonl').open('w') as f:
        for r in inputs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    protocol = {'sample_size': 500, 'seed': SEED,
                'sampling': 'Uniform random 500 from the two categories combined, excluding the previous 20 Qwen sample requests. Preserve original dataset order; no regrouping.',
                'counts': dict(collections.Counter(r['category'] for r in inputs)),
                'variant': 'name_description ONLY', 'rules': RULE,
                'qwen_cache': '16-entry LRU of exact-token fixed preambles. Reuse longest matching prefix from a previous request, including partial matches. Cache contains system and user-start marker only, never the changing question. Empty at start. Each request predicted once.',
                'clm_cache': 'Upstream Engine with 32MiB projected-action-vector arena, empty at start. State-vector cache bypassed so every request recomputes its state; some distinct dataset IDs share request wording. No separately cached encoder vectors.',
                'model': 'Original BF16 Qwen3-8B via MLX for both; original CLM PyTorch heads on CPU, four threads.',
                'qwen_output': 'Native output-head rows for single-token allowed letters only; argmax; thinking disabled. Algebraically identical allowed logits without computing unused vocabulary rows.',
                'latency': 'Single observation per request per method, alternating execution order. Includes tokenizer, cache lookup/update, all cache misses and output scoring. Excludes model load, file I/O, and one unrelated runtime warmup whose KV is discarded. No per-question warmup or repeats.',
                'scope': 'Function selection only; no argument evaluation. Earlier CLM reconstruction has seen these public requests; Qwen small sample excluded. No prompt tuning.',
                'inputs_sha256': sha(OUT / 'inputs.jsonl'), 'script_sha256': sha(__file__),
                'head_sha256': sha(ROOT / 'models/clm/CLM_v0.1-8B.pt')}
    save('protocol.json', protocol)
    print(json.dumps({'prepared': len(inputs), 'counts': protocol['counts'],
                      'distinct_preambles': len({r['qwen_preamble'] for r in inputs}),
                      'unique_state_texts': len({r['exact_state_text'] for r in inputs}),
                      'max_tokens': max(len(r['qwen_input_ids']) for r in inputs)}), flush=True)


class Runtime:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        mx.set_wired_limit(min(24 * 1024**3, mx.device_info()['max_recommended_working_set_size']))
        self.model, self.tok = load(str(ROOT / 'models/qwen3-8b'))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.prefixes = collections.OrderedDict()
        self.embed_calls = []

    def forward(self, ids, cache):
        import mlx.core as mx
        assert ids
        for offset in range(0, len(ids), 512):
            h = self.model.model(mx.array([ids[offset:offset + 512]]), cache=cache)[:, -1, :]
            mx.eval(h, [c.state for c in cache])
        return h

    def embed(self, texts):
        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.cache import make_prompt_cache
        from clm.embedder import l2
        vectors, spent = [], 0
        for text in texts:
            ids = self.tok.encode(text, add_special_tokens=True)
            h = self.forward(ids, make_prompt_cache(self.model))
            vectors.append(l2(np.asarray(h.astype(mx.float32))[0]))
            self.embed_calls.append({'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'tokens': len(ids)})
            spent += len(ids)
        return np.stack(vectors), spent

    def qwen(self, row):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        started = time.perf_counter()
        ids = self.tok.encode(row['qwen_prompt'], add_special_tokens=False)
        prefix = tuple(self.tok.encode(row['qwen_preamble'], add_special_tokens=False))
        best, matched = None, 0
        for key in self.prefixes:
            n = common(prefix, key)
            if n > matched:
                best, matched = key, n
        if best is None:
            cache = make_prompt_cache(self.model)
            from_id = None
        else:
            old, from_id = self.prefixes[best]
            cache = copy.deepcopy(old)
            self.prefixes.move_to_end(best)
            for layer in cache:
                layer.trim(layer.size() - matched)
        h = self.forward(ids[matched:], cache)
        allowed = mx.array(row['allowed_token_ids'])
        logits = (h @ self.model.lm_head.weight[allowed].T)[0].astype(mx.float32)
        probs = mx.softmax(logits)
        mx.eval(probs)
        values = probs.tolist()
        index = max(range(len(values)), key=values.__getitem__)
        # This request has already been evaluated once. Retain only its fixed preamble.
        for layer in cache:
            layer.trim(layer.size() - len(prefix))
            # Compact the retained tensors to exclude the changing request.
            layer.keys = layer.keys[..., :len(prefix), :]
            layer.values = layer.values[..., :len(prefix), :]
        mx.eval([layer.state for layer in cache])
        self.prefixes[prefix] = (cache, row['id'])
        self.prefixes.move_to_end(prefix)
        while len(self.prefixes) > CAPACITY:
            self.prefixes.popitem(last=False)
        elapsed = (time.perf_counter() - started) * 1000
        assert ids == row['qwen_input_ids'] and list(prefix) == row['qwen_prefix_ids']
        return {'choice': row['action_keys'][index], 'probabilities': dict(zip(row['action_keys'], values)),
                'ms': elapsed, 'total_tokens': len(ids), 'prefix_tokens': len(prefix),
                'cached_tokens': matched, 'processed_tokens': len(ids) - matched,
                'full_preamble_hit': matched == len(prefix), 'cache_source_request': from_id}


def run():
    import mlx.core as mx
    import numpy as np
    import torch
    from clm.engine import Engine
    from mlx_lm.models.cache import make_prompt_cache
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(__file__) == protocol['script_sha256']
    assert sha(OUT / 'inputs.jsonl') == protocol['inputs_sha256']
    assert not (OUT / 'predictions.jsonl').exists(), 'No reruns or silent cache reset on resume.'
    inputs = rows(OUT / 'inputs.jsonl')
    assert len(inputs) == 500
    torch.set_num_threads(4)
    rt = Runtime()
    class ActionCacheEngine(Engine):
        def _cached(self, namespace, dim, texts, tokens, project=None):
            if namespace.endswith('/state'):
                emb, spent = self.embedder.embed(texts)
                tokens.append(spent)
                return project(emb) if project else self._to_device(emb)
            return super()._cached(namespace, dim, texts, tokens, project)
    engine = ActionCacheEngine(embedder=rt, checkpoint=str(ROOT / 'models/clm/CLM_v0.1-8B.pt'), device='cpu', action_cache='32MiB')
    # Warm kernels once on unrelated text. All benchmark caches remain empty.
    h = rt.forward(rt.tok.encode('The sky is blue.', add_special_tokens=True), make_prompt_cache(rt.model))
    mx.eval(h @ rt.model.lm_head.weight[mx.array([32, 33])].T)
    engine.heads['clm-latest'].project(np.zeros((1, 4096), dtype=np.float32), np.zeros((1, 4096), dtype=np.float32))
    save('runtime.json', {'device': mx.device_info(), 'model_dtype': str(rt.model.model.embed_tokens.weight.dtype),
                          'clm_heads_device': 'cpu', 'threads': torch.get_num_threads(), 'qwen_cache_capacity': CAPACITY})
    seen_actions = set()
    began = time.perf_counter()
    for i, row in enumerate(inputs):
        result = {'id': row['id'], 'category': row['category'], 'sequence': i,
                  'first_method': 'qwen' if i % 2 == 0 else 'clm'}
        for method in (('qwen', 'clm') if i % 2 == 0 else ('clm', 'qwen')):
            if method == 'qwen':
                result['qwen'] = rt.qwen(row)
            else:
                rt.embed_calls = []
                started = time.perf_counter()
                answer = engine.answer(row['state'], row['questions'])
                elapsed = (time.perf_counter() - started) * 1000
                a = answer['answers']['function']
                new_actions = [s for s in dict.fromkeys(row['exact_action_texts']) if s not in seen_actions]
                result['clm'] = {'choice': a['choice'], 'probabilities': a['probabilities'], 'ms': elapsed,
                                 'processed_tokens': answer['usage']['input_tokens'], 'state_tokens': row['state_tokens'],
                                 'new_actions': len(new_actions), 'all_actions_cached': not new_actions,
                                 'encoder_calls': rt.embed_calls}
                assert len(rt.embed_calls) == 1 + len(new_actions)
                seen_actions.update(row['exact_action_texts'])
        with (OUT / 'predictions.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        if (i + 1) % 25 == 0:
            print(json.dumps({'completed': i + 1, 'total': 500, 'elapsed_seconds': round(time.perf_counter() - began, 1),
                              'last_qwen_cached_tokens': result['qwen']['cached_tokens'],
                              'last_qwen_processed_tokens': result['qwen']['processed_tokens']}), flush=True)
    print('COMPLETE: each model evaluated each of 500 requests once.', flush=True)


def latency(group, model):
    xs = sorted(r[model]['ms'] for r in group)
    return {'n': len(xs), 'median_ms': statistics.median(xs), 'mean_ms': statistics.mean(xs),
            'p95_ms': xs[math.ceil(.95 * len(xs)) - 1], 'total_seconds': sum(xs) / 1000} if xs else None


def score():
    inputs = rows(OUT / 'inputs.jsonl')
    predicted = rows(OUT / 'predictions.jsonl')
    assert len(predicted) == 500 and [r['id'] for r in inputs] == [r['id'] for r in predicted]
    gold = {}
    for category in ('multiple', 'live_multiple'):
        for row in rows(SOURCE / 'data' / f'answers_BFCL_v4_{category}.json'):
            assert len(row['ground_truth']) == 1 and len(row['ground_truth'][0]) == 1
            gold[row['id']] = next(iter(row['ground_truth'][0]))
    past = {}
    for row, inp in zip(predicted, inputs):
        row['gold'] = gold[row['id']]
        assert row['gold'] in inp['action_keys']
        for model in ('qwen', 'clm'):
            d = row[model]
            assert set(d['probabilities']) == set(inp['action_keys'])
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in d['probabilities'].values())
            assert abs(sum(d['probabilities'].values()) - 1) < 1e-5
            d['correct'] = d['choice'] == row['gold']
        q = row['qwen']
        assert q['cached_tokens'] + q['processed_tokens'] == q['total_tokens']
        if q['cache_source_request'] is not None:
            old = past[q['cache_source_request']]
            assert old['qwen_prefix_ids'][:q['cached_tokens']] == inp['qwen_prefix_ids'][:q['cached_tokens']]
        else:
            assert q['cached_tokens'] == 0
        past[row['id']] = inp
    groups = {'all': predicted, **{c: [r for r in predicted if r['category'] == c] for c in ('multiple', 'live_multiple')}}
    summary = {'accuracy': {}, 'latency': {}, 'cache': {}, 'inputs_sha256': sha(OUT / 'inputs.jsonl'),
               'predictions_sha256': sha(OUT / 'predictions.jsonl'), 'validated_requests': len(predicted)}
    for name, group in groups.items():
        summary['accuracy'][name] = {'n': len(group), **{m: sum(r[m]['correct'] for r in group) for m in ('qwen', 'clm')},
                                    'qwen_only_correct': sum(r['qwen']['correct'] and not r['clm']['correct'] for r in group),
                                    'clm_only_correct': sum(r['clm']['correct'] and not r['qwen']['correct'] for r in group)}
        summary['latency'][name] = {m: latency(group, m) for m in ('qwen', 'clm')}
    for name, group in [('both_reusable_parts_cached', [r for r in predicted if r['qwen']['full_preamble_hit'] and r['clm']['all_actions_cached']]),
                        ('qwen_preamble_not_fully_cached', [r for r in predicted if not r['qwen']['full_preamble_hit']])]:
        summary['latency'][name] = {m: latency(group, m) for m in ('qwen', 'clm')}
    summary['cache'] = {'qwen_full_preamble_hits': sum(r['qwen']['full_preamble_hit'] for r in predicted),
                        'qwen_partial_preamble_hits': sum(0 < r['qwen']['cached_tokens'] < r['qwen']['prefix_tokens'] for r in predicted),
                        'qwen_zero_cached_tokens': sum(r['qwen']['cached_tokens'] == 0 for r in predicted),
                        'qwen_cached_tokens': sum(r['qwen']['cached_tokens'] for r in predicted),
                        'qwen_total_tokens': sum(r['qwen']['total_tokens'] for r in predicted),
                        'clm_all_actions_cached': sum(r['clm']['all_actions_cached'] for r in predicted),
                        'clm_new_action_encodings': sum(r['clm']['new_actions'] for r in predicted)}
    save('scored_details.json', predicted)
    save('summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run', 'score'])
    globals()[ap.parse_args().command]()
