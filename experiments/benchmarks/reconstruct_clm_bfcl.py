"""Fixed, best-effort CLM BFCL function-selection reconstruction.

Uses the unmodified upstream Engine, schema serializer and PyTorch heads.
An MLX embedding adapter runs the original BF16 Qwen3-8B on this Mac.
Evaluation labels are read only by the separate score command.
"""
import argparse
import collections
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/bfcl_author_reproduction/reconstruction'
sys.path.insert(0, str(ROOT / 'upstream/src'))
from clm.schema import build_pairs

CATEGORIES = ('multiple', 'live_multiple')
VARIANTS = ('full_schema', 'name_description')
INSTRUCTIONS = 'Which function should be called to fulfill this request?'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def append(name, value):
    with (OUT / name).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()


def request(row, variant):
    # These categories contain one conversation turn. Keep all messages and roles.
    assert len(row['question']) == 1
    state = '\n\n'.join(m['role'].capitalize() + ': ' + m['content']
                        for m in row['question'][0])
    functions = row['function']
    names = [f['name'] for f in functions]
    assert len(set(names)) == len(names)
    criteria = {f['name']: (f if variant == 'full_schema'
                           else f['name'] + ': ' + f['description']) for f in functions}
    question = {'type': 'choice', 'instructions': INSTRUCTIONS, 'criteria': criteria}
    return state, {'function': question}


def prepare():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-8b')
    sources = json.loads((OUT / 'data_sources.json').read_text())
    inputs = []
    lengths = collections.defaultdict(list)
    for category in CATEGORIES:
        filename = f'BFCL_v4_{category}.json'
        assert sha(OUT / 'data' / filename) == sources['files'][filename]['sha256']
        for row in read_rows(OUT / 'data' / filename):
            # No answer-file access in preparation or inference.
            for variant in VARIANTS:
                state, questions = request(row, variant)
                text, keys, actions = build_pairs(state, questions)['function']
                state_n = len(tok.encode(text, add_special_tokens=True))
                action_ns = [len(tok.encode(a, add_special_tokens=True)) for a in actions]
                lengths[variant + '/state'].append(state_n)
                lengths[variant + '/action'].extend(action_ns)
                inputs.append({'id': row['id'], 'category': category, 'variant': variant,
                               'state': state, 'questions': questions,
                               'exact_state_text': text, 'action_keys': keys,
                               'exact_action_texts': actions,
                               'state_tokens': state_n, 'action_tokens': action_ns})
    with (OUT / 'inputs.jsonl').open('w') as f:
        for row in inputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {
        'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'best_effort_reconstruction_not_authors_exact_protocol',
        'primary_variant': 'full_schema', 'secondary_variant': 'name_description',
        'categories': {c: len([r for r in inputs if r['category'] == c]) // len(VARIANTS)
                       for c in CATEGORIES},
        'selection': 'Every row in both complete public categories; no row filters or development search.',
        'score': 'Exact function-name match. No arguments generated or scored. Not an official BFCL aggregate.',
        'candidate_set': 'All original functions, in original order. No added abstention option.',
        'out_of_scope': 'Irrelevance, parallel calls, multi-turn and agentic BFCL categories require an additional decision/generation protocol not released by the authors.',
        'instructions': INSTRUCTIONS,
        'state_format': 'All original message contents with their role labels, in source order.',
        'action_formats': {'full_schema': 'Original function dictionary passed to upstream Choice; upstream to_text renders all fields as prose.',
                           'name_description': 'Original function name + colon + original description; parameter schema omitted. Fixed secondary sensitivity check.'},
        'truncation': 'None. Full input text retained. This differs from the README 2048-token default where exceeded.',
        'latency': 'Local Apple Silicon reconstruction, not RTX 4090 or Jev latency. Primary first-encounter times include action-cache misses. Separate warm-action timing forces state recomputation on each call.',
        'runtime': 'Unmodified upstream Engine/schema/PyTorch heads on CPU; normalized BF16 Qwen3-8B last-token embeddings through MLX, no chat template.',
        'weights': {'head_sha256': sha(ROOT / 'models/clm/CLM_v0.1-8B.pt'),
                    'backbone_config_sha256': sha(ROOT / 'models/qwen3-8b/config.json')},
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': sha(__file__), 'inputs_sha256': sha(OUT / 'inputs.jsonl'),
        'sources': sources,
        'lengths': {k: {'max': max(v), 'median': statistics.median(v),
                        'over_2048': sum(n > 2048 for n in v), 'n': len(v)} for k, v in lengths.items()},
        'prior_exposure': 'Earlier experiments evaluated subsets of these public categories. This run is a fixed reconstruction, not a newly held-out prompt-selection test.',
    }
    save('protocol.json', manifest)
    print(json.dumps({'prepared': manifest['categories'], 'lengths': manifest['lengths']}), flush=True)


class MLXEmbedder:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        mx.set_wired_limit(min(24 * 1024**3, mx.device_info()['max_recommended_working_set_size']))
        self.model, self.tok = load(str(ROOT / 'models/qwen3-8b'))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.cache = {}
        self.calls = 0

    def embed(self, texts):
        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.cache import make_prompt_cache
        from clm.embedder import l2
        tokens = 0
        for text in dict.fromkeys(texts):
            if text in self.cache:
                continue
            ids = self.tok.encode(text, add_special_tokens=True)
            cache = make_prompt_cache(self.model)
            for offset in range(0, len(ids), 1024):
                h = self.model.model(mx.array([ids[offset:offset + 1024]]), cache=cache)[:, -1, :]
                mx.eval(h, [c.state for c in cache])
            self.cache[text] = l2(np.asarray(h.astype(mx.float32))[0])
            tokens += len(ids)
            self.calls += 1
        return np.stack([self.cache[text] for text in texts]), tokens


def run():
    import torch
    from clm.engine import Engine
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(__file__) == protocol['script_sha256']
    assert sha(OUT / 'inputs.jsonl') == protocol['inputs_sha256']
    assert sha(ROOT / 'models/clm/CLM_v0.1-8B.pt') == protocol['weights']['head_sha256']
    torch.set_num_threads(4)
    embedder = MLXEmbedder()
    engine = Engine(embedder=embedder, checkpoint=str(ROOT / 'models/clm/CLM_v0.1-8B.pt'),
                    device='cpu', action_cache='0')
    head = engine.heads['clm-latest']
    import importlib.metadata
    save('runtime.json', {'platform': platform.platform(), 'machine': platform.machine(),
                         'packages': {p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm', 'torch', 'transformers']},
                         'head_config': head.cfg, 'head_scale': head.scale,
                         'backbone_dtype': str(embedder.model.model.embed_tokens.weight.dtype),
                         'torch_threads': torch.get_num_threads()})
    # Unrelated warm-up. No BFCL labels used to choose any configuration.
    engine.rank('What color is the clear daytime sky?', ['Blue', 'Red'])
    embedder.cache.clear()
    inputs = read_rows(OUT / 'inputs.jsonl')
    path = OUT / 'predictions.jsonl'
    previous = read_rows(path) if path.exists() else []
    done = {(r['id'], r['variant']) for r in previous}
    began = time.perf_counter()
    for i, row in enumerate(inputs):
        if (row['id'], row['variant']) in done:
            continue
        state_text = row['exact_state_text']
        # A repeated question must still incur its state forward pass.
        embedder.cache.pop(state_text, None)
        start = time.perf_counter()
        out = engine.answer(row['state'], row['questions'])
        elapsed = (time.perf_counter() - start) * 1000
        a = out['answers']['function']
        append('predictions.jsonl', {'id': row['id'], 'category': row['category'],
                                    'variant': row['variant'], 'choice': a['choice'],
                                    'probabilities': a['probabilities'],
                                    'input_tokens_spent': out['usage']['input_tokens'],
                                    'total_decision_ms': elapsed})
        # Retain reusable action texts, not completed state representations.
        embedder.cache.pop(state_text, None)
        if (i + 1) % 20 == 0 or i + 1 == len(inputs):
            print(json.dumps({'completed': i + 1, 'total': len(inputs), 'category': row['category'],
                              'elapsed_seconds': round(time.perf_counter() - began, 1),
                              'unique_cached_texts': len(embedder.cache)}), flush=True)
    timing_inputs = []
    for c in CATEGORIES:
        timing_inputs += [r for r in inputs if r['category'] == c and r['variant'] == 'full_schema'][:5]
    timing_path = OUT / 'warm_action_timing.jsonl'
    timing_done = {(r['id'], r['repeat']) for r in read_rows(timing_path)} if timing_path.exists() else set()
    for row in timing_inputs:
        # Explicit action preparation outside measured region, including after resume.
        embedder.embed(row['exact_action_texts'])
        for repeat in range(5):
            if (row['id'], repeat) in timing_done:
                continue
            embedder.cache.pop(row['exact_state_text'], None)
            start = time.perf_counter()
            answer = engine.answer(row['state'], row['questions'])
            elapsed = (time.perf_counter() - start) * 1000
            assert answer['usage']['input_tokens'] == row['state_tokens']
            append('warm_action_timing.jsonl', {'id': row['id'], 'category': row['category'],
                                              'repeat': repeat, 'ms': elapsed,
                                              'input_tokens_spent': answer['usage']['input_tokens']})
    print('INFERENCE COMPLETE', flush=True)


def score():
    # Ground truth first enters here, after inference.
    gold = {}
    for c in CATEGORIES:
        for row in read_rows(OUT / 'data' / f'answers_BFCL_v4_{c}.json'):
            assert len(row['ground_truth']) == 1 and len(row['ground_truth'][0]) == 1
            gold[row['id']] = next(iter(row['ground_truth'][0]))
    inputs = read_rows(OUT / 'inputs.jsonl')
    expected = {(r['id'], r['variant']) for r in inputs}
    predictions = read_rows(OUT / 'predictions.jsonl')
    keys = [(r['id'], r['variant']) for r in predictions]
    assert len(set(keys)) == len(keys) and set(keys) == expected
    groups = collections.defaultdict(list)
    for row in predictions:
        row = {**row, 'gold': gold[row['id']], 'correct': row['choice'] == gold[row['id']]}
        groups[row['category'], row['variant']].append(row)
    result = {'complete': True, 'interpretation': 'Function selection reconstruction only; not the authors exact run or full BFCL scoring.',
              'results': [], 'inputs_sha256': sha(OUT / 'inputs.jsonl'),
              'predictions_sha256': sha(OUT / 'predictions.jsonl')}
    for (c, v), rs in sorted(groups.items()):
        result['results'].append({'category': c, 'variant': v, 'n': len(rs),
                                  'correct': sum(r['correct'] for r in rs),
                                  'accuracy': sum(r['correct'] for r in rs) / len(rs),
                                  'median_total_decision_ms': statistics.median(r['total_decision_ms'] for r in rs)})
    timings = read_rows(OUT / 'warm_action_timing.jsonl')
    result['warm_actions_latency'] = {'n_states': len({r['id'] for r in timings}),
                                     'repetitions': 5, 'median_ms': statistics.median(r['ms'] for r in timings),
                                     'hardware_scope': 'Local Mac + MLX backbone + upstream CPU heads, not authors hardware.'}
    save('summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'score'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.stage]()
