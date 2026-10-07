"""Bounded 20-request comparison; no prompt search and no full-run option."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/bfcl_author_reproduction/reconstruction'
OUT = SOURCE / 'qwen_small_sample'
CATEGORIES = ('multiple', 'live_multiple')
SEED = 20261003


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite frozen sample.'
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-8b')
    source = rows(SOURCE / 'inputs.jsonl')
    rng = random.Random(SEED)
    selected = {}
    for category in CATEGORIES:
        ids = [r['id'] for r in source if r['category'] == category and r['variant'] == 'full_schema']
        selected[category] = rng.sample(ids, 10)
    prepared = []
    for row in source:
        if row['id'] not in selected[row['category']]:
            continue
        labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(row['action_keys'])]
        assert len(labels) == len(row['action_keys'])
        label_ids = [tok.encode(x, add_special_tokens=False) for x in labels]
        assert all(len(x) == 1 for x in label_ids)
        system = ('Choose the available function that best fulfills the request. '
                  'Reply with exactly one option letter, without explanation.\n\nAvailable functions:\n\n'
                  + '\n\n'.join(f'{label}:\n{text}' for label, text in zip(labels, row['exact_action_texts'])))
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': row['exact_state_text']}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        prepared.append({'id': row['id'], 'category': row['category'], 'variant': row['variant'],
                         'action_keys': row['action_keys'], 'labels': labels,
                         'allowed_token_ids': [x[0] for x in label_ids],
                         'messages': messages, 'exact_input_text': text, 'input_ids': ids})
    assert len(prepared) == 40 and len({r['id'] for r in prepared}) == 20
    with (OUT / 'inputs.jsonl').open('w') as f:
        for r in prepared:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    save('protocol.json', {'seed': SEED, 'selection': '10 random requests per category, independent of labels and existing CLM predictions.',
                           'selected_ids': selected, 'variants': ['full_schema', 'name_description'],
                           'model': 'Local original BF16 Qwen3-8B, MLX', 'thinking': False,
                           'decoding': 'Argmax over single-token letter output logits; no additional generation.',
                           'information': 'Same state and action descriptions as corresponding CLM input; original choices/order; no examples or added abstention.',
                           'score': 'Function name only, not argument correctness or full BFCL.',
                           'scope': 'Exploratory small sample from already evaluated public data; no prompt tuning in this run.',
                           'latency': 'Not evaluated in this accuracy check.',
                           'inputs_sha256': digest(OUT / 'inputs.jsonl'), 'script_sha256': digest(Path(__file__))})
    (OUT / 'first_exact_prompt.txt').write_text(prepared[0]['exact_input_text'])
    print(json.dumps({'requests': 20, 'predictions': 40, 'max_tokens': max(len(r['input_ids']) for r in prepared)}))


def run():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert digest(Path(__file__)) == protocol['script_sha256']
    assert digest(OUT / 'inputs.jsonl') == protocol['inputs_sha256']
    inputs = rows(OUT / 'inputs.jsonl')
    assert len(inputs) == 40 and len({r['id'] for r in inputs}) == 20
    assert not (OUT / 'predictions.jsonl').exists(), 'Do not overwrite predictions.'
    mx.set_wired_limit(min(24 * 1024**3, mx.device_info()['max_recommended_working_set_size']))
    model, tok = load(str(ROOT / 'models/qwen3-8b'))
    model.eval()
    mx.eval(model.parameters())
    for i, row in enumerate(inputs):
        cache = make_prompt_cache(model)
        ids = row['input_ids']
        for offset in range(0, len(ids), 512):
            h = model.model(mx.array([ids[offset:offset + 512]]), cache=cache)[:, -1, :]
            mx.eval(h, [c.state for c in cache])
        allowed = mx.array(row['allowed_token_ids'])
        # Use the native output head and discard every non-permitted token.
        logits = model.lm_head(h)[0, allowed].astype(mx.float32)
        probs = mx.softmax(logits)
        mx.eval(probs)
        values = probs.tolist()
        index = max(range(len(values)), key=values.__getitem__)
        result = {k: row[k] for k in ('id', 'category', 'variant')}
        result.update(choice=row['action_keys'][index], letter=row['labels'][index],
                      probabilities=dict(zip(row['action_keys'], values)))
        with (OUT / 'predictions.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        print(f'{i + 1}/40 {row["id"]} {row["variant"]}', flush=True)


def score():
    inputs = rows(OUT / 'inputs.jsonl')
    qwen = rows(OUT / 'predictions.jsonl')
    assert {(r['id'], r['variant']) for r in inputs} == {(r['id'], r['variant']) for r in qwen}
    assert len(qwen) == 40
    clm = {(r['id'], r['variant']): r for r in rows(SOURCE / 'predictions.jsonl')}
    gold = {}
    for category in CATEGORIES:
        for r in rows(SOURCE / 'data' / f'answers_BFCL_v4_{category}.json'):
            assert len(r['ground_truth']) == 1 and len(r['ground_truth'][0]) == 1
            gold[r['id']] = next(iter(r['ground_truth'][0]))
    details = []
    for r in qwen:
        c = clm[r['id'], r['variant']]
        assert gold[r['id']] in r['probabilities']
        details.append({**r, 'gold': gold[r['id']], 'qwen_correct': r['choice'] == gold[r['id']],
                        'clm_choice': c['choice'], 'clm_correct': c['choice'] == gold[r['id']]})
    summary = []
    for category in CATEGORIES:
        for variant in ('full_schema', 'name_description'):
            group = [r for r in details if r['category'] == category and r['variant'] == variant]
            assert len(group) == 10
            summary.append({'category': category, 'variant': variant, 'n': len(group),
                            'qwen_correct': sum(r['qwen_correct'] for r in group),
                            'clm_correct': sum(r['clm_correct'] for r in group),
                            'qwen_only_correct': sum(r['qwen_correct'] and not r['clm_correct'] for r in group),
                            'clm_only_correct': sum(r['clm_correct'] and not r['qwen_correct'] for r in group)})
    save('scored_details.json', details)
    save('summary.json', {'results': summary, 'unique_requests': 20,
                          'scope': 'Small exploratory sample; accuracy only; function names only.'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run', 'score'])
    globals()[ap.parse_args().command]()
