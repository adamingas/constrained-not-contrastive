"""Check fixed CLM reconstruction cases with an independent encoder runtime."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/bfcl_author_reproduction/reconstruction'
sys.path.insert(0, str(ROOT / 'upstream/src'))
from clm.embedder import l2
from clm.engine import Engine


class TransformersEmbedder:
    def __init__(self):
        torch.set_num_threads(4)
        self.tok = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-8b')
        self.model = AutoModel.from_pretrained(
            ROOT / 'models/qwen3-8b', dtype=torch.bfloat16,
            attn_implementation='sdpa').to('mps').eval()
        self.cache = {}

    def embed(self, texts):
        tokens = 0
        for text in dict.fromkeys(texts):
            if text in self.cache:
                continue
            ids = self.tok(text, return_tensors='pt').to('mps')
            with torch.inference_mode():
                h = self.model(**ids, use_cache=False).last_hidden_state[:, -1].float()
            self.cache[text] = l2(h.cpu().numpy()[0])
            tokens += ids['input_ids'].shape[-1]
        return np.stack([self.cache[t] for t in texts]), tokens


def main():
    plan = json.loads((OUT / 'backend_check_plan.json').read_text())
    inputs = [json.loads(s) for s in (OUT / 'inputs.jsonl').read_text().splitlines()]
    predictions = [json.loads(s) for s in (OUT / 'predictions.jsonl').read_text().splitlines()]
    index = {(r['id'], r['variant']): r for r in predictions}
    embedder = TransformersEmbedder()
    engine = Engine(embedder=embedder, checkpoint=str(ROOT / 'models/clm/CLM_v0.1-8B.pt'),
                    device='cpu', action_cache='0')
    rows = []
    for r in inputs:
        if r['variant'] != plan['variant'] or r['id'] not in plan['case_ids']:
            continue
        start = time.perf_counter()
        answer = engine.answer(r['state'], r['questions'])['answers']['function']
        old = index[r['id'], r['variant']]
        result = {'id': r['id'], 'mlx_choice': old['choice'],
                  'transformers_choice': answer['choice'],
                  'same_choice': answer['choice'] == old['choice'],
                  'max_probability_difference': max(abs(answer['probabilities'][k] - old['probabilities'][k])
                                                    for k in r['action_keys']),
                  'transformers_probabilities': answer['probabilities'],
                  'check_seconds': time.perf_counter() - start}
        rows.append(result)
        print(json.dumps(result), flush=True)
    assert len(rows) == len(plan['case_ids'])
    out = {'plan': plan, 'rows': rows, 'all_same_choice': all(r['same_choice'] for r in rows),
           'scope': 'Five fixed cases; not a full second-backend benchmark or CUDA/vLLM parity claim.'}
    (OUT / 'backend_check.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
