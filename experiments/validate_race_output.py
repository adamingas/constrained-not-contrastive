"""Check selected-letter scoring against native full-vocabulary Qwen forward.

First ten frozen test cases, independent of correctness; no prompt changes.
Run after benchmark timing completes to avoid competing for the GPU.
"""
import json
import mlx.core as mx
from pilot_race_layout import RaceComparison,OUT


def main():
    cases=json.loads((OUT/'test_cases.json').read_text())[:10]
    primary={r['id']:r for r in map(json.loads,(OUT/'test_answers.jsonl').read_text().splitlines()) if r['variant']=='qwen_full'}
    c=RaceComparison();result=[]
    for case in cases:
        ids,_,keys,labels=c.prompt(case['state'],case['q'])
        logits=c.model(mx.array([ids]))[0,-1,:].astype(mx.float32)
        p=mx.softmax(logits);top=int(mx.argmax(logits).item())
        allowed=p[mx.array(labels)];restricted=keys[int(mx.argmax(allowed).item())]
        row={'id':case['id'],'unrestricted_top_token':c.tok.decode([top]),
             'top_token_is_allowed':top in labels,'allowed_probability_mass':float(mx.sum(allowed).item()),
             'restricted_choice':restricted,'primary_choice':primary[case['id']]['choice'],
             'matches_primary':restricted==primary[case['id']]['choice']}
        result.append(row);print(json.dumps(row),flush=True)
    (OUT/'native_output_validation.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
