"""Qualify RACE-M on development data, then evaluate a frozen test sample."""
import argparse
import hashlib
import json
import random
import time

import mlx.core as mx
from long_context_bench import LongComparison, append
from pilot_race_layout import RaceComparison
from compare import ROOT

OUT=ROOT/'results/race_context'


def run(c,split,original=False):
    filename='pilot_cases.json' if split=='pilot' else 'test_cases.json'
    cases=json.loads((OUT/filename).read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest()==manifest['files'][filename]
    path=OUT/(('pilot_answers.jsonl' if original else 'pilot_standard_layout.jsonl') if split=='pilot' else 'test_answers.jsonl')
    previous=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(r['id'],r['variant']) for r in previous}
    for index,case in enumerate(cases):
        variants=['qwen_full','qwen_no_context'] if split=='pilot' else ['clm_full','qwen_full','clm_no_context','qwen_no_context','qwen_cached']
        if split=='test' and case['order_audit']:variants.append('qwen_reversed')
        random.Random(20261003+index).shuffle(variants)
        if all((case['id'],v) in done for v in variants):continue
        c.action_cache.clear();c.prefix_cache.clear();mx.clear_cache()
        setup={}
        if split=='test':
            t=time.perf_counter();c.actions(list(case['q']['criteria'].values()))
            setup['clm_actions_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter();c.prepare_prefix(case['state'],case['q'])
            setup['qwen_prefix_ms']=(time.perf_counter()-t)*1000
        for variant in variants:
            if (case['id'],variant) in done:continue
            st='' if 'no_context' in variant else case['state']
            q=case['q']
            if variant=='qwen_reversed':q={**q,'criteria':dict(reversed(list(q['criteria'].items())))}
            if variant.startswith('clm'):a=c.clm(st,q)
            else:a=c.qwen(st,q,cached=(variant=='qwen_cached'))
            row={'id':case['id'],'variant':variant,'gold':case['gold'],'correct':a['choice']==case['gold'],
                 'evidence_question_tokens':case['evidence_question_tokens'],
                 'evidence_to_choices_ratio':case['evidence_to_choices_ratio'],'setup':setup,**a}
            append(path,row)
        print(json.dumps({'split':split,'case':index+1,'of':len(cases)}),flush=True)
    return [json.loads(x) for x in path.read_text().splitlines()]


def gate(rows,original=False):
    counts={v:sum(r['correct'] for r in rows if r['variant']==v) for v in ['qwen_full','qwen_no_context']}
    n=sum(r['variant']=='qwen_full' for r in rows)
    assert n==40
    result={'n':n,'layout':'options in system message' if original else 'passage, question, then options in user message','correct':counts,'full_accuracy':counts['qwen_full']/n,
            'context_gain':(counts['qwen_full']-counts['qwen_no_context'])/n,
            'rule':{'minimum_full_accuracy':.70,'minimum_context_gain':.15}}
    result['qualified']=result['full_accuracy']>=.70 and result['context_gain']>=.15
    (OUT/('pilot_original_layout_result.json' if original else 'pilot_result.json')).write_text(json.dumps(result,indent=2))
    print('PILOT',json.dumps(result),flush=True)
    return result['qualified']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-only',action='store_true')
    ap.add_argument('--original-pilot',action='store_true',help='Reproduce the development-only original layout diagnostic; never runs test.')
    args=ap.parse_args()
    c=LongComparison() if args.original_pilot else RaceComparison()
    c.meta.update({'primary_qwen_cache':False,'prefill_chunk_size':c.chunk_size,
                   'primary_prompt':'Standard passage/question/options order in user message, selected on development pilot only; thinking disabled.',
                   'cached_variant_prefix':'Shared instruction prefix only; options follow the passage and are always processed.',
                   'hardware':'Apple M4 Max, 64 GB unified memory'})
    if args.original_pilot:c.meta['primary_prompt']='Original options-in-system layout, development-only diagnostic.'
    (OUT/('runtime_original_pilot.json' if args.original_pilot else 'runtime.json')).write_text(json.dumps(c.meta,indent=2))
    q={'type':'choice','instructions':'Which color is the clear daytime sky?','criteria':{'0':'Blue','1':'Red'}}
    for _ in range(3):c.clm('A clear sunny day.',q);c.qwen('A clear sunny day.',q,cached=False)
    qualified=gate(run(c,'pilot',original=args.original_pilot),original=args.original_pilot)
    if args.original_pilot:
        print('DONE original development diagnostic',flush=True);return
    if not qualified:raise RuntimeError('Pilot failed the predeclared qualification rule; no test inference performed.')
    if not args.pilot_only:run(c,'test')
    print('DONE',flush=True)


if __name__=='__main__':main()
