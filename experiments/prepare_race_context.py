"""Freeze a RACE-M pilot and test sample based only on input token budgets.

The evidence+question must exceed FIVE TIMES ALL FOUR FORMATTED OPTIONS
combined, a stricter denominator than counting individual answer strings.
No padding, repetition, truncation, or selection using model outputs.
"""
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path

import requests
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/race_context'
DATA=ROOT/'data/race_context'
URL='https://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz'
SEED=20261003


def main():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    path=DATA/'RACE.tar.gz'
    if not path.exists():
        r=requests.get(URL,timeout=120);r.raise_for_status();path.write_bytes(r.content)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen3-8b')
    count=lambda x:len(tok.encode(x,add_special_tokens=False))
    pools={'dev':[],'test':[]}
    with tarfile.open(path,'r:gz') as tf:
        for m in sorted(tf.getmembers(),key=lambda x:x.name):
            if not m.isfile() or not m.name.endswith('.txt'):continue
            split=next((s for s in pools if f'/{s}/middle/' in m.name),None)
            if split is None:continue
            d=json.load(tf.extractfile(m))
            for i,(question,options,answer) in enumerate(zip(d['questions'],d['options'],d['answers'])):
                # Formatting whitespace is normalized identically for both methods.
                options=[o.strip() for o in options]
                evidence=count(d['article']+'\n\n'+question)
                choices=count('\n'.join(f'{l}: {o}' for l,o in zip('ABCD',options)))
                raw_choices=sum(map(count,options))
                denominator=max(choices,raw_choices)
                if not (128<=evidence<=1024 and evidence>5*denominator):continue
                pools[split].append({'dataset':'race_middle','split':split,'id':m.name+f'/q{i}',
                    'doc_id':m.name,'article_sha256':hashlib.sha256(d['article'].encode()).hexdigest(),
                    'state':d['article'],'q':{'type':'choice','instructions':question,'criteria':{str(i):x for i,x in enumerate(options)}},
                    'gold':str('ABCD'.index(answer)), 'context_tokens':count(d['article']),
                    'evidence_question_tokens':evidence,'raw_choices_tokens':raw_choices,
                    'formatted_choices_tokens':choices,'ratio_denominator_tokens':denominator,
                    'evidence_to_choices_ratio':evidence/denominator})
    manifest={'source':URL,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'dataset_home':'https://www.cs.cmu.edu/~glai1/data/race/',
              'seed':SEED,'ratio_rule':'tokens(article + question) > 5 * max(sum of raw option tokens, tokens of all four labeled options)',
              'evidence_question_token_range':[128,1024], 'truncation':False,
              'pilot_rule':'RACE-M dev: at least 70% Qwen full-context accuracy and at least 15 percentage points gain over no-context; pilot excluded from test.',
              'pools':{},'selected':{},'files':{}}
    rng=random.Random(SEED)
    used_articles=set()
    for split,target in [('dev',40),('test',200)]:
        pool=pools[split];rng.shuffle(pool)
        manifest['pools'][split]={'questions':len(pool),'passages':len({x['doc_id'] for x in pool})}
        selected=[]
        for c in pool:
            if c['article_sha256'] in used_articles:continue
            used_articles.add(c['article_sha256'])
            items=list(c['q']['criteria'].items());rng.shuffle(items);c['q']['criteria']=dict(items)
            # Assert the ratio again after randomizing the labeled option order.
            formatted=count('\n'.join(f'{l}: {o}' for l,o in zip('ABCD',c['q']['criteria'].values())))
            c['formatted_choices_tokens']=formatted
            c['ratio_denominator_tokens']=max(formatted,c['raw_choices_tokens'])
            c['evidence_to_choices_ratio']=c['evidence_question_tokens']/c['ratio_denominator_tokens']
            assert c['evidence_to_choices_ratio']>5
            selected.append(c)
            if len(selected)==target:break
        assert len(selected)==target,(split,len(selected),manifest['pools'])
        # Fixed order-sensitivity audit on test only; not used to select examples.
        audit={c['id'] for c in rng.sample(selected,20)} if split=='test' else set()
        for c in selected:c['order_audit']=c['id'] in audit
        name='pilot_cases.json' if split=='dev' else 'test_cases.json'
        payload=json.dumps(selected,ensure_ascii=False,indent=2).encode()
        (OUT/name).write_bytes(payload)
        manifest['selected'][split]=len(selected)
        manifest['files'][name]=hashlib.sha256(payload).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
