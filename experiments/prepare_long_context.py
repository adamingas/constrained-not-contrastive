"""Freeze public-data samples before inference; do not truncate any selected input."""
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/long_context'
OUT = ROOT / 'results/long_context'
SEED = 20261002
tok = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-8b')


def make_prompt(state, q, tokenizer=tok):
    keys = list(q['criteria'])
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(keys)])
    system = "Answer the question using the supplied document. Select the best option. Reply with exactly one option letter and nothing else.\nOptions:\n"
    system += '\n'.join(f'{label}: {q["criteria"][key]}' for label, key in zip(labels, keys))
    messages = [{'role':'system','content':system}, {'role':'user','content':state+'\n\n'+q['instructions']}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    blank = tokenizer.apply_chat_template([messages[0], {'role':'user','content':''}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    common = 0
    while common < min(len(ids),len(blank)) and ids[common] == blank[common]: common += 1
    label_ids = [tokenizer.encode(l,add_special_tokens=False) for l in labels]
    assert all(len(x)==1 for x in label_ids)
    return ids, tuple(ids[:common]), keys, [x[0] for x in label_ids]


def eligible(case):
    n = len(tok.encode(case['state'], add_special_tokens=False))
    case['context_tokens'] = n
    if not 1024 <= n <= 16300: return False
    return len(make_prompt(case['state'],case['q'])[0]) <= 16384 and len(tok.encode(case['state']+'\n\n'+case['q']['instructions'])) <= 16384


def question(options, instructions):
    return {'type':'choice','instructions':instructions,'criteria':{str(i):x for i,x in enumerate(options)}}


def main():
    rng = random.Random(SEED)
    manifest = {'seed':SEED,'max_prompt_tokens':16384,'min_context_tokens':1024,'truncation':False,
                'sampling':'Sample without model outputs. One question per source document within each dataset. QuALITY balanced hard/easy, contracts balanced labels; LongBench length-filtered random sample.',
                'sources':{
                 'quality':{'url':'https://github.com/nyu-mll/quality','revision':'f84977c40dbfef70c9cab48037b7becfc8e45f73','split':'dev'},
                 'contract_nli':{'url':'https://stanfordnlp.github.io/contract-nli/','revision':'eced6528dd3c1d14d73f9a87df8f7bdbc03126f9','split':'test'},
                 'longbench_v2':{'url':'https://huggingface.co/datasets/zai-org/LongBench-v2','revision':'2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9','split':'train (official evaluation questions; no model training performed)'}},'pools':{}}
    selected = []
    quality = []
    lengths = {}
    for line in (DATA/'quality.dev.jsonl').read_text().splitlines():
        d = json.loads(line)
        for q in d['questions']:
            demands = [a['untimed_eval2_context'] for a in q['validation']]
            # Median >=3 means annotators judge at least a third of the article necessary.
            if statistics.median(demands) < 3: continue
            c = {'dataset':'quality','id':q['question_unique_id'],'doc_id':d['article_id'],
                 'state':d['article'],'q':question(q['options'],q['question']),
                 'gold':str(int(q['gold_label'])-1),'difficulty':int(q['difficult']),
                 'human_context_requirement_median':statistics.median(demands)}
            if eligible(c): quality.append(c)
    rng.shuffle(quality)
    manifest['pools']['quality'] = dict(collections.Counter(x['difficulty'] for x in quality))
    seen = set()
    for hard in (1,0):
        candidates = [x for x in quality if x['difficulty']==hard and x['doc_id'] not in seen]
        count = 0
        for x in candidates:
            if x['doc_id'] in seen: continue
            selected.append(x); seen.add(x['doc_id']); count+=1
            if count==40: break
        assert count==40, ('quality pool',hard,count)
    contracts = json.loads((DATA/'contract_test.json').read_text())
    options = ['Entailment: The contract supports the hypothesis.',
               'Contradiction: The contract contradicts the hypothesis.',
               'Not mentioned: The contract neither supports nor contradicts the hypothesis.']
    mapping = {'Entailment':'0','Contradiction':'1','NotMentioned':'2'}
    pool = []
    for d in contracts['documents']:
        for h,a in d['annotation_sets'][0]['annotations'].items():
            hypothesis = contracts['labels'][h]['hypothesis']
            q = question(options, 'Determine whether the contract entails, contradicts, or does not mention the following hypothesis. Use the contract as the only source of evidence.\nHypothesis: '+hypothesis)
            c = {'dataset':'contract_nli','id':str(d['id'])+'/'+h,'doc_id':str(d['id']),
                 'state':d['text'],'q':q,'gold':mapping[a['choice']], 'label':a['choice']}
            if eligible(c): pool.append(c)
    manifest['pools']['contract_nli'] = dict(collections.Counter(x['label'] for x in pool))
    rng.shuffle(pool)
    seen = set()
    # Rare contradictions first, preventing abundant classes from exhausting their documents.
    for label in ('Contradiction','NotMentioned','Entailment'):
        count = 0
        for x in pool:
            if x['label']!=label or x['doc_id'] in seen: continue
            selected.append(x); seen.add(x['doc_id']); count+=1
            if count==30: break
        assert count==30, ('contract pool',label,count)
    pool = []
    lb = json.loads((DATA/'longbench_v2.json').read_text())
    for d in lb:
        # Restrict explicitly to the publisher's short-context category, then require
        # the complete prompt to fit 16K tokens. Other categories are not evaluated.
        if d['length'] != 'short': continue
        c = {'dataset':'longbench_v2','id':d['_id'],'doc_id':hashlib.sha256(d['context'].encode()).hexdigest(),
             'state':d['context'],'q':question([d['choice_'+l] for l in 'ABCD'],d['question']),
             'gold':str('ABCD'.index(d['answer'])),'domain':d['domain'],'difficulty':d['difficulty']}
        if eligible(c): pool.append(c)
    manifest['pools']['longbench_v2'] = {'all':len(lb),'eligible':len(pool),'domains':dict(collections.Counter(x['domain'] for x in pool))}
    rng.shuffle(pool)
    seen = set(); count=0
    for x in pool:
        if x['doc_id'] in seen: continue
        selected.append(x);seen.add(x['doc_id']);count+=1
        if count==30:break
    assert count>=20,('longbench pool too small',count)
    manifest['longbench_sampling_note']='Publisher short category only; all eligible unique documents if fewer than 30.'
    # Randomize candidate order without changing canonical gold IDs.
    for c in selected:
        items = list(c['q']['criteria'].items());rng.shuffle(items)
        c['q']['criteria'] = dict(items)
    for dataset in ('quality','contract_nli','longbench_v2'):
        members = [c for c in selected if c['dataset']==dataset]
        ids = {c['id'] for c in rng.sample(members,10)}
        for c in members:c['order_audit']=c['id'] in ids
    rng.shuffle(selected)
    manifest['selected'] = dict(collections.Counter(x['dataset'] for x in selected))
    manifest['context_tokens'] = {d:{'min':min(x['context_tokens'] for x in selected if x['dataset']==d),
                                    'median':statistics.median(x['context_tokens'] for x in selected if x['dataset']==d),
                                    'max':max(x['context_tokens'] for x in selected if x['dataset']==d)} for d in manifest['selected']}
    for path in DATA.glob('*json*'):manifest.setdefault('source_sha256',{})[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.dumps(selected,ensure_ascii=False,indent=2)
    (OUT/'cases.json').write_text(data)
    manifest['cases_sha256']=hashlib.sha256(data.encode()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
