"""Fixed RACE-M prompt ablations; no model changes or example selection.

Seven Qwen layouts, full and no-passage controls, 20 reversed-order audits
per layout, plus a contemporaneous CLM reference. Strict raw concatenation
and completion-style Answer: endings are both measured, avoiding an implicit
choice of answer boundary. Primary output restriction remains bare A/B/C/D.
"""
import hashlib
import json
import random
import time

import mlx.core as mx
from transformers import AutoTokenizer
from compare import ROOT
from long_context_bench import LongComparison, append

SOURCE=ROOT/'results/race_context'
OUT=ROOT/'results/race_ablations'
FULL_SYSTEM='Answer the multiple-choice reading comprehension question using the passage. Reply with exactly one option letter and nothing else.'
SHORT_SYSTEM='Reply with A, B, C, or D.'
VARIANTS={
    'original':{'system':FULL_SYSTEM,'chat':True,'answer_cue':False},
    'short_system':{'system':SHORT_SYSTEM,'chat':True,'answer_cue':False},
    'no_system':{'system':None,'chat':True,'answer_cue':False},
    'no_template':{'system':FULL_SYSTEM,'chat':False,'answer_cue':True},
    'neither':{'system':None,'chat':False,'answer_cue':True},
    'no_template_literal':{'system':FULL_SYSTEM,'chat':False,'answer_cue':False},
    'neither_literal':{'system':None,'chat':False,'answer_cue':False},
}


def render(tok,state,q,variant):
    spec=VARIANTS[variant]
    keys=list(q['criteria']);labels=list('ABCD'[:len(keys)])
    options='\n'.join(f'{l}: {q["criteria"][k]}' for l,k in zip(labels,keys))
    user='Passage:\n'+state+'\n\nQuestion: '+q['instructions']+'\n\nOptions:\n'+options
    if spec['chat']:
        messages=[]
        if spec['system'] is not None:messages.append({'role':'system','content':spec['system']})
        messages.append({'role':'user','content':user})
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    else:
        messages=None
        text=(spec['system']+'\n\n' if spec['system'] else '')+user
        if spec['answer_cue']:text+='\n\nAnswer:'
    ids=tok.encode(text,add_special_tokens=False)
    return ids,keys,text,messages


class Ablations(LongComparison):
    def __init__(self):
        super().__init__()
        self.bare_ids=[self.tok.encode(l,add_special_tokens=False) for l in 'ABCD']
        self.spaced_ids=[self.tok.encode(' '+l,add_special_tokens=False) for l in 'ABCD']
        assert all(len(x)==1 for x in self.bare_ids+self.spaced_ids)
        self.bare_ids=[x[0] for x in self.bare_ids]
        self.spaced_ids=[x[0] for x in self.spaced_ids]
        self.bare_rows=self.model.lm_head.weight[mx.array(self.bare_ids)]
        self.spaced_rows=self.model.lm_head.weight[mx.array(self.spaced_ids)]
        mx.eval(self.bare_rows,self.spaced_rows)

    def decision(self,state,q,variant,diagnostic=False):
        start=time.perf_counter()
        ids,keys,_,_=render(self.tok,state,q,variant)
        h=self.last(ids)
        logits=(h@self.bare_rows.T)[0].astype(mx.float32)
        probs=mx.softmax(logits);mx.eval(probs)
        choice=keys[int(mx.argmax(probs).item())];probabilities=dict(zip(keys,probs.tolist()))
        elapsed=(time.perf_counter()-start)*1000
        out={'choice':choice,'probabilities':probabilities,'ms':elapsed,'input_tokens':len(ids)}
        # Secondary grammar diagnostic, after stopping the primary timer.
        # Same forward pass, allow one leading-space letter token as an alias.
        spaced=(h@self.spaced_rows.T)[0].astype(mx.float32)
        both=mx.softmax(mx.concatenate([logits,spaced]));merged=both[:4]+both[4:]
        out['space_alias_choice']=keys[int(mx.argmax(merged).item())]
        if diagnostic:
            full=self.model.lm_head(h)[0].astype(mx.float32);p=mx.softmax(full)
            top=int(mx.argmax(full).item())
            out['diagnostic']={'unrestricted_top_token':self.tok.decode([top]),
                'bare_letters_probability_mass':float(p[mx.array(self.bare_ids)].sum().item()),
                'bare_or_spaced_letters_probability_mass':float(p[mx.array(self.bare_ids+self.spaced_ids)].sum().item())}
        return out


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cases=json.loads((SOURCE/'test_cases.json').read_text())
    original_manifest=json.loads((SOURCE/'manifest.json').read_text())
    sha=hashlib.sha256((SOURCE/'test_cases.json').read_bytes()).hexdigest()
    assert sha==original_manifest['files']['test_cases.json']
    manifest={'cases_file':str(SOURCE/'test_cases.json'),'cases_sha256':sha,'n':len(cases),
              'variants':VARIANTS,'primary_output_tokens':'bare A/B/C/D; one token; no thinking or generation loop',
              'secondary_output_grammar':'Also score bare or single-leading-space letter tokens from the same hidden state; aggregate aliases by probability sum.',
              'seed':20261004,'no_context_controls':True,'order_audit_per_variant':20,
              'diagnostic_sample':'First 10 frozen cases, each variant, full context; full-vocabulary diagnostics outside timed region.',
              'latency':'One decision per method and condition per case; randomized interleaving. No prefix or whole-state reuse. CLM actions cached.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Freeze literal prompt examples and counts before inference.
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen3-8b')
    examples={}
    for v in VARIANTS:
        ids,keys,text,msgs=render(tok,cases[0]['state'],cases[0]['q'],v)
        examples[v]={'input_tokens':len(ids),'messages':msgs,'rendered_input':text,'option_keys':keys}
        (OUT/(v+'_example.txt')).write_text(text)
    (OUT/'prompt_examples.json').write_text(json.dumps(examples,indent=2))
    # Verify identical baseline serialization against previous implementation.
    from pilot_race_layout import make_race_prompt
    assert render(tok,cases[0]['state'],cases[0]['q'],'original')[0]==make_race_prompt(cases[0]['state'],cases[0]['q'],tok)[0]
    c=Ablations()
    c.meta.update({'bare_letter_ids':c.bare_ids,'space_letter_ids':c.spaced_ids,'cases_sha256':sha})
    (OUT/'runtime.json').write_text(json.dumps(c.meta,indent=2))
    warmq={'type':'choice','instructions':'Which color is the sky?','criteria':{'0':'Blue','1':'Red','2':'Green','3':'Brown'}}
    for _ in range(2):
        for v in VARIANTS:c.decision('A clear sunny day.',warmq,v)
        c.clm('A clear sunny day.',warmq)
    path=OUT/'answers.jsonl'
    old=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(r['id'],r['variant'],r['condition']) for r in old}
    for index,case in enumerate(cases):
        jobs=[(v,condition) for v in list(VARIANTS)+['clm'] for condition in ['full','no_context']]
        if case['order_audit']:jobs += [(v,'reversed') for v in VARIANTS]
        random.Random(20261004+index).shuffle(jobs)
        if all((case['id'],v,cond) in done for v,cond in jobs):continue
        c.action_cache.clear();c.prefix_cache.clear();mx.clear_cache()
        c.actions(list(case['q']['criteria'].values()))
        for v,condition in jobs:
            if (case['id'],v,condition) in done:continue
            state='' if condition=='no_context' else case['state'];q=case['q']
            if condition=='reversed':q={**q,'criteria':dict(reversed(list(q['criteria'].items())))}
            a=c.clm(state,q) if v=='clm' else c.decision(state,q,v,diagnostic=(index<10 and condition=='full'))
            row={'id':case['id'],'variant':v,'condition':condition,'gold':case['gold'],
                 'correct':a['choice']==case['gold'],**a}
            if 'space_alias_choice' in a:row['space_alias_correct']=a['space_alias_choice']==case['gold']
            append(path,row)
        print(json.dumps({'case':index+1,'of':len(cases)}),flush=True)
    print('DONE',flush=True)


if __name__=='__main__':main()
