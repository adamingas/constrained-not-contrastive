"""Full-document accuracy, no-document controls, and context-length latency.

Uses matched BF16 Qwen3-8B backbones and exact released CLM heads. All selected
documents are complete. Chunked prefill bounds working memory without truncation.
Progress is append-only and resumable at the per-case/per-variant level.
"""
import argparse
import copy
import hashlib
import json
import random
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from compare import Comparison, ROOT, norm
from prepare_long_context import make_prompt

OUT = ROOT/'results/long_context'


class LongComparison(Comparison):
    chunk_size = 1024

    def __init__(self):
        # Process-local residency, no system setting changes. Avoid paging the 16GB
        # backbone between chunks on an active workstation. Same limit for both paths.
        self.wired_limit = min(24 * 1024**3, mx.device_info()['max_recommended_working_set_size'])
        mx.set_wired_limit(self.wired_limit)
        super().__init__()
        self.meta['wired_memory_limit_bytes'] = self.wired_limit

    def last(self, ids, cache=None):
        if cache is None:cache=make_prompt_cache(self.model)
        for start in range(0,len(ids),self.chunk_size):
            h=self.model.model(mx.array([ids[start:start+self.chunk_size]]),cache=cache)
            h=h[:,-1,:]
            mx.eval(h,[c.state for c in cache])
        return h

    def embed(self,text):
        ids=self.tok.encode(text,add_special_tokens=True)
        return norm(self.last(ids)),len(ids)

    def prompt(self,state,q):return make_prompt(state,q,self.tok)

    def prepare_prefix(self,state,q):
        ids,prefix,keys,labels=self.prompt(state,q)
        if prefix not in self.prefix_cache:
            cache=make_prompt_cache(self.model)
            self.last(list(prefix),cache=cache)
            self.prefix_cache[prefix]=cache
        k=tuple(labels)
        if k not in self.rows_cache:
            self.rows_cache[k]=self.model.lm_head.weight[mx.array(labels)]
            mx.eval(self.rows_cache[k])

    def qwen(self,state,q,cached=True,selected=True):
        start=time.perf_counter()
        ids,prefix,keys,labels=self.prompt(state,q)
        total=len(ids)
        cache=None
        if cached:
            self.prepare_prefix(state,q)
            cache=copy.deepcopy(self.prefix_cache[prefix])
            ids=ids[len(prefix):]
        h=self.last(ids,cache=cache)
        k=tuple(labels)
        if k not in self.rows_cache:
            self.rows_cache[k]=self.model.lm_head.weight[mx.array(labels)]
            mx.eval(self.rows_cache[k])
        logits=(h @ self.rows_cache[k].T)[0].astype(mx.float32)
        probs=mx.softmax(logits);mx.eval(probs)
        choice=keys[int(mx.argmax(probs).item())]
        values=probs.tolist()
        return {'choice':choice,'probabilities':dict(zip(keys,values)),
                'ms':(time.perf_counter()-start)*1000,'input_tokens':total,
                'prefix_tokens':len(prefix),'processed_tokens':len(ids)}


def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()


def accuracy(c):
    cases=json.loads((OUT/'cases.json').read_text())
    path=OUT/'answers.jsonl'
    old=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(x['id'],x['variant']) for x in old}
    total=0
    for index,case in enumerate(cases):
        c.prefix_cache.clear()
        c.action_cache.clear()
        q=case['q'];state=case['state']
        variants=['clm_full','qwen_full','clm_no_context','qwen_no_context']
        if case['order_audit']:variants.append('qwen_reversed')
        random.Random(index+20261002).shuffle(variants)
        for variant in variants:
            if (case['id'],variant) in done:continue
            st='' if 'no_context' in variant else state
            qq=q
            if variant=='qwen_reversed':qq={**q,'criteria':dict(reversed(list(q['criteria'].items())))}
            setup_start=time.perf_counter()
            if variant.startswith('clm'):
                c.actions(list(qq['criteria'].values()))
            else:c.prepare_prefix(st,qq)
            setup_ms=(time.perf_counter()-setup_start)*1000
            try:
                answer=c.clm(st,qq) if variant.startswith('clm') else c.qwen(st,qq)
                row={'dataset':case['dataset'],'id':case['id'],'doc_id':case['doc_id'],
                     'variant':variant,'gold':case['gold'],'context_tokens':case['context_tokens'],
                     'correct':answer['choice']==case['gold'],'cache_setup_ms':setup_ms,
                     'wired_memory_limit_bytes':c.wired_limit,**answer}
                if 'difficulty' in case:row['difficulty']=case['difficulty']
                append(path,row)
                print(json.dumps({'case':index+1,'of':len(cases),'dataset':case['dataset'],
                      'variant':variant,'correct':row['correct'],'tokens':answer['input_tokens'],
                      'ms':round(answer['ms'],1)}),flush=True)
            except Exception as e:
                append(OUT/'errors.jsonl',{'id':case['id'],'variant':variant,'error':repr(e)})
                raise
        c.prefix_cache.clear();c.action_cache.clear();mx.clear_cache()


def latency(c):
    """Controlled token-length sweep; accuracy is not scored on these excerpts."""
    cases=json.loads((OUT/'cases.json').read_text())
    longest=max(cases,key=lambda x:x['context_tokens'])
    # A natural document supplies real prose; repeated only to ensure 16K tokens available.
    token_pool=c.tok.encode(longest['state']+'\n\n'+longest['state'],add_special_tokens=False)
    q={'type':'choice','instructions':'Which action best addresses the document?',
       'criteria':{'review':'Review and summarize the key issues in the document.',
                   'delete':'Delete the document without reading it.',
                   'ignore':'Ignore the document entirely.',
                   'forward':'Forward the document for a detailed specialist assessment.'}}
    path=OUT/'latency.jsonl'
    existing=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={x['target_context_tokens'] for x in existing}
    for n in [256,512,1024,2048,4096,8192,16384]:
        if n in done:continue
        state=c.tok.decode(token_pool[:n])
        c.actions(list(q['criteria'].values()))
        c.prepare_prefix(state,q)
        methods={'clm_cached_actions':lambda:c.clm(state,q),
                 'qwen_cached_prefix':lambda:c.qwen(state,q),
                 'qwen_uncached':lambda:c.qwen(state,q,cached=False)}
        for f in methods.values():f()
        timings={k:[] for k in methods}
        info={}
        rng=random.Random(n)
        for _ in range(5):
            order=list(methods);rng.shuffle(order)
            for m in order:
                a=methods[m]();timings[m].append(a['ms']);info[m]=a
        row={'target_context_tokens':n,'context_tokens':len(c.tok.encode(state)),
             'wired_memory_limit_bytes':c.wired_limit,
             'methods':{k:{'p50_ms':float(np.median(v)),'samples_ms':v,'example':info[k]} for k,v in timings.items()}}
        append(path,row)
        print('LATENCY',n,json.dumps({k:round(v['p50_ms'],1) for k,v in row['methods'].items()}),flush=True)
        c.prefix_cache.clear();c.action_cache.clear();mx.clear_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['accuracy','latency','both'],default='both');a=ap.parse_args()
    c=LongComparison()
    c.meta.update({'prefill_chunk_size':c.chunk_size,'max_accuracy_prompt_tokens':16384,
                   'cases_sha256':hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest()})
    (OUT/'runtime.json').write_text(json.dumps(c.meta,indent=2))
    warmq={'type':'choice','instructions':'Which color is the sky?','criteria':{'blue':'Blue','red':'Red'}}
    for _ in range(3):c.clm('Clear sunny day.',warmq);c.qwen('Clear sunny day.',warmq)
    if a.mode in ('accuracy','both'):accuracy(c)
    if a.mode in ('latency','both'):latency(c)
    print('DONE',flush=True)


if __name__=='__main__':main()
