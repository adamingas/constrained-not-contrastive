"""Matched timing, exact decoder audit, repeated tool menus, and actual link races.

Run after the held-out suite; no prompt changes or new learned weights.
"""
import argparse,hashlib,json,random,time
from collections import defaultdict
from pathlib import Path
from run import Bench,OUT,append
from prepare import DATA,graph,wiki_case,rows

def repeated_cases(cases):
 groups=[]
 for category in ['live_multiple','live_irrelevance']:
  source=rows('BFCL_v4_'+category+'.json')
  answers={x['id']:x['ground_truth'] for x in rows('possible_answer__BFCL_v4_'+category+'.json')} if category=='live_multiple' else None
  by_schema=defaultdict(list)
  for r in source:
   fingerprint=hashlib.sha256(json.dumps(r['function'],sort_keys=True).encode()).hexdigest()
   by_schema[fingerprint].append(r)
  chosen=0
  for base in cases:
   if base['split']!='test' or base['dataset']!='bfcl_'+category:continue
   eligible=[]
   for r in by_schema[base['schema_id']]:
    if r['id']==base['id'] or len(r['question'])!=1:continue
    gt=answers[r['id']] if answers is not None else []
    if gt and (len(gt)!=1 or len(gt[0])!=1):continue
    target=list(gt[0])[0] if gt else 'NO_TOOL'
    if target not in base['actions']:continue
    state='\n'.join(m['role'].capitalize()+': '+str(m['content']) for m in r['question'][0])
    if len(state)>4000:continue
    eligible.append({**base,'id':r['id'],'state':state,'gold':[target]})
   if len(eligible)<2:continue
   groups.append({'schema_id':base['schema_id'],'dataset':base['dataset'],'cases':eligible[:5]})
   chosen+=1
   if chosen==3:break
 return groups

def main(stage):
 cases=json.loads((OUT/'cases.json').read_text());frozen=json.loads((OUT/'frozen.json').read_text())
 if stage=='prepare':
  groups=repeated_cases(cases)
  (OUT/'repeated_cases.json').write_text(json.dumps(groups,indent=2))
  print('Frozen repeated menus:',len(groups),'new requests:',sum(len(g['cases']) for g in groups));return
 c=Bench()
 if stage=='latency':
  selected=[]
  for ds,n in [('bfcl_multiple',4),('bfcl_live_multiple',3),('bfcl_live_irrelevance',3),('wikispeedia',10)]:
   selected.extend([x for x in cases if x['split']=='test' and x['dataset']==ds][:n])
  # Same forced-tool diagnostic, with matched timing on ten fixed positive cases.
  for ds in ['bfcl_multiple','bfcl_live_multiple']:
   for case in [x for x in cases if x['split']=='test' and x['dataset']==ds][:5]:
    selected.append({**case,'closed_set':True,'family':'bfcl','timing_family':'bfcl_forced_tool','actions':{k:v for k,v in case['actions'].items() if k!='NO_TOOL'},'question':'Which available function best fulfills this request?','choice_count':case['choice_count']-1})
  prompts=[]
  for i,case in enumerate(selected):
   variants=['clm_base','clm_prose_base','qwen_raw_base'] if case.get('closed_set') else ['clm_base','qwen_raw_base','qwen_raw_tuned']
   for v in variants:c.prepare(case,v);c.decide(case,v)
   for repeat in range(5):
    order=variants.copy();random.Random(800+i*5+repeat).shuffle(order)
    for v in order:
     ans=c.decide(case,v)
     append('latency.jsonl',{'id':case['id'],'family':case.get('timing_family',case['family']),'dataset':case['dataset'],'variant':v,'repeat':repeat,**ans})
   if i in [0,4,7,10,11,12,13,14,15]:
    for v in ['qwen_raw_base','qwen_raw_tuned']:
     a=c.decide(case,v);b=c.decide(case,v,full=True)
     error=max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))
     append('decoder_audit.jsonl',{'id':case['id'],'variant':v,'choice_count':case['choice_count'],'selected_choice':a['choice'],'full_vocab_choice':b['choice'],'same_choice':a['choice']==b['choice'],'max_probability_error':error})
     assert a['choice']==b['choice']
   if i in [0,10,20]:
    for v in variants:
     a,b=c.texts(case,v);ids,prefix=c.tokens(case,v)
     from clm.schema import to_text
     action_texts={k:to_text(json.loads(t)) if '_prose_' in v and case['family']=='bfcl' and k!='NO_TOOL' else t for k,t in case['actions'].items()}
     prompts.append({'case_id':case['id'],'variant':v,'closed_set':case.get('closed_set',False),'format':'raw text, no chat template or separate API roles','fixed_prefix_text':a,'changing_suffix_text':b,'exact_full_input_text':a+b,'full_token_ids':ids,'cached_prefix_token_ids':prefix,'allowed_labels':c.labels[:len(case['actions'])] if v.startswith('qwen') else None,'actions':case['actions'],'action_encoder_texts':action_texts if v.startswith('clm') else None,'gold_for_scoring_only':case['gold']})
   c.clear();print('latency',i+1,case['id'],flush=True)
  (OUT/'exact_prompts.json').write_text(json.dumps(prompts,indent=2,ensure_ascii=False))
 elif stage=='repeated':
  groups=json.loads((OUT/'repeated_cases.json').read_text())
  for i,g in enumerate(groups):
   variants=['clm_base','qwen_raw_base','qwen_raw_tuned'];random.Random(i+995).shuffle(variants)
   for v in variants:
    setup=c.prepare(g['cases'][0],v)
    for j,case in enumerate(g['cases']):
     ans=c.decide(case,v)
     append('repeated.jsonl',{'schema_id':g['schema_id'],'id':case['id'],'dataset':case['dataset'],'variant':v,'request_index':j,'menu_setup_ms':setup['setup_ms'] if j==0 else 0,**ans})
    c.clear()
   print('repeated menu',i+1,len(g['cases']),flush=True)
 elif stage=='navigation':
  g,rev=graph();starts=[x for x in cases if x['split']=='test' and x['family']=='wiki'][:5]
  for i,start in enumerate(starts):
   variants=frozen['navigation']['variants'].copy();random.Random(i+669).shuffle(variants)
   for v in variants:
    path=[start['src']];menu_cache={};steps=[];began=time.perf_counter()
    for step in range(frozen['navigation']['max_clicks']):
     case=wiki_case(path[-1],start['target'],g,rev,path)
     if not case['actions']:break
     if case['src'] in menu_cache:
      c.menus[v]=menu_cache[case['src']];setup_ms=0
     else:
      setup=c.prepare(case,v,reuse_actions=True);setup_ms=setup['setup_ms'];menu_cache[case['src']]=c.menus[v]
     ans=c.decide(case,v);path.append(ans['choice'])
     steps.append({'src':case['src'],'distance_before':case['distance'],'choice_count':case['choice_count'],'setup_ms':setup_ms,**ans})
     print('navigation',i+1,v,step+1,path[-2],'->',path[-1],flush=True)
     if path[-1]==start['target']:break
    append('navigation.jsonl',{'id':start['id'],'variant':v,'success':path[-1]==start['target'],'path':path,'shortest_possible_clicks':start['distance'],'steps':steps,'wall_ms':(time.perf_counter()-began)*1000})
    c.clear()
 print('DONE',stage,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','latency','repeated','navigation']);main(ap.parse_args().stage)
