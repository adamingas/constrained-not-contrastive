"""Exploratory raw Answer: ablation, keeping rules/actions/state/labels fixed."""
import json,sys,time,random
import bench
from bench import OUT,SPECS,CachedComparison,Trex,MarioClient,append
from mlx_lm.models.cache import make_prompt_cache
class RawComparison(CachedComparison):
 def texts(self,game,method,state):
  if method=='clm':return super().texts(game,method,state)
  msgs=super().texts(game,method,state)
  return msgs[0]['content']+'\n\nCurrent state:\n'+msgs[1]['content']+'\n\nAnswer:'
 def tokens(self,game,method,state):
  if method=='clm':return super().tokens(game,method,state)
  msgs=CachedComparison.texts(self,game,method,state)
  ids=self.tok.encode(self.texts(game,method,state),add_special_tokens=True)
  prefix=self.tok.encode(msgs[0]['content']+'\n\nCurrent state:\n',add_special_tokens=True)[:-1]
  assert ids[:len(prefix)]==prefix
  return ids,prefix
if __name__=='__main__':
 c=RawComparison();cases=json.loads((OUT/'cases.json').read_text())
 meta={'exploratory':True,'description':'Same rules/actions in raw prefix, same state/question, final Answer:; one bare letter; no chat template','setup':{g:{k:v for k,v in c.fixed[g]['qwen'].items() if k in ['setup_prefix_ms','labels']} for g in SPECS},'prompts':{g:c.texts(g,'qwen',next(s['state'] for s in cases if s['game']==g)) for g in SPECS}}
 (OUT/'raw_metadata.json').write_text(json.dumps(meta,indent=2))
 for g in SPECS:c.decide(g,'qwen',next(s['state'] for s in cases if s['game']==g))
 done={r['id'] for r in map(json.loads,(OUT/'raw_matched.jsonl').read_text().splitlines())} if (OUT/'raw_matched.jsonl').exists() else set()
 for case in cases:
  if case['id'] not in done:append('raw_matched.jsonl',{'id':case['id'],'game':case['game'],'method':'qwen_raw','cached':True,**c.decide(case['game'],'qwen',case['state'])})
  if case['id']%30==0:print('raw matched',case['id'],flush=True)
 done={(r['game'],str(r.get('seed',r.get('level')))) for r in map(json.loads,(OUT/'raw_episodes.jsonl').read_text().splitlines())} if (OUT/'raw_episodes.jsonl').exists() else set()
 for seed in range(5):
  if ('trex',str(seed)) in done:continue
  t=Trex(seed);records=[]
  while t.frame<1800 and not t.game.crashed:
   state=t.observe();a=c.decide('trex','qwen',state);records.append({'frame':t.frame,'state':state,**a});t.step(a['choice'])
   if len(records)%50==0:print('raw trex',seed,t.frame,flush=True)
  r={'game':'trex','seed':seed,'method':'qwen_raw','frames':t.frame,'score':t.game.score,'survived':not t.game.crashed,'records':records};append('raw_episodes.jsonl',r);print({k:v for k,v in r.items() if k!='records'},flush=True)
 m=MarioClient()
 for level in ['1-1','1-2','2-1']:
  if ('mario',level) in done:continue
  o=m.call('reset',level=level,record=str(OUT/f'mario_{level}_qwen_raw.gif'));records=[];last_progress=0;best=o['max_x'];stalled=False
  while o['frame']<1800 and not o['done']:
   a=c.decide('mario','qwen',o['state']);records.append({'frame':o['frame'],'state':o['state'],**a});o=m.call('step',action=a['choice'])
   if o['max_x']>best:best=o['max_x'];last_progress=o['frame']
   if o['frame']-last_progress>=300:stalled=True;break
   if len(records)%25==0:print('raw mario',level,o['frame'],o['max_x'],flush=True)
  m.call('save');r={'game':'mario','level':level,'method':'qwen_raw','frames':o['frame'],'max_x':o['max_x'],'flag_get':o['info']['flag_get'],'died':o['done'] and not o['info']['flag_get'],'stalled':stalled,'records':records};append('raw_episodes.jsonl',r);print({k:v for k,v in r.items() if k!='records'},flush=True)
 m.close()
