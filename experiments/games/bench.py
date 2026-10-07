"""Matched-state prefix-cache benchmark and unassisted lockstep game episodes."""
import copy,hashlib,json,random,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from long_context_bench import LongComparison
from compare import ROOT,norm
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from specs import SPECS
sys.path.insert(0,str(ROOT/'upstream/examples/t_rex'))
from trex.engine import Game
from trex.planner import Planner,snapshot
OUT=ROOT/'results/cached_games';OUT.mkdir(exist_ok=True)
def append(name,obj):
 with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n');f.flush()
class MarioClient:
 def __init__(self):self.p=subprocess.Popen([str(ROOT/'.venv-games/bin/python'),str(Path(__file__).with_name('mario_worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 def call(self,cmd,**kw):
  self.p.stdin.write(json.dumps({'cmd':cmd,**kw})+'\n');self.p.stdin.flush();r=json.loads(self.p.stdout.readline())
  if 'error' in r:raise RuntimeError(r)
  return r
 def close(self):self.p.stdin.close();self.p.wait()
class Trex:
 def __init__(self,seed):self.game=Game(seed);self.game.press_jump();self.held='run';self.frame=0;self.planner=Planner()
 def observe(self):
  g=self.game;t=g.trex
  obstacles='; '.join(f'{o.kind.name}, left_x={o.x}, top_y={o.y}, width={o.width}, height={o.kind.height}, leftward speed={g.speed+o.speed_offset:.2f}, gap from dinosaur front={o.x-94}' for o in g.obstacles if not o.remove and o.x<600) or 'none'
  return f'Dinosaur top_y={t.y:.1f}, vertical velocity={t.velocity:.1f}, airborne={t.jumping}, ducking={t.ducking}, rapid descent={t.speed_drop}. Running speed={g.speed:.2f} pixels/frame. Previous held action={self.held}.\nVisible obstacles: {obstacles}.\nFrame {self.frame}; score={g.score}.'
 def plan(self):return self.planner.plan(snapshot(self.game,self.held),(0,0),(6,6))
 def step(self,action):
  g=self.game
  if action=='jump':
   if not g.trex.jumping:
    if g.trex.ducking:g.release_duck()
    g.press_jump()
   self.held='run'
  else:self.held=action
  for _ in range(6):
   if self.held=='duck':
    if g.trex.jumping:
     if not g.trex.speed_drop:g.press_duck()
    elif not g.trex.ducking:g.press_duck()
   elif g.trex.speed_drop or (not g.trex.jumping and g.trex.ducking):g.release_duck()
   g.step();self.frame+=1
   if g.crashed:break
class CachedComparison(LongComparison):
 def __init__(self):
  super().__init__();self.fixed={}
  for game,s in SPECS.items():
   self.fixed[game]={}
   for method in ['qwen','clm']:
    start=time.perf_counter();ids,prefix=self.tokens(game,method,'DYNAMIC_STATE_SENTINEL')
    cache=make_prompt_cache(self.model);self.last(prefix,cache)
    d={'prefix':prefix,'cache':cache,'setup_prefix_ms':(time.perf_counter()-start)*1000}
    if method=='clm':
     start=time.perf_counter();d['actions']=self.actions(list(s['actions'].values()));mx.eval(d['actions']);d['setup_actions_ms']=(time.perf_counter()-start)*1000
    else:
     labels=[self.tok.encode(chr(65+i),add_special_tokens=False) for i in range(len(s['actions']))];assert all(len(l)==1 for l in labels)
     d['labels']=[l[0] for l in labels];d['rows']=self.model.lm_head.weight[mx.array(d['labels'])];mx.eval(d['rows'])
    self.fixed[game][method]=d
 def texts(self,game,method,state):
  s=SPECS[game]
  if method=='clm':return s['rules']+'\n\nCurrent state:\n'+state+'\n\n'+s['question']
  system=s['rules']+'\n\nChoose exactly one action letter.\n'+'\n'.join(f'{chr(65+i)}: {k}: {v}' for i,(k,v) in enumerate(s['actions'].items()))
  return [{'role':'system','content':system},{'role':'user','content':state+'\n\n'+s['question']}]
 def tokens(self,game,method,state):
  if method=='clm':
   prefix_text=SPECS[game]['rules']+'\n\nCurrent state:\n'
   ids=self.tok.encode(self.texts(game,method,state),add_special_tokens=True)
   prefix=self.tok.encode(prefix_text,add_special_tokens=True)
   # Leave the potentially merge-sensitive final token uncached.
   prefix=prefix[:-1]
  else:
   messages=self.texts(game,method,state)
   ids=self.tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
   prefix=self.tok.encode(self.tok.apply_chat_template([messages[0]],tokenize=False,add_generation_prompt=False)+'<|im_start|>user\n',add_special_tokens=False)
  assert ids[:len(prefix)]==prefix
  return ids,prefix
 def decide(self,game,method,state,cached=True,full_vocab=False):
  start=time.perf_counter();ids,prefix=self.tokens(game,method,state);total=len(ids);d=self.fixed[game][method]
  assert prefix==d['prefix'];cache=copy.deepcopy(d['cache']) if cached else None
  if cached:ids=ids[len(prefix):]
  h=self.last(ids,cache)
  if method=='qwen':
   logits=(self.model.lm_head(h)[0,mx.array(d['labels'])] if full_vocab else (h@d['rows'].T)[0]).astype(mx.float32)
  else:logits=self.head.scale*(self.project(norm(h),'state')@d['actions'].T)[0]
  probs=mx.softmax(logits);mx.eval(probs);values=probs.tolist();index=int(np.argmax(values));elapsed=(time.perf_counter()-start)*1000
  return {'choice':list(SPECS[game]['actions'])[index],'probabilities':values,'ms':elapsed,'input_tokens':total,'prefix_tokens':len(prefix),'processed_tokens':len(ids)}
def make_cases():
 cases=[];references=[]
 for seed in range(5):
  t=Trex(seed);states=[]
  while t.frame<1800 and not t.game.crashed:
   p=t.plan();states.append({'game':'trex','state':t.observe(),'seed':seed,'frame':t.frame,'safe':p.safe,'best':p.best,'robust':p.robust});t.step(p.best)
  # 12 near-obstacle and 4 quiet samples per seed, sampled deterministically.
  near=[s for s in states if 'Visible obstacles: none' not in s['state']];quiet=[s for s in states if s not in near]
  rng=random.Random(101+seed);chosen=rng.sample(near,min(12,len(near)))+rng.sample(quiet,min(4,len(quiet)))
  cases.extend(chosen);references.append({'game':'trex','seed':seed,'method':'planner','frames':t.frame,'score':t.game.score,'survived':not t.game.crashed})
 m=MarioClient()
 for level in ['1-1','1-2','2-1']:
  o=m.call('reset',level=level,record=str(OUT/f'reference_{level}.gif'));states=[]
  while o['frame']<1800 and not o['done']:
   states.append({'game':'mario','state':o['state'],'level':level,'frame':o['frame']});a=m.call('reference')['choice'];o=m.call('step',action=a)
  cases.extend(random.Random(2026).sample(states,min(24,len(states))));m.call('save')
  references.append({'game':'mario','level':level,'method':'heuristic','frames':o['frame'],'max_x':o['max_x'],'flag_get':o['info']['flag_get'],'died':o['done'] and not o['info']['flag_get']})
 m.close()
 for i,s in enumerate(cases):s['id']=i
 (OUT/'cases.json').write_text(json.dumps(cases,indent=2));(OUT/'references.json').write_text(json.dumps(references,indent=2));print('prepared',len(cases),references,flush=True)
def matched(c):
 cases=json.loads((OUT/'cases.json').read_text());path=OUT/'matched.jsonl';done={(r['id'],r['method'],r['cached']) for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
 for game in SPECS:
  state=next(x['state'] for x in cases if x['game']==game)
  for method in ['clm','qwen']:
   for cached in [False,True]:c.decide(game,method,state,cached)
 for case in cases:
  variants=[('clm',False),('clm',True),('qwen',False),('qwen',True)];random.Random(case['id']).shuffle(variants)
  for method,cached in variants:
   if (case['id'],method,cached) in done:continue
   a=c.decide(case['game'],method,case['state'],cached)
   append('matched.jsonl',{'id':case['id'],'game':case['game'],'method':method,'cached':cached,**a})
  if case['id']%10==0:print('matched',case['id'],'/',len(cases),flush=True)
 # Audit selected LM rows vs full-vocabulary output, same cached next-token decision.
 audited={(r['game'],r['id']) for r in map(json.loads,(OUT/'audit.jsonl').read_text().splitlines())} if (OUT/'audit.jsonl').exists() else set()
 for game in SPECS:
  for case in [s for s in cases if s['game']==game][:5]:
   if (game,case['id']) in audited:continue
   a=c.decide(game,'qwen',case['state']);b=c.decide(game,'qwen',case['state'],full_vocab=True)
   append('audit.jsonl',{'game':game,'id':case['id'],'selected':a,'full_vocab':b});assert a['choice']==b['choice']
def episodes(c):
 path=OUT/'episodes.jsonl';done={(r['game'],str(r.get('seed',r.get('level'))),r['method']) for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
 for seed in range(5):
  methods=['clm','qwen'] if seed%2==0 else ['qwen','clm']
  for method in methods:
   if ('trex',str(seed),method) in done:continue
   t=Trex(seed);records=[]
   while t.frame<1800 and not t.game.crashed:
    state=t.observe();a=c.decide('trex',method,state);records.append({'frame':t.frame,'state':state,**a});t.step(a['choice'])
    if len(records)%50==0:print('play trex',seed,method,t.frame,flush=True)
   row={'game':'trex','seed':seed,'method':method,'frames':t.frame,'score':t.game.score,'survived':not t.game.crashed,'records':records}
   append('episodes.jsonl',row);print('episode',json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
 m=MarioClient()
 for li,level in enumerate(['1-1','1-2','2-1']):
  for method in (['clm','qwen'] if li%2==0 else ['qwen','clm']):
   if ('mario',level,method) in done:continue
   o=m.call('reset',level=level,record=str(OUT/f'mario_{level}_{method}.gif'));records=[];last_progress=0;best=o['max_x'];stalled=False
   while o['frame']<1800 and not o['done']:
    a=c.decide('mario',method,o['state']);records.append({'frame':o['frame'],'state':o['state'],**a});o=m.call('step',action=a['choice'])
    if o['max_x']>best:best=o['max_x'];last_progress=o['frame']
    if o['frame']-last_progress>=300:stalled=True;break
    if len(records)%25==0:print('play mario',level,method,o['frame'],o['max_x'],flush=True)
   m.call('save');row={'game':'mario','level':level,'method':method,'frames':o['frame'],'max_x':o['max_x'],'flag_get':o['info']['flag_get'],'died':o['done'] and not o['info']['flag_get'],'stalled':stalled,'records':records}
   append('episodes.jsonl',row);print('episode',json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
 m.close()
def main():
 if not (OUT/'cases.json').exists():make_cases()
 if '--prepare-only' in sys.argv:return
 c=CachedComparison();meta={**c.meta,'specs':SPECS,'seed':20261003,'mode':'lockstep; no request memoization; no planner in model inputs','cached':{g:{m:{k:v for k,v in d.items() if k in ('setup_prefix_ms','setup_actions_ms','labels')} for m,d in ms.items()} for g,ms in c.fixed.items()}}
 meta['hardware']={'model':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))}
 meta['game_packages']=Path(__file__).with_name('requirements-games.txt').read_text()
 meta['artifact_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('specs.py'),Path(__file__).with_name('mario_worker.py'),OUT/'cases.json']}
 (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
 for game in SPECS:
  state=next(x['state'] for x in json.loads((OUT/'cases.json').read_text()) if x['game']==game)
  (OUT/f'{game}_prompts.json').write_text(json.dumps({m:{'text':c.texts(game,m,state),'token_ids':c.tokens(game,m,state)[0],'prefix_token_ids':c.tokens(game,m,state)[1]} for m in ['clm','qwen']},indent=2))
 matched(c);episodes(c)
def controls_only_pilot():
 """One bounded episode with worked examples and no live action hints."""
 sys.path.insert(0, str(ROOT/'experiments/benchmarks'))
 from bfcl_500_noschema import Runtime
 from specs import TREX_RULES, TREX_ACTIONS, TREX_EXAMPLES, CLM_PROMPT_ADDITIONS, CLM_RULES, CLM_ACTION_TEXTS, CLM_QUESTION, clm_observation, clm_example
 from trex.engine import obstacle_step
 rt = Runtime()
 seed = int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 42
 actions = list(TREX_ACTIONS)
 system = TREX_RULES + '\n\nControls:\n' + '\n'.join(
  f'{chr(65+i)} = {a}: {desc}' for i,(a,desc) in enumerate(TREX_ACTIONS.items()))
 system += '\n\n' + '\n'.join(TREX_EXAMPLES)
 clm_mode = '--clm' in sys.argv
 examples = list(TREX_EXAMPLES)
 questions = None
 if clm_mode:
  assert '--plain-state' in sys.argv and '--nearest-only' in sys.argv
  assert not any(flag in sys.argv for flag in ('--word-actions','--geometry'))
  examples = [clm_example(example) for example in CLM_PROMPT_ADDITIONS]
  system = CLM_RULES + ('\n\n'+'\n\n'.join(examples) if examples else '')
  import torch
  from clm.engine import Engine
  from clm.schema import build_pairs
  torch.set_num_threads(4)
  class ActionCacheEngine(Engine):
   def _cached(self,namespace,dim,texts,tokens,project=None):
    if namespace.endswith('/state'):
     emb,spent=self.embedder.embed(texts);tokens.append(spent)
     return project(emb) if project else self._to_device(emb)
    return super()._cached(namespace,dim,texts,tokens,project)
  engine = ActionCacheEngine(embedder=rt,checkpoint=str(ROOT/'models/clm/CLM_v0.1-8B.pt'),device='cpu',action_cache='32MiB')
  questions = {'action':{'type':'choice','instructions':'' if '--clm-chat' in sys.argv else CLM_QUESTION,'criteria':dict(CLM_ACTION_TEXTS)}}
  def clm_input(observation,previous=None):
   observed=clm_observation(observation)
   earlier=''
   if '--history' in sys.argv and previous is not None:
    earlier='\n\nPrevious observation, six frames earlier:\n'+clm_observation(previous['observation'])
   if '--clm-chat' in sys.argv:
    return rt.tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':earlier+'\n\nCurrent observation:\n'+observed+'\n\n'+CLM_QUESTION}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   return system+earlier+'\n\nCurrent observation:\n'+observed
  if '--clm-probe' in sys.argv:
   saved=[json.loads(line) for line in (OUT/'raw_episodes.jsonl').read_text().splitlines()]
   source=next(row for row in reversed(saved) if row['method']=='qwen_worked_examples_progressive' and row['seed']==42 and row['frames']==1800)
   if '--history' in sys.argv:
    source=max((row for row in saved if row['method']=='clm_worked_examples_natural_actions' and row['seed']==42),key=lambda row:row['frames'])
   probes=[]
   source_name='clm_reference' if '--history' in sys.argv else 'qwen_reference'
   selected=[(source_name,row) for row in source['records'] if row['frame'] in (120,270,288,300,312,318,324,336,348,378,384,390,420,438,444,996,1002,1008,1014)]
   latest_clm=next(row for row in reversed(saved) if row['method']=='clm_worked_examples_natural_actions')
   grounded=[row for row in latest_clm['records'] if 'dinosaur is on the ground' in row['observation']]
   selected += [('previous_clm_run',row) for row in grounded[-5:]]
   speed_case=next(row for row in saved if row['method']=='clm_worked_examples_natural_actions' and row['seed']==42 and row['frames']==1122)
   selected += [('faster_cactus_regression',row) for row in speed_case['records'][-5:]]
   histories={source_name:{row['frame']:row for row in source['records']},'previous_clm_run':{row['frame']:row for row in latest_clm['records']},'faster_cactus_regression':{row['frame']:row for row in speed_case['records']}}
   for source_name,row in selected:
    previous=histories[source_name].get(row['frame']-6)
    if previous is not None:assert previous['frame']+6==row['frame']
    state=clm_input(row['observation'],previous)
    exact_state,_,exact_actions=build_pairs(state,questions)['action']
    a=engine.answer(state,questions)['answers']['action']
    probes.append({'source':source_name,'frame':row['frame'],'observation':row['observation'],'previous_observation':previous['observation'] if '--history' in sys.argv and previous else None,'exact_state_input':exact_state,**a})
    print(json.dumps({'source':source_name,'frame':row['frame'],'choice':a['choice'],'p_run':round(a['probabilities']['run'],3),'p_jump':round(a['probabilities']['jump'],3)}),flush=True)
   append('audit.jsonl',{'type':'clm_prompt_probe','history':('--history' in sys.argv),'system':system,'prompt_additions':CLM_PROMPT_ADDITIONS,'questions':questions,'exact_action_inputs':exact_actions,'probes':probes})
   return
 def output_words(text):
  import re
  for letter,word in zip('ABC',actions):text=re.sub(r'\b'+letter+r'\b',word,text)
  text=text.replace('run jump rises', 'A jump rises').replace('run grounded dinosaur', 'A grounded dinosaur')
  return text.replace('one letter','one action word').replace('chosen letter','chosen action word').replace('allowed letter','allowed action word')
 def geometric(text):
  import re
  text=text.replace('T-Rex runner', 'a scrolling geometry game').replace('dinosaur','player body').replace('Dinosaur','Player body')
  return re.sub(r'cactus(?:Large|Small)?', 'rectangle', text)
 geometry_mode = '--geometry' in sys.argv
 if geometry_mode:system=geometric(system)
 word_actions = '--word-actions' in sys.argv
 if word_actions:system=output_words(system)
 allowed = [rt.tok.encode(actions[i] if word_actions else chr(65+i), add_special_tokens=False) for i in range(3)]
 assert all(len(ids)==1 for ids in allowed)
 prefix = rt.tok.apply_chat_template([{'role':'system','content':system}], tokenize=False, add_generation_prompt=False) + '<|im_start|>user\n'
 print(json.dumps({'protocol': 'One prompt, at most 1800 frames, stop on first collision; thinking disabled; no planner or action corrections. One startup jump activates the unmodified original game.', 'seed':seed, 'example_blocks':len(examples)}), flush=True)
 g=Game(seed); g.press_jump(); frame=0; records=[]; seen={}; passed=set()
 while frame < 1800 and not g.crashed:
  t=g.trex
  visible=[o for o in g.obstacles if not o.remove and o.x<600 and o.x+o.width>0]
  if '--nearest-only' in sys.argv:visible=sorted((o for o in visible if o.x+o.width>=t.x),key=lambda o:o.x)[:1]
  state=(f'Dinosaur: '+('airborne' if t.jumping else 'grounded')+
         f'; feet {140-(t.y+47):.0f} pixels above ground; vertical speed {-t.velocity:.1f} pixels/frame upward; ducking={t.ducking}.\n')
  state+='Obstacles, nearest first:\n'+('\n'.join(
   f'{o.kind.name}: front gap {o.x-(t.x+44)} pixels; width {o.width} pixels; lower edge {140-(o.y+o.kind.height)} and upper edge {140-o.y} pixels above ground; approaching at {obstacle_step(g.speed,o.speed_offset)} pixels/frame ({(o.x-(t.x+44))/obstacle_step(g.speed,o.speed_offset):.1f} frames until front edges meet; {(o.x+o.width-t.x)/obstacle_step(g.speed,o.speed_offset):.1f} frames until its back edge passes the player rear).' for o in visible) or 'None.')
  state+='\nChoose for the next 6 frames only: A = continue running/current flight, B = initiate jump, C = duck. Consider when the obstacle arrives, not just whether it is visible. Reply with one letter.'
  if '--plain-state' in sys.argv:
   if t.jumping:
    state=f'State: The dinosaur is in the air, with its feet {93-t.y:.0f} pixels above the ground, '+('rising' if t.velocity<0 else 'falling')+f' at {abs(t.velocity):.1f} pixels per frame. '
   else:state='State: The dinosaur is on the ground. '+('It is ducking. ' if t.ducking else '')
   for o in visible:
    kind='cactus' if o.kind.name.startswith('cactus') else 'bird'
    state+=f'A {kind} is {o.x-(t.x+44)} pixels beyond its front, {o.width} pixels wide and {o.kind.height} pixels tall, moving toward it at {obstacle_step(g.speed,o.speed_offset)} pixels per frame. '
    if kind=='bird':state+=f'The bird bottom is {140-o.y-o.kind.height} pixels above the ground. '
   if not visible:state+='No obstacles are visible. '
   state+='\nAnswer:'
  observation=state
  if '--history' in sys.argv and records and not clm_mode:
   prior=records[-1]
   state='Previous observation:\n'+prior['observation']+'\nControl you chose: '+prior['choice']+'\n\nCurrent observation after 6 frames:\n'+state
  if geometry_mode:state=geometric(state)
  if word_actions:state=output_words(state)
  if clm_mode:
   state=clm_input(state,records[-1] if records else None)
   exact_state,_,exact_actions=build_pairs(state,questions)['action']
   started=time.perf_counter()
   output=engine.answer(state,questions)
   answer={**output['answers']['action'],'ms':(time.perf_counter()-started)*1000,
           'processed_tokens':output['usage']['input_tokens'],'exact_state_input':exact_state}
  else:
   prompt=rt.tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':state}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   ids=rt.tok.encode(prompt,add_special_tokens=False); prefix_ids=rt.tok.encode(prefix,add_special_tokens=False)
   assert ids[:len(prefix_ids)]==prefix_ids
   answer=rt.qwen({'id':str(frame),'qwen_prompt':prompt,'qwen_preamble':prefix,'qwen_input_ids':ids,'qwen_prefix_ids':prefix_ids,'allowed_token_ids':[x[0] for x in allowed],'action_keys':actions})
  record={'frame':frame,'observation':observation,'state':state,**answer};records.append(record)
  if (answer['choice']=='jump' and not t.jumping) or (visible and frame%120==0): print(json.dumps({'frame':frame,'choice':answer['choice'],'probabilities':answer['probabilities']}),flush=True)
  action=answer['choice']
  if action in ('run','jump'):g.release_duck()
  if action=='jump' and not t.jumping:g.press_jump()
  if action=='duck':g.press_duck()
  for _ in range(6):
   for o in g.obstacles:seen[o.id]=o
   g.step();frame+=1
   if g.crashed:break
   for oid,o in seen.items():
    if o.x+o.width < g.trex.x:passed.add(oid)
  if frame%120==0:print(json.dumps({'progress_frames':frame,'passed':len(passed)}),flush=True)
 from collections import Counter
 result={'frames':frame,'game_seconds':frame/60,'crashed':g.crashed,'score':g.score,'obstacles_passed':len(passed),'decisions':len(records),'actions':dict(Counter(r['choice'] for r in records)),'last_decision':records[-1]}
 append('raw_episodes.jsonl', {'game':'trex','method':'clm_worked_examples_natural_actions' if clm_mode else 'qwen_worked_examples_progressive','prompt_additions':CLM_PROMPT_ADDITIONS if clm_mode else [],'seed':seed,'word_actions':word_actions,'history':('--history' in sys.argv),'geometry_mode':geometry_mode,'nearest_only':('--nearest-only' in sys.argv),'plain_state':('--plain-state' in sys.argv),'system':system,'examples':examples,'questions':questions,'exact_action_inputs':exact_actions if clm_mode else None,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,**result})
 print(json.dumps({'result':{k:v for k,v in result.items() if k!='last_decision'}}),flush=True)

def warm_cache_comparison():
 """Bounded games per model, frozen individual prompts, fixed-prefix KV on both."""
 import math, statistics, torch
 from datetime import datetime, timezone
 sys.path.insert(0, str(ROOT/'experiments/benchmarks'))
 from bfcl_500_noschema import Runtime
 from specs import clm_observation
 from clm.engine import Engine
 from clm.embedder import l2
 from clm.schema import build_pairs
 from trex.engine import obstacle_step
 torch.set_num_threads(4)
 saved=[json.loads(line) for line in (OUT/'raw_episodes.jsonl').read_text().splitlines()]
 fresh_seed=int(sys.argv[sys.argv.index('--new-seed')+1]) if '--new-seed' in sys.argv else None
 seeds=[fresh_seed] if fresh_seed is not None else [42,314159,271828]
 if fresh_seed is not None:
  assert not any(r.get('game')=='trex' and r.get('seed')==fresh_seed for r in saved), 'Requested seed has already been tested'
 sources={m:next(r for r in reversed(saved) if r['method']==name and r['seed']==42 and r['frames']==1800)
          for m,name in [('qwen','qwen_worked_examples_progressive'),('clm','clm_worked_examples_natural_actions')]}
 systems={m:r['system'] for m,r in sources.items()}
 questions=sources['clm']['questions']
 assert sources['clm']['exact_action_inputs']==['Run.','Jump.','Duck.']
 rt=Runtime(); actions=['run','jump','duck']
 run_id=datetime.now(timezone.utc).isoformat()
 qprefix=rt.tok.apply_chat_template([{'role':'system','content':systems['qwen']}],tokenize=False,add_generation_prompt=False)+'<|im_start|>user\n'
 cprefix=systems['clm']+'\n\nCurrent observation:\n'
 prefixes={'qwen':rt.tok.encode(qprefix,add_special_tokens=False),
           'clm':rt.tok.encode(cprefix,add_special_tokens=True)[:-1]}
 caches={};setup={}
 for method,ids in prefixes.items():
  began=time.perf_counter();cache=make_prompt_cache(rt.model);rt.forward(ids,cache)
  for layer in cache:
   layer.keys=layer.keys[...,:len(ids),:];layer.values=layer.values[...,:len(ids),:]
  mx.eval([layer.state for layer in cache]);caches[method]=cache
  setup[method]={'prefix_ms':(time.perf_counter()-began)*1000,'prefix_tokens':len(ids)}
 allowed=[rt.tok.encode(x,add_special_tokens=False) for x in 'ABC']
 assert all(len(x)==1 for x in allowed)
 label_rows=rt.model.lm_head.weight[mx.array([x[0] for x in allowed])];mx.eval(label_rows)

 class PrefixEmbedder:
  def __init__(self):self.enabled=True;self.calls=[]
  def embed(self,texts):
   vectors=[];spent=0
   for text in texts:
    ids=rt.tok.encode(text,add_special_tokens=True)
    is_state=text.startswith(systems['clm'])
    n=len(prefixes['clm']) if self.enabled and is_state else 0
    if is_state:assert ids[:len(prefixes['clm'])]==prefixes['clm']
    cache=copy.deepcopy(caches['clm']) if n else make_prompt_cache(rt.model)
    h=rt.forward(ids[n:],cache)
    vectors.append(l2(np.asarray(h.astype(mx.float32))[0]));spent+=len(ids)-n
    self.calls.append({'total_tokens':len(ids),'cached_tokens':n,'processed_tokens':len(ids)-n,'state':is_state})
   return np.stack(vectors),spent
 class StateUncachedEngine(Engine):
  def _cached(self,namespace,dim,texts,tokens,project=None):
   if namespace.endswith('/state'):
    emb,spent=self.embedder.embed(texts);tokens.append(spent)
    return project(emb) if project else self._to_device(emb)
   return super()._cached(namespace,dim,texts,tokens,project)
 embedder=PrefixEmbedder()
 engine=StateUncachedEngine(embedder=embedder,checkpoint=str(ROOT/'models/clm/CLM_v0.1-8B.pt'),device='cpu',action_cache='32MiB')
 head=engine.heads['clm-latest']
 began=time.perf_counter()
 engine._cached(f'{head.namespace}/action',head.proj_dim,sources['clm']['exact_action_inputs'],[],head.project_actions)
 setup['clm']['actions_ms']=(time.perf_counter()-began)*1000

 def decide(method,observation,cached=True):
  # Includes prompt construction, tokenization, KV copy, inference, scoring and selection.
  began=time.perf_counter()
  if method=='qwen':
   prompt=rt.tok.apply_chat_template([{'role':'system','content':systems['qwen']},{'role':'user','content':observation}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   ids=rt.tok.encode(prompt,add_special_tokens=False);n=len(prefixes['qwen']) if cached else 0
   assert ids[:len(prefixes['qwen'])]==prefixes['qwen']
   cache=copy.deepcopy(caches['qwen']) if cached else make_prompt_cache(rt.model)
   h=rt.forward(ids[n:],cache)
   probs=mx.softmax((h@label_rows.T)[0].astype(mx.float32));mx.eval(probs)
   values=probs.tolist();choice=actions[int(np.argmax(values))]
   ms=(time.perf_counter()-began)*1000
   result={'choice':choice,'probabilities':dict(zip(actions,values)),'ms':ms,'total_tokens':len(ids),'cached_tokens':n,'processed_tokens':len(ids)-n,'exact_input':prompt}
  else:
   state=cprefix+clm_observation(observation)
   embedder.enabled=cached;embedder.calls=[]
   output=engine.answer(state,questions)
   ms=(time.perf_counter()-began)*1000
   assert len(embedder.calls)==1 and embedder.calls[0]['state'], 'Action cache missed or state computation skipped'
   result={**output['answers']['action'],'ms':ms,**embedder.calls[0],'exact_input':build_pairs(state,questions)['action'][0]}
  assert all(layer.size()==len(prefixes[method]) for layer in caches[method]), 'Fixed cache mutated'
  return result

 # Small numerical check on earlier saved states. Not part of measured episodes.
 validation=[]
 for method in (('clm','qwen') if '--validate-only' in sys.argv else ()):
  for frame in (0,300,324,348,390):
   record=next(r for r in sources[method]['records'] if r['frame']==frame)
   cold=decide(method,record['observation'],False);warm=decide(method,record['observation'],True)
   delta=max(abs(cold['probabilities'][a]-warm['probabilities'][a]) for a in actions)
   check={'method':method,'frame':frame,'cold_choice':cold['choice'],'warm_choice':warm['choice'],'max_probability_delta':delta}
   validation.append(check);print(json.dumps({'validation':check}),flush=True)
   assert cold['choice']==warm['choice'] and delta<0.03, 'Stop and inspect numerical cache mismatch'
 if '--validate-only' not in sys.argv:
  prior=next(json.loads(s) for s in reversed((OUT/'audit.jsonl').read_text().splitlines()) if json.loads(s).get('type')=='trex_warm_cache_protocol')
  assert prior['system']==systems and prior['questions']==questions
  validation=prior['validation'];assert len(validation)==10
  # Only warm scoring kernels on unrelated text; no measured state is answered twice.
  h=rt.forward(rt.tok.encode('The sky is blue.',add_special_tokens=True),make_prompt_cache(rt.model))
  mx.eval(h@label_rows.T)
  head.project_states(l2(np.asarray(h.astype(mx.float32))))
 protocol={'type':'trex_warm_cache_protocol','run_id':run_id,'seeds':seeds,'max_frames':1800,'frames_per_decision':6,
           'system':systems,'questions':questions,'exact_action_inputs':sources['clm']['exact_action_inputs'],
           'setup':setup,'validation':validation,'device':mx.device_info(),'dtype':str(rt.model.model.embed_tokens.weight.dtype),
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'timing':'Fixed prefixes and CLM projected actions prepared once, before gameplay. Each live state evaluated once. Identical timer scope: prompt construction, tokenization, fixed KV copy, suffix inference, final scoring and selection. Excludes model load, one-time setup, simulator observation and step, file output. Both models use shared BF16 Qwen3-8B via MLX; CLM original heads on CPU, four threads. Alternate model execution order at each decision.',
           'scope':'Three 30-second capped courses, both prompts frozen from latest successful saved pilots. Seeds 42 and 314159 were development courses; 271828 was already checked once with final CLM, never with Qwen. No planner, live hints, history or action corrections. Different prompts: Qwen four examples, CLM ten blocks plus distance diagram. Game pauses for inference.'}
 if fresh_seed is not None:
  protocol['scope']='One new seed, not previously tested by either model in saved experiment records. One 30-second capped attempt per model; fixed individual prompts from the previous comparison. No tuning, planner, live hints, history, action corrections, or reruns. Both fixed prefixes and CLM actions cached before play. Game pauses for inference.'
 append('audit.jsonl',protocol);print(json.dumps({'run_id':run_id,'setup':setup,'validation_checks':len(validation),'seeds':protocol['seeds']}),flush=True)
 if '--validate-only' in sys.argv:return

 def observation(g):
  t=g.trex
  visible=sorted((o for o in g.obstacles if not o.remove and o.x<600 and o.x+o.width>0 and o.x+o.width>=t.x),key=lambda o:o.x)[:1]
  if t.jumping:
   state=f'State: The dinosaur is in the air, with its feet {93-t.y:.0f} pixels above the ground, '+('rising' if t.velocity<0 else 'falling')+f' at {abs(t.velocity):.1f} pixels per frame. '
  else:state='State: The dinosaur is on the ground. '+('It is ducking. ' if t.ducking else '')
  for o in visible:
   kind='cactus' if o.kind.name.startswith('cactus') else 'bird'
   state+=f'A {kind} is {o.x-(t.x+44)} pixels beyond its front, {o.width} pixels wide and {o.kind.height} pixels tall, moving toward it at {obstacle_step(g.speed,o.speed_offset)} pixels per frame. '
   if kind=='bird':state+=f'The bird bottom is {140-o.y-o.kind.height} pixels above the ground. '
  if not visible:state+='No obstacles are visible. '
  return state+'\nAnswer:'

 completed=[]
 for seed_index,seed in enumerate(protocol['seeds']):
  games={m:Game(seed) for m in ('clm','qwen')};records={m:[] for m in games};frames={m:0 for m in games}
  seen={m:{} for m in games};passed={m:set() for m in games}
  for g in games.values():g.press_jump()
  for turn in range(300):
   order=('clm','qwen') if (turn+seed_index)%2==0 else ('qwen','clm')
   for method in order:
    g=games[method]
    if g.crashed:continue
    text=observation(g);answer=decide(method,text)
    assert answer['cached_tokens']==len(prefixes[method])>0
    records[method].append({'frame':frames[method],'observation':text,'first_method':order[0],**answer})
    action=answer['choice']
    if action in ('run','jump'):g.release_duck()
    if action=='jump' and not g.trex.jumping:g.press_jump()
    if action=='duck':g.press_duck()
    for _ in range(6):
     for o in g.obstacles:seen[method][o.id]=o
     g.step();frames[method]+=1
     if g.crashed:break
     for oid,o in seen[method].items():
      if o.x+o.width<g.trex.x:passed[method].add(oid)
   if (turn+1)%50==0:
    print(json.dumps({'seed':seed,'turn':turn+1,'frames':frames,'crashed':{m:g.crashed for m,g in games.items()}}),flush=True)
   if all(g.crashed for g in games.values()):break
  for method,g in games.items():
   xs=sorted(r['ms'] for r in records[method])
   result={'game':'trex','method':method+'_warm_cache_final','run_id':run_id,'seed':seed,'system':systems[method],
           'records':records[method],'frames':frames[method],'game_seconds':frames[method]/60,'crashed':g.crashed,
           'score':g.score,'obstacles_passed':len(passed[method]),'decisions':len(xs),
           'median_ms':statistics.median(xs),'mean_ms':statistics.mean(xs),'p95_ms':xs[math.ceil(.95*len(xs))-1]}
   append('raw_episodes.jsonl',result);completed.append(result)
   print(json.dumps({'result':{k:v for k,v in result.items() if k not in ('records','system')}}),flush=True)
 summary={'type':'trex_warm_cache_summary','run_id':run_id,'courses':[{k:v for k,v in r.items() if k not in ('records','system')} for r in completed]}
 for method in ('clm','qwen'):
  xs=sorted(d['ms'] for r in completed if r['method']==method+'_warm_cache_final' for d in r['records'])
  summary[method]={'decisions':len(xs),'median_ms':statistics.median(xs),'mean_ms':statistics.mean(xs),'p95_ms':xs[math.ceil(.95*len(xs))-1]}
 append('audit.jsonl',summary);print(json.dumps({'summary':summary}),flush=True)

if __name__=='__main__':
 if '--warm-cache-comparison' in sys.argv:warm_cache_comparison()
 elif '--controls-only-pilot' in sys.argv:controls_only_pilot()
 else:main()
