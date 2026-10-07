"""Same backbone decision-head comparison on public benchmark adaptations."""
import argparse,copy,hashlib,itertools,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from long_context_bench import LongComparison
from compare import ROOT,norm
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
OUT=ROOT/'results/decision_benchmarks'
RULES={
 'bfcl':{'base':'Choose the available function that best matches the user request. Choose NO_TOOL if no function can fulfill it.',
 'tuned':'''Select a tool by its documented capability, not just a similar name or keyword. Check the requested operation, domain, parameter meanings, and allowed values. Do not assume a tool supports operations absent from its documentation. Choose NO_TOOL when the available tools cannot fulfill the request. Pick a suitable function when one exists; do not abstain merely because optional arguments are unspecified.
Examples: A weather-query tool cannot book a flight: choose NO_TOOL for flight booking. A stock-price lookup cannot execute a stock purchase: choose NO_TOOL for buying shares. Between a hotel-search tool and a hotel-cancellation tool, choose cancellation for cancelling a reservation. Between food-edit and drink-edit tools, choose drink-edit for changing a latte.'''},
 'wiki':{'base':'Choose a linked Wikipedia article that helps reach the target article.',
 'tuned':'''Navigate to the target Wikipedia article using only the listed outgoing links. Select the target itself when listed. Otherwise choose the best bridge: a directly related concept, relevant field, geographic region, or broad hub likely to contain a link toward the target. Prefer a useful bridge over superficial word overlap. Avoid returning to an article already visited when a plausible unvisited route exists. You do not know future link lists, so use your factual knowledge of the relationships between topics.
Examples: to reach a marine animal, a biology or ocean article may be a better bridge than an unrelated location. To reach a country's capital, its country is often a good bridge. To reach a scientist, their scientific field may lead there.'''}
}
VARIANTS=['clm_base','clm_tuned','clm_prose_base','clm_prose_tuned','qwen_raw_base','qwen_raw_tuned','qwen_chat_base','qwen_chat_tuned']
def append(name,row):
 with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
class Bench(LongComparison):
 def __init__(self):
  super().__init__();self.menus={}
  self.labels=[''.join(cs) for n in [1,2] for cs in itertools.product('ABCDEFGHIJKLMNOPQRSTUVWXYZ',repeat=n)]
  self.labels=[x for x in self.labels if len(self.tok.encode(x,add_special_tokens=False))==1]
 def texts(self,case,variant):
  rules=RULES[case['family']]['tuned' if variant.endswith('tuned') else 'base']
  if case.get('closed_set'):rules='Choose the available function that best matches the user request.'
  if variant.startswith('clm'):
   prefix=rules+'\n\nState:\n' if variant.endswith('tuned') else ''
   if '_menu_' in variant:
    from clm.schema import to_text
    texts=[to_text(json.loads(v)) if '_prose_' in variant and case['family']=='bfcl' and k!='NO_TOOL' else v for k,v in case['actions'].items()]
    prefix=rules+'\nAvailable options:\n'+'\n'.join(f'{self.labels[i]}: {v}' for i,v in enumerate(texts))+'\n\nState:\n'
   return prefix,case['state']+'\n\n'+case['question']
  opts='\n'.join(f'{self.labels[i]}: {v}' for i,v in enumerate(case['actions'].values()))
  system=rules+'\nReply with exactly one option label, without explanation.\nAvailable options:\n'+opts
  suffix=case['state']+'\n\n'+case['question']
  if '_raw_' in variant:return system+'\n\nState:\n',suffix+'\nAnswer:'
  return system,suffix
 def tokens(self,case,variant):
  a,b=self.texts(case,variant)
  if '_chat_' in variant:
   msgs=[{'role':'system','content':a},{'role':'user','content':b}]
   ids=self.tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False)
   prefix=self.tok.encode(self.tok.apply_chat_template([msgs[0]],tokenize=False,add_generation_prompt=False)+'<|im_start|>user\n',add_special_tokens=False)
  else:
   ids=self.tok.encode(a+b,add_special_tokens=True);prefix=self.tok.encode(a,add_special_tokens=True)[:-1] if a else []
  assert ids[:len(prefix)]==prefix
  return ids,prefix
 def prepare(self,case,variant,reuse_actions=False):
  start=time.perf_counter();ids,prefix=self.tokens(case,variant);entry={'prefix':prefix,'cache':None}
  if prefix:
   entry['cache']=make_prompt_cache(self.model);self.last(prefix,entry['cache'])
  if variant.startswith('clm'):
   if not reuse_actions:self.action_cache.clear()
   from clm.schema import to_text
   action_texts=[to_text(json.loads(v)) if '_prose_' in variant and case['family']=='bfcl' and k!='NO_TOOL' else v for k,v in case['actions'].items()]
   entry['actions']=self.actions(action_texts);mx.eval(entry['actions'])
  else:
   assert len(case['actions'])<=len(self.labels)
   label_ids=[self.tok.encode(self.labels[i],add_special_tokens=False) for i in range(len(case['actions']))];assert all(len(x)==1 for x in label_ids)
   entry['labels']=[x[0] for x in label_ids];entry['rows']=self.model.lm_head.weight[mx.array(entry['labels'])];mx.eval(entry['rows'])
  self.menus[variant]=entry
  return {'setup_ms':(time.perf_counter()-start)*1000,'prefix_tokens':len(prefix),'action_tokens':sum(len(self.tok.encode(x,add_special_tokens=True)) for x in case['actions'].values())}
 def decide(self,case,variant,full=False):
  start=time.perf_counter();ids,prefix=self.tokens(case,variant);d=self.menus[variant];assert prefix==d['prefix'];cache=copy.deepcopy(d['cache']) if prefix else None
  h=self.last(ids[len(prefix):],cache)
  if variant.startswith('clm'):logits=self.head.scale*(self.project(norm(h),'state')@d['actions'].T)[0]
  else:logits=(self.model.lm_head(h)[0,mx.array(d['labels'])] if full else (h@d['rows'].T)[0]).astype(mx.float32)
  probs=mx.softmax(logits);mx.eval(probs);values=probs.tolist();index=int(mx.argmax(probs).item());keys=list(case['actions']);choice=keys[index]
  return {'choice':choice,'correct':choice in case['gold'],'ms':(time.perf_counter()-start)*1000,'probabilities':values,'input_tokens':len(ids),'prefix_tokens':len(prefix),'processed_tokens':len(ids)-len(prefix)}
 def clear(self):self.menus.clear();self.action_cache.clear();mx.clear_cache()
def run(split):
 cases=[c for c in json.loads((OUT/'cases.json').read_text()) if c['split']==split]
 c=Bench();(OUT/'model_metadata.json').write_text(json.dumps(c.meta,indent=2));(OUT/'prompt_rules.json').write_text(json.dumps(RULES,indent=2))
 frozen=json.loads((OUT/'frozen.json').read_text()) if split=='test' else None
 path=OUT/f'{split}.jsonl';done={(r['id'],r['variant']) for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
 for i,case in enumerate(cases):
  variants=VARIANTS if split=='dev' else frozen['variants'][case['family']]
  variants=list(variants);random.Random(i+71).shuffle(variants)
  for v in variants:
   if (case['id'],v) in done:continue
   setup=c.prepare(case,v);answer=c.decide(case,v)
   row={'id':case['id'],'dataset':case['dataset'],'family':case['family'],'variant':v,'split':split,'gold':case['gold'],'choice_count':case['choice_count'],**setup,**answer,'first_query_total_ms':setup['setup_ms']+answer['ms']}
   append(f'{split}.jsonl',row)
   print(split,i+1,case['dataset'],v,'ok',answer['correct'],'ms',round(answer['ms'],1),'setup',round(setup['setup_ms'],1),flush=True)
  c.clear()
 print('DONE',split,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['dev','test'],default='dev');run(ap.parse_args().split)
