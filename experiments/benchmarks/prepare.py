"""Freeze outcome-blind samples from public BFCL and Wikispeedia sources."""
import collections,hashlib,json,random,sys
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/decision_benchmarks';OUT=ROOT/'results/decision_benchmarks'
def rows(file):return [json.loads(l) for l in (DATA/file).read_text().splitlines() if l.strip()]
def title(s):return unquote(s).replace('_',' ')
def graph():
 g=collections.defaultdict(set);rev=collections.defaultdict(set)
 for line in (DATA/'wikispeedia_paths-and-graph/links.tsv').read_text().splitlines():
  if not line or line.startswith('#'):continue
  a,b=line.split('\t');g[a].add(b);rev[b].add(a)
 return {k:sorted(v) for k,v in g.items()},rev
def distances(target,rev):
 d={target:0};todo=collections.deque([target])
 while todo:
  node=todo.popleft()
  for parent in rev.get(node,[]):
   if parent not in d:d[parent]=d[node]+1;todo.append(parent)
 return d
def wiki_case(src,target,g,rev,path=None):
 links=list(g.get(src,[]));random.Random(int(hashlib.sha256(src.encode()).hexdigest()[:8],16)).shuffle(links)
 d=distances(target,rev)
 state=f'Current Wikipedia article: {title(src)}.\nTarget article: {title(target)}.\nVisited articles: '+(' -> '.join(title(x) for x in path) if path else title(src))+'.'
 return {'family':'wiki','dataset':'wikispeedia','id':src+'__'+target,'src':src,'target':target,'state':state,'question':'Which linked article should we visit next to reach the target?',
         'actions':{k:title(k) for k in links},'gold':[k for k in links if src in d and d.get(k,999)==d[src]-1],'distance':d.get(src),'choice_count':len(links)}
def main():
 allcases=[];manifest={}
 for category,ndev in [('multiple',2),('live_multiple',4),('live_irrelevance',4)]:
  raw=rows('BFCL_v4_'+category+'.json');answers={r['id']:r['ground_truth'] for r in rows('possible_answer__BFCL_v4_'+category+'.json')} if 'irrelevance' not in category else None
  unique={}
  for r in raw:
   funcs=r['function'];names=[f['name'] for f in funcs]
   if not 1<=len(funcs)<=30 or len(set(names))!=len(names):continue
   conversation=r['question']
   if len(conversation)!=1:continue
   gt=answers[r['id']] if answers is not None else []
   if gt and (len(gt)!=1 or len(gt[0])!=1):continue
   target=list(gt[0])[0] if gt else 'NO_TOOL'
   if target!='NO_TOOL' and target not in names:continue
   # Keep every schema field and description; only compact JSON whitespace.
   actions={f['name']:json.dumps(f,ensure_ascii=False,separators=(',',':')) for f in funcs}
   actions['NO_TOOL']='No available function can satisfy this request. Do not call a tool.'
   text='\n'.join(m['role'].capitalize()+': '+str(m['content']) for m in conversation[0])
   if len(text)>4000 or sum(len(v) for v in actions.values())>24000:continue
   fingerprint=hashlib.sha256(json.dumps(funcs,sort_keys=True).encode()).hexdigest()
   if fingerprint not in unique:
    unique[fingerprint]={'family':'bfcl','dataset':'bfcl_'+category,'id':r['id'],'schema_id':fingerprint,'state':text,'question':'Which available function can fulfill this request? Choose NO_TOOL if none can.', 'actions':actions,'gold':[target],'choice_count':len(actions)}
  pool=list(unique.values());random.Random(20261003+len(category)).shuffle(pool)
  assert len(pool)>=ndev+30
  chosen=pool[:ndev+30]
  for i,c in enumerate(chosen):c['split']='dev' if i<ndev else 'test';allcases.append(c)
  manifest[category]={'raw_rows':len(raw),'eligible_unique_schemas':len(pool),'dev':ndev,'test':30}
 g,rev=graph();pairs=set()
 for line in (DATA/'wikispeedia_paths-and-graph/paths_finished.tsv').read_text().splitlines():
  if not line or line.startswith('#'):continue
  path=line.split('\t')[3].split(';');pairs.add((path[0],path[-1]))
 pool=sorted(pairs);random.Random(20261003).shuffle(pool);selected=[];targets=set()
 for src,target in pool:
  if src not in g or not 2<=len(g[src])<=999 or target in targets:continue
  d=distances(target,rev)
  if not 2<=d.get(src,999)<=4:continue
  c=wiki_case(src,target,g,rev);c['split']='dev' if len(selected)<10 else 'test';selected.append(c);targets.add(target)
  if len(selected)==50:break
 allcases+=selected
 (OUT/'cases.json').write_text(json.dumps(allcases,indent=2,ensure_ascii=False))
 manifest['wiki']={'nodes':len(set(g)|set(rev)),'links':sum(map(len,g.values())),'dev':10,'test':40,'target_disjoint':True,'shortest_distance_range':[2,4],'all_outgoing_links_retained':True}
 manifest['file_sha256']={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.json')}
 manifest['limitations']=['BFCL measures function selection/abstention only, not argument generation or the full BFCL v4 score.','Wikispeedia is a public WikiRacing-style adaptation, not the authors undisclosed 30 races.','Repeated-menu latency assumes full schema/link list is cached for Qwen and action vectors for CLM. Cold setup will be reported too.']
 (OUT/'sampling.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2));print('cases',len(allcases),'wiki choices',[c['choice_count'] for c in selected])
if __name__=='__main__':main()
