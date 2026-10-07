"""Prespecified diagnostic contrasts after observing CLM's abstention bias.

Do not mix these post-primary-design diagnostics into untouched test selection.
Apply each contrast to every eligible held-out case, with no prompt search.
"""
import json,random
from run import Bench,OUT,append

def main():
 c=Bench();cases=[x for x in json.loads((OUT/'cases.json').read_text()) if x['split']=='test' and x['family']=='bfcl']
 for i,case in enumerate(cases):
  configurations=[('menu_context',case,['clm_menu_base','clm_menu_prose_base'])]
  if case['dataset']!='bfcl_live_irrelevance':
   closed={**case,'closed_set':True,'actions':{k:v for k,v in case['actions'].items() if k!='NO_TOOL'},'question':'Which available function best fulfills this request?','choice_count':case['choice_count']-1}
   configurations.append(('forced_tool',closed,['clm_base','clm_prose_base','qwen_raw_base']))
  for diagnostic,item,variants in configurations:
   random.Random(i+884).shuffle(variants)
   for v in variants:
    setup=c.prepare(item,v);ans=c.decide(item,v)
    append('diagnostics.jsonl',{'diagnostic':diagnostic,'id':item['id'],'dataset':item['dataset'],'variant':v,'gold':item['gold'],'choice_count':item['choice_count'],**setup,**ans})
    print(diagnostic,i+1,v,ans['correct'],flush=True)
   c.clear()
 print('DONE diagnostics',flush=True)
if __name__=='__main__':main()
