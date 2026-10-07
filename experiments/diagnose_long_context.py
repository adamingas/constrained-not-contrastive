"""Post-primary spot checks of forced-token validity and prompt placement.

Use the first two frozen cases per dataset, independent of correctness.
These six cases are an implementation/prompt diagnostic, not a new benchmark.
"""
import json
import mlx.core as mx
from long_context_bench import LongComparison,OUT

c=LongComparison()
cases=json.loads((OUT/'cases.json').read_text())
rows=[json.loads(x) for x in (OUT/'answers.jsonl').read_text().splitlines()]
primary={x['id']:x for x in rows if x['variant']=='qwen_full'}
sample=[]
for dataset in ['quality','contract_nli','longbench_v2']:
    sample += [x for x in cases if x['dataset']==dataset][:2]
output=[]
for case in sample:
    q=case['q'];keys=list(q['criteria']);labels=list('ABCD'[:len(keys)])
    label_ids=[c.tok.encode(x,add_special_tokens=False)[0] for x in labels]
    for layout in ['original','options_at_end']:
        if layout=='original':ids,_,_,_=c.prompt(case['state'],q)
        else:
            options='\n'.join(f'{l}: {q["criteria"][k]}' for l,k in zip(labels,keys))
            msgs=[{'role':'system','content':'Answer the question using the supplied document. Select the best option. Reply with exactly one option letter and nothing else.'},
                  {'role':'user','content':case['state']+'\n\nQuestion: '+q['instructions']+'\n\nOptions:\n'+options}]
            ids=c.tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        h=c.last(ids)
        logits=c.model.lm_head(h)[0].astype(mx.float32)
        p=mx.softmax(logits)
        allowed=p[mx.array(label_ids)]
        selected=mx.softmax(logits[mx.array(label_ids)])
        top=mx.argsort(logits)[-8:][::-1]
        mx.eval(p,selected,top)
        choice=keys[int(mx.argmax(selected).item())]
        row={'id':case['id'],'dataset':case['dataset'],'layout':layout,'gold':case['gold'],
             'choice':choice,'correct':choice==case['gold'],'allowed_token_probability_mass':float(allowed.sum().item()),
             'top_tokens':[{'text':c.tok.decode([i]),'probability':float(p[i].item())} for i in top.tolist()],
             'matches_primary':choice==primary[case['id']]['choice'] if layout=='original' else None}
        output.append(row)
        (OUT/'prompt_diagnostics.json').write_text(json.dumps(output,indent=2))
        print(json.dumps(row),flush=True)
        mx.clear_cache()
