"""Development-only comparison of conventional passage/question/options layout."""
import json
import mlx.core as mx
from compare import ROOT
from long_context_bench import LongComparison, append

OUT=ROOT/'results/race_context'


def make_race_prompt(state,q,tok):
    keys=list(q['criteria']);labels=list('ABCD'[:len(keys)])
    options='\n'.join(f'{l}: {q["criteria"][k]}' for l,k in zip(labels,keys))
    system='Answer the multiple-choice reading comprehension question using the passage. Reply with exactly one option letter and nothing else.'
    user='Passage:\n'+state+'\n\nQuestion: '+q['instructions']+'\n\nOptions:\n'+options
    msgs=[{'role':'system','content':system},{'role':'user','content':user}]
    ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    blank=tok.apply_chat_template([msgs[0],{'role':'user','content':''}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    common=0
    while common<min(len(ids),len(blank)) and ids[common]==blank[common]:common+=1
    label_ids=[tok.encode(l,add_special_tokens=False) for l in labels]
    assert all(len(x)==1 for x in label_ids)
    return ids,tuple(ids[:common]),keys,[x[0] for x in label_ids]


class RaceComparison(LongComparison):
    def prompt(self,state,q):return make_race_prompt(state,q,self.tok)


def main():
    c=RaceComparison()
    cs=json.loads((OUT/'pilot_cases.json').read_text())
    path=OUT/'pilot_standard_layout.jsonl'
    old=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(r['id'],r['variant']) for r in old}
    for i,case in enumerate(cs):
        for v in ['qwen_full','qwen_no_context']:
            if (case['id'],v) in done:continue
            a=c.qwen(case['state'] if v=='qwen_full' else '',case['q'],cached=False)
            append(path,{'id':case['id'],'variant':v,'gold':case['gold'],'correct':a['choice']==case['gold'],**a})
        print('PILOT',i+1,flush=True)
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    print('SCORES',{v:sum(r['correct'] for r in rows if r['variant']==v) for v in ['qwen_full','qwen_no_context']},flush=True)


if __name__=='__main__':main()
