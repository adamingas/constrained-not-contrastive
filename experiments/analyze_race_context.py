"""Audit and summarize the fixed RACE-M experiment."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_long_context import paired, wilson, table

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/race_context'
PRIMARY=['clm_full','qwen_full','clm_no_context','qwen_no_context']


def dist(values):
    return {'min':float(np.min(values)),'median':float(np.median(values)),
            'mean':float(np.mean(values)),'max':float(np.max(values))}


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    cases=json.loads((OUT/'test_cases.json').read_text())
    pilot_cases=json.loads((OUT/'pilot_cases.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==expected
    assert len({c['article_sha256'] for c in cases+pilot_cases})==240
    assert all(c['evidence_question_tokens']>5*c['ratio_denominator_tokens'] for c in cases+pilot_cases)
    rows=[json.loads(x) for x in (OUT/'test_answers.jsonl').read_text().splitlines()]
    by={(r['id'],r['variant']):r for r in rows}
    assert len(by)==len(rows)==1020
    cb={c['id']:c for c in cases}
    for r in rows:
        assert r['gold']==cb[r['id']]['gold']
        assert r['correct']==(r['choice']==r['gold'])
        assert r['choice']==max(r['probabilities'],key=r['probabilities'].get)
        assert abs(sum(r['probabilities'].values())-1)<1e-5
    pilot=json.loads((OUT/'pilot_result.json').read_text());assert pilot['qualified']
    validation=json.loads((OUT/'native_output_validation.json').read_text())
    assert len(validation)==10 and all(r['matches_primary'] for r in validation)
    s={'n':len(cases),'prediction_count':len(rows),'pilot':pilot,'accuracy':{},
       'token_distributions':{k:dist([c[k] for c in cases]) for k in ['context_tokens','evidence_question_tokens','ratio_denominator_tokens','evidence_to_choices_ratio']},
       'latency':{},'chance_accuracy':.25,
       'native_output_validation':{'n':len(validation),'matches':sum(r['matches_primary'] for r in validation),
          'native_top_is_allowed':sum(r['top_token_is_allowed'] for r in validation),
          'minimum_allowed_probability_mass':min(r['allowed_probability_mass'] for r in validation)}}
    for v in PRIMARY+['qwen_cached']:
        r=[by[c['id'],v] for c in cases];k=sum(x['correct'] for x in r)
        s['accuracy'][v]={'correct':k,'n':len(r),'accuracy':k/len(r),'ci95':wilson(k,len(r))}
    s['paired_qwen_minus_clm']=paired([by[c['id'],'clm_full']['correct'] for c in cases],[by[c['id'],'qwen_full']['correct'] for c in cases])
    for m in ['clm','qwen']:
        s[m+'_context_gain']=paired([by[c['id'],m+'_no_context']['correct'] for c in cases],[by[c['id'],m+'_full']['correct'] for c in cases])
    for v in ['clm_full','qwen_full','qwen_cached']:
        s['latency'][v]={'ms':dist([by[c['id'],v]['ms'] for c in cases]),
             'paired_ratio_to_clm':dist([by[c['id'],v]['ms']/by[c['id'],'clm_full']['ms'] for c in cases])}
    s['latency']['clm_setup_ms']=dist([by[c['id'],'clm_full']['setup']['clm_actions_ms'] for c in cases])
    s['latency']['clm_setup_plus_decision_ms']=dist([by[c['id'],'clm_full']['setup']['clm_actions_ms']+by[c['id'],'clm_full']['ms'] for c in cases])
    s['latency']['qwen_prefix_setup_ms']=dist([by[c['id'],'qwen_cached']['setup']['qwen_prefix_ms'] for c in cases])
    audit=[c for c in cases if c['order_audit']]
    s['order_audit']={'n':len(audit),'original_correct':sum(by[c['id'],'qwen_full']['correct'] for c in audit),
        'reversed_correct':sum(by[c['id'],'qwen_reversed']['correct'] for c in audit),
        'changed_answers':sum(by[c['id'],'qwen_full']['choice']!=by[c['id'],'qwen_reversed']['choice'] for c in audit)}
    s['cached_uncached_choice_disagreements']=sum(by[c['id'],'qwen_full']['choice']!=by[c['id'],'qwen_cached']['choice'] for c in cases)
    s['ratio_bins']={}
    for lo,hi in [(5,7),(7,10),(10,100)]:
        sub=[c for c in cases if lo<c['evidence_to_choices_ratio']<=hi]
        s['ratio_bins'][f'({lo},{hi}]']={'n':len(sub),**{v:sum(by[c['id'],v]['correct'] for c in sub) for v in PRIMARY}}
    (OUT/'summary.json').write_text(json.dumps(s,indent=2))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,rat)=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1.2,1]})
    xx=np.arange(2);width=.32
    for offset,variants,label,alpha in [(-.16,['clm_full','qwen_full'],'Full passage',1),(.16,['clm_no_context','qwen_no_context'],'Passage removed',.28)]:
        vals=np.array([s['accuracy'][v]['accuracy'] for v in variants])*100
        cis=np.array([s['accuracy'][v]['ci95'] for v in variants])*100
        bars=ax.bar(xx+offset,vals,width,color=['#b95c45','#087c8b'],alpha=alpha,label=label,
                    yerr=[vals-cis[:,0],cis[:,1]-vals],capsize=3)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=4,fontsize=10)
    ax.axhline(25,color='#777',linestyle='--',linewidth=1)
    ax.set_xticks(xx,['CLM','Constrained Qwen']);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)')
    ax.set_title('RACE-M · 200 held-out questions',loc='left',fontweight='bold');ax.legend(loc='lower center',ncol=2,bbox_to_anchor=(.5,-.25),frameon=False)
    rat.hist([c['evidence_to_choices_ratio'] for c in cases],bins=[5,7,9,11,13,15,18,22,28],color='#087c8b',alpha=.8)
    rat.axvline(5,color='#b95c45',linestyle='--',label='Required minimum: 5×')
    rat.set_xlabel('(Passage + question) / all four choices\nQwen tokenizer tokens');rat.set_ylabel('Questions')
    rat.set_title('Every input meets the token ratio',loc='left',fontweight='bold');rat.legend(frameon=False,fontsize=9)
    fig.text(.02,.01,'Same BF16 Qwen3-8B · no thinking tokens · 95% Wilson intervals · selected test subset',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'results.png',dpi=170,bbox_inches='tight');plt.close(fig)

    a=s['accuracy'];p=s['paired_qwen_minus_clm'];order=s['order_audit'];t=s['token_distributions']
    rows_accuracy=[]
    for m,label in [('clm','CLM'),('qwen','Constrained Qwen (uncached)')]:
        full=a[m+'_full'];none=a[m+'_no_context'];gain=s[m+'_context_gain']
        rows_accuracy.append([label,f"{full['correct']}/200 ({full['accuracy']:.1%})",f"{full['ci95'][0]:.1%}–{full['ci95'][1]:.1%}",
                              f"{none['correct']}/200 ({none['accuracy']:.1%})",f"{100*gain['difference']:+.1f} pp"])
    rows_latency=[[label,f"{s['latency'][v]['ms']['median']:.1f} ms",f"{s['latency'][v]['paired_ratio_to_clm']['median']:.3f}×"]
                  for v,label in [('clm_full','CLM, cached actions'),('qwen_full','Qwen, complete prompt'),('qwen_cached','Qwen, cached instruction prefix')]]
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLM versus constrained Qwen on context-dependent RACE-M</title>
<style>body{{font:17px/1.6 system-ui;color:#23323e;background:#f2f6f8;margin:0}}main{{max-width:1120px;margin:32px auto;padding:38px;background:white;border-radius:14px}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dbe2e7}}th{{background:#edf4f6}}img{{max-width:100%}}a{{color:#086f91}}.scroll{{overflow:auto}}.note{{background:#eaf4f5;padding:20px;border-left:4px solid #087c8b}}pre{{overflow:auto;background:#f4f6f7;padding:16px}}@media(max-width:700px){{main{{margin:0;padding:20px}}}}</style><main>
<p>Local experiment · 2 October 2026</p><h1>Short, useful context with a small action set</h1>
<p class="note">200 held-out RACE-M questions. Every passage plus question exceeds five times the token count of all four labeled choices combined. No document truncation, padding, extra reasoning tokens, or test-example selection based on correctness.</p>
<p>With full passages, constrained Qwen scored <strong>{a['qwen_full']['accuracy']:.1%}</strong> and CLM scored <strong>{a['clm_full']['accuracy']:.1%}</strong>. Their paired difference is {100*p['difference']:+.1f} percentage points (paired bootstrap 95% interval {100*p['ci95'][0]:+.1f} to {100*p['ci95'][1]:+.1f}). Uniform random choice is 25%.</p>
<h2>Accuracy and actual context use</h2>{table(['Method','Full passage','95% Wilson interval','Passage removed','Context gain'],rows_accuracy)}<img src="results.png" alt="Accuracy with and without passages, and evidence-to-options token ratios">
<p>The same questions and choices remain in the passage-removal control. The original gold answer is retained to measure how much information the passage supplies. Paired tests compare the identical 200 questions; exact McNemar p={p['mcnemar_exact_p']:.4g}. The bootstrap uses 10,000 paired resamples. These intervals describe this selected population and do not address training-data contamination.</p>
<h2>Why this benchmark was selected</h2><p><a href="https://www.cs.cmu.edu/~glai1/data/race/">RACE</a> is a reading-comprehension benchmark drawn from school English exams. Its middle-school subset tests much shorter passages than the previous long-document experiment. A separate 40-question development pilot qualified the benchmark before any test inference: Qwen answered {pilot['correct']['qwen_full']}/40 with the passage versus {pilot['correct']['qwen_no_context']}/40 without it. The predeclared acceptance rule required at least 70% accuracy and a gain of at least 15 percentage points. No test examples were retained or removed based on model performance.</p>
<p>The first pilot retained the earlier experiment's options-in-system layout and failed qualification: 24/40 with the passage versus 26/40 without it. A second, conventional layout placed passage, question, and then options together in the user message; it passed. This prompt selection was performed on development data only, before any test inference. Both pilot outputs are retained as <a href="pilot_answers.jsonl">original layout</a> and <a href="pilot_standard_layout.jsonl">standard layout</a>. This substantial prompt effect limits general conclusions from the earlier long-context experiment; its numerical results remain valid for that specific prompt.</p>
<p>The test sample was frozen before the pilot ran. It uses one question per distinct passage, with no passage text overlap with the pilot. Eligible test pool: {manifest['pools']['test']['questions']} questions from {manifest['pools']['test']['passages']} passages. Sampling and option-order seed: {manifest['seed']}. This is a ratio-filtered RACE-M test subset, not an official full-benchmark result.</p>
<h2>Token-budget audit</h2><p>The numerator is the Qwen tokenizer count of the complete passage, two newlines, and question. The denominator is the larger of (a) the sum of the four raw answer-string token counts and (b) the token count of all four choices formatted as A/B/C/D lines. Shared system instructions are not credited to the evidence. Every case satisfies a strict ratio &gt;5.</p>
{table(['Quantity','Minimum','Median','Maximum'],[[k,f"{v['min']:.2f}",f"{v['median']:.2f}",f"{v['max']:.2f}"] for k,v in t.items()])}
<h2>Latency on the actual 200 questions</h2>{table(['Method','Median decision latency','Median paired ratio to CLM'],rows_latency)}
<p>These are one timed decision per method on each of 200 distinct inputs, with method order randomized per question; they are not repeated trials of one prompt. CLM action embeddings and the cached Qwen instruction prefix are prepared outside decision timing. The primary Qwen result processes the complete prompt including options on each call, with no precomputed prefix. Because options follow the passage in this layout, even the cached Qwen variant processes all option tokens on every call; only its common instruction prefix is cached. Model loading, HTTP serving and candidate setup are excluded. Tokenization, cache-copy work where applicable, inference and output selection are included; GPU outputs are materialized before stopping each timer.</p>
<p>For new action sets, CLM also paid a median {s['latency']['clm_setup_ms']['median']:.1f} ms to embed the four actions separately. The median of action-setup time plus CLM decision time was {s['latency']['clm_setup_plus_decision_ms']['median']:.1f} ms. This uses the reference harness's serial action setup, not an optimized batched action encoder. Qwen prefix setup took a median {s['latency']['qwen_prefix_setup_ms']['median']:.1f} ms. The cached Qwen variant scored {a['qwen_cached']['correct']}/200, with {s['cached_uncached_choice_disagreements']} answer disagreements versus the uncached primary; BF16 numerical differences can change close decisions.</p>
<h2>Option-order audit</h2><p>Twenty test questions were selected before inference for reversal of all four options. Qwen scored {order['original_correct']}/20 in the original randomized order and {order['reversed_correct']}/20 after reversal, changing {order['changed_answers']}/20 answers. This audit does not replace the primary answers or ensemble them. CLM action scores are independently computed and permutation-equivariant.</p>
<h2>Implementation and limitations</h2><p>Both methods use the original BF16 Qwen/Qwen3-8B weights. CLM uses its released FP32 heads, original state/question serializer, normalized last-token pooling and checkpoint scaling. Qwen uses its original language-model output weights restricted to verified single-token option letters. Thinking is disabled and no explanation tokens are generated. Qwen's user message contains the passage, question, and then options; the system message asks for exactly one letter. No task-specific fine-tuning or few-shot examples are used.</p>
<p>A separate check of the first ten frozen test cases used the native Qwen model call with its full-vocabulary output projection. All ten restricted choices matched the primary optimized scoring path, and the unrestricted top token was an allowed option letter in every case. See <a href="native_output_validation.json">native-forward validation</a>. This check ran after all benchmark timing was finished.</p>
<p>MLX runs on an M4 Max with 64 GB unified memory and a process-local 24 GiB wired-memory limit. The inherited 1024-token prefill chunks preserve full context; these short prompts generally fit a single chunk. This is not an NVIDIA/vLLM speed measurement. Public RACE data may occur in model training; contamination was not audited. Selecting a benchmark where Qwen succeeds on development data was deliberate and does not establish superiority across all tasks. Both test methods use exactly the same selected evidence, questions and candidate text.</p>
<h2>Artifacts and reproduction</h2><p><a href="summary.json">Statistics</a> · <a href="test_answers.jsonl">Every test prediction</a> · <a href="test_cases.json">Frozen test cases</a> · <a href="pilot_result.json">Pilot qualification</a> · <a href="manifest.json">Source hash and sample hashes</a> · <a href="runtime.json">Runtime metadata</a></p>
<pre>.venv/bin/python experiments/prepare_race_context.py
.venv/bin/python -u experiments/bench_race_context.py --original-pilot
.venv/bin/python -u experiments/bench_race_context.py
.venv/bin/python experiments/validate_race_output.py
.venv/bin/python experiments/analyze_race_context.py</pre>
<p>The runner resumes existing logs. Preserve this results directory before a fresh replication. The source archive was downloaded from the authors' site and hashed in the manifest. RACE's stated terms are non-commercial research use; see the linked dataset page.</p></main></html>'''
    (OUT/'report.html').write_text(body)
    print(json.dumps(s,indent=2))


if __name__=='__main__':main()
