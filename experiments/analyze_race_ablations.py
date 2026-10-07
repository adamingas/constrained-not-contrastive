"""Paired analysis of all fixed RACE-M prompt ablations."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_long_context import paired,wilson,table

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/race_ablations'
SOURCE=ROOT/'results/race_context'
NAMES={'original':'Original: full system + chat','short_system':'Short system + chat',
       'no_system':'No system + chat','no_template':'Full instruction + raw + Answer:',
       'neither':'No instruction + raw + Answer:',
       'no_template_literal':'Full instruction + raw, literal',
       'neither_literal':'No instruction + raw, literal','clm':'CLM (cached actions)'}


def distribution(xs):
    return {'min':float(np.min(xs)),'median':float(np.median(xs)),
            'mean':float(np.mean(xs)),'max':float(np.max(xs))}


def main():
    cases=json.loads((SOURCE/'test_cases.json').read_text());n=len(cases)
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((SOURCE/'test_cases.json').read_bytes()).hexdigest()==manifest['cases_sha256']
    rows=list(map(json.loads,(OUT/'answers.jsonl').read_text().splitlines()))
    by={(r['id'],r['variant'],r['condition']):r for r in rows}
    expected={(c['id'],v,cond) for c in cases for v in NAMES for cond in ['full','no_context']}
    expected|={(c['id'],v,'reversed') for c in cases if c['order_audit'] for v in NAMES if v!='clm'}
    assert len(by)==len(rows)==3340
    assert set(by)==expected
    cb={c['id']:c for c in cases}
    for r in rows:
        assert r['gold']==cb[r['id']]['gold']
        assert r['correct']==(r['choice']==r['gold'])
        assert abs(sum(r['probabilities'].values())-1)<1e-5
        assert r['choice']==max(r['probabilities'],key=r['probabilities'].get)
    old={(r['id'],r['variant']):r for r in map(json.loads,(SOURCE/'test_answers.jsonl').read_text().splitlines())}
    s={'n_questions':n,'n_predictions':len(rows),'variants':{},'baseline_reproduction':{}}
    for v,prior in [('original','qwen_full'),('clm','clm_full')]:
        s['baseline_reproduction'][v]={'same_choices':sum(by[c['id'],v,'full']['choice']==old[c['id'],prior]['choice'] for c in cases),
                                        'previous_correct':sum(old[c['id'],prior]['correct'] for c in cases)}
    for v in NAMES:
        full=[by[c['id'],v,'full'] for c in cases];none=[by[c['id'],v,'no_context'] for c in cases]
        k=sum(r['correct'] for r in full);kn=sum(r['correct'] for r in none)
        z={'full':{'correct':k,'accuracy':k/n,'ci95':wilson(k,n)},
           'no_context':{'correct':kn,'accuracy':kn/n,'ci95':wilson(kn,n)},
           'context_gain':paired([r['correct'] for r in none],[r['correct'] for r in full]),
           'difference_from_original':paired([by[c['id'],'original','full']['correct'] for c in cases],[r['correct'] for r in full]),
           'tokens':distribution([r['input_tokens'] for r in full]),'latency_ms':distribution([r['ms'] for r in full]),
           'paired_latency_ratio_to_clm':distribution([by[c['id'],v,'full']['ms']/by[c['id'],'clm','full']['ms'] for c in cases]),
           'paired_latency_ratio_to_original':distribution([by[c['id'],v,'full']['ms']/by[c['id'],'original','full']['ms'] for c in cases])}
        if v!='clm':
            z['space_alias_full_correct']=sum(r['space_alias_correct'] for r in full)
            z['space_alias_no_context_correct']=sum(r['space_alias_correct'] for r in none)
            audit=[c for c in cases if c['order_audit']]
            z['order_audit']={'n':len(audit),'original_correct':sum(by[c['id'],v,'full']['correct'] for c in audit),
                'reversed_correct':sum(by[c['id'],v,'reversed']['correct'] for c in audit),
                'changed_answers':sum(by[c['id'],v,'full']['choice']!=by[c['id'],v,'reversed']['choice'] for c in audit)}
            diag=[r['diagnostic'] for r in full if 'diagnostic' in r]
            assert len(diag)==10
            z['output_format_diagnostic']={'n':10,'bare_mass':distribution([d['bare_letters_probability_mass'] for d in diag]),
                'bare_or_spaced_mass':distribution([d['bare_or_spaced_letters_probability_mass'] for d in diag]),
                'unrestricted_top_tokens':[d['unrestricted_top_token'] for d in diag]}
        s['variants'][v]=z
    (OUT/'summary.json').write_text(json.dumps(s,indent=2))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,lat)=plt.subplots(1,2,figsize=(13,6.4),gridspec_kw={'width_ratios':[1.6,1]})
    labels=list(NAMES);yy=np.arange(len(labels));height=.34
    for offset,condition,label,alpha in [(-.17,'full','Full passage',1),(.17,'no_context','Passage removed',.28)]:
        vals=np.array([s['variants'][v][condition]['accuracy'] for v in labels])*100
        bars=ax.barh(yy+offset,vals,height,color=['#087c8b']*7+['#b95c45'],alpha=alpha,label=label)
        ax.bar_label(bars,labels=[f'{x:.1f}' for x in vals],padding=3,fontsize=9)
    ax.axvline(25,color='#777',linestyle='--',linewidth=1);ax.set_xlim(0,108)
    ax.set_yticks(yy,[NAMES[v] for v in labels]);ax.invert_yaxis();ax.set_xlabel('Accuracy (%)')
    ax.set_title('Accuracy and context use',loc='left',fontweight='bold');ax.legend(loc='lower center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
    vals=[s['variants'][v]['latency_ms']['median'] for v in labels]
    bars=lat.barh(yy,vals,.55,color=['#087c8b']*7+['#b95c45']);lat.bar_label(bars,fmt='%.0f',padding=3,fontsize=9)
    lat.set_yticks(yy,[]);lat.invert_yaxis();lat.set_xlim(0,max(vals)*1.2);lat.set_xlabel('Median decision latency (ms)')
    lat.set_title('Contemporaneous timings',loc='left',fontweight='bold')
    fig.text(.02,.012,'200 fixed RACE-M questions · BF16 Qwen3-8B · one bare A/B/C/D token · no Qwen prefix cache',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(OUT/'results.png',dpi=170,bbox_inches='tight');plt.close(fig)

    main_rows=[];paired_rows=[];format_rows=[];order_rows=[]
    for v,z in s['variants'].items():
        main_rows.append([NAMES[v],f"{z['full']['correct']}/200 ({z['full']['accuracy']:.1%})",f"{z['no_context']['accuracy']:.1%}",
                          f"{100*z['context_gain']['difference']:+.1f} pp",f"{z['tokens']['median']:.1f}",
                          f"{z['latency_ms']['median']:.1f} ms",f"{z['paired_latency_ratio_to_clm']['median']:.3f}×"])
        if v!='clm':
            p=z['difference_from_original'];paired_rows.append([NAMES[v],f"{100*p['difference']:+.1f} pp",f"{100*p['ci95'][0]:+.1f} to {100*p['ci95'][1]:+.1f} pp"])
            d=z['output_format_diagnostic'];format_rows.append([NAMES[v],f"{z['space_alias_full_correct']}/200",f"{d['bare_mass']['median']:.3g}",f"{d['bare_or_spaced_mass']['median']:.3g}",repr(d['unrestricted_top_tokens'])])
            a=z['order_audit'];order_rows.append([NAMES[v],a['original_correct'],a['reversed_correct'],a['changed_answers']])
    examples=json.loads((OUT/'prompt_examples.json').read_text())
    prompts=''.join(f'<details><summary>{html.escape(NAMES[v])} — first example, {examples[v]["input_tokens"]} tokens</summary><pre>{html.escape(examples[v]["rendered_input"])}</pre></details>' for v in examples)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RACE-M prompt ablations</title>
<style>body{{font:17px/1.55 system-ui;color:#24323d;background:#f3f6f8;margin:0}}main{{max-width:1250px;margin:30px auto;padding:36px;background:white;border-radius:14px}}h1{{font-size:34px}}h2{{margin-top:34px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #dbe2e8}}th{{background:#edf4f6}}img{{max-width:100%}}a{{color:#076b90}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:18px;font-size:13px}}details{{margin:14px 0}}.note{{background:#eaf5f5;padding:18px;border-left:4px solid #087c8b}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main>
<p>2 October 2026 · matched prompt ablations</p><h1>What do the system prompt and chat template contribute?</h1>
<p class="note">Seven Qwen prompt variants on the same 200 RACE-M test questions, with no-passage controls and 20 option-reversal audits per variant. The original Qwen prompt and CLM are rerun in the same process, interleaved randomly. Total: {len(rows):,} predictions.</p>
<p>Removing both the system instruction and chat template, while retaining a two-token <code>Answer:</code> cue, scored {s['variants']['neither']['full']['accuracy']:.1%} versus {s['variants']['original']['full']['accuracy']:.1%} for the original. Its median paired latency ratio was {s['variants']['neither']['paired_latency_ratio_to_original']['median']:.3f}, a {100*(1-s['variants']['neither']['paired_latency_ratio_to_original']['median']):.1f}% reduction relative to the original on matched questions. Literal stripping without the cue scored {s['variants']['neither_literal']['full']['accuracy']:.1%}. The small apparent improvement over original accuracy needs fresh-data validation; the exact paired McNemar p-value is {s['variants']['neither']['difference_from_original']['mcnemar_exact_p']:.3f}, before multiple-comparison adjustment.</p>
{table(['Method','Full-passage accuracy','No passage','Context gain','Median input tokens','Median latency','Paired latency / CLM'],main_rows)}
<img src="results.png" alt="Prompt ablation accuracy, no-passage controls and latency">
<h2>Exact interventions</h2><ul><li><strong>Original:</strong> the prior 21-token system instruction, Qwen's normal chat template and non-thinking assistant prefix.</li><li><strong>Short system:</strong> replace the instruction with <code>Reply with A, B, C, or D.</code>, retaining the template.</li><li><strong>No system:</strong> remove the system message entirely; retain the user message and non-thinking assistant prefix. The tokenizer inserts no replacement system instruction.</li><li><strong>No template:</strong> retain the full instruction as plain text before the unchanged user body, remove role delimiters and the assistant/thinking prefix, then append <code>\n\nAnswer:</code> as a conventional completion cue.</li><li><strong>Neither:</strong> remove the instruction and template; retain the unchanged user body plus <code>\n\nAnswer:</code>.</li><li><strong>Literal raw controls:</strong> repeat both raw versions without adding the answer cue. These end immediately after the final option's text.</li></ul>
<p>The answer cue costs two tokens on the first example. It is reported explicitly because deleting a chat template also deletes its assistant answer boundary. Testing both raw endings avoids hiding that choice inside a template ablation. The passage, question, four choices, labels and original option order are identical across primary conditions. CLM retains the released head and canonical raw state/action inputs.</p>
<h2>Paired accuracy changes versus the original prompt</h2>{table(['Variant','Difference','Paired bootstrap 95% interval'],paired_rows)}
<p>Intervals use 10,000 paired resamples of the same questions. They are exploratory and not adjusted for multiple comparisons. The dataset and variants were fixed before this run; no test cases were filtered based on any outputs. These ablations reuse a previously evaluated test subset, so the best observed variant has not been independently validated on fresh data. The sample is the prior ratio-filtered RACE-M test subset: complete passage plus question has more than five times the token count of all four formatted choices combined. This is not an official full RACE leaderboard score.</p>
<h2>Output-format diagnostics</h2>{table(['Variant','Accuracy allowing space aliases','Median mass: bare A–D','Median mass: bare or spaced A–D','Unrestricted top tokens (10 cases)'],format_rows)}
<p>The primary experiment always restricts the next token to exactly the bare letters A/B/C/D, matching the original benchmark. Removing the chat template can change the model's preferred continuation format. For a secondary diagnostic, the same hidden state is scored with either a bare letter or a single-leading-space letter token allowed; token-alias probabilities are summed per answer. This requires no additional backbone pass and does not replace primary scores. Full-vocabulary diagnostics use the first ten frozen test cases, independent of correctness. All diagnostics run after the primary decision timer stops.</p>
<h2>Option-order sensitivity</h2>{table(['Variant','Original correct /20','Reversed correct /20','Changed answers /20'],order_rows)}
<h2>Timing and reproducibility</h2><p>BF16 Qwen3-8B on the same M4 Max/MLX setup; thinking disabled, one output-token decision, no generation loop. Each full state is recomputed. No Qwen prefix, document or whole-request cache is reused. CLM action embeddings are cached outside timed decisions. Methods and no-passage controls are randomly interleaved within each question. Latencies include prompt construction, tokenization, prefill and selecting the allowed label; GPU outputs are materialized before stopping the timer. Model loading, CLM action setup and secondary output diagnostics are excluded. There is one timed call per condition on each of 200 different questions, not repeated identical-prompt microbenchmarks.</p>
<p>The new original-prompt run agrees with {s['baseline_reproduction']['original']['same_choices']}/200 prior Qwen choices; CLM agrees with {s['baseline_reproduction']['clm']['same_choices']}/200 prior choices. Small BF16 differences can affect close decisions. Compare latency within this new interleaved run rather than attributing differences between separate sessions entirely to prompt changes.</p>
<h2>Literal first-example prompts</h2>{prompts}
<h2>Files</h2><p><a href="summary.json">Statistics</a> · <a href="answers.jsonl">All predictions and timings</a> · <a href="manifest.json">Conditions and fixed sample hash</a> · <a href="prompt_examples.json">Exact prompt examples</a> · <a href="runtime.json">Runtime metadata</a> · <a href="../race_context/report.html">Original RACE-M report</a></p>
<pre>.venv/bin/python -u experiments/bench_race_ablations.py
.venv/bin/python experiments/analyze_race_ablations.py</pre>
<p>The runner resumes its append-only log. Preserve this results directory before running a fresh replication. Public-data training contamination is not audited. These zero-shot results compare existing released models and do not measure what task-specific CLM training could achieve.</p></main></html>'''
    (OUT/'report.html').write_text(body)
    print(json.dumps(s,indent=2))


if __name__=='__main__':main()
