"""Summaries, paired statistics and plots; no inference or sample selection."""
import collections
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/long_context'
NAMES={'quality':'QuALITY','contract_nli':'ContractNLI','longbench_v2':'LongBench v2 (≤16K)'}
VARIANTS=['clm_full','qwen_full','clm_no_context','qwen_no_context']


def wilson(k,n):
    z=1.959963984540054
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-delta),min(1.,center+delta)]


def paired(a,b):
    """b minus a; exact two-sided McNemar plus paired bootstrap interval."""
    a=np.array(a,dtype=int);b=np.array(b,dtype=int)
    only_a=int(((a==1)&(b==0)).sum());only_b=int(((b==1)&(a==0)).sum())
    discord=only_a+only_b
    p=min(1.,2*sum(math.comb(discord,i) for i in range(min(only_a,only_b)+1))/2**discord) if discord else 1.
    rng=np.random.default_rng(42)
    diff=b-a
    boot=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(axis=1)
    return {'n':len(a),'difference':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'only_first_correct':only_a,'only_second_correct':only_b,'both_correct':int(((a==1)&(b==1)).sum()),
            'both_wrong':int(((a==0)&(b==0)).sum()),'mcnemar_exact_p':p}


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    cases=json.loads((OUT/'cases.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    answers=[json.loads(x) for x in (OUT/'answers.jsonl').read_text().splitlines()]
    by={(x['id'],x['variant']):x for x in answers}
    assert len(by)==len(answers),'Duplicate outputs'
    expected=sum(4+int(c['order_audit']) for c in cases)
    assert len(answers)==expected,(len(answers),expected)
    case_by_id={c['id']:c for c in cases}
    assert hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest()==manifest['cases_sha256']
    for row in answers:
        assert row['gold']==case_by_id[row['id']]['gold']
        assert row['correct']==(row['choice']==row['gold'])
        assert row['choice']==max(row['probabilities'],key=row['probabilities'].get)
        assert abs(sum(row['probabilities'].values())-1)<1e-5
    result={'n_questions':len(cases),'n_predictions':len(answers),'datasets':{},'length_bins':{},'subgroups':{}}
    for dataset in NAMES:
        subset=[c for c in cases if c['dataset']==dataset]
        stats={'n':len(subset),'chance_accuracy':1/len(subset[0]['q']['criteria']),'context_tokens':manifest['context_tokens'][dataset], 'accuracy':{}}
        for variant in VARIANTS:
            rows=[by[c['id'],variant] for c in subset]
            k=sum(x['correct'] for x in rows)
            stats['accuracy'][variant]={'correct':k,'total':len(rows),'accuracy':k/len(rows),'wilson_ci95':wilson(k,len(rows)),
                                        'median_ms':float(np.median([x['ms'] for x in rows]))}
        stats['paired_qwen_vs_clm']=paired([by[c['id'],'clm_full']['correct'] for c in subset],[by[c['id'],'qwen_full']['correct'] for c in subset])
        for m in ['clm','qwen']:
            stats[m+'_context_gain']=paired([by[c['id'],m+'_no_context']['correct'] for c in subset],[by[c['id'],m+'_full']['correct'] for c in subset])
        audit=[c for c in subset if c['order_audit']]
        stats['option_order_audit']={'n':len(audit),'original_correct':sum(by[c['id'],'qwen_full']['correct'] for c in audit),
                                      'reversed_correct':sum(by[c['id'],'qwen_reversed']['correct'] for c in audit),
                                      'changed_answers':sum(by[c['id'],'qwen_full']['choice']!=by[c['id'],'qwen_reversed']['choice'] for c in audit)}
        stats['confusion_matrices']={m:dict(collections.Counter(f"gold={c['gold']},pred={by[c['id'],m+'_full']['choice']}" for c in subset)) for m in ['clm','qwen']}
        result['datasets'][dataset]=stats
    for lo,hi in [(1024,2048),(2048,4096),(4096,8192),(8192,16385)]:
        subset=[c for c in cases if lo<=c['context_tokens']<hi]
        result['length_bins'][f'{lo}-{hi-1}']={'n':len(subset),'datasets':dict(collections.Counter(c['dataset'] for c in subset)),
            **{m:sum(by[c['id'],m]['correct'] for c in subset)/len(subset) for m in VARIANTS}}
    for label in [0,1]:
        subset=[c for c in cases if c['dataset']=='quality' and c['difficulty']==label]
        result['subgroups']['quality_hard' if label else 'quality_easy']={'n':len(subset),**{m:sum(by[c['id'],m]['correct'] for c in subset) for m in VARIANTS}}
    # Post-hoc data-format audit, without changing inputs or primary results.
    # A few original answer strings retain surrounding whitespace, which can
    # affect a final-token embedding. Report the unaffected subset separately.
    subset=[c for c in cases if c['dataset']=='quality' and all(v==v.strip() for v in c['q']['criteria'].values())]
    result['subgroups']['quality_no_option_surrounding_whitespace_exploratory']={'n':len(subset),**{m:sum(by[c['id'],m]['correct'] for c in subset) for m in VARIANTS}}
    for label in ['Entailment','Contradiction','NotMentioned']:
        subset=[c for c in cases if c['dataset']=='contract_nli' and c['label']==label]
        result['subgroups']['contract_'+label]={'n':len(subset),**{m:sum(by[c['id'],m]['correct'] for c in subset) for m in VARIANTS}}
    lb=[c for c in cases if c['dataset']=='longbench_v2']
    for domain in sorted({c['domain'] for c in lb}):
        subset=[c for c in lb if c['domain']==domain]
        result['subgroups']['longbench_'+domain]={'n':len(subset),**{m:sum(by[c['id'],m]['correct'] for c in subset) for m in VARIANTS}}
    lat=[json.loads(x) for x in (OUT/'latency.jsonl').read_text().splitlines()]
    assert len(lat)==7
    result['latency_sweep']=lat
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.5))
    x=np.arange(3);w=.19
    colors=['#b95c45','#087c8b','#b95c45','#087c8b']
    labels=['CLM · full context','Qwen · full context','CLM · no document','Qwen · no document']
    for i,m in enumerate(VARIANTS):
        vals=np.array([result['datasets'][d]['accuracy'][m]['accuracy'] for d in NAMES])*100
        cis=np.array([result['datasets'][d]['accuracy'][m]['wilson_ci95'] for d in NAMES])*100
        bars=ax.bar(x+(i-1.5)*w,vals,w,color=colors[i],alpha=1 if i<2 else .28,label=labels[i],
                    yerr=[vals-cis[:,0],cis[:,1]-vals],capsize=3,error_kw={'elinewidth':1,'alpha':.6})
        ax.bar_label(bars,labels=[f'{v:.0f}%' for v in vals],padding=5,fontsize=9)
    ax.plot(x,[25,100/3,25],linestyle='none',marker='_',markersize=60,color='#444',label='Uniform random choice')
    ax.set_xticks(x,[f'{NAMES[d]}\nn={result["datasets"][d]["n"]}' for d in NAMES]);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)')
    ax.set_title('Full-document decisions versus document-removal controls',loc='left',pad=16,fontweight='bold')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,markerscale=.3)
    fig.text(.015,.015,'Same BF16 Qwen3-8B backbone. 95% Wilson intervals. Selected subsets; not official leaderboard scores.',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'accuracy.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,(ax,ratio)=plt.subplots(1,2,figsize=(12,4.6))
    xx=[a['context_tokens'] for a in lat]
    for m,label,color in [('clm_cached_actions','CLM · cached actions','#b95c45'),('qwen_cached_prefix','Qwen · cached options','#087c8b'),('qwen_uncached','Qwen · uncached options','#6757a2')]:
        yy=[a['methods'][m]['p50_ms']/1000 for a in lat]
        ax.plot(xx,yy,'o-',color=color,label=label)
        if m!='clm_cached_actions':ratio.plot(xx,[a['methods'][m]['p50_ms']/a['methods']['clm_cached_actions']['p50_ms'] for a in lat],'o-',color=color,label=label)
    for a in (ax,ratio):a.set_xscale('log',base=2);a.set_xticks(xx,[f'{n//1024}K' if n>=1024 else str(n) for n in xx]);a.set_xlabel('Context tokens');a.grid(alpha=.2)
    ax.set_ylabel('Median decision latency (s)');ratio.set_ylabel('Qwen latency / CLM latency');ratio.axhline(1,color='#333',linestyle='--',linewidth=1)
    ax.set_title('Reading the shared context dominates',loc='left',fontweight='bold');ratio.set_title('Relative overhead',loc='left',fontweight='bold')
    ax.legend(frameon=False,fontsize=9);fig.text(.02,.015,'M4 Max · BF16 · 1024-token prefill chunks · fixed four options · five timed trials per point',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUT/'latency.png',dpi=170,bbox_inches='tight');plt.close(fig)
    rows=[];contextrows=[];pairedrows=[]
    for d,s in result['datasets'].items():
        rows.append([NAMES[d],s['n'],f"{s['context_tokens']['median']:,.0f}",f"{s['chance_accuracy']:.1%}"]+[f"{s['accuracy'][m]['correct']}/{s['n']} ({s['accuracy'][m]['accuracy']:.1%})" for m in VARIANTS])
        p=s['paired_qwen_vs_clm'];pairedrows.append([NAMES[d],f"{100*p['difference']:+.1f} pp",f"{100*p['ci95'][0]:+.1f} to {100*p['ci95'][1]:+.1f} pp",p['only_first_correct'],p['only_second_correct'],f"{p['mcnemar_exact_p']:.3g}"])
        contextrows.append([NAMES[d]]+[f"{100*s[m+'_context_gain']['difference']:+.1f} pp" for m in ['clm','qwen']])
    latencyrows=[]
    for a in lat:
        clm=a['methods']['clm_cached_actions']['p50_ms']
        latencyrows.append([a['context_tokens']]+[f"{a['methods'][m]['p50_ms']:.1f} ms" for m in ['clm_cached_actions','qwen_cached_prefix','qwen_uncached']]+[f"{a['methods'][m]['p50_ms']/clm:.3f}×" for m in ['qwen_cached_prefix','qwen_uncached']])
    subgroup_names={'quality_easy':'QuALITY: easy','quality_hard':'QuALITY: hard',
                    'quality_no_option_surrounding_whitespace_exploratory':'QuALITY: whitespace-free options (exploratory)',
                    'contract_Entailment':'ContractNLI: entailment','contract_Contradiction':'ContractNLI: contradiction',
                    'contract_NotMentioned':'ContractNLI: not mentioned'}
    subrows=[[subgroup_names.get(k,k.replace('longbench_','LongBench: ')),s['n']]+[f"{s[m]}/{s['n']}" for m in VARIANTS] for k,s in result['subgroups'].items()]
    auditrows=[[NAMES[d],s['option_order_audit']['n'],s['option_order_audit']['original_correct'],s['option_order_audit']['reversed_correct'],s['option_order_audit']['changed_answers']] for d,s in result['datasets'].items()]
    last=lat[-1]['methods']
    final_overhead=100*(last['qwen_uncached']['p50_ms']/last['clm_cached_actions']['p50_ms']-1)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Long-context CLM versus constrained Qwen</title>
    <style>body{{font:17px/1.6 system-ui;color:#21303e;background:#f3f6f8;margin:0}}main{{max-width:1160px;margin:35px auto;padding:40px;background:white;border-radius:14px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:38px}}p{{max-width:1000px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #d9e0e7}}th{{background:#eef4f5}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}a{{color:#07689c}}code,pre{{background:#f0f3f5}}pre{{padding:18px;overflow:auto}}.note{{padding:18px;background:#f1f7f7;border-left:4px solid #087c8b}}@media(max-width:700px){{main{{padding:20px;margin:0}}h1{{font-size:27px}}}}</style><main>
    <p>2 October 2026 · local reproducible evaluation</p><h1>Does constrained Qwen retain accuracy when the decision needs a long document?</h1>
    <p class="note"><strong>Follow-up qualification:</strong> a <a href="../race_context/report.html">RACE-M experiment with shorter, context-dependent questions</a> found a large Qwen prompt-layout effect on a separate development sample: 60% with this report's options-in-system layout versus 92.5% with passage–question–options order. The long-document results below remain valid for the recorded prompt, but should not be interpreted as prompt-independent model-capacity measurements. These long-document samples have not been rerun with the new layout.</p>
    <p class="note">{len(cases)} public-dataset questions, {len(answers)} predictions, complete documents, the same frozen BF16 Qwen3-8B backbone, no model fine-tuning or hidden reasoning tokens. These are explicitly selected diagnostic subsets, not official benchmark scores.</p>
    <p>For the fixed four-option latency test, at 16,384 context tokens CLM took {last['clm_cached_actions']['p50_ms']/1000:.2f} seconds and uncached constrained Qwen took {last['qwen_uncached']['p50_ms']/1000:.2f} seconds: only {final_overhead:.1f}% overhead. The measurements support the claim that option-processing cost becomes relatively small as the shared context grows.</p>
    <p>Constrained Qwen is numerically more accurate on all three selected datasets. Only the ContractNLI paired 95% interval excludes zero before adjustment for multiple comparisons. Neither approach is a reliable long-document solver here: Qwen benefits clearly from the contracts, but gets no accuracy gain from the LongBench documents and scores lower with the QuALITY articles than without them. CLM predicts “not mentioned” on 88 of 90 contracts.</p>
    <h2>Accuracy</h2>{table(['Dataset','n','Median context tokens','Chance','CLM full','Qwen full','CLM no document','Qwen no document'],rows)}<img src="accuracy.png" alt="Accuracy with and without documents, with confidence intervals">
    <h2>Paired comparison</h2>{table(['Dataset','Qwen minus CLM','Paired bootstrap 95% CI','Only CLM correct','Only Qwen correct','Exact McNemar p'],pairedrows)}
    <p>The same questions are scored for both methods. Bootstrap intervals use 10,000 paired resamples. Exact McNemar tests use only discordant pairs. Results are exploratory; intervals and p-values do not account for training-data contamination, prompt selection, or generalization beyond these subsets. No multiple-comparison correction is applied.</p>
    <h2>Did the context help?</h2>{table(['Dataset','CLM full minus no-document','Qwen full minus no-document'],contextrows)}
    <p>In the document-removal control, the question and identical answer choices remain. The document is replaced with an empty string. Full and control predictions share the same gold answer. A positive gain supports use of the document on this sample; it does not prove that every correct answer requires the entire document. A negative gain indicates that this scoring setup became less accurate when given the document.</p>
    <h2>Latency as context grows</h2><p>For a fixed option budget A and shared context length N, the rough comparison is CLM ≈ F(N) + head cost versus uncached constrained Qwen ≈ F(N+A) + output cost. If A stays fixed while N grows, their relative prefill costs should approach one under comparable kernels. This is a relative-cost claim, not a guarantee of equal absolute latency on every implementation, and it does not apply when the options themselves grow with the context.</p>{table(['Context tokens','CLM cached actions','Qwen cached prefix','Qwen uncached','Cached / CLM','Uncached / CLM'],latencyrows)}<img src="latency.png" alt="Latency and Qwen to CLM ratio versus context length">
    <p>The controlled sweep holds four option descriptions fixed and varies natural document text from 256 to 16,384 tokens. It is a compute benchmark, not an accuracy test: shorter excerpts may omit evidence, and the text is repeated only when needed to fill the longest point. All methods use a process-local 24 GiB wired-memory limit to avoid paging the backbone, use 1024-token prefill chunks, preserve all tokens and materialize GPU output before stopping the timer. Each point has one warm-up per method followed by five timed trials in randomized order. CLM action vectors and the cached Qwen prefix are prepared outside the timing. Qwen uncached processes its option text on every call. There are no whole-state cache hits or HTTP calls. Finite-length kernel effects and ordinary device variance can produce ratios slightly above or below one.</p>
    <h2>Subgroups and option order</h2>{table(['Subset','n','CLM full','Qwen full','CLM no document','Qwen no document'],subrows)}{table(['Dataset','Audit n','Qwen original correct','Qwen reversed correct','Answer changes'],auditrows)}
    <p>The primary option order was randomized once with the fixed seed. Reversal was preselected for ten questions per dataset before any model outputs existed. CLM independent action scores are permutation-equivariant by construction; the extra inference audit measures Qwen's prompt sensitivity. There is no answer-order ensembling in the primary result.</p>
    <p>Qwen changed its answer on 15/30 reversed-order cases: 7/10 QuALITY, 1/10 ContractNLI, and 7/10 LongBench. This is a material sensitivity even though aggregate accuracy barely changed. Original answer strings were preserved; nine QuALITY cases contain surrounding whitespace in at least one option. The separately labeled 71-case whitespace-free subgroup is a post-hoc formatting audit, not a replacement primary sample.</p>
    <h2>Data selection, fixed before inference</h2>
    <ul><li><a href="https://github.com/nyu-mll/quality">QuALITY v1.0.1 development set</a>: 80 distinct articles, 40 hard and 40 easy questions. Only questions whose median human annotation says at least one-third of the passage is needed are eligible. The full HTML-stripped articles are supplied.</li>
    <li><a href="https://stanfordnlp.github.io/contract-nli/">ContractNLI test set</a>: 90 distinct contracts, 30 examples each of entailment, contradiction and not-mentioned. Documents below 1024 tokens are excluded. This balances the original imbalanced labels and is not the official test distribution. Only NLI accuracy is evaluated, not evidence-span extraction.</li>
    <li><a href="https://huggingface.co/datasets/zai-org/LongBench-v2">LongBench v2</a>: all 27 eligible unique documents from the publisher's short category whose complete prompts fit within 16,384 tokens. The source has 503 questions; this is a small, length-selected subset concentrated in document QA, not an evaluation of its full task coverage. The Hugging Face split is named train, but it contains the published evaluation questions; no training was performed.</li></ul>
    <p>All selected contexts contain at least 1024 tokens. No selected document was truncated, summarized, retrieved in pieces or padded for the accuracy evaluation. One question is selected per document within each dataset, and candidates are randomized with seed 20261002. Data hashes, source revisions and case IDs are in <a href="manifest.json">the manifest</a>. The frozen case file SHA-256 is <code>{manifest['cases_sha256']}</code>.</p>
    <h2>Model and prompt controls</h2><p>A six-case implementation diagnostic selected the first two frozen questions from each dataset without conditioning on correctness. Recomputing Qwen with a full-vocabulary output projection and uncached prompt reproduced all six primary choices. Allowed letters carried at least 99.978% of the unrestricted next-token probability. Thus these errors were not caused by forcing an implausible first-token format. Moving the options to the end changed predictions but scored 2/6 versus 3/6 for the original layout; this small exploratory check was not used to change the primary prompt. See <a href="prompt_diagnostics.json">the diagnostic outputs</a>.</p>
    <p>The released <a href="https://github.com/Contrastive-LM/CLM">CLM-v0.1 heads</a> use their original last-token representation, normalized input embeddings, FP32 projection heads, output normalization and checkpoint logit-scale clamp. The backbone is the original <a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen/Qwen3-8B</a> in BF16. Qwen uses the same backbone and its original output weights, <code>enable_thinking=False</code>, and a single forced choice among verified one-token A/B/C/D labels. Options appear in the system message, document and question in the user message. No answer explanation, extra autoregressive step or external model is used.</p>
    <p>Chunked prefill is a memory-management choice, not context compression: the KV cache retains all earlier tokens. A 2050-token spot-check of chunked versus unchunked computation gave cosine agreement 0.999970. The prior investigation also checked backbone embeddings against Transformers and the CLM head port against PyTorch. Full-document timing includes tokenization, prefix-copy overhead and scoring, but excludes candidate/prefix setup. Setup durations are preserved separately in the answer log. Earlier accuracy predictions were collected before the process-local wired-memory setting was enabled; this does not change the computation, but their per-question timings can contain paging noise. Only the separate, consistently configured latency sweep is used for the speed comparison.</p>
    <p>This evaluates off-the-shelf released models. The CLM server defaults to a 2048-token limit, which its documentation allows increasing; the long-context run increases the effective limit and does not train new heads. Thus it measures whether the existing head generalizes to these long inputs, not the best achievable long-context CLM after fine-tuning. Model training sets may overlap public benchmarks; contamination is not audited. A single prompt template per task is used without answer-driven tuning.</p>
    <h2>Artifacts and reproduction</h2><p><a href="summary.json">Summary and statistics</a> · <a href="answers.jsonl">Every prediction and probability</a> · <a href="cases.json">Frozen input cases</a> · <a href="latency.jsonl">Every timing trial</a> · <a href="runtime.json">Runtime metadata</a></p>
    <pre>uv pip install --python .venv/bin/python -r experiments/requirements-long-context-lock.txt
.venv/bin/python experiments/download_long_context.py
.venv/bin/python experiments/prepare_long_context.py
.venv/bin/python -u experiments/long_context_bench.py
.venv/bin/python experiments/analyze_long_context.py</pre>
    <p>The benchmark resumes existing outputs. To perform a separate fresh run, preserve this results directory and use a separate copy of the project or change the output path. Sources: Pang et al., QuALITY (NAACL 2022); Koreeda and Manning, ContractNLI (Findings of EMNLP 2021); Bai et al., LongBench v2 (ACL 2025). See linked dataset repositories for licenses and full citations.</p></main></html>'''
    (OUT/'report.html').write_text(body)
    print(json.dumps({d:s for d,s in result['datasets'].items()},indent=2))


if __name__=='__main__':main()
