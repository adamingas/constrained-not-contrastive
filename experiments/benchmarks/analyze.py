"""Build an auditable report; show all outcomes, including counterexamples."""
import collections,datetime,hashlib,html,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/decision_benchmarks'
LABELS={'clm_base':'CLM · direct action text','clm_prose_base':'CLM · prose schemas','clm_menu_base':'CLM · menu in state, JSON','clm_menu_prose_base':'CLM · menu in state, prose','qwen_raw_base':'Qwen · short prompt','qwen_raw_tuned':'Qwen · longer prompt'}
def load(name):
 p=OUT/name
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def stats(rows):
 vals=np.array([r['ms'] for r in rows]);n=len(rows);k=sum(r['correct'] for r in rows)
 z=1.95996398454;center=(k/n+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
 return {'n':n,'correct':k,'accuracy':k/n,'wilson95':[center-half,center+half],'median_ms':float(np.median(vals)),'p95_ms':float(np.percentile(vals,95)), 'median_setup_ms':float(np.median([r.get('setup_ms',0) for r in rows])),'median_first_query_total_ms':float(np.median([r.get('first_query_total_ms',r.get('setup_ms',0)+r['ms']) for r in rows]))}
def grouped(rows,fields):
 g=collections.defaultdict(list)
 for r in rows:g[tuple(r[f] for f in fields)].append(r)
 return [{'group':list(k),**stats(v)} for k,v in sorted(g.items())]
def paired_accuracy(rows,a,b):
 da={r['id']:r for r in rows if r['variant']==a};db={r['id']:r for r in rows if r['variant']==b};ids=sorted(set(da)&set(db))
 dif=np.array([int(db[i]['correct'])-int(da[i]['correct']) for i in ids]);rng=np.random.default_rng(3709);boot=np.mean(dif[rng.integers(0,len(ids),(10000,len(ids)))],axis=1)
 return {'a':a,'b':b,'n':len(ids),'b_minus_a_percentage_points':float(100*dif.mean()),'paired_bootstrap95_pp':(100*np.percentile(boot,[2.5,97.5])).tolist(),'b_only_correct':int(np.sum(dif==1)),'a_only_correct':int(np.sum(dif==-1))}
def paired_latency(rows,a,b):
 g=collections.defaultdict(list)
 for r in rows:g[r['id'],r['variant']].append(r['ms'])
 ids=sorted({k[0] for k in g if (k[0],a) in g and (k[0],b) in g});x=np.array([np.median(g[i,a]) for i in ids]);y=np.array([np.median(g[i,b]) for i in ids]);ratios=y/x
 rng=np.random.default_rng(8291);boot=np.median(ratios[rng.integers(0,len(ids),(10000,len(ids)))],axis=1)
 return {'a':a,'b':b,'states':len(ids),'repetitions_per_state':len(g[ids[0],a]),'a_median_ms':float(np.median(x)),'b_median_ms':float(np.median(y)),'paired_median_overhead_percent':float((np.median(ratios)-1)*100),'state_bootstrap95_overhead_percent':((np.percentile(boot,[2.5,97.5])-1)*100).tolist(),'ratios_by_state':dict(zip(ids,ratios.tolist()))}
def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def acc(s):return f"{s['correct']}/{s['n']} ({100*s['accuracy']:.1f}%)"
def metric_table(groups):
 return table(['Task / condition','Method','Correct','95% Wilson interval','Cached decision median'],[[html.escape(' / '.join(x['group'][:-1])),LABELS.get(x['group'][-1],x['group'][-1]),acc(x),f"{x['wilson95'][0]*100:.1f}–{x['wilson95'][1]*100:.1f}%",f"{x['median_ms']:.1f} ms"] for x in groups])
def main():
 test=load('test.jsonl');diag=load('diagnostics.jsonl');lat=load('latency.jsonl');rep=load('repeated.jsonl');nav=load('navigation.jsonl');audit=load('decoder_audit.jsonl');dev=load('dev.jsonl')
 cases=json.loads((OUT/'cases.json').read_text())
 summary={'built_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary':grouped(test,['family','variant']),'by_dataset':grouped(test,['dataset','variant']),'development':grouped(dev,['family','variant']),'diagnostics':grouped(diag,['diagnostic','dataset','variant']),'diagnostic_overall':grouped(diag,['diagnostic','variant']),'repeated_requests':grouped(rep,['variant']),'paired_accuracy':[],'matched_latency':[]}
 for family in ['bfcl','wiki']:
  rs=[r for r in test if r['family']==family]
  for a,b in [('clm_base','qwen_raw_base'),('qwen_raw_base','qwen_raw_tuned')]:summary['paired_accuracy'].append({'family':family,**paired_accuracy(rs,a,b)})
  if lat:
   for a,b in [('clm_base','qwen_raw_base'),('clm_base','qwen_raw_tuned'),('qwen_raw_base','qwen_raw_tuned')]:summary['matched_latency'].append({'family':family,**paired_latency([r for r in lat if r['family']==family],a,b)})
 if lat:
  for a in ['clm_base','clm_prose_base']:
   summary['matched_latency'].append({'family':'bfcl_forced_tool',**paired_latency([r for r in lat if r['family']=='bfcl_forced_tool'],a,'qwen_raw_base')})
 summary['navigation']=[]
 summary['paired_accuracy'].append({'family':'bfcl_forced_tool',**paired_accuracy([r for r in diag if r['diagnostic']=='forced_tool'],'clm_base','qwen_raw_base')})
 for v in ['clm_base','qwen_raw_base','qwen_raw_tuned']:
  rs=[r for r in nav if r['variant']==v]
  if rs:summary['navigation'].append({'variant':v,'n':len(rs),'success':sum(r['success'] for r in rs),'total_clicks':sum(len(r['steps']) for r in rs),'total_setup_ms':sum(s['setup_ms'] for r in rs for s in r['steps']),'total_decision_ms':sum(s['ms'] for r in rs for s in r['steps']),'paths':[{'id':r['id'],'success':r['success'],'path':r['path']} for r in rs]})
 summary['decoder_audit']={'n':len(audit),'all_same_choice':all(x['same_choice'] for x in audit),'max_probability_error':max((x['max_probability_error'] for x in audit),default=None)}
 summary['token_distributions']=[]
 for family in ['bfcl','wiki']:
  for variant in ['clm_base','qwen_raw_base','qwen_raw_tuned']:
   rs=[r for r in test if r['family']==family and r['variant']==variant]
   summary['token_distributions'].append({'family':family,'variant':variant,**{k:{'median':float(np.median([r[k] for r in rs])),'max':max(r[k] for r in rs)} for k in ['input_tokens','prefix_tokens','processed_tokens','choice_count']}})
 summary['run_counts']={k:len(v) for k,v in [('dev',dev),('test',test),('diagnostics',diag),('latency',lat),('repeated',rep),('navigation',nav)]}
 summary['random_next_step_accuracy']={family:float(np.mean([len(x['gold'])/x['choice_count'] for x in cases if x['split']=='test' and x['family']==family])) for family in ['bfcl','wiki']}
 summary['random_navigation']=json.loads((OUT/'random_navigation.json').read_text())
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 # Two panels separate function ranking from the synthetic abstention test.
 fig,axes=plt.subplots(1,3,figsize=(13,4.5));fig.patch.set_facecolor('#fafaf7')
 panels=[('Tool choice\n60 requests, no abstention', [r for r in diag if r['diagnostic']=='forced_tool'],['clm_base','clm_prose_base','qwen_raw_base']),('Tool choice + abstention\n90 requests',test and [r for r in test if r['family']=='bfcl'],['clm_base','clm_prose_base','qwen_raw_base','qwen_raw_tuned']),('Wikipedia shortest next step\n40 source–target pairs',[r for r in test if r['family']=='wiki'],['clm_base','qwen_raw_base','qwen_raw_tuned'])]
 colors={'clm_base':'#8391a3','clm_prose_base':'#596d85','qwen_raw_base':'#087f8c','qwen_raw_tuned':'#d58b29'}
 short={'clm_base':'CLM','clm_prose_base':'CLM\nprose','qwen_raw_base':'Qwen\nshort','qwen_raw_tuned':'Qwen\nlonger'}
 for panel_index,(ax,(title,rows,vs)) in enumerate(zip(axes,panels)):
  ss=[stats([r for r in rows if r['variant']==v]) for v in vs];ys=[100*s['accuracy'] for s in ss]
  ax.bar(range(len(vs)),ys,color=[colors[v] for v in vs],width=.66)
  ax.errorbar(range(len(vs)),ys,yerr=[[y-100*s['wilson95'][0] for y,s in zip(ys,ss)],[100*s['wilson95'][1]-y for y,s in zip(ys,ss)]],fmt='none',color='#243447',capsize=3)
  for j,s in enumerate(ss):ax.text(j,min(104,100*s['wilson95'][1]+2),f"{s['correct']}/{s['n']}",ha='center',fontsize=10)
  ax.set_xticks(range(len(vs)),[short[v] for v in vs]);ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  if panel_index==1:ax.axhline(100/3,ls='--',color='#a05e00',lw=1,label='Always abstain');ax.legend(loc='upper left',fontsize=8)
  if panel_index==2:ax.axhline(100*summary['random_next_step_accuracy']['wiki'],ls='--',color='#a05e00',lw=1,label='Uniform random link');ax.legend(loc='upper left',fontsize=8)
 axes[0].set_ylabel('Accuracy (%)');fig.suptitle('Constrained Qwen versus the released CLM head',fontsize=15);fig.tight_layout();fig.savefig(OUT/'accuracy.png',dpi=180);plt.close(fig)
 author=table(['Reported comparison','Availability and treatment here'],[
 ['BFCL v4','Public BFCL requests and function schemas available. Authors’ selected cases/prompts unavailable; our experiment tests function selection, not argument generation or the full BFCL score.'],
 ['WikiRacing','Authors’ 30 races and harness unavailable. We use the public Wikispeedia directed graph, all outgoing links, title-only observations, and five short actual races.'],
 ['Super Mario / T-Rex','Previously tested in this workspace. The planner-assisted T-Rex result is excluded from this report’s evidence.'],
 ['DeepSWE / Terminal-Bench verifier','Task-specific fine-tuned heads and sampled candidate solutions are involved. The accessible DeepSWE evaluation dataset contains embeddings and metadata, not the original state/action text needed by Qwen. No fair text-based reproduction completed.']])
 matched=table(['Task','Comparison','Median ms, A → B','Paired median overhead','95% interval'],[[x['family'],LABELS[x['a']]+' → '+LABELS[x['b']],f"{x['a_median_ms']:.1f} → {x['b_median_ms']:.1f}",f"{x['paired_median_overhead_percent']:+.1f}%",f"{x['state_bootstrap95_overhead_percent'][0]:+.1f} to {x['state_bootstrap95_overhead_percent'][1]:+.1f}%"] for x in summary['matched_latency']])
 navtable=table(['Method','Reached target within 8 clicks','Total clicks','Total cache setup','Total decision time'],[[LABELS[x['variant']],f"{x['success']}/{x['n']}",x['total_clicks'],f"{x['total_setup_ms']/1000:.2f} s",f"{x['total_decision_ms']/1000:.2f} s"] for x in summary['navigation']])
 cold=table(['Task','Method','Median menu setup','Median setup + first decision'],[[x['group'][0],LABELS[x['group'][1]],f"{x['median_setup_ms']:.1f} ms",f"{x['median_first_query_total_ms']:.1f} ms"] for x in summary['primary']])
 htmltext='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLM versus constrained Qwen: public benchmark adaptations</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#243447;background:#fafaf7;max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:32px;line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0;display:block;overflow:auto}th,td{padding:10px 13px;border-bottom:1px solid #cfd7dd;text-align:left}th{background:#e7edf0}a{color:#076c83}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf0f1;padding:18px;font-size:13px}.note{border-left:4px solid #d58b29;padding:12px 18px;background:#fff4df}.lead{font-size:20px}details{margin:18px 0}</style></head><body><h1>Constrained Qwen versus CLM</h1><p class="lead">Public tool-selection and Wikipedia-navigation experiments, with frozen prompts, cached decisions, and no planner-provided answers.</p>'''
 htmltext+='''<p class="note">These are adaptations of benchmark families cited by the CLM authors, not reproductions of their reported scores. We compare the released CLM head to its original Qwen3-8B backbone on the same Mac; Jev is not run here. A desired conclusion was not used to select test cases or suppress unfavorable outcomes.</p>'''
 htmltext+='''<p><strong>Main finding:</strong> constrained Qwen selected the correct tool on 55/60 requests (91.7%), versus CLM’s 40/60 (66.7%), at approximately 81 versus 79 ms per cached decision. Qwen retained every CLM-correct answer and added 15. The paired accuracy difference is +25 percentage points, with a 95% bootstrap interval of +15 to +36.7 points.</p><p>The broader claim needs restraint: Wikipedia next-step scores were close to the random-link baseline, and the five actual races ended 1/5 for Qwen and 0/5 for CLM. Longer Qwen instructions changed BFCL from 79/90 to 78/90 and Wiki next-step accuracy from 15/40 to 16/40. The strongest supported result is a competitive constrained-decoding baseline with little cached latency overhead, not a general guarantee that longer prompts greatly improve accuracy.</p>'''
 htmltext+='<h2>What was available</h2>'+author+'<p>Sources: <a href="https://github.com/Contrastive-LM/CLM">CLM release</a>, <a href="https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html">BFCL v4</a>, <a href="https://snap.stanford.edu/data/wikispeedia.html">Wikispeedia</a>, <a href="https://huggingface.co/datasets/tarsur385/deepswe-prm-embeddings-8k">DeepSWE embedding dataset</a>.</p>'
 htmltext+='<details><summary>Authors’ reported zero-shot comparison with Jev</summary>'+table(['Benchmark','CLM success','Jev success','CLM latency','Jev latency'],[['T-Rex','5/5','5/5','16.5 ms','149.8 ms'],['BFCL v4','95.2%','99.2%','76.8 ms','125.5 ms'],['WikiRacing','26/30','30/30','79.8 ms','225 ms'],['Super Mario','5/5','5/5','33.5 ms','132.6 ms']])+'<p>Transcribed from the release’s <a href="https://github.com/Contrastive-LM/CLM/blob/main/assets/zero-shot.png">zero-shot figure</a>. These are author-reported measurements with different inputs and deployment conditions; they are not directly comparable to the local latencies below.</p></details>'
 htmltext+='''<h2>Accuracy</h2><img src="accuracy.png" alt="Accuracy with 95% Wilson intervals"><p>The no-abstention comparison uses the original function candidates on 60 positive requests. It was added as a diagnostic after discovering that CLM always selected the synthetic NO_TOOL candidate in the primary JSON-action test. The primary 90-request test has 30 requests from each of multiple, live_multiple, and live_irrelevance; NO_TOOL is available to both models on every request. Its 33.3% always-abstain baseline is therefore essential context.</p>'''+metric_table(summary['primary'])
 htmltext+='<details><summary>Primary results by BFCL category</summary>'+metric_table(summary['by_dataset'])+'</details>'
 htmltext+='''<p>Wikispeedia scores whether the selected outgoing link is on any shortest route to the target. This is a strict graph metric: a plausible but longer route counts as wrong. Neither model sees graph distances or future link lists. The episode experiment below instead measures actual arrival within eight clicks.</p>'''
 htmltext+=f"<p><strong>Wiki next-step baseline:</strong> a uniformly random outgoing link has {100*summary['random_next_step_accuracy']['wiki']:.1f}% expected accuracy on these starting states, since several links may belong to equally short routes. The 35–40% model scores therefore give little evidence of a useful advantage on this metric alone.</p>"
 htmltext+='<h2>Fairness diagnostics</h2>'+metric_table(summary['diagnostic_overall'])+'<p>All eligible held-out cases receive each diagnostic; there is no test-driven prompt search. Menu-in-state CLM receives the same available-function information that Qwen sees. CLM action serialization is tested both as compact JSON and using the release’s prose serializer. These diagnostics are explicitly separate from the development-selected primary comparison.</p>'
 htmltext+='<h2>Matched cached latency</h2>'+matched+'''<p>Five interleaved repetitions on each of ten held-out states per task family. Compute each state’s median, then compare paired ratios; 95% intervals bootstrap states, not individual repetitions. A ≤10% median overhead was specified before held-out results as a descriptive “near latency” threshold, not a universal guarantee. Both methods reuse their actual caches: CLM caches action vectors; Qwen caches the instruction/menu KV prefix. The changing request is recomputed on every call. Complete-answer memoization is disabled.</p>'''
 htmltext+='<h2>Different requests sharing the same tool menu</h2>'+metric_table(summary['repeated_requests'])+'<p>Six held-out schema groups receive 24 additional public BFCL requests in total, in source order, with one cache setup per group. These requests were not used to tune prompts. They supplement the timing microbenchmark with genuine fixed-menu reuse. Requests within a menu are correlated; this is a small feasibility test, not 24 independent task families.</p>'
 htmltext+='<h2>Actual Wikipedia races</h2>'+navtable+'<p>The first five frozen held-out pairs, at most eight clicks each, all outgoing links retained. Each model follows its own actions; no action is replaced by an oracle. Page-menu KV caches and individual CLM article vectors are reused where possible within each race. Different paths require different work, so cumulative times are application outcomes, not matched per-state latency comparisons.</p>'
 htmltext+=f"<p>A uniform-random policy reached the target in {summary['random_navigation']['success_rate']*100:.3f}% of 50,000 eight-click simulations on these same five starting pairs. Unlike the first-step metric, actual arrival is very unlikely by chance. Five model races still provide only pilot-scale evidence.</p>"
 for x in summary['navigation']:
  htmltext+='<details><summary>'+LABELS[x['variant']]+' paths</summary><pre>'+html.escape(json.dumps(x['paths'],indent=2,ensure_ascii=False))+'</pre></details>'
 htmltext+='<h2>Setup costs and limits of the timing claim</h2>'+cold+'''<p>Model loading is excluded. Setup includes Qwen prefix construction or CLM action embedding; the latter currently uses sequential embedding calls. Thus these cold CLM timings describe this implementation, not an optimized batched server. A changing Wikipedia link menu cannot automatically reuse the entire previous Qwen prefix. The cached-decision numbers apply after the required menu cache exists. No claim about H100 serving, concurrency, remote API latency, or Jev’s implementation follows from this local measurement.</p>'''
 htmltext+='<h2>Prompt engineering: what the experiment establishes</h2><p>The short and longer Qwen prompts use identical weights and identical one-token constraints. Both were evaluated on the same held-out cases. A longer prompt is not automatically better; see the paired differences below. This experiment does not measure MLP training time or cost, so it cannot quantify how much easier prompt engineering is. No training is needed to deploy the Qwen baseline tested here.</p><pre>'+html.escape(json.dumps(summary['paired_accuracy'],indent=2))+'</pre>'
 htmltext+='''<h2>Implementation and sampling</h2><p>BF16 Qwen3-8B on Apple M4 Max, 64 GB, MLX; same loaded backbone for both paths. CLM uses the released approximately 18.9-million-parameter projection-head pair with normalized final hidden states; the port is checked against PyTorch. Qwen uses its original LM-head rows for verified single-token labels. It performs one state forward pass and argmax over only allowed labels, with no generation loop, thinking tokens, or additional completion pass. A–Z and verified single-token two-letter labels support larger menus. The full-vocabulary restricted-argmax audit is recorded below.</p><p>Development: 10 BFCL and 10 Wiki cases. Held-out: 90 BFCL and 40 Wiki cases. BFCL schemas are disjoint across development and test; Wiki targets are disjoint. Source-order deduplication and fixed random sampling happen before model scoring. Wiki source/target pairs come from human successful paths, have shortest distance 2–4, and retain every outgoing link. These are title-only observations, not full Wikipedia article reading. Most tool requests are short relative to their schemas; this suite does not replace the earlier RACE context-heavy experiment.</p>'''
 htmltext+='<pre>'+html.escape(json.dumps(summary['decoder_audit'],indent=2))+'</pre>'
 htmltext+='<details><summary>Prompt and cache token counts</summary><pre>'+html.escape(json.dumps(summary['token_distributions'],indent=2))+'</pre></details>'
 htmltext+='<h2>Exact inputs and reproducibility</h2><p><a href="exact_prompts.json">Exact text, token IDs, and choices</a> · <a href="prompt_rules.json">All prompt rules</a> · <a href="frozen.json">Prespecified primary selection</a> · <a href="diagnostic_plan.json">Diagnostic rationale</a> · <a href="cases.json">Frozen cases and scoring labels</a> · <a href="summary.json">Machine-readable results</a></p><p>Scoring labels, graph distances, and expected answers in the saved case files are evaluation metadata only; input construction accesses state, question, and actions, never the gold field.</p>'
 htmltext+='<p>Held-out means excluded from this prompt-development sample; it does not establish that these public examples were absent from either model’s pretraining.</p>'
 htmltext+='<p>Raw outputs: '+ ' · '.join(f'<a href="{n}">{n}</a>' for n in ['dev.jsonl','test.jsonl','diagnostics.jsonl','latency.jsonl','repeated.jsonl','navigation.jsonl','decoder_audit.jsonl','sampling.json','model_metadata.json'])+'</p>'
 htmltext+='<p>Scripts: prepare.py, run.py, diagnostics.py, supplement.py, analyze.py under experiments/benchmarks. Re-run with .venv/bin/python; run model stages sequentially to avoid GPU contention. The earlier numeric-label development pilot stopped when a two-digit label was not a single token; it is retained for audit but excluded from all reported results.</p></body></html>'
 (OUT/'report.html').write_text(htmltext)
 (OUT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted((ROOT/'experiments/benchmarks').glob('*.py')),*sorted(OUT.glob('*.jsonl'))]},indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
