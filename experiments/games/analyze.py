"""Summarize cached-game experiment; HTML report with exact prompts and replays."""
import collections,html,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'results/cached_games'
def read(name):return [json.loads(s) for s in (OUT/name).read_text().splitlines()]
def med(xs):return float(np.median(xs))
def ci(xs):
 a=np.array(xs);r=np.random.default_rng(20261003);z=np.median(a[r.integers(0,len(a),(5000,len(a)))],axis=1)
 return [float(x) for x in np.percentile(z,[2.5,97.5])]
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'
def main():
 cases=json.loads((OUT/'cases.json').read_text());index={c['id']:c for c in cases};matched=read('matched.jsonl');episodes=read('episodes.jsonl');metadata=json.loads((OUT/'metadata.json').read_text());refs=json.loads((OUT/'references.json').read_text());audit=read('audit.jsonl')
 assert len(matched)==len(cases)*4
 assert len({(r['id'],r['method'],r['cached']) for r in matched})==len(matched)
 assert len(episodes)==16
 summary={'matched':{},'comparisons':{},'trex_safety':{},'episodes':[{k:v for k,v in r.items() if k!='records'} for r in episodes],'setup':metadata['cached'],'audits':{}}
 lookup={(r['id'],r['method'],r['cached']):r for r in matched};timingrows=[];tokenrows=[]
 for game in ['trex','mario']:
  ids=[c['id'] for c in cases if c['game']==game]
  for method in ['clm','qwen']:
   for cached in [False,True]:
    rows=[lookup[(i,method,cached)] for i in ids];lat=[r['ms'] for r in rows];s={'n':len(rows),'median_ms':med(lat),'p95_ms':float(np.percentile(lat,95)),'median_input_tokens':med([r['input_tokens'] for r in rows]),'prefix_tokens':rows[0]['prefix_tokens'],'median_processed_tokens':med([r['processed_tokens'] for r in rows]),'fraction_within_decision_interval':sum(x<= (100 if game=='trex' else 200) for x in lat)/len(lat)}
    summary['matched'][f'{game}_{method}_{cached}']=s;timingrows.append([game,method,'cached' if cached else 'uncached',len(rows),f"{s['median_ms']:.1f}",f"{s['p95_ms']:.1f}"])
    if cached:tokenrows.append([game,method,s['median_input_tokens'],s['prefix_tokens'],s['median_processed_tokens']])
   ratios=[lookup[(i,method,True)]['ms']/lookup[(i,method,False)]['ms'] for i in ids]
   agreement=sum(lookup[(i,method,True)]['choice']==lookup[(i,method,False)]['choice'] for i in ids)
   probdiff=max(max(abs(a-b) for a,b in zip(lookup[(i,method,True)]['probabilities'],lookup[(i,method,False)]['probabilities'])) for i in ids)
   summary['comparisons'][f'{game}_{method}_cached_over_uncached']={'median_paired_ratio':med(ratios),'bootstrap95':ci(ratios),'choice_agreement':agreement,'n':len(ids),'max_probability_difference':probdiff}
  ratios=[lookup[(i,'qwen',True)]['ms']/lookup[(i,'clm',True)]['ms'] for i in ids]
  summary['comparisons'][f'{game}_cached_qwen_over_clm']={'median_paired_ratio':med(ratios),'bootstrap95':ci(ratios)}
 for method in ['clm','qwen']:
  for subset in ['all','critical','forced']:
   cs=[c for c in cases if c['game']=='trex' and c['robust'] and (subset=='all' or (sum(c['safe'].values())<3 if subset=='critical' else sum(c['safe'].values())==1))]
   n=len(cs);safe=sum(c['safe'][lookup[(c['id'],method,True)]['choice']] for c in cs)
   summary['trex_safety'][method+'_'+subset]={'safe':safe,'n':n,'percent':100*safe/n if n else None}
 summary['audits']={'full_vocab_choice_agreement':sum(r['selected']['choice']==r['full_vocab']['choice'] for r in audit),'n':len(audit),'label_ids':{g:metadata['cached'][g]['qwen']['labels'] for g in ['trex','mario']}}
 raw_matched=read('raw_matched.jsonl');raw_episodes=read('raw_episodes.jsonl')
 assert len(raw_matched)==len(cases) and len(raw_episodes)==8
 summary['raw_ablation']={'exploratory':True,'episodes':[{k:v for k,v in r.items() if k!='records'} for r in raw_episodes],
  'timings':{g:{'n':len([r for r in raw_matched if r['game']==g]),'median_ms':med([r['ms'] for r in raw_matched if r['game']==g])} for g in ['trex','mario']},
  'choices':{g:dict(collections.Counter(r['choice'] for r in raw_matched if r['game']==g)) for g in ['trex','mario']}}
 summary['aggregate_play']={m:{'trex_mean_survival_s':float(np.mean([r['frames']/60 for r in episodes if r['game']=='trex' and r['method']==m])),
  'trex_completed':sum(r['survived'] for r in episodes if r['game']=='trex' and r['method']==m),
  'mario_cleared':sum(r['flag_get'] for r in episodes if r['game']=='mario' and r['method']==m),
  'mario_forward_progress_pixels':[r['max_x']-40 for r in episodes if r['game']=='mario' and r['method']==m]} for m in ['clm','qwen']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,game in zip(axes,['trex','mario']):
  vals=[summary['matched'][f'{game}_{m}_{c}']['median_ms'] for m,c in [('clm',False),('qwen',False),('clm',True),('qwen',True)]]
  ax.bar(['CLM\nuncached','Qwen\nuncached','CLM\ncached','Qwen\ncached'],vals,color=['#99b8cb','#bcaadd','#24759c','#7751b5']);ax.set_title(game.upper() if game=='trex' else 'Super Mario Bros.');ax.set_ylabel('Median decision latency (ms)')
  for i,v in enumerate(vals):ax.text(i,v+6,f'{v:.0f}',ha='center')
  ax.set_ylim(0,max(vals)*1.17)
 fig.tight_layout();fig.savefig(OUT/'latency.png',dpi=160);plt.close(fig)
 erows=[]
 for r in episodes:
  if r['game']=='trex':erows.append(['T-Rex',r['seed'],r['method'],f"{r['frames']/60:.2f} s",r['score'],'survived' if r['survived'] else 'crashed',len(r['records'])])
  else:erows.append(['Mario',r['level'],r['method'],f"{r['frames']/60:.2f} s",f"x={r['max_x']}",'cleared' if r['flag_get'] else 'died' if r['died'] else 'stalled' if r['stalled'] else 'time cap',len(r['records'])])
 safetyrows=[[m,subset, f"{summary['trex_safety'][m+'_'+subset]['safe']}/{summary['trex_safety'][m+'_'+subset]['n']}"] for m in ['clm','qwen'] for subset in ['all','critical','forced']]
 promptsections=''
 for game in ['trex','mario']:
  p=json.loads((OUT/f'{game}_prompts.json').read_text())
  for m in ['qwen','clm']:
   txt=json.dumps(p[m]['text'],indent=2,ensure_ascii=False) if isinstance(p[m]['text'],list) else p[m]['text']
   promptsections+=f'<details><summary>{game}: literal {m} input</summary><pre>{html.escape(txt)}</pre></details>'
  promptsections+=f'<details><summary>{game}: fixed CLM action texts</summary><pre>{html.escape(json.dumps(metadata["specs"][game]["actions"],indent=2))}</pre></details>'
 replay=''.join(f'<figure><figcaption>Level {l} · {m}</figcaption><img width="512" src="mario_{l}_{m}.gif"></figure>' for l in ['1-1','1-2','2-1'] for m in ['clm','qwen'])
 setuprows=[[g,m,f"{d['setup_prefix_ms']:.0f}",f"{d.get('setup_actions_ms',0):.0f}"] for g,ms in metadata['cached'].items() for m,d in ms.items()]
 doc='''<!doctype html><html><head><meta charset="utf-8"><title>Cached game decisions: CLM vs constrained Qwen</title><style>body{max-width:1120px;margin:40px auto;padding:0 24px;color:#24334b;background:#f8fafc;font:16px/1.55 system-ui}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;margin:20px 0;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e0e9}th{background:#e8edf5}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#e8edf5;font-size:13px}img{max-width:100%}.replays{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figure img{image-rendering:pixelated}details{margin:12px 0}.note{padding:18px;background:#e5edf8;border-left:4px solid #477ba1}</style></head><body>'''
 doc+='<p class="note">Follow-up: a better cached Qwen prompt now survives a held-out 30-second T-Rex run with about 3% additional decision latency. <a href="../trex_prompt_tuning/report.html">Read the prompt-tuning experiment</a>. The results below retain the original prompts.</p>'
 doc+='<h1>Cached game decisions: CLM vs constrained Qwen</h1><p>Qwen3-8B BF16 · Apple M4 Max · 3 October 2026 · actual NES Mario and upstream T-Rex physics</p>'
 doc+='<p class="note">Both models reuse their fixed prefixes. Qwen caches rules and action definitions; CLM caches rules and action embeddings. Every changing state is processed afresh. This is one-token greedy constrained selection, with no generation loop, thinking tokens, planner hints, safety shield, or whole-request memoization.</p>'
 doc+='<p><strong>Caching removes almost all of Qwen’s measured latency disadvantage.</strong> Cached Qwen/CLM paired median latency ratios are 1.011 on T-Rex and 1.010 on Mario. Game quality remains poor: neither model finished any test episode successfully. Qwen made forward progress in Mario, while CLM stalled; CLM survived longer in T-Rex.</p>'
 doc+='<h2>Matched-state latency</h2><img src="latency.png" alt="Cached and uncached decision latency">'+table(['Game','Method','Prefix','States','Median ms','95th percentile ms'],timingrows)
 doc+='<p>136 frozen states: 80 T-Rex states from five planner trajectories and 56 Mario states from three heuristic trajectories. Each is evaluated under all four conditions, with deterministic randomized order. Latency includes tokenization, cache branching, model evaluation, head scoring, and synchronized result extraction. It excludes environment observation extraction and one-time setup. Uncached CLM still retains its fixed action embeddings: only its state prefix is recomputed.</p>'
 doc+='<h2>Token work per decision</h2>'+table(['Game','Method','Median total','Cached prefix','Median newly processed'],tokenrows)
 doc+='<p>Cached tokens remain in attention; caching avoids recomputing their transformer states, but does not make attention to them free. The same BF16 backbone, MLX implementation, and process-local 24 GiB wired-memory limit are used for both methods.</p>'
 doc+='<h2>Closed-loop play</h2>'+table(['Game','Seed / level','Method','Game time','Score / furthest x','Outcome','Decisions'],erows)
 doc+='<p>Five T-Rex seeds (0–4), up to 30 simulated seconds; three Mario levels (1-1, 1-2, 2-1), up to 30 seconds, stopping after five seconds without new maximum x. T-Rex acts every 6 frames (100 ms); Mario every 12 (200 ms). The environment is paused during inference: these are lockstep action-quality tests, not claims of successful real-time deployment. Mario is deterministic, so repeating the identical start would not create independent trials.</p>'
 doc+='<h2>T-Rex matched-state safety check</h2>'+table(['Method','Subset','Planner-safe choices'],safetyrows)
 doc+='<p>The planner evaluates whether an action permits future survival under its physics/search horizon. “Critical” excludes states where all three actions are safe; “forced” has exactly one safe action. The fixed sample contains only one forced state, so the overall safety percentage is a weak measure of game skill. Closed-loop survival is the stronger test. Planner results were used only for evaluation and generating reference trajectories, never supplied to either model.</p>'
 doc+='<h2>Exploratory raw-prompt check</h2><p>After seeing the main results, I reran Qwen with the same fixed rules/actions and changing state, removed the chat template, and appended <code>Answer:</code>. It still used one bare constrained letter and a cached prefix. This was an exploratory check, not a newly selected primary result. It did not improve play: all five T-Rex runs again crashed at 5.55 seconds; all three Mario runs stalled at the starting x=40. Later-run latency measurements are not interleaved with the primary comparison.</p>'
 doc+=table(['Game','Raw Qwen median ms','Matched states'],[[g,f"{summary['raw_ablation']['timings'][g]['median_ms']:.1f}",summary['raw_ablation']['timings'][g]['n']] for g in ['trex','mario']])
 doc+='<p><a href="raw_metadata.json">Exact raw prompts</a> · <a href="raw_matched.jsonl">Raw matched predictions</a> · <a href="raw_episodes.jsonl">Raw episode traces</a></p>'
 doc+='<h2>Reference controllers</h2>'+table(['Game','Seed / level','Reference','Game time','Score / x','Outcome'],[[r['game'],r.get('seed',r.get('level')),r['method'],f"{r['frames']/60:.2f} s",r.get('score',r.get('max_x')),'survived' if r.get('survived') else 'cleared' if r.get('flag_get') else 'died'] for r in refs])
 doc+='<p>The Mario reference is a simple jump heuristic, not an optimal policy or accuracy oracle. Mario observations are structured prose from visible RAM terrain/enemy geometry, with current motion and button state. T-Rex observations use visible engine geometry. Neither model receives images; this tests decisions given symbolic state, not visual perception.</p>'
 doc+='<h2>Cache setup (one observed measurement)</h2>'+table(['Game','Method','Prefix setup ms','Action embedding setup ms'],setuprows)
 doc+='<p>Setup excludes model loading. These single cold measurements are descriptive rather than stable latency estimates. Qwen caches its instruction/action prefix once; CLM separately caches its rules prefix and each action vector.</p>'
 doc+='<h2>Implementation checks</h2><pre>'+html.escape(json.dumps({'cache_equivalence':summary['comparisons'],'output_head':summary['audits'],'CLM_head_max_abs_error':metadata['head_port_max_abs_error']},indent=2))+'</pre>'
 doc+='<p>Qwen uses the original LM-head rows for the allowed letters (A–C or A–F), equivalent to masking all other tokens before greedy decoding. Ten checks against the full-vocabulary LM head must choose the same action. Cached and uncached BF16 computations can differ slightly because matrix shapes and reduction order change; the agreement and largest probability differences above disclose this.</p>'
 doc+='<h2>Exact example prompts</h2>'+promptsections
 doc+='<h2>Mario replays</h2><p>Actual emulator frames at 10 frames per second, shown at simulated game speed. Inference waiting time is omitted.</p><div class="replays">'+replay+'</div>'
 doc+='<h2>Reproduce and inspect</h2><p>Run <code>.venv/bin/python -u experiments/games/bench.py</code>, then <code>.venv/bin/python -u experiments/games/raw_check.py</code>, then <code>.venv/bin/python experiments/games/analyze.py</code>. The NES worker uses the separate <code>.venv-games</code> environment and the pinned package list in <code>experiments/games/requirements-games.txt</code>. Outputs are resumable; use a fresh output directory for an independent repeat.</p><p>Raw data: <a href="summary.json">summary</a>, <a href="metadata.json">metadata</a>, <a href="cases.json">frozen states</a>, <a href="matched.jsonl">matched predictions</a>, <a href="episodes.jsonl">complete episode decisions</a>, <a href="audit.jsonl">head audits</a>. Sources: <a href="https://github.com/Kautenja/gym-super-mario-bros">NES environment</a>, <a href="https://github.com/Contrastive-LM/CLM">CLM repository and T-Rex engine</a>.</p></body></html>'
 (OUT/'report.html').write_text(doc)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
