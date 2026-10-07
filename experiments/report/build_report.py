"""A short reader-facing report with the detailed evidence kept behind links."""
import hashlib,html,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/reusable_prompt_report'
def read(path):return json.loads((ROOT/path).read_text())
def main():
 fair=read('results/reusable_prompt_report/trex_fairness_audit.json')
 bench=read('results/decision_benchmarks/summary.json')
 trex=read('results/trex_prompt_tuning/summary.json')
 replay=read('results/reusable_prompt_report/replay_audit.json')
 clm=fair['median_ms']['clm:timing_v4'];qwen=fair['median_ms']['timing_v4']
 tool=next(r for r in bench['matched_latency'] if r['family']=='bfcl_forced_tool' and r['a']=='clm_base')
 assert all(r['all_recorded_states_exact_match'] for r in replay['runs'])
 assert fair['same_instructions_and_examples']
 source=ROOT/'results/trex_prompt_tuning'
 for src,dst in [('episode_holdout_17_clm:timing_v4.json','trex_clm_trace.json'),('episode_holdout_17_timing_v4.json','trex_qwen_trace.json')]:shutil.copyfile(source/src,OUT/dst)
 evidence={'trex':{'gameplay':[r for r in trex['holdout'] if r['prompt'] in ['clm:timing_v4','timing_v4']],'fair_timing':fair,'replay_audit':replay},'bfcl':{'tool_selection':[x for x in bench['diagnostic_overall'] if x['group'][0]=='forced_tool'],'timing':tool,'genuine_reuse':bench['repeated_requests']},'wiki':{'episodes':[{k:v for k,v in r.items() if k!='paths'} for r in bench['navigation']],'first_step':[r for r in bench['primary'] if r['group'][0]=='wiki']},'sources':{'clm':'https://github.com/Contrastive-LM/CLM','bfcl':'https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard','wikispeedia':'https://snap.stanford.edu/data/wikispeedia.html'},'scope':'These are local public-data adaptations, not reproductions of the authors’ Jev comparison. Timing excludes one-time cache setup and model loading.'}
 (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
 # Preserve the exact paired timing rows and tested tool cases in the portable bundle.
 shutil.copyfile(ROOT/'results/decision_benchmarks/diagnostics.jsonl',OUT/'bfcl_diagnostics.jsonl')
 shutil.copyfile(ROOT/'results/decision_benchmarks/repeated_cases.json',OUT/'bfcl_reused_tools.json')
 doc='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Constrained not Contrastive Language Models</title>
<style>
:root{--paper:#fafaf7;--ink:#233747;--muted:#5e707c;--line:#d6dfdf;--teal:#087f8c;--wash:#e9f3f2}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:17px;line-height:1.65}
main{max-width:1000px;margin:auto;padding:58px 28px 64px}.copy{max-width:760px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--teal)}h1{font-family:Georgia,serif;font-weight:400;letter-spacing:-.035em;font-size:clamp(38px,6vw,62px);line-height:1.08;margin:16px 0 22px}h2{font-family:Georgia,serif;font-size:30px;line-height:1.25;font-weight:400;letter-spacing:-.015em;margin:42px 0 16px}p{margin:0 0 17px}.dek{font-size:23px;line-height:1.45;color:var(--muted);max-width:830px}.meta{font-size:13px;color:var(--muted);padding-bottom:24px;border-bottom:1px solid var(--line);margin-bottom:28px}strong{font-weight:650}a{color:var(--teal);text-underline-offset:3px}a:hover{color:#034c58}.takeaway{margin:28px 0;padding:22px 26px;background:var(--wash);border-left:4px solid var(--teal);max-width:850px;font-size:20px;line-height:1.5}.takeaway p:last-child{margin:0}
.prompt{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin:24px 0;max-width:850px}.prompt>div{border:1px solid var(--line);border-radius:8px;padding:20px}.prompt>div:first-child{background:var(--wash)}.tag{font-size:12px;letter-spacing:.06em;text-transform:uppercase;font-weight:700;color:var(--teal);display:block;margin-bottom:8px}.prompt p{margin:0;font-size:16px}.example{border-left:2px solid var(--line);padding:4px 0 4px 20px;font-size:16px;margin:21px 0}.example p{margin:3px 0}figure{margin:26px 0}figure img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:8px}figcaption{font-size:13px;line-height:1.55;color:var(--muted);margin-top:10px;max-width:900px}.table-wrap{overflow-x:auto;margin:22px 0;max-width:850px}table{width:100%;border-collapse:collapse;font-size:16px;font-variant-numeric:tabular-nums}th,td{padding:13px 16px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:13px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);background:#f0f3f1}td:first-child,th:first-child{padding-left:12px}td:last-child{font-weight:650;color:var(--teal)}.note{font-size:14px;color:var(--muted)}details{margin-top:34px;border-top:1px solid var(--line);padding-top:18px;max-width:850px}summary{cursor:pointer;font-weight:650}details p{margin:16px 0;font-size:15px}footer{border-top:1px solid var(--line);margin-top:36px;padding-top:18px;font-size:13px;color:var(--muted)}.links{display:flex;flex-wrap:wrap;gap:10px 22px;margin:16px 0}code{font-size:.88em}button{border:1px solid var(--line);background:var(--paper);color:var(--ink);padding:7px 12px;border-radius:5px;cursor:pointer;font:inherit;font-size:13px}.animation-controls{display:flex;gap:8px;margin-top:10px}
@media(max-width:640px){main{padding:32px 18px 40px}.prompt{grid-template-columns:1fr}.dek{font-size:20px}th,td{padding:10px 8px;font-size:14px}h2{font-size:27px}.takeaway{padding:18px;font-size:18px}}
@media print{body{background:white}main{padding:0}details{display:block}button{display:none}a{color:inherit}.animation-controls{display:none}figure{break-inside:avoid}}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f0;border:1px solid var(--line);border-radius:8px;padding:20px;font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;margin:20px 0}.prompt-expander{margin:24px 0 28px}.prompt-expander pre{max-height:560px;overflow:auto}
h3{font-family:Georgia,serif;font-size:26px;line-height:1.3;font-weight:400;margin:30px 0 16px}.explanation{margin:24px 0 28px}.explanation p{font-size:16px}.results-heading{border-top:1px solid var(--line);padding-top:30px;margin-top:40px}
</style></head><body><main>
<h1>Constrained not Contrastive Language Models</h1>
<div class="copy">
<p>A new wave of “decision models”—classifiers, in plain terms—has landed on the scene. <a href="https://typesafe.ai/">TypeSafe AI’s Jev</a> offers a commercial implementation, while open-source releases such as <a href="https://github.com/Contrastive-LM/CLM">Contrastive Language Model (CLM)</a> position themselves as competitors. Their promise is to make decisions faster than conventional language models.</p>
<p>This prompted our investigation: does a new decision model offer enough to justify the added machinery? In our function-selection and T-Rex experiments, CLM performed worse than a language model restricted to the same choices. When the instructions and choices could be reused, the latency gap was small. For these cases, we argue that the extra trained component did not earn its place.</p>
<p>We compare CLM with constrained decoding using the same Qwen3-8B backbone. We use the released CLM scoring heads and T-Rex simulator, and follow the repository’s implementation where possible. For the other tasks, we use public benchmark data because the authors’ exact test setups were not released. We are testing CLM against constrained Qwen, not reproducing the published comparison with Jev.</p>
<details class="explanation"><summary>What is a decision model?</summary>
<p>A decision model takes an input and selects from a fixed set of choices. The choices might be true or false, named categories, or ordered labels such as low, medium, and high. In a game, they are the available actions.</p>
<p>CLM uses an existing language model to encode the question and each possible answer—the “actions” in the paper—separately as vectors. It then passes them through two trained projection networks, one for the question and one for the answers. These are small multilayer perceptrons, or MLPs. A dot product scores each question–answer pair in their shared space, and the highest score selects the answer.</p>
</details>
<h2>Constrain the language model instead</h2>
<p>Our approach needs no additional trained component. We put the choices in the prompt and restrict the model’s output to one of them. To avoid generating a multi-word answer over several forward passes, we assign each choice a letter that is a single token. The model then chooses one permitted letter, such as A, B, C, or D.</p>
<p>For T-Rex, the choice mapping is simply:</p>
<pre aria-label="Abbreviated T-Rex prompt layout">[System prompt: game instructions and worked examples]

A = run
B = jump
C = duck

…</pre>
<p>We choose the output token among the allowed choices with the highest probability mass. Thinking is not enabled.</p>
</div>
<div class="copy">
<details class="explanation"><summary>Caching the actions</summary>
<p>We put the instructions and choices at the start of Qwen’s prompt, before the changing request. This lets it save their processed form and reuse it on later requests. CLM also reuses work: it saves representations of the actions and any fixed instructions supplied with the state.</p>
<p><strong>Every latency quoted here is measured after cache preparation.</strong> The first request also pays that preparation cost. A changed set of choices may require new work. These results describe repeated use, not a free speed-up for unrelated questions.</p>
</details>
<h2 class="results-heading">Results</h2>
<h3>T-Rex</h3>
<p>We use the T-Rex simulator from the CLM repository to compare the two approaches in a game where quick decisions matter. Both models receive the same game state—the dinosaur’s position and motion, and the positions of visible obstacles—as text. Qwen chooses among run, jump, and duck using the action choices in its prompt; CLM scores separately encoded actions. We give both models instructions and action examples to help them make better decisions. These stay fixed; only the game state changes.</p>
<p>This makes T-Rex a natural place for fixed examples. We added instructions about jump timing and examples such as:</p>
<div class="example"><p>On the ground, cactus 55 pixels ahead → jump.</p><p>On the ground, cactus 127 pixels ahead → keep running.</p><p>Already airborne → let the current jump continue.</p></div>
<p><strong>CLM received the same instructions and all the same examples.</strong> They appeared before its current-state input; Qwen received them in its system prompt. Neither model received a recommended action for the live state, and neither had its chosen action corrected.</p>
<details class="explanation"><summary>How the released T-Rex benchmark supplies the answer</summary>
<p>The CLM repository’s T-Rex benchmark runs a physics planner before asking the model to choose. The planner predicts which moves avoid a collision and picks its preferred action. The default prompt then labels each action:</p>
<pre>Planner’s preferred action → “Safe. … Best.”
Another safe action       → “Safe. …”
Unsafe action             → “Unsafe. … Collision.”</pre>
<p>The alternative “guided” prompt goes further: it explicitly adds “Recommended action: …” to the state. Both CLM and Jev receive these hints. You can see this in the <a href="https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/examples/t_rex/trex/backends.py#L35-L62">released prompt-building code</a>.</p>
<p><strong>This leaks the planner’s preferred answer into the input.</strong> A model can select the option marked “Best” without working out which move avoids the obstacle. That does not establish independent game-playing ability. The default harness also replaces unsafe choices and can intervene before a collision, as the <a href="https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/examples/t_rex/README.md">repository’s own description explains</a>. Its survival results therefore measure the model and planner together.</p>
<p>Our comparison removes these live hints and action overrides. Both models receive the observable game state and the same fixed examples, then make their own choice. The game pauses while they answer; decision latency is measured separately.</p>
</details>
<details class="prompt-expander"><summary>Read the full T-Rex prompt</summary>
<p>This is the exact fixed prompt used for Qwen and supplied in full to CLM, including every worked example. It stays unchanged throughout the game.</p>
<pre id="full-trex-prompt">__TREX_PROMPT__</pre>
<p>After this fixed text, each model receives the current game state and the question: “Which action should the dinosaur take now to survive?”</p>
<p><a href="trex_prompt.txt">Download the prompt as plain text</a></p>
</details>
</div>
<figure>
<img id="replay" src="trex_comparison.gif" width="1224" height="272" alt="Synchronized replay: CLM with examples crashes at 5.55 seconds; Qwen with the same examples continues to the 30-second limit.">
<div class="animation-controls"><button type="button" id="restart">Replay from start</button><button type="button" id="pause">Show still image</button></div>
<figcaption>Same test course, same examples. This animation reconstructs the recorded actions in the original simulator; it is not a screen recording. Playback follows game time. During the experiment, the game paused while each model made its decision.</figcaption>
</figure>
<div class="table-wrap"><table><thead><tr><th>T-Rex · one test course</th><th>CLM + examples</th><th>Qwen + same examples</th></tr></thead><tbody>
<tr><td>Time survived · 30-second limit</td><td>5.55 seconds</td><td>30 seconds</td></tr>
<tr><td>Game score</td><td>46</td><td>305</td></tr>
<tr><td>Median decision time · cache ready</td><td>__CLM_MS__ ms</td><td>__QWEN_MS__ ms</td></tr>
</tbody></table></div>
<div class="copy">
<p>The instructions and examples took Qwen from 5.55 seconds with the original prompt to the full 30-second test. Giving those examples to CLM did not improve its result; CLM’s original, shorter instructions lasted 7.27 seconds.</p>
<p class="note">The prompt was developed on a different course and fixed before this test. It was developed for Qwen, then supplied unchanged to CLM—not separately optimized for each model. This is a fair check of access to the examples, not proof that CLM cannot benefit from a better prompt. One test course also does not establish reliable performance across the game.</p>
<h3>Choosing a software function</h3>
<p>The Berkeley Function Calling Leaderboard (BFCL) provides user requests and descriptions of available functions. For example, a request to solve a quadratic equation comes with functions for solving equations, converting colours, and reversing text. The model must choose the equation solver.</p>
<p>We tested that choice—not whether the model supplied the right arguments or executed the function. On 60 requests where a suitable function existed:</p>
</div>
<div class="table-wrap"><table><thead><tr><th>BFCL · function choice</th><th>CLM</th><th>Constrained Qwen</th></tr></thead><tbody>
<tr><td>Correct choices</td><td>40 / 60 · 66.7%</td><td>55 / 60 · 91.7%</td></tr>
<tr><td>Median decision time · cache ready</td><td>__TOOL_CLM__ ms</td><td>__TOOL_QWEN__ ms</td></tr>
</tbody></table></div>
<div class="copy">
<p>Qwen got every CLM-correct case right and added 15 more. We also checked actual reuse: six sets of functions received 24 different requests, with the descriptions unchanged within each set. Qwen answered 21 correctly; CLM answered 12. That test also allowed “none” when no function could help.</p>
<h3>Wikipedia: both struggled</h3>
<p>We asked the models to reach a target Wikipedia article by following links in the public Wikispeedia dataset. Across five test routes, with at most eight clicks each, Qwen finished one and CLM finished none. Both results were poor. They do not establish a useful navigation system or a reliable advantage for Qwen.</p>
<h2>What this supports</h2>
<p>For tasks that reuse instructions and choices, test constrained decoding before training a separate decision network. It can use written rules and examples, and their cost can be spread across many decisions. Here, it improved function selection and one T-Rex run with little extra time per cached decision. That advantage did not turn either model into a capable Wikipedia navigator.</p>
</div>
<details><summary>How we checked the comparison</summary>
<p>Both methods used the same full-precision-format BF16 Qwen3-8B weights on an Apple M4 Max with 64 GB of memory. CLM used its released scoring heads. Qwen chose one permitted token using its original output weights; it generated no hidden reasoning or explanation. Model loading and cache preparation are excluded from the reported decision times.</p>
<p>T-Rex timing used the same ten saved states for both models, with five interleaved measurements per state. CLM’s timing includes the full examples, not its shorter original prompt. The paired median overhead for Qwen was __TREX_OVERHEAD__%, with a 95% interval of __TREX_LOWER__–__TREX_UPPER__%. “Little extra time” describes the measured size of the difference; it does not mean we established statistical equivalence.</p>
<p>The GIF replay was checked against every saved state, final score, and collision frame. The 30-second course tests cacti; it does not establish performance on later birds or indefinite survival.</p>
<p>The 60 BFCL requests were excluded from prompt development. Their choice lists contained only the supplied functions. This comparison was added after an earlier test exposed a strong CLM preference for an added “none” option; we removed that option for both models on all 60 eligible requests. CLM was also tested with prose function descriptions, which scored 35/60; we report the stronger 40/60 result above. Timing used ten fixed requests and five repetitions per method.</p>
<p>These are public-data adaptations of tasks cited by the CLM authors, not reproductions of their published Jev comparison. The evidence supports these specific cases, not a claim that prompt engineering always beats training.</p>
<div class="links"><a href="trex_prompt.txt">Full shared T-Rex prompt</a><a href="evidence.json">Results and measurement details</a><a href="trex_fairness_audit.json">Equal-example check</a><a href="replay_audit.json">Replay verification</a></div>
<div class="links"><a href="trex_clm.gif">CLM GIF</a><a href="trex_qwen.gif">Qwen GIF</a><a href="trex_clm_trace.json">CLM action record</a><a href="trex_qwen_trace.json">Qwen action record</a></div>
</details>
<footer>Sources: <a href="https://typesafe.ai/">TypeSafe AI and Jev</a> · <a href="https://github.com/Contrastive-LM/CLM">CLM release and T-Rex simulator</a> · <a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard">Berkeley Function Calling Leaderboard</a> · <a href="https://snap.stanford.edu/data/wikispeedia.html">Wikispeedia</a>.</footer>
</main><script>
const replay=document.getElementById('replay');
document.getElementById('restart').onclick=()=>{replay.src='trex_comparison.gif?replay='+Date.now()};
document.getElementById('pause').onclick=()=>{replay.src='trex_preview.png'};
if(window.matchMedia('(prefers-reduced-motion: reduce)').matches)replay.src='trex_preview.png';
</script></body></html>'''
 replacements={'__CLM_MS__':f'{clm:.1f}','__QWEN_MS__':f'{qwen:.1f}','__TOOL_CLM__':f"{tool['a_median_ms']:.1f}",'__TOOL_QWEN__':f"{tool['b_median_ms']:.1f}",'__TREX_OVERHEAD__':f"{100*(fair['paired_median_qwen_over_clm_same_examples']-1):.1f}",'__TREX_LOWER__':f"{100*(fair['paired_state_bootstrap95'][0]-1):.1f}",'__TREX_UPPER__':f"{100*(fair['paired_state_bootstrap95'][1]-1):.1f}"}
 for key,value in replacements.items():doc=doc.replace(key,value)
 doc=doc.replace('__TREX_PROMPT__',html.escape((OUT/'trex_prompt.txt').read_text().rstrip('\n')))
 # BF16 is reduced precision; name it directly without an inaccurate qualifier.
 doc=doc.replace('same full-precision-format BF16','same BF16')
 (OUT/'report.html').write_text(doc)
 files=['report.html','trex_comparison.gif','trex_clm.gif','trex_qwen.gif','trex_preview.png','trex_prompt.txt','evidence.json','trex_fairness_audit.json','replay_audit.json','trex_clm_trace.json','trex_qwen_trace.json','trex_fair_timing.json','bfcl_diagnostics.jsonl','bfcl_reused_tools.json']
 manifest={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));files.append('manifest.json')
 with zipfile.ZipFile(OUT/'report_with_gifs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in files:z.write(OUT/name,arcname='reusable-prompt-report/'+name)
 print('Report and portable GIF bundle created.')
def restore_trex():
 """Restore the final, individually prompted warm-cache experiment without changing BFCL."""
 import sys, re, math, statistics
 from PIL import Image, ImageDraw, ImageFont
 sys.path.insert(0,str(ROOT/'upstream/examples/t_rex'))
 sys.path.insert(0,str(ROOT/'experiments/games'))
 from trex.engine import Game, obstacle_step
 from specs import clm_observation
 dist=ROOT/'sites/constrained-not-contrastive/dist'
 audits=[json.loads(s) for s in (ROOT/'results/cached_games/audit.jsonl').read_text().splitlines()]
 summary=next(r for r in reversed(audits) if r.get('type')=='trex_warm_cache_summary')
 protocol=next(r for r in reversed(audits) if r.get('type')=='trex_warm_cache_protocol' and r['run_id']==summary['run_id'])
 episodes=[json.loads(s) for s in (ROOT/'results/cached_games/raw_episodes.jsonl').read_text().splitlines()]
 episodes=[r for r in episodes if r.get('run_id')==summary['run_id']]
 assert len(episodes)==6
 def get(method,seed):return next(r for r in episodes if r['method']==method+'_warm_cache_final' and r['seed']==seed)
 def observe(g):
  t=g.trex
  visible=sorted((o for o in g.obstacles if not o.remove and o.x<600 and o.x+o.width>0 and o.x+o.width>=t.x),key=lambda o:o.x)[:1]
  if t.jumping:state=f'State: The dinosaur is in the air, with its feet {93-t.y:.0f} pixels above the ground, '+('rising' if t.velocity<0 else 'falling')+f' at {abs(t.velocity):.1f} pixels per frame. '
  else:state='State: The dinosaur is on the ground. '+('It is ducking. ' if t.ducking else '')
  for o in visible:
   kind='cactus' if o.kind.name.startswith('cactus') else 'bird'
   state+=f'A {kind} is {o.x-(t.x+44)} pixels beyond its front, {o.width} pixels wide and {o.kind.height} pixels tall, moving toward it at {obstacle_step(g.speed,o.speed_offset)} pixels per frame. '
   if kind=='bird':state+=f'The bird bottom is {140-o.y-o.kind.height} pixels above the ground. '
  if not visible:state+='No obstacles are visible. '
  return state+'\nAnswer:'
 def control(g,action):
  if action in ('run','jump'):g.release_duck()
  if action=='jump' and not g.trex.jumping:g.press_jump()
  if action=='duck':g.press_duck()
 replay_audit=[]
 for ep in episodes:
  method=ep['method'].split('_')[0];g=Game(ep['seed']);g.press_jump();frame=0;seen={};passed=set()
  for r in ep['records']:
   assert r['frame']==frame and r['observation']==observe(g)
   assert r['choice']==max(r['probabilities'],key=r['probabilities'].get)
   assert r['cached_tokens']==protocol['setup'][method]['prefix_tokens']
   assert r['processed_tokens']==r['total_tokens']-r['cached_tokens']>0
   if method=='clm':
    expected=(protocol['system']['clm']+'\n\nCurrent observation:\n'+clm_observation(r['observation'])).strip()+'\n\n'+protocol['questions']['action']['instructions']
    assert r['exact_input']==expected
   control(g,r['choice'])
   for _ in range(6):
    for o in g.obstacles:seen[o.id]=o
    g.step();frame+=1
    if g.crashed:break
    for oid,o in seen.items():
     if o.x+o.width<g.trex.x:passed.add(oid)
  assert frame==ep['frames'] and g.crashed==ep['crashed'] and g.score==ep['score'] and len(passed)==ep['obstacles_passed']
  replay_audit.append({'method':method,'seed':ep['seed'],'decisions_verified':len(ep['records']),'frames':frame,'obstacles_passed':len(passed),'all_observations_match':True,'all_choices_are_argmax':True,'all_fixed_prefixes_cached':True})
 for method in ('clm','qwen'):
  xs=sorted(d['ms'] for ep in episodes if ep['method']==method+'_warm_cache_final' for d in ep['records'])
  assert len(xs)==summary[method]['decisions']
  assert statistics.median(xs)==summary[method]['median_ms']
  assert statistics.mean(xs)==summary[method]['mean_ms']
  assert xs[math.ceil(.95*len(xs))-1]==summary[method]['p95_ms']
  (dist/f'trex_{method}_trace.json').write_text(json.dumps({'run_id':summary['run_id'],'episodes':[ep for ep in episodes if ep['method']==method+'_warm_cache_final']},indent=2)+'\n')
 (dist/'trex_fair_timing.json').write_text(json.dumps({'protocol':protocol,'summary':summary},indent=2)+'\n')
 (dist/'trex_fairness_audit.json').write_text(json.dumps({'run_id':summary['run_id'],'different_prompts':True,'qwen_example_blocks':4,'clm_example_blocks':10,'same_backbone':True,'both_fixed_prefixes_cached':True,'clm_action_inputs':protocol['exact_action_inputs'],'replay_checks':replay_audit,'cache_numerical_checks':protocol['validation']},indent=2)+'\n')
 (dist/'replay_audit.json').write_text(json.dumps({'artwork':'Schematic rendering reconstructed from original simulator positions, not a screen recording.','displayed_course':42,'fps':20,'runs':replay_audit},indent=2)+'\n')

 # Reconstruct the first course, selected in advance, from the actions actually measured.
 ink='#233747';paper='#fafaf7';muted='#5e707c';teal='#087f8c'
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
 bold=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',18)
 small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
 def panel(draw,g,method,frame,action,xoff):
  draw.text((xoff+18,34),'CLM' if method=='clm' else 'Constrained Qwen',font=bold,fill=ink)
  draw.text((xoff+18,59),('CRASHED' if g.crashed else action.upper()),font=small,fill=teal)
  draw.text((xoff+418,36),f'{frame/60:05.2f} s  |  score {g.score:03d}',font=small,fill=muted)
  yy=80
  draw.line((xoff+10,yy+140,xoff+600,yy+140),fill='#d6dfdf')
  for cloud in g.clouds:
   x=xoff+cloud.x;y=yy+cloud.y
   draw.line([(x,y+8),(x+7,y+8),(x+12,y+2),(x+24,y+2),(x+28,y-3),(x+35,y-3),(x+39,y+8),(x+46,y+8)],fill='#d6dfdf')
  t=g.trex;x=xoff+t.x;y=yy+t.y
  if t.ducking:
   draw.rectangle((x+10,y+22,x+58,y+38),fill=ink);draw.rectangle((x+19,y+38,x+25,y+47),fill=ink)
   draw.rectangle((x+49,y+25,x+52,y+28),fill=paper)
  else:
   shape=[(0,18),(5,24),(13,30),(18,28),(18,3),(21,0),(43,0),(43,17),(28,17),(28,23),(36,23),(36,27),(27,27),(27,35),(22,40),(22,47),(16,47),(16,37),(7,31),(0,25)]
   draw.polygon([(x+a,y+b) for a,b in shape],fill=ink)
   draw.rectangle((x+33,y+4,x+36,y+7),fill=paper)
  for o in g.obstacles:
   if o.remove or o.x>600 or o.x+o.width<0:continue
   ox=xoff+o.x;oy=yy+o.y;w=o.width;h=o.kind.height
   count=max(1,round(w/(25 if 'Large' in o.kind.name else 17)));unit=w/count
   for j in range(count):
    cx=ox+j*unit;stem=max(3,unit*.2)
    draw.rectangle((cx+unit*.45,oy,cx+unit*.45+stem,oy+h),fill=ink)
    draw.line([(cx+unit*.1,oy+h*.28),(cx+unit*.1,oy+h*.58),(cx+unit*.45,oy+h*.58)],fill=ink,width=3)
    draw.line([(cx+unit*.85,oy+h*.15),(cx+unit*.85,oy+h*.44),(cx+unit*.6,oy+h*.44)],fill=ink,width=3)
 games={m:Game(42) for m in ('clm','qwen')};traces={m:get(m,42) for m in games};current={m:'run' for m in games};images=[]
 for g in games.values():g.press_jump()
 for frame in range(1801):
  if frame%6==0 and frame<1800:
   for m,g in games.items():
    if not g.crashed and frame<traces[m]['frames']:
     current[m]=traces[m]['records'][frame//6]['choice'];control(g,current[m])
  if frame%3==0:
   im=Image.new('RGB',(1224,262),paper);draw=ImageDraw.Draw(im)
   draw.text((18,10),'SAME COURSE / INDIVIDUAL PROMPTS / BOTH CACHES PREPARED',font=small,fill=muted)
   draw.line((612,31,612,228),fill='#d6dfdf')
   for i,(m,g) in enumerate(games.items()):panel(draw,g,m,min(frame,traces[m]['frames']),current[m],i*612)
   draw.text((18,241),'Recorded actions reconstructed in the simulator. 1x game time; model waiting time omitted.',font=small,fill=muted)
   images.append(im.quantize(colors=32))
  if frame<1800:
   for g in games.values():
    if not g.crashed:g.step()
 durations=[50]*len(images);durations[-1]=1050
 images[0].save(dist/'trex_comparison.gif',save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=False,disposal=2)
 images[100].convert('RGB').save(dist/'trex_preview.png')
 for i,m in enumerate(games):
  cropped=[im.crop((i*612,28,(i+1)*612,230)) for im in images]
  cropped[0].save(dist/f'trex_{m}.gif',save_all=True,append_images=cropped[1:],duration=durations,loop=0,optimize=False,disposal=2)

 pre=lambda text:'<pre>'+html.escape(text)+'</pre>'
 example={m:next(r for r in get(m,42)['records'] if r['frame']==300) for m in games}
 timing_rows=''
 for stat,label in [('median_ms','Median decision'),('mean_ms','Mean decision'),('p95_ms','95th percentile')]:
  vals=[summary[m][stat] for m in games]
  cells=''.join(f'<td class="best-value">{v:.1f} ms</td>' if v==min(vals) else f'<td>{v:.1f} ms</td>' for v in vals)
  timing_rows+=f'<tr><td>{label}</td>{cells}</tr>\n'
 course_rows=''
 for i,seed in enumerate(protocol['seeds'],1):
  cells=''.join(f'<td>{get(m,seed)["game_seconds"]:g} s · {get(m,seed)["obstacles_passed"]} cleared'+(' · crashed' if get(m,seed)['crashed'] else '')+'</td>' for m in games)
  course_rows+=f'<tr><td>Course {i}</td>{cells}</tr>\n'
 successes={m:sum(not get(m,s)['crashed'] and get(m,s)['frames']==1800 for s in protocol['seeds']) for m in games}
 outcome='Both models completed all three 30-second courses without crashing.' if all(n==3 for n in successes.values()) else f'CLM completed {successes["clm"]} of three courses; constrained Qwen completed {successes["qwen"]}.'
 section=f'''<section id="trex">
<h3>T-Rex</h3>
<p>We use the T-Rex simulator from the CLM repository to compare the two approaches on repeated game decisions. Both receive the dinosaur’s position and motion and the nearest obstacle as text. Each chooses run, jump, or duck for the next six frames.</p>
<p>We use different prompts for each model because both needed extensive prompt tuning to play for 30 seconds without crashing. CLM needed many more worked examples and had much higher latency, even with cached input.</p>
<figure><img id="replay" src="trex_comparison.gif" width="1224" height="262" alt="Side-by-side replay of CLM and constrained Qwen on the first measured T-Rex course.">
<div class="animation-controls"><button type="button" id="restart">Replay from start</button><button type="button" id="pause">Show still image</button></div>
<figcaption>Course 1, reconstructed from the recorded actions. Playback follows game time and omits model waiting time. During the experiment, the game paused for each decision.</figcaption></figure>
<h3>Decision time with warm caches</h3>
<p>Across three tries with different seeds, median decision time was <strong>{summary['clm']['median_ms']:.1f} ms for CLM</strong> and <strong>{summary['qwen']['median_ms']:.1f} ms for constrained Qwen</strong>. These timings compare the two working prompts, including CLM’s longer prompt and distance diagram.</p>
<div class="table-wrap"><table aria-label="T-Rex warm-cache decision latency"><thead><tr><th>{summary['clm']['decisions']} decisions per model</th><th>CLM</th><th>Constrained Qwen</th></tr></thead><tbody>{timing_rows}</tbody></table></div>
<p class="note">Measured on an Apple M4 Max with the same BF16 Qwen3-8B backbone. CLM’s released scoring heads ran on the CPU. Timings include prompt construction, tokenization, cache copying, processing the new state, and selecting the action. One-time cache preparation and model loading are excluded. The game paused during inference, so this is not a test of uninterrupted real-time play.</p>
<details class="prompt-expander" id="qwen-trex-prompt"><summary>Exact constrained Qwen prompt</summary>
<p>The fixed system message is below. Qwen’s chat template used thinking disabled, and code restricted its output to the single-token letters A, B, and C.</p>
<p><strong>System message</strong></p>{pre(protocol['system']['qwen'])}
<p><strong>User message at frame 300, course 1</strong></p>{pre(example['qwen']['observation'])}
</details>
<details class="prompt-expander" id="clm-trex-prompt"><summary>Exact CLM inputs</summary>
<p>CLM used no chat template or system-message role. This is its complete state encoder input at the same point in course 1. The text before “Current observation” stayed fixed. The distance diagram represents the observed gap; it does not label an action safe or unsafe.</p>
{pre(example['clm']['exact_input'])}
<p><strong>Separately encoded action inputs</strong></p>{''.join(pre(x) for x in protocol['exact_action_inputs'])}
</details>
</section>
'''
 old=(dist/'index.html').read_text()
 start=old.index('<details class="explanation" id="trex-archived">');end=old.index('<h3>Wikipedia: both struggled</h3>',start)
 old=old[:start]+old[end:]
 marker='<h2 class="results-heading">Results</h2>'
 old=old.replace(marker,marker+'\n'+section,1)
 old=old.replace('We use the released CLM scoring heads and public BFCL data, following the repository’s implementation where possible.', 'We use the released CLM scoring heads, its T-Rex simulator, and public BFCL data, following the repository’s implementation where possible.')
 old=old.replace('Both caches start empty in the experiment below.', 'Both caches start empty in the BFCL experiment below. For T-Rex, we prepare both fixed prompts before play, then time decisions with warm caches.')
 old=old.replace("document.getElementById('pause').onclick=()=>{replay.src='trex_preview.png'};", "document.getElementById('pause').onclick=()=>{replay.src='trex_preview.png'};\nif(window.matchMedia('(prefers-reduced-motion: reduce)').matches)replay.src='trex_preview.png';")
 old=old.replace('<h2>What this supports</h2>', '<h2>What this supports</h2>\n<p>In T-Rex, both models completed our short courses with their own prompts. Constrained Qwen made faster warm-cache decisions in this implementation. The result shows that both approaches can play these courses; it does not establish performance on longer games or other hardware.</p>')
 for filename in ('index.html','report.html'):(dist/filename).write_text(old)
 (dist/'trex_prompt.txt').write_text('QWEN SYSTEM MESSAGE\n'+protocol['system']['qwen']+'\n\nQWEN USER MESSAGE (course 1, frame 300)\n'+example['qwen']['observation']+'\n\nQWEN EXACT CHAT-TEMPLATED INPUT\n'+example['qwen']['exact_input']+'\n\nCLM COMPLETE STATE INPUT (course 1, frame 300)\n'+example['clm']['exact_input']+'\n\nCLM SEPARATE ACTION INPUTS\n'+json.dumps(protocol['exact_action_inputs'])+'\n')
 evidence=json.loads((dist/'evidence.json').read_text());evidence['trex']={'protocol':protocol,'summary':summary,'replay_checks':replay_audit};(dist/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 files=[p for p in dist.iterdir() if p.is_file() and p.name not in ('manifest.json','report_with_gifs.zip')]
 (dist/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
 with zipfile.ZipFile(dist/'report_with_gifs.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[dist/'manifest.json']:z.write(p,arcname='reusable-prompt-report/'+p.name)
 for p in dist.iterdir():
  if p.is_file():shutil.copyfile(p,OUT/p.name)
 print(json.dumps({'run_id':summary['run_id'],'decisions_replayed':sum(r['decisions_verified'] for r in replay_audit),'courses':successes,'latency':{m:summary[m] for m in games},'updated':str(dist)}),flush=True)

if __name__=='__main__':
 import sys
 if '--restore-trex' in sys.argv:restore_trex()
 else:main()
