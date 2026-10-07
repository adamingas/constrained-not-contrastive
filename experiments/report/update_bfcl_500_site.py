"""Publishable report copy drawn from the saved 500-request experiment."""
import html
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/bfcl_author_reproduction/reconstruction/noschema_500'
DIST = ROOT / 'sites/constrained-not-contrastive/dist'
old = (DIST / 'index.html').read_text()
result = json.loads((DATA / 'summary.json').read_text())
validation = json.loads((DATA / 'validation.json').read_text())
example = json.loads((DATA / 'inputs.jsonl').read_text().splitlines()[0])
assert example['id'] == 'multiple_0'
assert result['accuracy']['all'] == {'n': 500, 'qwen': 477, 'clm': 362, 'qwen_only_correct': 131, 'clm_only_correct': 16}
escape = html.escape
pre = lambda text: '<pre>' + escape(text) + '</pre>'

prefix = old[:old.index('<h2 class="results-heading">Results</h2>')]
prefix = prefix.replace('A comparison of constrained Qwen and Contrastive Language Model on T-Rex, software function selection, and Wikipedia navigation, with cached prompts and verified game replays.',
                        'A 500-request BFCL comparison: constrained Qwen achieves 95.4% function-selection accuracy versus CLM’s 72.4%, with prompt reuse measured across successive requests.')
prefix = prefix.replace('In our function-selection and T-Rex experiments, CLM performed worse than a language model restricted to the same choices. When the instructions and choices could be reused, the latency gap was small. For these cases, we argue that the extra trained component did not earn its place.',
                        'In our 500-request function-selection experiment, CLM performed worse than a language model restricted to the same choices. When successive requests reused the same instructions and choices, the difference in median latency was small. In this setting, the extra trained component did not earn its place.')
prefix = prefix.replace('We use the released CLM scoring heads and T-Rex simulator, and follow the repository’s implementation where possible. For the other tasks, we use public benchmark data because the authors’ exact test setups were not released.',
                        'We use the released CLM scoring heads and public BFCL data, following the repository’s implementation where possible. The authors’ exact BFCL test setup was not released.')
layout_start = prefix.index('<p>For T-Rex, the choice mapping is simply:</p>')
layout_end = prefix.index('<p>We choose the output token', layout_start)
prefix = prefix[:layout_start] + '''<p>For a request about a triangle, the choice mapping might be:</p>
<pre aria-label="Abbreviated function-selection prompt">[System prompt: choose the function that fulfills the request]

A = triangle_properties.get — triangle dimensions and properties
B = circle_properties.get — circle dimensions and properties

…</pre>
''' + prefix[layout_end:]
cache_start = prefix.index('<details class="explanation"><summary>Caching the actions</summary>')
prefix = prefix[:cache_start] + '''<details class="explanation"><summary>Caching the actions</summary>
<p>Qwen receives the instructions and available functions before the changing request. If a later request starts with the same tokens, it can reuse the model’s saved work for that prefix, called the KV cache. CLM can save each function’s action vector and reuse it whenever that function appears again.</p>
<p>Both caches start empty in the experiment below. Each model answers each request once. A first encounter pays the cost of processing new choices; later requests can reuse them. We report both the overall timings and the requests where both models reused all their fixed inputs.</p>
</details>
'''

accuracy_rows = ''
for key, name in [('multiple', 'Multiple'), ('live_multiple', 'Live multiple'), ('all', 'Combined')]:
    r = result['accuracy'][key]
    accuracy_rows += f'<tr><td>{name}</td><td>{r["n"]}</td><td>{r["clm"]}/{r["n"]} · {100*r["clm"]/r["n"]:.1f}%</td><td>{r["qwen"]}/{r["n"]} · {100*r["qwen"]/r["n"]:.1f}%</td></tr>\n'
timing_rows = ''
for group, stat, label in [('all', 'median_ms', 'Median · all 500 requests'), ('all', 'mean_ms', 'Mean · all 500 requests'), ('both_reusable_parts_cached', 'median_ms', 'Median · 264 requests with both fixed inputs cached')]:
    r = result['latency'][group]
    timing_rows += f'<tr><td>{label}</td><td>{r["clm"][stat]:.1f} ms</td><td>{r["qwen"][stat]:.1f} ms</td></tr>\n'

clm_actions = ''.join(f'<p><strong>Separate action input {i+1}: <code>{escape(key)}</code></strong></p>{pre(text)}'
                      for i, (key, text) in enumerate(zip(example['action_keys'], example['exact_action_texts'])))
trex_prompt = (DIST / 'trex_prompt.txt').read_text()
body = f'''<h2 class="results-heading">Results</h2>
<h3 id="bfcl">BFCL: choosing a software function</h3>
<p>The Berkeley Function Calling Leaderboard (BFCL) tests whether models can use software functions to answer a request. We focus on one part of that task: choosing the right function from the available choices. For example, a request for a triangle’s properties should select the triangle function rather than the circle function.</p>
<p>We sampled <strong>500 requests</strong> from two BFCL categories. Both require a single function choice:</p>
<ul>
<li><strong>Multiple</strong> is the original benchmark’s category for choosing one function from several supplied functions. “Multiple” refers to the available choices, not to making several calls. Our sample includes 78 requests. <a href="https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/README.md#python">Category definition</a>.</li>
<li><strong>Live multiple</strong> tests the same choice using the user-contributed requests and function documentation in BFCL’s Live collection. Our sample includes 422 requests. “Live” describes where the data came from; we evaluated saved examples, without calling live services. <a href="https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html">About the Live dataset</a>.</li>
</ul>
<p>Both models received the same request text and each function’s <strong>name and description only</strong>. We omitted parameter schemas—the fields describing a function’s arguments—from both models. We kept the supplied choices and their order, added no “no tool” option, and gave neither model examples or answer hints. Qwen used a short instruction to select a letter; CLM scored the separately encoded function descriptions.</p>
<p class="note">This is a function-selection adaptation of BFCL. We check the selected function’s name; we do not generate or score its arguments, execute it, or report an official full-BFCL score.</p>
</div>
<div class="table-wrap"><table aria-label="Accuracy on 500 BFCL requests"><thead><tr><th>BFCL category</th><th>Requests</th><th>CLM</th><th>Constrained Qwen</th></tr></thead><tbody>
{accuracy_rows}</tbody></table></div>
<div class="copy">
<p><strong>Constrained Qwen answered 477 of 500 requests correctly (95.4%); CLM answered 362 (72.4%).</strong> Qwen was correct where CLM was wrong on 131 requests. CLM was correct where Qwen was wrong on 16.</p>
<h3>Latency with reuse across requests</h3>
<p>Each model answered each request once. We selected the requests randomly, then kept their order in the dataset. We did not group similar requests to create more cache hits, or answer a question twice to warm its cache. Both caches began empty, and the reported timings include the cost of processing new choices.</p>
<p>Qwen reused its entire fixed preamble on <strong>264 requests</strong>. It reused only part of an earlier preamble on another 235; the first request had no cached tokens. CLM reused all its action vectors on 268 requests. On the 264 requests where Qwen’s whole preamble was cached, CLM’s actions were also cached.</p>
</div>
<div class="table-wrap"><table aria-label="BFCL latency including first-use costs"><thead><tr><th>Decision time</th><th>CLM</th><th>Constrained Qwen</th></tr></thead><tbody>
{timing_rows}</tbody></table></div>
<div class="copy">
<p>On those <strong>264 requests with both fixed inputs cached</strong>, Qwen answered 248 correctly (93.9%), versus CLM’s 185 (70.1%). Their median times were 86.7 and 83.0 milliseconds—a difference of 3.7 milliseconds.</p>
<p>Across all 500 requests, CLM had the lower median latency, while Qwen had the lower mean. CLM’s larger first-use costs raised its average. These figures describe this run; the small median gap in the cached subset is not a claim of statistical equivalence or equal latency in every case.</p>
<p class="note">Measured locally on an Apple M4 Max. CLM’s new actions were encoded sequentially, and its released scoring heads ran on the CPU. Its first-use timings should not be treated as an optimized batched server result.</p>

<details class="prompt-expander" id="qwen-bfcl-prompt"><summary>Exact constrained Qwen prompt</summary>
<p>This is <code>{escape(example['id'])}</code>, the first request in the 500-request run. The instructions stayed fixed. The listed functions changed when the dataset supplied a different set; an unchanged prefix could be reused from an earlier request.</p>
<p><strong>System message</strong></p>
{pre(example['qwen_messages'][0]['content'])}
<p><strong>User message</strong></p>
{pre(example['qwen_messages'][1]['content'])}
<p>The standard Qwen chat template wrapped these messages. Thinking was disabled; the template supplied an empty thinking block before the answer. For this request, the code chose the highest-scoring output token from <code>A</code> and <code>B</code> only, then mapped it to the function name. The model generated no explanation.</p>
<p><a href="bfcl_qwen_exact_prompt.txt">Download the exact input, including chat-template markers</a></p>
</details>

<details class="prompt-expander" id="clm-bfcl-prompt"><summary>Exact CLM inputs</summary>
<p>The same request, <code>{escape(example['id'])}</code>. CLM used no system message or chat template. It encoded the state below and each action separately. It received no A/B labels or letter-output instruction.</p>
<p><strong>State encoder input</strong></p>
{pre(example['exact_state_text'])}
{clm_actions}
<p>The released state and action heads transformed these vectors, and their scores selected a function name.</p>
<p><a href="bfcl_clm_exact_inputs.txt">Download these exact inputs</a> · <a href="bfcl_500_inputs.jsonl">Read both models’ inputs for all 500 requests</a></p>
</details>

<details class="explanation"><summary>Experiment details and saved results</summary>
<p>Both approaches used the same original BF16 Qwen3-8B weights through MLX. CLM used its released PyTorch scoring heads on the CPU with four threads. Qwen selected among the permitted single-token labels using its native output weights. We alternated which model ran first for each request.</p>
<p>The sample used a fixed random seed of 20261004 and excluded the 20 requests in the preceding Qwen pilot. The 500 requests were drawn from the two categories combined, giving 78 Multiple and 422 Live multiple cases. These public data had been used in earlier CLM checks; this was not a newly held-out benchmark. The prompt was fixed before this run and was not tuned on its results.</p>
<p>Qwen retained up to 16 fixed preambles, discarding the least recently used when full. It could reuse any identical opening tokens from a retained preamble. It never cached the changing question for reuse. CLM retained projected action vectors in a 32 MiB cache and recomputed each request’s state. Four request texts occurred under different dataset IDs; no completed state-vector cache was used.</p>
<p>Each timing includes tokenization, cache lookup and maintenance, previously unseen inputs, and the final choice. Model loading, file writing, and a single unrelated runtime warmup are excluded. No benchmark request was repeated for timing. Cache-source records were checked against an independent simulation of the request order.</p>
<p>The saved data come from the pinned <a href="https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data">BFCL repository snapshot</a>. The authors’ exact CLM-versus-Jev evaluation setup is unavailable, so this is not a reproduction of their published score. This 500-request run replaces the earlier 60-request BFCL comparison in this report.</p>
<div class="links"><a href="bfcl_500_summary.json">Results</a><a href="bfcl_500_protocol.json">Fixed setup</a><a href="bfcl_500_scored_details.json">Every prediction and timing</a><a href="bfcl_500_validation.json">Cache verification</a></div>
</details>

<details class="explanation" id="trex-archived"><summary>Earlier T-Rex experiment — excluded from the conclusions</summary>
<p><strong>The earlier T-Rex comparison was not a fair test of the two approaches.</strong> Its prompt spelled out when to jump, including a numerical distance threshold. It was written for Qwen and copied into CLM’s state input with A/B/C output instructions. Supplying the same text did not make it equally suited to both interfaces.</p>
<p>We withdraw the earlier claim that this established an advantage for constrained Qwen. The prompt and replay remain available to show what was tested; they are not evidence for the BFCL findings above.</p>
<details class="prompt-expander"><summary>Read the earlier T-Rex prompt</summary>{pre(trex_prompt)}<p><a href="trex_prompt.txt">Download the earlier prompt</a></p></details>
<figure><img id="replay" src="trex_preview.png" width="1224" height="272" alt="Archived T-Rex comparison, excluded because of the prompt design.">
<div class="animation-controls"><button type="button" id="restart">Play archived replay</button><button type="button" id="pause">Show still image</button></div>
<figcaption>Reconstructed from recorded actions. The game paused during model decisions. This replay documents the flawed experiment and is excluded from the conclusions.</figcaption></figure>
</details>

<h3>Wikipedia: both struggled</h3>
<p>In the earlier Wikispeedia experiment, we asked the models to reach a target Wikipedia article by following links. Across five test routes, with at most eight clicks each, Qwen finished one and CLM finished none. Both results were poor. They do not establish a useful navigation system or a reliable advantage for Qwen.</p>
<h2>What this supports</h2>
<p>For repeated function-selection tasks, constrained decoding is worth testing before adding a trained decision network. In this 500-request sample, it selected the correct function more often. When earlier requests had already populated both models’ caches, the difference between median decision times was 3.7 milliseconds. That conclusion is specific to this setup; it does not establish an advantage on every task or deployment.</p>
</div>
<footer>Updated 3 October 2026. Sources: <a href="https://typesafe.ai/">TypeSafe AI and Jev</a> · <a href="https://github.com/Contrastive-LM/CLM">CLM release</a> · <a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard">Berkeley Function Calling Leaderboard</a> · <a href="https://snap.stanford.edu/data/wikispeedia.html">Wikispeedia</a>.</footer>
</main><script>
const replay=document.getElementById('replay');
document.getElementById('restart').onclick=()=>{{replay.src='trex_comparison.gif?replay='+Date.now()}};
document.getElementById('pause').onclick=()=>{{replay.src='trex_preview.png'}};
</script></body></html>
'''

for name in ('index.html', 'report.html'):
    (DIST / name).write_text(prefix + body)
for name in ('inputs.jsonl', 'summary.json', 'protocol.json', 'scored_details.json', 'validation.json'):
    shutil.copy2(DATA / name, DIST / ('bfcl_500_' + name))
(DIST / 'bfcl_qwen_exact_prompt.txt').write_text(example['qwen_prompt'])
(DIST / 'bfcl_clm_exact_inputs.txt').write_text('STATE ENCODER INPUT\n\n' + example['exact_state_text'] + '\n\nSEPARATELY ENCODED ACTIONS\n\n' + '\n\n'.join(key + '\n' + text for key, text in zip(example['action_keys'], example['exact_action_texts'])))
old_evidence = json.loads((DIST / 'evidence.json').read_text())
(DIST / 'evidence.json').write_text(json.dumps({'current_bfcl_run': result, 'current_bfcl_validation': validation,
    'superseded_results_note': 'Earlier BFCL figures are superseded by the 500-request experiment. Earlier T-Rex comparison is excluded from conclusions because of prompt design.',
    'historical_evidence': old_evidence}, indent=2) + '\n')
with zipfile.ZipFile(DIST / 'report_with_gifs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DIST.iterdir()):
        if p.is_file() and p.suffix != '.zip':
            z.write(p, p.name)
print(json.dumps({'updated': str(DIST), 'example': example['id'], 'requests': result['validated_requests']}))
