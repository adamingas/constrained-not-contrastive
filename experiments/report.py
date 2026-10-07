"""Render the measured results as a self-contained HTML research report."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "results/comparison.json").read_text())
t = json.loads((ROOT / "results/trex.json").read_text())
v = json.loads((ROOT / "results/backbone_validation.json").read_text())


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


methods = ["clm_cached_actions", "qwen_prefix_cached_selected", "qwen_uncached_selected", "qwen_uncached_full_vocab"]
names = {"billing_2": "Billing · 2 options", "mario_5": "Mario text · 5 options", "routing_26_long_options": "26 long options", "billing_long_state": "Long state · 2 options"}
latency = table(["Scenario", "CLM / cached actions", "Qwen / cached prefix", "Qwen / no prefix cache", "Qwen / full vocabulary"], [[names[k]] + [f"{d[m]['p50_ms']:.1f} ms" for m in methods] for k,d in r["latency"].items()])
latency_detail = table(["Scenario", "Method", "p50", "p95", "Prompt tokens", "New tokens processed"], [[names[k], m, f"{d[m]['p50_ms']:.2f} ms", f"{d[m]['p95_ms']:.2f} ms", d[m]['example']['input_tokens'], d[m]['example'].get('processed_tokens',d[m]['example']['input_tokens'])] for k,d in r['latency'].items() for m in methods])
quality = []
for group in ["routing", "mario_text", "relative_numbers"]:
    rows = [x for x in r['quality'] if x['group']==group]
    quality.append([group, len(rows)] + [f"{sum(x[m]['choice']==x['gold'] for x in rows)}/{len(rows)}" for m in ['clm','qwen','qwen_reversed']])
quality = table(["Diagnostic", "Cases", "CLM", "Qwen", "Qwen, reversed options"], quality)
games = table(["Seed", "Method", "Survived 20 s", "Time reached", "Planner agreement", "Unsafe choices"], [[x['seed'], x['method'], 'Yes' if x['survived'] else 'No', f"{x['simulated_seconds']:.1f} s", f"{100*x['agreement']:.1f}%", x['unsafe_choices']] for x in t['runs']])

content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLM vs constrained Qwen3: investigation and local measurements</title>
<style>
body{{font:17px/1.65 system-ui,sans-serif;color:#1a2938;background:#f4f6f8;margin:0}}main{{max-width:1120px;margin:40px auto;padding:40px;background:white;border-radius:16px}}h1{{font-size:38px;line-height:1.2}}h2{{margin-top:40px;font-size:25px}}p{{max-width:950px}}a{{color:#0758a0}}.lead{{font-size:21px;border-left:4px solid #087b76;padding-left:20px}}.muted{{color:#536273}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.4;margin:20px 0}}th,td{{padding:13px 12px;text-align:left;border-bottom:1px solid #dbe1e8}}th{{background:#edf4f5}}.scroll{{overflow-x:auto}}code,pre{{font-family:ui-monospace,monospace;background:#f0f3f5}}pre{{padding:20px;overflow:auto;font-size:14px}}li{{margin:8px 0}}.pill{{display:inline-block;background:#e3f5ee;color:#166149;padding:4px 12px;border-radius:20px;font-size:13px}}@media(max-width:700px){{main{{margin:0;padding:22px}}h1{{font-size:29px}}}}
</style><main>
<span class="pill">Measured locally · 2 October 2026</span>
<h1>CLM versus a single constrained output token</h1>
<p class="lead">The proposed A/B/C/D/E baseline is valid. In this experiment, with a reusable options-prefix KV cache, it was within 11% of CLM's median latency across four workloads. The evidence inspected does not establish a 9× advantage over this baseline.</p>
<p class="muted">Apple M4 Max · 64 GB · original BF16 Qwen3-8B · MLX · released CLM checkpoint. Exploratory measurements, not a general accuracy leaderboard or a reproduction of NVIDIA/vLLM timings.</p>

<h2>What was actually compared</h2>
<p>Both paths use the same frozen Qwen3-8B transformer weights. CLM processes plain state-plus-question text, takes the final-token representation, normalizes it, and applies the released state head. Each action description is separately encoded and projected by the action head. Candidate vectors are cached, then scored using scaled cosine similarity.</p>
<p>The constrained baseline places the option definitions in the system message, puts the state and question afterward, disables thinking with <code>enable_thinking=False</code>, and reads the logits for single-token option letters at the end of that one prompt forward pass. It never feeds the selected letter back through the transformer. A further optimized version computes only the corresponding rows of the LM output matrix. The full-vocabulary version computes the last-position output projection normally, then selects the allowed labels.</p>
<pre>CLM:
  z_state = normalize(state_head(normalize(Qwen(state + question)[last])))
  z_action[i] = normalize(action_head(normalize(Qwen(action[i])[last])))
  score[i] = scale * dot(z_state, z_action[i])

Constrained Qwen:
  h = Qwen(options + state + question + answer_prefix)[last]
  score[i] = dot(lm_head.weight[token_id(label[i])], h)

Both:
  probabilities = softmax(scores over allowed options)
  choice = argmax(scores)</pre>
<p>The output weight for “A” is not an embedding of “jump right.” Its meaning comes from the prompt, which contextualizes the final hidden state. In this Qwen checkpoint, input embeddings and LM output weights are not tied. The semantic comparison is implemented through the transformer's joint processing of the options and state.</p>

<h2>Latency measurements</h2>
{latency}
<p>Each entry is the median of 25 timed calls after three warm-ups, with randomized method order. Every call recomputes the state. CLM's action cache and Qwen's reusable system-prefix KV cache are warm; full-request/state memoization is excluded. Cache population, model loading, HTTP transport and serving concurrency are outside these timings. Prompt construction, tokenization, prefix-cache copying and output materialization are included. The final column is also constrained to the same labels; only its output projection is larger.</p>
<p>For billing, CLM processes 19 tokens; Qwen's full prompt has 75, of which 47 are cached and 28 are newly processed. For the 26-option scenario, the full prompt has 760 tokens, of which 734 can be reused. With a long state, the shared cost of reading that state dominates both approaches.</p>
<p>The CLM state projection plus scoring took <strong>{r['head_only']['p50_ms']:.3f} ms</strong> in a separate warmed microbenchmark. The small MLP is not the dominant inference cost. Restricting the LM output projection saves only a few milliseconds on the short workloads here; prefix reuse matters much more.</p>
<details><summary>All latency statistics and token counts</summary>{latency_detail}</details>

<h2>Decision quality: small diagnostic checks</h2>
{quality}
<p>These 35 questions were authored for this investigation before running the models. They cover customer routing, textual Mario situations, and numerical comparisons. They are too small and informal to establish broad superiority. The two systems also have different training objectives and data; this is an off-the-shelf comparison, not a controlled training ablation.</p>
<p>Both models classify the user's duplicate-invoice example as billing. The local CLM distribution is approximately 98.8% billing and 1.2% technical. This need not exactly equal the illustrative README probabilities. Some CLM errors favor “jump left” across quite different Mario states. Qwen also makes mistakes, and reversing option order changes 3/10 Mario answers and all 5 numerical answers. Constraining the format guarantees a valid option, not correctness or calibration. Neither method's softmax should automatically be interpreted as a calibrated probability of success.</p>

<h2>T-Rex: what the original demo measures</h2>
<p>The <a href="https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/README.md">published T-Rex harness</a> computes action safety with an exact-physics planner, includes “Safe,” “Unsafe,” and “Best” in the candidate descriptions, and enables a shield that can replace unsafe answers or intervene at execution time. Thus it is a planner-assisted selector benchmark. Survival is a property of the combined model, planner, scheduling and shield.</p>
<p>The committed reports give CLM 5/5 surviving courses, 65.8% planner agreement and 4,883 combined shield interventions; Jev also survives 5/5, with 98.7% agreement and 28 interventions. Their median client-side request latencies are 16.5 ms and 149.8 ms. CLM is served locally on an RTX 4090 while Jev is accessed through its hosted API. These observations can support a deployment-level speed comparison, but they do not isolate architecture or compare against local constrained Qwen. Sources: <a href="https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/results/clm_realtime.json">CLM results</a>, <a href="https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/results/jev_realtime.json">Jev results</a>.</p>

<h2>Our T-Rex run with the shield removed</h2>
{games}
<p>Both methods receive the upstream planner's labeled candidate descriptions. The same deterministic engine runs two seeds for up to 20 simulated seconds, with one choice every six frames. No safety shield or emergency override changes the answer. A planner-only control survives both courses. Game time pauses during inference, and identical full requests are memoized for both models; these game runs are not a latency benchmark. Reported crash times are rounded to the six-frame decision interval.</p>
<p>Constrained Qwen follows the planner's best action on all 400 decisions and survives both courses. CLM crashes in both. This shows that the proposed constrained decoder can perform the supplied demo's selection task well on these courses; it does not establish unaided game intelligence or real-time survival at these local inference latencies. The Mario test above uses textual situations only; an actual Mario emulator was not run.</p>

<h2>What is genuinely different about CLM?</h2>
<ul>
<li><strong>Different supervision.</strong> CLM's heads are trained to align state–action pairs and reject alternatives through a contrastive objective. Qwen's output head was trained to predict text. A relatively small nonlinear head can substantially change a ranking by learning which backbone features matter. Small parameter count does not imply no useful learning.</li>
<li><strong>Independent candidate reuse.</strong> CLM can encode an action once and reuse it across states, candidate reorderings and different subsets. A causal options-prefix cache is tied to the exact token prefix and its order. Each new state token still attends to that cached prefix; CLM's state transformer does not attend to the candidate texts.</li>
<li><strong>Compact candidate storage.</strong> The released head emits 512-dimensional vectors: approximately 2 KiB per projected FP32 action. Qwen3-8B's ordinary BF16 KV cache costs about 144 KiB per cached token, before allocation overhead (36 layers × K/V × 8 KV heads × 128 dimensions × 2 bytes). Memory, large candidate sets and repeated candidate subsets can matter even when small-set latency is close.</li>
<li><strong>A representational tradeoff.</strong> In the independent-input form, a CLM action's raw score cannot change when another candidate is added; only its normalized probability changes. Joint prompting can reason about relations among alternatives. For example, identifying the middle-priced item requires knowing which other items are offered. Putting the options into CLM's state can expose that information, but changes the independent-state caching advantage.</li>
<li><strong>Cheap head training.</strong> The frozen backbone permits precomputing embeddings and training only small heads. This is a real training and adaptation benefit, separate from whether one-token inference is faster.</li>
</ul>
<p>The released checkpoint contains <strong>{r['metadata']['head_parameters_total']:,} total head parameters</strong>, split equally between state and action heads. Its approximately 75 MB file stores FP32 weights; it is not a 75-million-parameter MLP. Its actual topology is 4096 → 1536 → 1536 → 512 with GELU and an intermediate LayerNorm.</p>

<h2>How strong is this evidence?</h2>
<p>The strongest result here is the direct latency measurement: after prefix caching, single-token constrained Qwen is close to CLM on these small and medium candidate sets. It refutes the assumption that choosing through an LLM necessarily requires a costly autoregressive generation loop. It does not prove that Qwen is universally as fast, particularly for huge action catalogs, dynamic candidate subsets, many concurrent requests or different hardware.</p>
<p>The weak point in the published comparison, for this question, is the missing same-backbone constrained baseline. A convincing broader study would match hardware, serving stack, precision, inputs and training budget; distinguish cold and warm caches; measure throughput and cache memory; and compare actual downstream success without a shield masking model errors.</p>
<p>Numerical validation: the CLM head port's maximum absolute difference against PyTorch was below 4×10⁻⁷ on checked normalized inputs. Six backbone embeddings compared against Transformers/PyTorch MPS had cosine similarity from {min(x['cosine'] for x in v['embedding_comparison']):.6f} to {max(x['cosine'] for x in v['embedding_comparison']):.6f}. The duplicate-invoice CLM probabilities were 0.98797 with MLX and 0.98830 with Transformers. This substantially reduces the likelihood of a basic porting error, but is not an exhaustive vLLM equivalence test.</p>
<p>The linked research artifact is cited by the authors as a Notion blog. Its root page failed to load through both web retrieval and the browser during this investigation. Architectural and training claims were therefore checked against the public repository, released weights, model card, and runnable benchmark code. The complete blog's scaling experiments were not independently reviewed.</p>

<h2>Reproduce and inspect</h2>
<p>All code, inputs, outputs, environment versions and timing samples are preserved in the workspace. Start with <a href="../README.md">README.md</a>. The source commit is <code>{r['metadata']['upstream_commit']}</code>. The original repository has not been modified.</p>
<ul><li><a href="comparison.json">Raw decision and latency results</a></li><li><a href="cases.json">Diagnostic questions and gold answers</a></li><li><a href="trex.json">Full T-Rex decision traces</a></li><li><a href="backbone_validation.json">Numerical validation</a></li></ul>
<p>Primary external references: <a href="https://github.com/Contrastive-LM/CLM">CLM repository</a>, <a href="https://huggingface.co/Contrastive-LM/CLM-v0.1-8B">released CLM model card</a>, <a href="https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py">head implementation</a>, <a href="https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py">state and candidate formatting</a>, <a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B documentation</a>.</p>
</main></html>"""
(ROOT / "results/report.html").write_text(content)
print(ROOT / "results/report.html")
