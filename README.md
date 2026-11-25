# Constrained not Contrastive Language Models

A new wave of “decision models”, classifiers in plain terms, has landed on the
scene. [TypeSafe AI’s Jev](https://typesafe.ai/) offers a commercial implementation,
while open-source releases such as [Contrastive Language Model (CLM)](https://github.com/Contrastive-LM/CLM)
position themselves as competitors. Their promise is to make decisions faster
than conventional language models.

This prompted our investigation: does a new decision model offer enough to justify
the added machinery? In our 500-request function-selection experiment, CLM performed
worse than a language model restricted to the same choices. When successive requests
reused the same instructions and choices, the difference in median latency was small.
In this setting, the extra trained component did not earn its place.

We compare CLM with constrained decoding using the same Qwen3-8B backbone. We use the
released CLM scoring heads, its T-Rex simulator, and public BFCL data, following the
repository’s implementation where possible. The authors’ exact BFCL test setup was
not released. We are testing CLM against constrained Qwen, not reproducing the
published comparison with Jev.

<details>
<summary>What is a decision model?</summary>

A decision model takes an input and selects from a fixed set of choices. These
might be true or false, named categories, or ordered labels such as low, medium,
and high. In a game, they are the available actions.

CLM encodes the question and each possible answer separately as vectors using an
existing language model. Two trained projection networks, or MLPs, transform the
question and answer vectors. A dot product scores each pair, and the highest
score selects the answer.

</details>

## Constrain the language model instead

Our approach needs no additional trained component. We put the choices in the
prompt and restrict the model’s output to one of them. To avoid generating a
multi-word answer over several forward passes, we assign each choice a letter
that is a single token. The model chooses one permitted letter.

For a request about a triangle, the choice mapping might be:

```text
[System prompt: instructions and available functions]

A: triangle_properties.get: Retrieve the dimensions of a triangle…
B: circle_properties.get: Retrieve the dimensions of a circle…

…
```

We choose the output token among the allowed choices with the highest probability.
Thinking is disabled. We use the corresponding rows of Qwen’s native output head,
so the restriction is enforced in code.

<details>
<summary>Caching the actions</summary>

Qwen receives the instructions and available functions before the changing request.
If a later request starts with the same tokens, it can reuse the model’s saved work
for that prefix, called the KV cache. CLM can save each function’s action vector
and reuse it whenever that function appears again.

Both caches start empty in the BFCL experiment. For T-Rex, we prepare both fixed
prompts before play, then time decisions with warm caches. Each model answers each
request once. A first encounter pays the cost of processing new choices; later
requests can reuse them.

</details>

## Results

### T-Rex

We use the T-Rex simulator from the CLM repository to compare the two approaches
on repeated game decisions. Both receive the dinosaur’s position and motion and
the nearest obstacle as text. Each chooses run, jump, or duck for the next six frames.

We use different prompts for each model because both needed extensive prompt tuning
to play for 30 seconds without crashing. CLM needed many more worked examples and
had much higher latency, even with cached input.

Across three tries with different seeds:

| 900 decisions per model | CLM | Constrained Qwen |
|---|---:|---:|
| Median decision | 182.5 ms | **113.4 ms** |
| Mean decision | 182.1 ms | **124.7 ms** |
| 95th percentile | 229.7 ms | **192.3 ms** |

Measured on an M4 Max with 64 GB memory and BF16 Qwen3-8B; CLM’s scoring heads ran
on the CPU. Timings cover each decision, excluding model loading and initial cache
preparation. The game paused during inference.

[Watch the replay on the website](https://constrained-not-contrastive.deploy.chatgpt.site/#trex).
The animation reconstructs recorded actions and omits model waiting time.

<details>
<summary>Exact constrained Qwen prompt</summary>

**System message**

```text
You are playing T-Rex runner. Stay alive by avoiding collisions. At each turn, choose A, B, or C for the next 6 frames, then you will receive a new state and choose again. The game pauses while you decide.

A continues automatic running, or lets an existing jump continue.
B starts a jump from the ground. It does nothing extra in the air.
C crouches on the ground, or rapidly descends in the air.

Obstacles move left. The state gives distances, sizes, and heights above ground. The dinosaur is 44 pixels wide and 47 pixels tall when standing. Feet height zero means grounded. Contact ends the game. A jump rises, reaches its peak, falls, and ends; it cannot be extended by repeatedly selecting B. Duck does not stop horizontal motion.

Output only the chosen letter.

Controls:
A = run: Release duck. Running is automatic; an existing jump continues.
B = jump: Release duck and start a jump if grounded. No second jump while airborne.
C = duck: Crouch if grounded; start a rapid descent if airborne.

State: The dinosaur is on the ground. A cactus is 500 pixels beyond its front, moving toward it at 6 pixels per frame.
Answer: A (run).
Explanation: A jump now would finish before the cactus arrived. Continue running for this turn and reconsider at the next observation.
State: The dinosaur is on the ground. A cactus is 60 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame.
Answer: B (jump).
Explanation: Begin rising now so the dinosaur can pass above the approaching cactus. Continuing on the ground would bring it into the cactus.
State: The dinosaur is on the ground. A cactus is 180 pixels beyond its front, 25 pixels wide and 50 pixels tall, approaching at 6 pixels per frame.
Answer: A (run).
Explanation: This is still too early to start the jump. The dinosaur would descend while the cactus was passing. Keep running for this six-frame turn, then reassess the smaller distance.
State: The dinosaur is in the air, with its feet 80 pixels above the ground, falling at 2 pixels per frame. A cactus is 20 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame.
Answer: A (run).
Explanation: Let the current jump carry the dinosaur over the cactus. Ducking would accelerate its descent into the obstacle; another jump command cannot add height while airborne.
```

**User message at frame 300 of the first course**

```text
State: The dinosaur is on the ground. A cactus is 192 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame.
Answer:
```

[Prompt file](fixtures/trex_qwen.txt)

</details>

<details>
<summary>Exact CLM inputs</summary>

State input at the same frame, without a chat template. Everything before
“Current observation” is fixed.

```text
T-Rex runner: avoid collisions. Choose the next control for six frames. Run continues forward motion and any existing jump. Jump starts a jump from the ground. Duck crouches on the ground or accelerates descent in the air. Cacti are ground obstacles; birds may leave room underneath. The game pauses while the model decides. Obstacles move left. The dinosaur is 44 pixels wide and 47 pixels tall when standing. Feet height zero means grounded. Contact ends the game. A jump rises, reaches its peak, falls, and ends; it cannot be extended by repeatedly choosing jump. Duck does not stop horizontal motion. The examples below describe other situations. Choose for the current observation at the end, using the examples to understand timing.

Example state: The dinosaur is on the ground. A cactus is 60 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame. Answer: jump. Explanation: Start rising to clear the approaching cactus; staying on the ground would cause a collision.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |------| obstacle.

Example state: The dinosaur is on the ground. A cactus is 500 pixels beyond its front, moving toward it at 6 pixels per frame. Answer: run. Explanation: A jump now would finish before the cactus arrived.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |--------------------------------------------------| obstacle.

Example state: The dinosaur is on the ground. A cactus is 180 pixels beyond its front, 25 pixels wide and 50 pixels tall, approaching at 6 pixels per frame. Answer: run. Explanation: Wait for this six-frame turn. Jumping now would put the descent too early, while the cactus is still passing.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |------------------| obstacle.

Example state: The dinosaur is in the air with its feet 80 pixels above the ground, falling at 2 pixels per frame. A cactus is 20 pixels beyond its front, 25 pixels wide and 50 pixels tall, approaching at 6 pixels per frame. Answer: run. Explanation: Let the current jump continue over the cactus. Ducking would force the dinosaur down into it.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |--| obstacle.

Compare two cases with the same grounded dinosaur, cactus size, and speed of 6 pixels per frame. At a distance of 180 pixels, choose run: the cactus arrives too late for a jump started now. At a distance of 60 pixels, choose jump: the cactus will arrive soon enough for this jump to clear it. The distance changes the answer even though both observations mention a cactus.

Example state: The dinosaur is on the ground. A cactus is 40 pixels beyond its front, approaching at 6 pixels per frame. Answer: jump. Explanation: Waiting another six frames would leave almost no clearance; begin rising now.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |----| obstacle.

Example state: The dinosaur is on the ground. A cactus is 300 pixels beyond its front, approaching at 6 pixels per frame. Answer: run. Explanation: An obstacle being visible does not mean it is time to jump. It will still be far ahead after this six-frame turn.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |------------------------------| obstacle.

Example state: The dinosaur is on the ground. A cactus is 120 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame. Answer: run. Explanation: Keep running for this six-frame turn. There is time to get closer before starting a jump, so it will not land too early.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |------------| obstacle.

Example state: The dinosaur is on the ground. A cactus is 200 pixels beyond its front, 34 pixels wide and 35 pixels tall, moving toward it at 6 pixels per frame. Answer: run. Explanation: This shorter cactus is still too far away. Continue running for six frames and look again before jumping.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |--------------------| obstacle.

Example state: The dinosaur is on the ground. A cactus is 150 pixels beyond its front, 17 pixels wide and 35 pixels tall, moving toward it at 6 pixels per frame. Answer: run. Explanation: This small cactus is still too far away. A jump started now would finish while the cactus was passing the dinosaur. Wait for the next observation before starting the jump.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |---------------| obstacle.

Current observation:
State: The dinosaur is on the ground. A cactus is 192 pixels beyond its front, 25 pixels wide and 50 pixels tall, moving toward it at 6 pixels per frame.
Side view of the gap, each dash is approximately 10 pixels: dinosaur |-------------------| obstacle.

What should the dinosaur do for the next six frames only? It will get another observation afterward. Decide whether to keep running, jump now, or duck now.
```

**Separately encoded action inputs**

```text
Run.
```

```text
Jump.
```

```text
Duck.
```

[Prompt file](fixtures/trex_clm.txt) · [Question and actions](fixtures/trex_question.json) ·
[CLM decisions](results/published/trex_clm_trace.json.gz) ·
[Qwen decisions](results/published/trex_qwen_trace.json.gz)

These seeds were used during prompt development or checks. A later
[unseen-seed run](results/published/trex_new_seed.json) is separate from these timings.

</details>

### BFCL: choosing a software function

We tested 500 examples from the Berkeley Function Calling Leaderboard (BFCL),
giving each model a question and a set of tools and asking it to select the right
tool. Both models received the tool names and descriptions, without parameter schemas.

We score tool selection only, not arguments or execution. This is not a full BFCL score.

| 500 requests | CLM | Constrained Qwen |
|---|---:|---:|
| Correct function choices | 362/500 · 72.4% | **477/500 · 95.4%** |

We could not reproduce the CLM repository’s reported
[95.2% BFCL accuracy](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/assets/zero-shot.png)
in this experiment.

Both caches started empty and warmed as earlier requests supplied reusable
instructions and tools. On the 264 requests where both models’ fixed inputs were
cached, median latency differed by 3.7 ms.

| Decision time | CLM | Constrained Qwen |
|---|---:|---:|
| Median, all 500 requests | **92.9 ms** | 134.3 ms |
| Mean, all 500 requests | 202.9 ms | **148.8 ms** |
| Median, 264 requests with both fixed inputs cached | **83.0 ms** | 86.7 ms |

CLM’s mean was higher because new tools were encoded one at a time. These timings
reflect our Mac implementation, not an optimized batched server.

<details>
<summary>Exact BFCL prompts: the first request</summary>

**Qwen system message**

```text
Choose the available function that best fulfills the request. Reply with exactly one option letter, without explanation.

Available functions:

A:
triangle_properties.get: Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.

B:
circle_properties.get: Retrieve the dimensions, such as area and circumference, of a circle if radius is given.
```

**Qwen user message and CLM state encoder input**

```text
User: Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?

Which function should be called to fulfill this request?
```

Qwen uses its non-thinking chat template and is restricted to the A and B tokens.
CLM sees the raw state above and separately encodes these actions:

```text
triangle_properties.get: Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.
```

```text
circle_properties.get: Retrieve the dimensions, such as area and circumference, of a circle if radius is given.
```

[Complete sample](fixtures/bfcl.jsonl)

</details>

<details>
<summary>BFCL experiment details</summary>

We kept the original choices and their order, added no “no tool” option, and gave
neither model examples or answer hints. Each model answered each request once.

Both categories ask for one tool from several choices. **Live multiple** uses
user-contributed requests and tool descriptions.

| BFCL category | Requests | CLM | Constrained Qwen |
|---|---:|---:|---:|
| Multiple | 78 | 63/78 · 80.8% | **77/78 · 98.7%** |
| Live multiple | 422 | 299/422 · 70.9% | **400/422 · 94.8%** |
| Combined | 500 | 362/500 · 72.4% | **477/500 · 95.4%** |

We sampled with seed 20261004, excluded the 20-request pilot, and kept dataset
order. The prompt was fixed before this run; these public data had been used in
earlier CLM checks. Neither model reused completed answers.

Timings include input processing, caching, and scoring. Model loading and file
writing are excluded. The models ran in alternating order on the same M4 Max.

[Original protocol and source revisions](results/published/bfcl_protocol.json) ·
[Recorded predictions](results/published/bfcl_predictions.json.gz) ·
[Recorded summary](results/published/bfcl_summary.json)

</details>

### Wikipedia: both struggled

We asked the models to reach a target Wikipedia article by following links.
Across five routes with at most eight clicks each, Qwen finished one and CLM none.

[Routes](fixtures/wiki_routes.json) · [Recorded decisions](results/published/wiki.json.gz)

## Reproduce

Requires an Apple Silicon Mac and Python 3.12. Model weights take about 16.5 GB,
with additional memory needed for inference.

**Installation blocker:** upstream’s vLLM dependency requires
`nvidia-cudnn-frontend`, which has no macOS distribution. This must be resolved
before the commands below can run.

```sh
git submodule update --init --recursive
uv sync --locked
uv run compare download
```

View saved results:

```sh
uv run compare results
```

Start with small samples:

```sh
uv run compare bfcl --limit 5 --output runs/bfcl-smoke
uv run compare trex --seed 42 --seconds 1 --output runs/trex-smoke
uv run compare wiki --routes 1 --clicks 1 --output runs/wiki-smoke
```

Reproduce the reported experiments:

```sh
uv run compare bfcl --limit 500 --output runs/bfcl-500
uv run compare trex --seed 42 --seed 314159 --seed 271828 --seconds 30 --output runs/trex-three-seeds
uv run compare wiki --routes 5 --clicks 8 --output runs/wiki-five-routes
```

[Dataset sources and checksums](fixtures/sources.json) · [Upstream license](upstream/LICENSE)
