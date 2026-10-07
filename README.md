# CLM versus single-token constrained Qwen3

An October 2, 2026 investigation using an Apple M4 Max with 64 GB unified memory.
See the [RACE-M prompt ablations](results/race_ablations/report.html) for the
latest system-prompt and chat-template comparison, the
[RACE-M context-ratio benchmark](results/race_context/report.html) for the
latest controlled reading-comprehension comparison, the
[long-context benchmark report](results/long_context/report.html) for the
197-question follow-up, and the [initial investigation](results/report.html)
for the small-task and game experiments.

The original repository is included as a Git submodule under `upstream/`, pinned to commit
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`. No upstream source was modified.

After cloning this repository, initialize the upstream checkout:

```sh
git submodule update --init --recursive
```

Alternatively, clone this repository with `git clone --recurse-submodules`.

## Reproduce on Apple Silicon

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r experiments/requirements-lock.txt
.venv/bin/python experiments/download.py
.venv/bin/python -u experiments/compare.py
.venv/bin/python -u experiments/validate_backbone.py
.venv/bin/python -u experiments/trex_compare.py
.venv/bin/python experiments/report.py
```

The downloads are approximately 16.5 GB. Both decision paths share the original
BF16 Qwen3-8B weights. The released CLM heads are evaluated in FP32 with the same
last-token pooling, input normalization, output normalization and temperature
clamp used by the repository. MLX replaces vLLM for local Apple Silicon execution.

`compare.py` runs 35 hand-authored diagnostic questions, reversed-option checks,
and four latency scenarios. Each timing has three warm-ups and 25 timed samples;
method order is shuffled with seed 42. CLM action projections and Qwen system-prefix
KV tensors are warmed before timing. State computation is repeated on every timed
call; whole-state/whole-request cache hits are excluded. Timings include prompt
construction, tokenization, cache copying, inference, scoring and extraction of
probabilities, but not model loading, cache setup or HTTP serving. GPU results are
materialized before stopping the timer. The full-vocabulary baseline computes the
output projection at the last position only, then restricts it to allowed labels.

The Qwen prefix contains the system options and user-message opening; no changing
state tokens are cached. `enable_thinking=False` is used. Each option letter is
checked to be exactly one tokenizer token. Selected-row projection uses the actual
LM output weights, not the input token embeddings. No autoregressive decoding loop
or second transformer pass is used.

`validate_backbone.py` compares six MLX embeddings with Transformers/PyTorch MPS
using the same BF16 weights and compares the billing probabilities. This is a
numerical spot-check, not a complete vLLM parity test.

`trex_compare.py` uses the upstream game physics, question construction and planner.
Two seeds, 20 simulated seconds each, one decision per six frames, no shield or
emergency override. Game time pauses during inference. Planner Safe/Unsafe/Best
labels remain visible to both models. Identical full requests are memoized for
both methods in this game-only experiment; gameplay timing is not used as a latency
benchmark. A planner-only control checks that the courses can be survived.

## Outputs

- `results/comparison.json`: every diagnostic answer and all timing samples.
- `results/cases.json`: diagnostic inputs and expected answers.
- `results/backbone_validation.json`: numerical cross-check.
- `results/trex.json`: full game decision traces and summaries.
- `results/report.html`: readable research report.

These are exploratory local experiments, not a representative accuracy benchmark,
an NVIDIA/vLLM speed comparison, a training-data-controlled ablation, or a claim
that either approach is universally better. The Mario questions are textual
scenarios, not an actual Mario game. The T-Rex test is planner-assisted.

## Long-context follow-up

The second evaluation uses 197 public-dataset questions: 80 QuALITY articles,
90 ContractNLI contracts, and 27 LongBench v2 documents. Every selected document
is complete; no accuracy input is truncated. The same model is evaluated with
the document removed, and ten questions per dataset receive an option-order
audit. Full methodology and statistics are in `results/long_context/report.html`.

```sh
uv pip install --python .venv/bin/python -r experiments/requirements-long-context-lock.txt
.venv/bin/python experiments/download_long_context.py
.venv/bin/python experiments/prepare_long_context.py
.venv/bin/python -u experiments/long_context_bench.py
.venv/bin/python experiments/analyze_long_context.py
```

The primary samples, option order, no-document controls and order-audit samples
were fixed before inference. `manifest.json` records source revisions and hashes.
QuALITY is balanced across easy/hard questions with high annotated context needs;
ContractNLI is label-balanced; LongBench is the complete eligible subset of the
publisher's short category within a 16K-token prompt limit. These are selected
diagnostic subsets, not official leaderboard scores.

The inference runner resumes its append-only `answers.jsonl`; it does not overwrite
prior outputs. Preserve the existing results and change the output path or use a
separate project copy to perform a fresh replication. It uses 1024-token prefill
chunks with a full KV cache, not context-window truncation. The seven-point latency
sweep is separate from the accuracy test and holds four candidate descriptions fixed.

## Context-rich, moderate-length reading comprehension

The RACE-M follow-up tests the user's precise ratio condition: passage plus question
must contain more than five times the tokens of **all four formatted choices
combined**. The frozen test sample contains 200 distinct passages, 131–524 tokens
including the question, with a median ratio of 9.02. No input is padded or truncated.

A separate 40-question development pilot selected the standard passage–question–
options prompt before test inference. It scored 37/40 with context and 27/40 without;
the original options-in-system layout scored 24/40 and 26/40. Both pilot results are
retained. The acceptance rule (≥70% full accuracy, ≥15-point context gain) was set
before the pilot. No test questions were filtered using model outputs.

```sh
.venv/bin/python experiments/prepare_race_context.py
.venv/bin/python -u experiments/bench_race_context.py --original-pilot
.venv/bin/python -u experiments/bench_race_context.py
.venv/bin/python experiments/validate_race_output.py
.venv/bin/python experiments/analyze_race_context.py
```

The primary Qwen result processes the entire prompt including choices, with one
constrained output letter and no thinking tokens. A secondary variant caches only
the shared instruction prefix; choices follow the passage and cannot be reused as
a prefix. CLM retains its released heads and canonical state/question format.
Four main full/no-context accuracy conditions, the cached-prefix comparison and
20 preselected option-order audits produce 1,020 test predictions. See
`results/race_context/report.html`, `summary.json`, and the raw JSONL predictions.

## System-prompt and template ablations

The same 200 test cases are rerun under seven Qwen prompt conditions: original,
short system instruction, no system message, raw text with/without the full
instruction and an `Answer:` cue, and literal template stripping with/without
the instruction. CLM is rerun as a contemporaneous cached-action reference.
Every condition has a no-passage control; each Qwen condition also has 20
preselected option-reversal audits. No Qwen prefix caching is used.

```sh
.venv/bin/python -u experiments/bench_race_ablations.py
.venv/bin/python experiments/analyze_race_ablations.py
```

The primary restriction is always one bare A/B/C/D token. A secondary diagnostic
allows a bare or single-leading-space letter token, aggregating aliases from the
same hidden state. Full-vocabulary checks on the first ten frozen cases reveal
whether template removal changes the preferred output format. These diagnostics
are outside the primary timing region and do not alter primary scores.
The 3,340 predictions, exact first-example prompts, metadata, plots and statistics
are saved under `results/race_ablations/`. All cases and conditions are fixed
before inference; these are exploratory ablations on a previously evaluated subset.

## Repeated-state games with prefix caching

`experiments/games/bench.py` compares CLM and bare-letter constrained Qwen on
identical changing game states with both prefixes cached and uncached. CLM also
caches its action vectors in both conditions. Both models use fixed rules;
Qwen additionally caches its fixed action definitions inside the system message.
No per-state planner labels, shield, or whole-request memoization is used.

The same script runs unassisted closed-loop T-Rex (five seeds) and actual NES
Super Mario Bros. (levels 1-1, 1-2, 2-1), using symbolic visible observations.
Inference is lockstep: game time pauses while the model chooses an action.
Mario runs in a separate Python 3.13 worker environment:

```sh
uv venv --python 3.13 .venv-games
uv pip install --python .venv-games/bin/python -r experiments/games/requirements-games.txt
.venv/bin/python -u experiments/games/bench.py
.venv/bin/python -u experiments/games/raw_check.py
.venv/bin/python experiments/games/analyze.py
```

Results, exact prompts, synchronized timings, complete action traces, and actual
Mario GIF replays are in `results/cached_games/`. Setup times exclude model load;
decision timings include tokenization, cache copy, backbone, and output scoring.
The CLM state sequence uses raw rules + state + final question; the Qwen sequence
uses the same rules/state/question with the standard non-thinking chat template.

## T-Rex prompt tuning with a 5% latency limit

The follow-up in `results/trex_prompt_tuning/` changes only Qwen's cached system
prompt. The selected prompt includes explicit jump-timing guidance and worked
examples, with no live planner answers or shielding. Ten diagnostic states and
one development seed were used; the prompt was frozen before a single fresh
holdout seed. Original CLM and CLM supplied the same improved instructions are
both retained as controls. The changing state, question, chat template, backbone,
and one-token A/B/C restriction remain the same.

The final prompt is `results/trex_prompt_tuning/final_system_prompt.txt`.
A reproducible run uses the persistent worker's JSON-lines input:

```sh
.venv/bin/python -u experiments/games/tune_prompt.py <<'JSONL'
{"op":"register_file","path":"results/trex_prompt_tuning/candidate_timing_v4.json"}
{"op":"play","name":"timing_v4","seed":17,"seconds":30,"tag":"reproduce"}
JSONL
```

`validate_tuned.py` runs the frozen-prompt output and latency audit;
`analyze_tuning.py` builds the report. The latency gate measures matched-state
median and paired-median steady-state latency, excluding one-time prefix setup.
Raw timings retain outliers; the report does not promise a per-call deadline.

## Same-planner T-Rex comparison

`experiments/games/planner_assisted.py` gives both models the exact released
`build_question(prompt="labeled")` state/question/candidate annotations. CLM
keeps its canonical state/question encoding and caches exact candidate texts;
Qwen caches a fixed instruction and action mapping, then reads the changing
annotations. Cache misses are timed, and entire requests are not memoized.
Ten matched cases and one 30-second course isolate model decisions with the shield
off for both. This is not the repository's full real-time, shielded configuration.
Run `experiments/games/analyze_assisted.py` to generate the report under
`results/trex_planner_assisted/`.
