---
license: mit
task_categories: [feature-extraction]
tags: [deepswe, swe-agent, embeddings, process-reward-model]
pretty_name: DeepSWE PRM evaluation embeddings (Qwen3-8B, 8k)
size_categories: [10K<n<100K]
configs:
- config_name: default
  data_files: deepswe_eval_opus5max.parquet
---

# DeepSWE PRM evaluation embeddings (Qwen3-8B, 8k)

Qwen3-8B last-token-pooled embeddings (4096-d) of every step of the DeepSWE evaluation
rollouts (mini-swe-agent, Claude-Opus-5, 4 rollouts per task): 44,409 steps, 449 rollouts,
113 tasks. Embedded with `preprocessing/deepswe/embed_shard.py` at `max_model_len 8192`.

| column | description |
|---|---|
| `trajectory_id` | rollout id |
| `step_idx` | step order within the rollout |
| `task_id` | DeepSWE task |
| `model`, `config` | policy model / rollout config |
| `reward` | rollout outcome (1 = passed) |
| `fold` | fold whose head excludes this task (`tarsur385/deepswe-prm-heads-8k`) |
| `prm_score` | per-step score of that fold's head |
| `state_embedding`, `action_embedding` | 4096-d |

```bash
python evaluation/bon_deepswe_eval.py --hf-dataset tarsur385/deepswe-prm-embeddings-8k \
    --fold-spec heads/fold_spec.json
```
