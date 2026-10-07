"""Cross-check MLX last-token embeddings against Transformers/PyTorch MPS."""
import gc
import json
import numpy as np
import mlx.core as mx
import torch
from transformers import AutoModel, AutoTokenizer
from compare import Comparison, ROOT, BILLING

c = Comparison()
texts = ["Customer: my invoice was charged twice and nobody answers the phone!\n\nWhich team should handle this?",
         "Charges, invoices, refunds", "Bugs and outages",
         "Mario is on flat safe ground. The goal is to his right with no obstacles. Walking is preferred to jumping when safe.\n\nWhich action should Mario take to reach the goal safely?",
         "Jump up and move left", "Walk right"]
embeddings = [np.asarray(c.embed(t)[0]) for t in texts]
head = c.head
del c
gc.collect()
mx.clear_cache()
torch.set_num_threads(8)
model = AutoModel.from_pretrained(str(ROOT / "models/qwen3-8b"), dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()
tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/qwen3-8b"))
hf_embeddings = []
results = []
for text, e in zip(texts, embeddings):
    ids = tokenizer(text, return_tensors="pt").to("mps")
    with torch.inference_mode():
        h = model(**ids, use_cache=False).last_hidden_state[:, -1].float()
        h = torch.nn.functional.normalize(h, dim=-1)
    h = h.cpu().numpy()
    hf_embeddings.append(h)
    row = {"text": text, "cosine": float((e * h).sum()), "max_abs_error": float(np.max(np.abs(e-h)))}
    results.append(row)
    print(row, flush=True)
def scores(embs):
    with torch.inference_mode():
        s = head.project_states(embs[0])
        a = head.project_actions(np.concatenate(embs[1:3]))
        return torch.softmax(head.scale * (s @ a.T), dim=-1).tolist()
output = {"embedding_comparison": results, "mlx_billing_probabilities": scores(embeddings),
          "transformers_billing_probabilities": scores(hf_embeddings)}
(ROOT / "results/backbone_validation.json").write_text(json.dumps(output, indent=2))
print(output, flush=True)
