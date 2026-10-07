"""Download the exact model revisions used in this investigation."""
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt",
                revision="e939398d4556fcd9400c76fa8c5a513202f42b0a", local_dir=ROOT / "models/clm")
snapshot_download("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218",
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
                  local_dir=ROOT / "models/qwen3-8b")
