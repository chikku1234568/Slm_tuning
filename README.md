# Slm_tuning

Starter kit for fine-tuning a small language model (SLM) to turn **Indian bank-statement narrations** (UPI / NACH / ACH / reversal-style lines) into structured JSON. Statement narrations only — no SMS.

## Contents
- `SLM_Narration_to_JSON_Colab_Smoke.ipynb` — component-wise smoke notebook: A data · A2 humanize · B QLoRA SFT · B2 DPO · C eval · D infer · E vLLM serving.
- `docs/SLM_RnD_Starter_Guide.md` (+ `.docx`) — R&D starter guide for people new to SLMs / fine-tuning, with cited resources.

## Open in Colab
1. Go to https://colab.research.google.com → **File → Open notebook → GitHub**, paste this repo URL, pick the notebook (or use the `colab.research.google.com/github/chikku1234568/Slm_tuning/blob/main/SLM_Narration_to_JSON_Colab_Smoke.ipynb` URL).
2. **Runtime → Change runtime type → T4 GPU**, then **Run all**.
3. Flags in the notebook: `USE_UNSLOTH=True` (try Unsloth, fall back to PEFT + TRL), `USE_VLLM=True` (vLLM serving in Component E, installed only on Colab + GPU), `SKIP_GPU=1` for a CPU-only plumbing check.

Training steps are **1-step smoke tests**, not real fine-tunes. No model-quality metrics are claimed; on CPU, eval uses a regex fallback and only validates plumbing. Data is synthetic.

## Model & license
Default base: `Qwen/Qwen2.5-0.5B-Instruct` (scale to `Qwen2.5-1.5B-Instruct`). Qwen2.5 0.5B/1.5B Instruct are Apache-2.0 per their model cards. Code in this repo is released under Apache-2.0 (`LICENSE`).
