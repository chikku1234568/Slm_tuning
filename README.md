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

## Serving (vLLM + Streamlit)
Serve the trained model at low latency and call it from a Streamlit UI. Concepts (prefill/decode, KV cache, PagedAttention, continuous batching, prefix caching, structured outputs, scaling): [`docs/Serving_and_KV_Cache_Explained.md`](docs/Serving_and_KV_Cache_Explained.md).

`serving/`:
- `schema.py` — the exact training `SYS` prompt, JSON `SCHEMA`, `FIELDS`, sample narrations.
- `export_weights.py` — optional helper: merge a LoRA adapter into the fp16 base, save safetensors, push adapter + merged model to private HF repos (optional W&B Artifact).
- `serve_vllm.sh` — `vllm serve` with every flag explained.
- `client.py` — `parse(text)` / `parse_batch(lines)`: OpenAI-compatible, temperature 0, schema-guided decoding, jsonschema check, latency stats, regex fallback if the server is down.
- `streamlit_app.py` — demo UI (one or many lines, JSON per line, schema badges, p50/p95 latency, throughput).
- `colab_serve.ipynb` — Colab T4: load merged weights (HF / Drive / merge an adapter) → launch vLLM → test → cloudflared public URL; GGUF/Ollama CPU fallback outline. (Built by `build_colab_serve.py`.)
- `requirements-serving.txt` — client-side deps.
- `cpu_ollama/` — **no-GPU path: CPU + Ollama** (GGUF conversion scripts, `Modelfile` with explicit Qwen ChatML template, `run_ollama.sh/.ps1`, `check_setup.py`, README with install / quantization / structured-output / troubleshooting). See [`serving/cpu_ollama/README.md`](serving/cpu_ollama/README.md).

Quick start:
1. Train in your own notebook; save the LoRA adapter (e.g. to Drive).
2. Export: `HF_TOKEN=... python serving/export_weights.py --adapter <adapter_dir> --base Qwen/Qwen2.5-0.5B-Instruct --adapter-repo <user>/finai-0.5b-lora --merged-repo <user>/finai-0.5b-merged` (or let `colab_serve.ipynb` merge it).
3. Open `serving/colab_serve.ipynb` in Colab (T4), set `MODEL_SOURCE` / `HF_REPO`, Run all, copy the printed `https://….trycloudflare.com` URL.
4. On your laptop: `pip install -r serving/requirements-serving.txt` then `SLM_URL=https://….trycloudflare.com streamlit run serving/streamlit_app.py`.

**No GPU?** Use CPU + Ollama: [`serving/cpu_ollama/README.md`](serving/cpu_ollama/README.md). Same Streamlit app: `SLM_URL=http://localhost:11434 SLM_MODEL=finai SLM_GUIDED=ollama streamlit run serving/streamlit_app.py` (Ollama takes the schema via OpenAI-style `response_format` json_schema).

Guided decoding uses vLLM's current `extra_body={"structured_outputs": {"json": SCHEMA}}`; for vLLM < 0.12 set `SLM_GUIDED=guided_json`. The tunnel URL is public — add `--api-key` to `vllm serve` (and `SLM_API_KEY` on the client) for anything beyond a quick demo.
