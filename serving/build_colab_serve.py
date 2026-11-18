"""Builds colab_serve.ipynb with nbformat (run: python serving/build_colab_serve.py)."""
import os
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Fin AI SLM — serve the fine-tuned model with vLLM (Colab T4)
Serving only. Training happens in your own notebook; this one **loads already-trained weights**,
starts a vLLM server, tests it, and exposes it with a public tunnel so the Streamlit app on your laptop can call it.

Steps: **1** install → **2** pick weights (HF repo / Drive merged folder / optional merge of an adapter) → **3** launch vLLM in background → **4** health + test call → **5** tunnel → **6** (optional) GGUF/Ollama CPU fallback.

Runtime → Change runtime type → **T4 GPU**. Concepts (KV cache, prefix caching, batching): `docs/Serving_and_KV_Cache_Explained.md`."""),
code("""# 1) Install vLLM (+ helpers). Takes a few minutes on Colab.
!pip -q install vllm openai jsonschema requests
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
md("""## 2) Choose the weights to serve
vLLM wants a **merged** model folder or HF repo (config.json + *.safetensors + tokenizer).
- `MODEL_SOURCE='hf'`: a (private) HF repo, e.g. `your-user/finai-narration-0.5b-merged`. Needs `HF_TOKEN` for private repos (Colab → 🔑 Secrets).
- `MODEL_SOURCE='drive'`: a merged folder on Google Drive.
- `MODEL_SOURCE='adapter'`: a LoRA adapter folder → merged here into the fp16 base (same as `serving/export_weights.py`).
- `MODEL_SOURCE='base'`: untuned base model, useful as a "before" baseline."""),
code("""import os
MODEL_SOURCE = 'hf'          # 'hf' | 'drive' | 'adapter' | 'base'
BASE_MODEL   = 'Qwen/Qwen2.5-0.5B-Instruct'      # switch to Qwen/Qwen2.5-1.5B-Instruct later
HF_REPO      = 'your-hf-user/finai-narration-0.5b-merged'
DRIVE_DIR    = '/content/drive/MyDrive/finai/merged_model'
ADAPTER_DIR  = '/content/drive/MyDrive/finai/adapter'
MERGED_DIR   = '/content/merged_model'

try:  # private HF repos: put HF_TOKEN in Colab Secrets
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN') or ''
except Exception:
    pass
if MODEL_SOURCE in ('drive', 'adapter'):
    from google.colab import drive; drive.mount('/content/drive')

if MODEL_SOURCE == 'hf':
    MODEL = HF_REPO
elif MODEL_SOURCE == 'drive':
    MODEL = DRIVE_DIR
elif MODEL_SOURCE == 'adapter':
    # Merge LoRA into an fp16 base (never into a 4-bit base), save safetensors, free GPU memory.
    import torch, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload().save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)
    del base; gc.collect(); torch.cuda.empty_cache()
    MODEL = MERGED_DIR
else:
    MODEL = BASE_MODEL
print('serving:', MODEL)"""),
md("""## 3) Launch the vLLM server in the background
Same flags as `serving/serve_vllm.sh` (each explained there). `nohup … &` keeps it running while we use other cells; logs go to `vllm.log`.
- `--gpu-memory-utilization 0.6`: vLLM grabs 60% of the T4 up front; after weights, the rest is the **KV-cache pool**. Look for the `KV cache` / `Maximum concurrency` lines in the log.
- `--enable-prefix-caching`: our system prompt is identical for every request, so its KV cache is computed once and reused."""),
code("""import subprocess
os.environ['MODEL'] = MODEL
!nohup vllm serve "$MODEL" --served-model-name finai --dtype half --max-model-len 512 \\
   --gpu-memory-utilization 0.6 --enable-prefix-caching --max-num-seqs 64 --port 8000 > vllm.log 2>&1 &
print('started; tail of log:')
!sleep 5; tail -n 5 vllm.log"""),
md("## 4) Wait for /health, then test a call"),
code("""import time, requests
for i in range(120):   # up to ~10 minutes (first run downloads weights + compiles)
    try:
        if requests.get('http://localhost:8000/health', timeout=2).status_code == 200:
            print('server ready after', i*5, 's'); break
    except Exception:
        pass
    time.sleep(5)
else:
    print('not ready yet -> check the log below'); 
!grep -iE "KV cache|concurrency|error" vllm.log | tail -n 8"""),
code("""# Test with the repo's client (same SYS prompt + schema as training).
!git clone -q https://github.com/chikku1234568/Slm_tuning.git /content/Slm_tuning 2>/dev/null || git -C /content/Slm_tuning pull -q
import sys, json; sys.path.insert(0, '/content/Slm_tuning/serving')
os.environ['SLM_URL'] = 'http://localhost:8000'
import client; from schema import SAMPLES
r = client.parse(SAMPLES[2]); print(r['backend'], f"{r['latency_ms']:.0f} ms", json.dumps(r['json']))
res, stats = client.parse_batch(SAMPLES * 5)   # 50 lines concurrently -> continuous batching
print(json.dumps(stats, indent=1))
print('schema ok:', sum(x['schema_ok'] for x in res), '/', len(res))
# If you get a 400 about 'structured_outputs', your vLLM is older: client.parse(..., guided='guided_json')"""),
md("""## 5) Expose it with a public tunnel (cloudflared)
Gives a temporary `https://xxxx.trycloudflare.com` URL. On your laptop:
`SLM_URL=https://xxxx.trycloudflare.com streamlit run serving/streamlit_app.py`

⚠️ The URL is public while the tunnel runs. For anything beyond a quick demo add `--api-key SECRET` to `vllm serve` and set `SLM_API_KEY=SECRET` on the client.
Alternative: **ngrok** (`pip install pyngrok`, `ngrok.set_auth_token(...)`, `ngrok.connect(8000)`) — needs a free ngrok account."""),
code("""!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:8000 > cloudflared.log 2>&1 &
import re, time
url = None
for _ in range(30):
    time.sleep(2)
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('cloudflared.log').read())
    if m: url = m.group(0); break
print('PUBLIC URL:', url)
print(f'On your laptop:  SLM_URL={url} streamlit run serving/streamlit_app.py')"""),
md("""## 6) (Optional) CPU fallback: GGUF + Ollama
For demos without a GPU. Run on any machine with the **merged** model folder. Slower than vLLM on GPU — measure it.
Docs: https://docs.ollama.com/import · llama.cpp: https://github.com/ggml-org/llama.cpp

Ollama also serves an OpenAI-compatible API at `http://localhost:11434/v1`, so the same client works:
`SLM_URL=http://localhost:11434 SLM_MODEL=finai SLM_GUIDED=ollama streamlit run serving/streamlit_app.py`

**Full step-by-step (Modelfile with the correct Qwen ChatML template, run scripts, troubleshooting): `serving/cpu_ollama/README.md`.**
Flip `RUN_GGUF = True` below, run the cell, download `finai-q8_0.gguf`, and put it next to `serving/cpu_ollama/Modelfile` (use that Modelfile, not the minimal one this cell writes)."""),
code("""RUN_GGUF = False   # outline only; flip to True to try on Colab
if RUN_GGUF:
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
    !pip -q install -r /content/llama.cpp/requirements.txt
    # Convert merged HF model -> GGUF (q8_0 is a safe quality/size tradeoff; f16 = no quantization)
    !python /content/llama.cpp/convert_hf_to_gguf.py {MODEL if os.path.isdir(MODEL) else MERGED_DIR} --outfile /content/finai-q8_0.gguf --outtype q8_0
    # Modelfile for Ollama (temperature 0 for deterministic extraction)
    open('/content/Modelfile','w').write('FROM /content/finai-q8_0.gguf\\nPARAMETER temperature 0\\n')
    print('Download /content/finai-q8_0.gguf, put it next to serving/cpu_ollama/Modelfile, then run serving/cpu_ollama/run_ollama.sh (or .ps1)')
    try:
        from google.colab import files; files.download('/content/finai-q8_0.gguf')   # ~0.5 GB for 0.5B
    except Exception as e: print('download manually from the Files panel:', e)"""),
md("""## Stop
`!pkill -f "vllm serve"; pkill -f cloudflared` — Colab also stops everything when the runtime disconnects."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_serve.ipynb")
nbf.write(nb, out)
print("wrote", out)
