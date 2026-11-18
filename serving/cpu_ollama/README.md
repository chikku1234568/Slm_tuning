# Fin AI SLM on a laptop CPU: Ollama + Streamlit

No GPU needed. You turn the merged fp16 model into one **GGUF** file, Ollama serves it on
`http://localhost:11434`, and the same Streamlit app / `client.py` you use with vLLM talk to it.

```
merged HF folder ──convert──▶ finai-q8_0.gguf ──ollama create──▶ Ollama server :11434 ◀── Streamlit (SLM_URL)
 (fp16 safetensors)            (+ Modelfile)                      (OpenAI-style /v1 API)
```

Files in this folder:

| File | What it is |
|---|---|
| `Modelfile` | `FROM ./finai-q8_0.gguf`, explicit Qwen2.5 ChatML `TEMPLATE`, `SYSTEM` = the exact training prompt, `temperature 0`, `num_ctx 512`, `num_predict 256`, stop `<|im_end|>` |
| `convert_to_gguf.sh` / `.ps1` | clone llama.cpp, install converter deps, merged folder → `finai-<quant>.gguf` |
| `run_ollama.sh` / `.ps1` | `ollama create finai`, then smoke-test one narration through the OpenAI-compatible endpoint and print latency |
| `check_setup.py` | self-check: Modelfile matches `schema.SYS`; if Ollama is up, checks the rendered prompt, `client.py` and the Streamlit app |

## 1. Install Ollama
Download from **https://ollama.com/download** (Windows installer, macOS app, or on Linux
`curl -fsSL https://ollama.com/install.sh | sh`). Check: `ollama --version`. It runs a local server on port 11434
(the Windows/Mac apps start it for you; otherwise run `ollama serve`). The model never leaves your machine.

## 2. Get a GGUF of your fine-tuned model
You need the **merged** fp16 model (config.json + model.safetensors + tokenizer files), **not** the LoRA adapter
(see `serving/export_weights.py`, or the merge step in `serving/colab_serve.ipynb`).

### Option A: convert on Colab (no installs on the office laptop)
1. Open `serving/colab_serve.ipynb`, run steps 1-2 (load your merged weights), then jump to **section 6**.
2. Set `RUN_GGUF = True` and run the cell. It clones llama.cpp, converts to `finai-q8_0.gguf` and offers a download.
   (Or download `/content/finai-q8_0.gguf` from the Colab Files panel / copy it to Drive.)
3. Put the `.gguf` in **this folder** (next to `Modelfile`). Ignore the minimal `Modelfile` that cell writes: use the one in this folder.

### Option B: convert locally with llama.cpp
```bash
# Linux / macOS (python3, git; use a venv if you like)
./convert_to_gguf.sh /path/to/finai-merged q8_0        # -> ./finai-q8_0.gguf
```
```powershell
# Windows PowerShell (python, git on PATH)
.\convert_to_gguf.ps1 -ModelDir C:\models\finai-merged -OutType q8_0
```
This runs llama.cpp's `convert_hf_to_gguf.py` under the hood. For `q4_K_M`, `convert_to_gguf.sh <dir> q4_K_M` converts to f16 and then runs
`llama-quantize` (it builds it with cmake if it isn't on your PATH; on Windows download a release zip from
https://github.com/ggml-org/llama.cpp/releases and run `llama-quantize finai-f16.gguf finai-q4_K_M.gguf Q4_K_M`).
Ollama itself does **not** quantize GGUF on import ([docs](https://docs.ollama.com/import)), so quantize before `ollama create`.

### q8_0 or q4_K_M?
| | q8_0 (default here) | q4_K_M |
|---|---|---|
| Size for 0.5B | ~0.5 GB | ~0.4 GB |
| Quality | essentially the fp16 model | small loss expected; for tiny models it can matter more than for big ones |
| Speed on CPU | usually fine | smaller to read from RAM, but **not automatically faster** (see numbers below) |

Start with **q8_0**. Try q4_K_M only if RAM/disk is tight, and **re-run your eval after quantizing**: field exact-match on `val` / `ood`
with the quantized model, compared with the fp16 numbers from the notebook. A 0.5B model has little spare capacity,
so do not assume quantization is free. If you change quant, edit the `FROM` line in `Modelfile`.

## 3. Create the model in Ollama and test
```bash
./run_ollama.sh                 # Linux / macOS  (or:  ollama create finai -f Modelfile)
```
```powershell
.\run_ollama.ps1                # Windows
```
`run_ollama.sh` creates `finai`, does one warm-up request (the first call loads the model; ignore its time), then three timed requests and prints the JSON.

Manual test with curl:
```bash
curl http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "finai", "temperature": 0, "max_tokens": 200,
  "messages": [
    {"role": "system", "content": "Convert the bank statement narration into JSON with keys: category, sub_category, counterparty_name, counterparty_type, is_emi, lender_name, is_bounce_or_return, is_reversal, p2p_or_p2m, confidence. Output JSON only."},
    {"role": "user", "content": "NACH-DR-BAJAJFIN-1234567890"}]}'
```
(On Windows use `curl.exe` and put the JSON in a file: `curl.exe ... -d "@req.json"`.) The system prompt must be the training prompt;
it is also baked into the Modelfile `SYSTEM`, so requests without a system message still get it.

## 4. Run Streamlit against Ollama
```bash
pip install -r serving/requirements-serving.txt
SLM_URL=http://localhost:11434 SLM_MODEL=finai SLM_GUIDED=ollama SLM_CONCURRENCY=1 streamlit run serving/streamlit_app.py
```
```powershell
$env:SLM_URL="http://localhost:11434"; $env:SLM_MODEL="finai"; $env:SLM_GUIDED="ollama"; $env:SLM_CONCURRENCY="1"
streamlit run serving/streamlit_app.py
```
Use `http://localhost:11434` (no `/v1`; the client adds it). `SLM_CONCURRENCY=1` just sets the sidebar slider's default (see CPU notes).
The sidebar shows "server up" when Ollama answers. Then **Load sample narrations → Parse**. Self-check any time: `python serving/cpu_ollama/check_setup.py`.

## 5. Structured (schema-constrained) output with Ollama
`SLM_GUIDED=ollama` sends the standard OpenAI field the Ollama docs list as supported:
```python
response_format={"type": "json_schema", "json_schema": {"name": "narration", "schema": SCHEMA}}
```
(`SLM_GUIDED=response_format` is the same request; `ollama` just also labels results `backend="ollama"`.)
Ollama's native API takes the schema in a top-level **`format`** field instead (`/api/chat` with `"format": {<schema>}`).
Docs: [Structured outputs](https://docs.ollama.com/capabilities/structured-outputs) ("Structured outputs work through the OpenAI-compatible API via `response_format`") and
[OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) (lists `response_format`, `temperature`, `max_tokens`, `stop`, `seed` for `/v1/chat/completions`).
vLLM's `structured_outputs` / `guided_json` fields do not exist in Ollama; use `SLM_GUIDED=none` for no constraint.

What we saw when we tested it (Ollama 0.40.16, base model; re-check on your version):
- Output is always parseable JSON with all required keys, and enums (`counterparty_type`, `p2p_or_p2m`) are enforced.
- **Numeric bounds are not enforced** (`confidence` came back as `99` although the schema says 0-1). The client's `jsonschema` check still flags it (the UI shows "schema ✗").
- **Keys came out in alphabetical order**, not the training order (`category, sub_category, ...`). Dicts do not care, but the model was fine-tuned on one order, so compare `SLM_GUIDED=ollama` against `SLM_GUIDED=none` on your eval set before deciding which to use.

## 6. Alternative: llama.cpp `llama-server`
Same GGUF, no Modelfile, OpenAI-compatible, and it can constrain output with a JSON schema/grammar:
```bash
python -c "import json,sys; sys.path.insert(0,'serving'); import schema; print(json.dumps(schema.SCHEMA))" > schema.json
llama-server -m finai-q8_0.gguf --alias finai -c 512 --port 8080 --chat-template chatml --json-schema-file schema.json
SLM_URL=http://localhost:8080 SLM_MODEL=finai SLM_GUIDED=none streamlit run serving/streamlit_app.py
```
(`--json-schema` / `-jf` apply one schema to every request; per-request `response_format` also works, in which case use `SLM_GUIDED=response_format`.)
Get `llama-server` from https://github.com/ggml-org/llama.cpp/releases or `brew install llama.cpp`. Check `llama-server --help` for your version's flags.
*We did not run llama-server in our test; the commands above follow its documented options.*

## 7. What to expect on CPU (measure, don't trust)
- One request at a time is the sweet spot. A CPU does not batch like a GPU, so more concurrent requests mostly means a longer queue, and in our test 4 concurrent requests were **not** faster than sequential (sometimes much slower). Hence `SLM_CONCURRENCY=1`.
- Bulk jobs (e.g. 1.5K statements) will be slow. That is where you ask for a GPU + vLLM.
- Latency depends on your CPU cores/clock, RAM speed, quant, output length, and what else is running. Ollama also unloads an idle model after about 5 minutes, so the first request after a pause is slower.

**How to measure:** `./run_ollama.sh` prints per-call wall time; the Streamlit app shows p50/p95 latency and lines/s for a batch, with concurrency 1 (use your real narrations, 50+ lines, ignore the first/warm-up call).

**Our box measurement (reference only):** a shared 8-vCPU Intel Xeon (AVX-512/AMX) Linux VM, Ollama 0.40.16 CPU-only, **base** `Qwen2.5-0.5B-Instruct`
(not your fine-tune, so quality is meaningless and it writes longer JSON, ~140 tokens, than a fine-tune should), 20 sequential requests after warm-up:
q8_0 (converted by `convert_to_gguf.sh`) p50 ~1.2 s, p95 ~1.4 s, ~95 output tokens/s; q4_K_M (Qwen's official file) p50 ~1.46 s, ~82 tokens/s, i.e. slower here than q8_0.
Your laptop will differ.

## 8. Troubleshooting
| Symptom | Fix |
|---|---|
| Streamlit says "server unreachable → regex fallback" | Ollama not running (`ollama serve`), wrong `SLM_URL` (no `/v1`), or a VPN/proxy. Test `curl http://localhost:11434/`. Fallback rows are labelled `regex_fallback`, never the model. |
| `model "finai" not found` | Run `ollama create finai -f Modelfile` (or `run_ollama.sh`), check `ollama list`, and set `SLM_MODEL=finai`. |
| `ollama create` fails: file not found | The `.gguf` must be next to `Modelfile` and match the `FROM` line. |
| Output has chatty text / never stops / odd tokens | Wrong template or stop tokens. Use this folder's `Modelfile` (explicit template + stop `<|im_end|>`). Inspect with `ollama show --modelfile finai`. Run `check_setup.py`. |
| Answers are worse than in the notebook | Different system prompt than training (keep `schema.SYS`), quantization loss (try q8_0 or f16), or `SLM_GUIDED=ollama` key order (try `none`). Re-run your eval. |
| Output cut off | Raise `num_predict` in the Modelfile (and `max_tokens` in `client.py`); `num_ctx 512` is tight if you send very long narrations. |
| Very slow / laptop freezes | Close other apps, use q8_0 on 0.5B (1.5B is ~3x heavier), `OLLAMA_NUM_PARALLEL=1`, send one line at a time. |
| `convert_to_gguf` says adapter folder | You pointed it at the LoRA adapter. Merge first (`export_weights.py`). |
| Windows: script blocked | `powershell -ExecutionPolicy Bypass -File .\run_ollama.ps1` |
| Corporate proxy blocks `git clone` / pip | Use Option A (convert on Colab), then only the `.gguf` is moved to the laptop. |
| Port 11434 busy | Another Ollama is running; or set `OLLAMA_HOST=127.0.0.1:11500` and update `SLM_URL`. |

## What was tested (and what was not)
Tested on a Linux box with Ollama on CPU: converting an HF Qwen2.5-0.5B-Instruct folder with `convert_to_gguf.sh` (q8_0); `ollama create` from this `Modelfile`; rendered prompt == expected ChatML (same token count; `check_setup.py`); `run_ollama.sh`; `client.py` in `ollama` and `none` modes; the Streamlit app via `streamlit.testing`; fallback when the server is down.
Not tested: your fine-tuned weights, Windows/macOS (the `.ps1` scripts were written, not executed), `q4_K_M` conversion via `convert_to_gguf.sh` (the q4_K_M timing used Qwen's official GGUF), `llama-server`, Option A on a real Colab.
