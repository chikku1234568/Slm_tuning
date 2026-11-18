#!/usr/bin/env bash
# Create the `finai` model in Ollama from ./Modelfile, then smoke-test one narration through
# Ollama's OpenAI-compatible endpoint and print the latency.
#
#   ./run_ollama.sh                      # model name finai, Modelfile in this folder
#   MODEL=finai OLLAMA_URL=http://localhost:11434 ./run_ollama.sh
#   SKIP_CREATE=1 MODEL=qwen2.5:0.5b ./run_ollama.sh   # just test an already-pulled model
#
# Requires: ollama (https://ollama.com/download) running, a .gguf next to the Modelfile, curl.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-finai}"
URL="${OLLAMA_URL:-http://localhost:11434}"
NARRATION="${NARRATION:-NACH-DR-BAJAJFIN-1234567890}"

command -v ollama >/dev/null || { echo "ollama not found. Install: https://ollama.com/download" >&2; exit 1; }
curl -fsS "$URL/" >/dev/null 2>&1 || { echo "Ollama server not reachable at $URL. Start the app, or run:  ollama serve" >&2; exit 1; }

if [ -z "${SKIP_CREATE:-}" ]; then
  FROM_FILE="$(sed -n 's/^FROM[[:space:]]\{1,\}//p' "$HERE/Modelfile" | head -1)"
  [ -f "$HERE/$FROM_FILE" ] || { echo "GGUF '$FROM_FILE' not found in $HERE. Convert one first (README, step 2)." >&2; exit 1; }
  echo ">> ollama create $MODEL -f Modelfile"
  (cd "$HERE" && ollama create "$MODEL" -f Modelfile)
fi

# Same system prompt as training (the Modelfile SYSTEM carries it, but we send it explicitly too,
# exactly like serving/client.py does).
SYS='Convert the bank statement narration into JSON with keys: category, sub_category, counterparty_name, counterparty_type, is_emi, lender_name, is_bounce_or_return, is_reversal, p2p_or_p2m, confidence. Output JSON only.'
# Same JSON schema as serving/schema.py (kept inline so this script has no Python dependency).
SCHEMA='{"type":"object","additionalProperties":false,"required":["category","sub_category","counterparty_name","counterparty_type","is_emi","lender_name","is_bounce_or_return","is_reversal","p2p_or_p2m","confidence"],"properties":{"category":{"type":"string"},"sub_category":{"type":"string"},"counterparty_name":{"type":["string","null"]},"counterparty_type":{"enum":["merchant","individual","lender","employer","bank","unknown"]},"is_emi":{"type":"boolean"},"lender_name":{"type":["string","null"]},"is_bounce_or_return":{"type":"boolean"},"is_reversal":{"type":"boolean"},"p2p_or_p2m":{"enum":["P2P","P2M","NA"]},"confidence":{"type":"number","minimum":0,"maximum":1}}}'
BODY=$(cat <<JSON
{"model":"$MODEL","temperature":0,"max_tokens":200,
 "messages":[{"role":"system","content":"$SYS"},{"role":"user","content":"$NARRATION"}],
 "response_format":{"type":"json_schema","json_schema":{"name":"narration","schema":$SCHEMA}}}
JSON
)

call() { curl -sS -o /tmp/finai_resp.json -w '%{time_total}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY"; }
echo ">> warm-up call (first call loads the model into RAM; it is NOT representative latency)"
T0=$(call)
echo "   first call: ${T0}s"
echo ">> timed calls (narration: $NARRATION)"
for i in 1 2 3; do echo "   call $i: $(call)s"; done
echo ">> response:"
cat /tmp/finai_resp.json; echo
echo
echo "Latency above = wall time of one request on THIS machine (CPU, 1 request at a time). Measure with your own narrations."
echo "Next: SLM_URL=$URL SLM_MODEL=$MODEL SLM_GUIDED=ollama streamlit run ../streamlit_app.py"
