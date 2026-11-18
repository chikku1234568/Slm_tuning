#!/usr/bin/env bash
# Convert the MERGED fp16 Hugging Face model folder -> a single GGUF file for Ollama / llama.cpp.
#
#   ./convert_to_gguf.sh <merged_model_dir> [outtype] [out_dir]
#
#   merged_model_dir  folder with config.json + model.safetensors + tokenizer files
#                     (NOT the LoRA adapter folder - merge first; see serving/export_weights.py)
#   outtype           q8_0 (default) | f16 | bf16 | q4_K_M
#                     q8_0/f16/bf16 are produced directly by llama.cpp's convert_hf_to_gguf.py.
#                     q4_K_M (and other k-quants) need the `llama-quantize` tool: we convert to f16 first,
#                     then quantize. llama-quantize comes from a llama.cpp release binary or a build.
#   out_dir           where to write finai-<outtype>.gguf (default: this folder)
#
# Needs: git, python3 (+pip). For q4_K_M also: cmake + a C++ compiler (or a prebuilt llama-quantize on PATH).
set -euo pipefail

MODEL_DIR="${1:?usage: $0 <merged_model_dir> [q8_0|f16|bf16|q4_K_M] [out_dir]}"
OUTTYPE="${2:-q8_0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${3:-$HERE}"
LLAMA_DIR="${LLAMA_CPP_DIR:-$HERE/llama.cpp}"      # override with LLAMA_CPP_DIR=/path/to/existing/clone
PY="${PYTHON:-python3}"

[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: $MODEL_DIR/config.json not found - pass the merged model folder." >&2; exit 1; }
[ -f "$MODEL_DIR/adapter_config.json" ] && { echo "ERROR: $MODEL_DIR looks like a LoRA adapter; merge it into the base model first." >&2; exit 1; }
mkdir -p "$OUT_DIR"

if [ ! -d "$LLAMA_DIR" ]; then
  echo ">> cloning llama.cpp into $LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
echo ">> installing converter requirements (use a venv if you prefer: python3 -m venv .venv && . .venv/bin/activate)"
"$PY" -m pip install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"

case "$OUTTYPE" in
  q8_0|f16|bf16)
    OUT="$OUT_DIR/finai-$OUTTYPE.gguf"
    "$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$OUT" --outtype "$OUTTYPE"
    ;;
  q4_K_M|Q4_K_M|q4_k_m)
    F16="$OUT_DIR/finai-f16.gguf"
    OUT="$OUT_DIR/finai-q4_K_M.gguf"
    "$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$F16" --outtype f16
    QUANT="$(command -v llama-quantize || true)"
    [ -z "$QUANT" ] && [ -x "$LLAMA_DIR/build/bin/llama-quantize" ] && QUANT="$LLAMA_DIR/build/bin/llama-quantize"
    if [ -z "$QUANT" ]; then
      command -v cmake >/dev/null || { echo "ERROR: need llama-quantize (prebuilt: https://github.com/ggml-org/llama.cpp/releases) or cmake to build it." >&2; exit 1; }
      echo ">> building llama-quantize (one-time, a few minutes)"
      cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DLLAMA_CURL=OFF >/dev/null
      cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
      QUANT="$LLAMA_DIR/build/bin/llama-quantize"
    fi
    "$QUANT" "$F16" "$OUT" Q4_K_M
    ;;
  *) echo "ERROR: unsupported outtype '$OUTTYPE' (use q8_0, f16, bf16, q4_K_M)" >&2; exit 1 ;;
esac

echo
echo "Done: $OUT ($(du -h "$OUT" | cut -f1))"
echo "Next: put it next to the Modelfile (edit the FROM line if the name differs), then:  ./run_ollama.sh"
