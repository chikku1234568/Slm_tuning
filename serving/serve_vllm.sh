#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for the Fin AI narration -> JSON SLM.
#
#   bash serving/serve_vllm.sh <model>            # model = HF repo id or local folder
#   bash serving/serve_vllm.sh your-hf-user/finai-narration-0.5b-merged
#   bash serving/serve_vllm.sh /content/merged_model
#
# <model> must be a MERGED model (LoRA already folded into the base, see export_weights.py),
# or the base model plus --enable-lora/--lora-modules (see note at the bottom).
# Needs a GPU (Colab T4 is fine). Endpoints: http://localhost:8000/v1/chat/completions, /health, /metrics
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"   # default = untuned base, handy for a "before" baseline
PORT="${PORT:-8000}"

# Flag-by-flag:
#  --served-model-name finai     Name clients send in "model": ...; lets you swap weights without
#                                changing client code.
#  --dtype half                  fp16. The T4 (Turing) has no bfloat16 support, Qwen's default dtype.
#  --max-model-len 512           Max tokens per request (prompt + output). Our system prompt + one
#                                narration + JSON is well under 512. Smaller = less KV cache reserved
#                                per sequence, so more sequences fit at once.
#  --gpu-memory-utilization 0.6  vLLM pre-allocates this fraction of GPU memory up front:
#                                weights + activations, and EVERYTHING LEFT becomes the KV-cache pool
#                                (paged blocks). 0.6 leaves room on a shared Colab GPU; raise it
#                                (0.85-0.9) on a dedicated GPU for a bigger KV pool = more concurrency.
#  --enable-prefix-caching       Reuse KV blocks for identical prompt prefixes. Every request starts
#                                with the same SYS prompt, so its prefill is computed once and reused.
#                                (On by default in recent vLLM; kept explicit for teaching.)
#  --max-num-seqs 64             Max sequences batched together in one step (continuous batching).
#                                Higher = more throughput but each request may wait a bit more.
#                                Bounded in practice by KV-cache pool size.
#  --port                        HTTP port.
exec vllm serve "$MODEL" \
  --served-model-name finai \
  --dtype half \
  --max-model-len 512 \
  --gpu-memory-utilization 0.6 \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --port "$PORT"

# Optional extras:
#  --api-key SECRET              require "Authorization: Bearer SECRET" (set SLM_API_KEY in client).
#                                Do this before exposing the server through a public tunnel.
#  --quantization awq            if you serve an AWQ-quantized checkpoint (less memory, often faster).
#  Serving the ADAPTER without merging:
#    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora --lora-modules finai=/path/to/adapter ...
#    (clients then use model="finai"; slightly slower than merged, but many adapters can share one base)
