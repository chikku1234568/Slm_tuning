---
title: "Fin AI SLM — R&D Starter Guide"
subtitle: "Bank statement narrations → structured JSON with small language models"
date: "2 November 2026 (IST)"
---

# Fin AI SLM — R&D Starter Guide

*For data scientists new to small language models (SLMs) and fine-tuning. Fin AI team, SG Analytics / Straive, Bengaluru. Dated 2 Nov 2026 IST; all links checked live (1–2 Nov 2026 IST).*

**How to read this.** Section 1 tells you what we are building. Section 2 gives you the vocabulary. Section 3 is your week-by-week plan. Sections 4–7 are the reference shelf you come back to. A glossary sits at the end. Companion files (read after this one):

- `SLM_Similar_Repos_Research.md` — what open-source projects already did, repo by repo.
- `SLM_Narration_to_JSON_Deep_Dive.md` — the full technical deep-dive (schema, baselines, serving).
- `SLM_Narration_to_JSON_Colab_Smoke.ipynb` — the runnable notebook (data → humanize → SFT → DPO → eval → infer).
- `brainstorm-newbot/fin-ai/use-cases-slm.md` §S2 — the original use-case write-up.

**Number rule used throughout.** Every number is either cited to a source, or marked **To measure** (we will measure it ourselves) or **Unverified** (claimed somewhere but not checked by us). Please keep this rule in your own notes and slides.

---

## 1. What you're building (one page)

**Input:** one bank *statement* narration line — the short, cryptic text in the description column of a statement. Examples (synthetic):

```
NACH-DR-BAJAJFIN-EMI-0042317
UPI/DR/431209876543/PRIYA N/okaxis/payment
ACH RTN CHG 590.00 INSUFF FUNDS
REV-UPI/CR/431209876543/SWIGGY
```

**Output:** one JSON object with 10 fields:

| Field | Type of problem |
|---|---|
| `category`, `sub_category` | classification (closed lists) |
| `counterparty_name` | extract + normalise (`swiggy@icici` → `Swiggy`) |
| `counterparty_type` | person / merchant / lender / govt / self |
| `is_emi`, `is_bounce_or_return`, `is_reversal` | yes/no flags |
| `lender_name` | extract + map to our lender list, else `null` |
| `p2p_or_p2m` | p2p / p2m / not-applicable |
| `confidence` | 0–1 score (best computed, not generated — see Deep Dive §2.5) |

Scope is **statement narrations only**. SMS parsing and PDF/OCR extraction are out of scope for this track.

**Model plan.** Prototype on **Qwen2.5-0.5B-Instruct** (0.49B parameters), demo on **Qwen2.5-1.5B-Instruct** (1.54B parameters). Both are Apache-2.0 licensed with a 32,768-token context, per their model cards [R9, R10]. We fine-tune with **QLoRA** (default: Hugging Face PEFT + TRL; optional faster path: Unsloth [R8]), serve with **vLLM** [R20], optionally add a **DPO** pass, and score on valid-JSON %, schema-pass %, and per-field exact match.

**Why a small model instead of a big hosted LLM?** Three reasons, stated carefully:

1. **Data residency and privacy.** Statement narrations contain names, UPI handles and lender relationships — personal data. India's Digital Personal Data Protection framework [R40] governs how such data is processed, and RBI's 2018 directive on *Storage of Payment System Data* [R41] requires payment-system data to be stored in India. Whether a given hosted API is acceptable is a question for our compliance team, not this guide — but a model we run on our own GPU/CPU removes the question entirely. *(This is a design motivation, not legal advice.)*
2. **Cost.** A 0.5–1.5B model fine-tunes on a free Colab T4. TARizvi reports a 1.5B QLoRA run of **40.4 min on a T4** [R14]. Per-line inference cost for our setup: **To measure**.
3. **Latency and volume.** A statement can have hundreds of lines. TARizvi reports **140 ms/line on a T4** for the tuned 1.5B vs 687 ms for few-shot prompting [R14]. Our numbers: **To measure**.

**The honest caveat:** for many fields, regex + a classic classifier (TF-IDF + logistic regression) is a strong baseline. TARizvi's regex+dictionary baseline got 80.0% all-field exact match on seen formats but only 31.8% on unseen formats [R14] — the SLM's value is generalising to wording it hasn't seen. We always report the baseline next to the SLM.

---

## 2. Concepts primer

### 2.1 How a transformer generates text
A language model reads tokens (word pieces) and predicts the **next token**, one at a time; each predicted token is appended and fed back in. This loop is *autoregressive decoding* [R1, R5]. Two consequences for us: (a) output length drives latency — a 10-field JSON answer is longer than the narration itself; (b) the model can, in principle, produce *anything*, including broken JSON, unless we train or constrain it. **Greedy decoding** (always take the most likely token) is what we use for extraction: deterministic and repeatable.

### 2.2 Prompting vs supervised fine-tuning (SFT)
- **Prompting / few-shot:** you show the model instructions and a few examples at run time. No training. Works well for big models; small models follow it poorly. TARizvi's untuned 1.5B with 4-shot prompting scored 17.0% all-field exact match [R14].
- **SFT:** you train the model on thousands of (narration → correct JSON) pairs so the behaviour is built in. Prompts become short, outputs consistent. TRL's `SFTTrainer` is the standard tool [R4]. Key detail: compute the loss **only on the answer (JSON) tokens**, not the prompt [R4, R14].

### 2.3 LoRA vs QLoRA
- **Full fine-tuning** updates every weight — too heavy for a T4.
- **LoRA** [R11] freezes the original weights and learns small low-rank "adapter" matrices beside them. The paper reports up to 10,000× fewer trainable parameters and 3× less GPU memory than full fine-tuning of GPT-3 175B, with comparable quality. But the frozen base still sits in GPU memory at 16-bit.
- **QLoRA** [R12] keeps the frozen base in **4-bit (NF4)** and trains LoRA adapters on top. It introduced NF4, double quantisation and paged optimisers, and fine-tuned a 65B model on a single 48 GB GPU. For us, it means the 1.5B model plus training state fits comfortably on a 16 GB T4 with room for larger batches.
- **Rule of thumb:** use QLoRA on Colab; after training, merge adapters into a normal fp16 model for serving or GGUF export [R2, R3].

### 2.4 Why "Instruct" models (instruction / JSON priors)
Qwen ships a *base* model (raw next-token predictor) and an *Instruct* model (already trained to follow chat instructions) [R9, R10, R13]. Plain-language version: the Instruct model already "knows" how to answer a question in the requested format and has seen lots of JSON, so our fine-tune only has to teach *our* fields and labels, not basic manners. That is why it needs less data and fewer steps. Always use the model's **chat template** when formatting training and inference prompts [R7].

### 2.5 Structured output and JSON Schema
We describe the target with a **JSON Schema** (field names, types, enums, required) [R27] and check every output with the `jsonschema` Python library [R26]. Schema checks catch type/enum errors; business rules (e.g. `lender_name` must be null unless `counterparty_type == "lender"`) are checked in our own Python post-validation.

### 2.6 Constrained decoding — and its limits
Libraries such as **Outlines** [R28], **lm-format-enforcer** [R29], vLLM's structured outputs [R21] and llama.cpp GBNF grammars [R23] mask out tokens that would break the schema, so the output is *always* parseable. Important limit: **a grammar guarantees shape, not truth.** It can force `is_emi` to be a boolean; it cannot make it the *right* boolean. A public receipt benchmark found grammar improved validity but not field accuracy [R34]. Fine-tuning drives field accuracy; constraints are the safety net.

### 2.7 DPO (optional)
**Direct Preference Optimization** [R13b] trains on pairs (*chosen* good answer vs *rejected* bad answer) without a separate reward model; TRL has `DPOTrainer` [R4b]. We use it for format discipline (no markdown fences, no chatty preamble, no hallucinated merchant). It is a second stage, after SFT works.

### 2.8 Serving: vLLM vs llama.cpp/Ollama
- **vLLM** [R20]: GPU server with continuous batching and paged KV cache; the right default for a multi-user demo; supports structured outputs [R21].
- **llama.cpp / Ollama** [R22, R24]: runs quantised **GGUF** models on CPU (or GPU). Good offline/laptop backup. TARizvi reports their Q8_0 GGUF lost ≤2 exact matches per 100 vs the full model [R14] — always re-check after quantising.
- **Plain `model.generate()`** in Transformers [R6] is fine for notebooks and smoke tests.
- **In our notebook:** Component E is a real vLLM step (`USE_VLLM=True`): on Colab + GPU it installs vLLM in that cell, merges the LoRA adapter into the fp16 base (or uses the base model if training was skipped), and generates JSON with vLLM. On failure it falls back to Transformers, then regex, and prints `INFER_BACKEND=vllm|transformers|regex`.

### 2.9 Where Unsloth fits
Unsloth [R8] is a *training* accelerator, not a serving engine: it wraps 4-bit loading + LoRA (`FastLanguageModel`) with lower memory use and faster steps on a T4. In our notebook `USE_UNSLOTH=True` tries Unsloth first in Component B; on any failure it falls back to BitsAndBytes + PEFT + TRL and prints `TRAIN_BACKEND=unsloth|peft|none`. Speed-up on our data: **To measure**.

---

## 3. Week-by-week plan (≈5 weeks incl. learning + R&D buffer)

| Week | Learn (≈40%) | Build (≈60%) | Exit check |
|---|---|---|---|
| **0 (2–3 days)** | HF LLM Course ch. 1–3 [R5]; this guide §2 | Freeze **schema v1** (fields, enums, null rules); write 100 hand-labelled gold lines from real-*like* narration formats | Schema doc reviewed by team |
| **1** | Transformers generate + chat templates [R6, R7]; LoRA paper abstract/§4 [R11] | Run the smoke notebook end-to-end on Colab; extend the synthetic generator (UPI/NACH/ACH/IMPS/NEFT/ATM/REV/charges); humanize pass; build ID + **OOD split** (hold out formats and merchants, as TARizvi does [R14, R15]); regex + TF-IDF baseline | Baseline numbers on gold + OOD (**To measure**) |
| **2** | QLoRA paper §2–3 [R12]; PEFT + SFTTrainer docs [R2, R4]; Unsloth notebooks [R8] | Real QLoRA SFT on **0.5B** (≈1–2 epochs); full eval harness; error-slice table per field and per narration type | 0.5B beats baseline on OOD? (**To measure**) |
| **3** | vLLM / structured outputs [R20, R21]; jsonschema [R26] | Scale to **1.5B**; add schema-constrained inference; first serving test (vLLM on Colab, or GGUF on CPU) | Valid JSON, schema pass, per-field EM for 1.5B (**To measure**) |
| **4** | DPO paper/TRL docs [R13b, R4b] | Optional DPO pass (only if format errors remain); demo app (Colab ↔ simple local UI); latency per line | Demo dry-run with the team |
| **5 — R&D buffer** | Read failures, not papers | Fix the worst slices (usually NACH/e-mandate/ECS wording [R14]), re-run eval, write up results | Final report with every number sourced |

If week 2 stalls, the documented fallback is Qwen3-1.7B (see Deep Dive) — but change one variable at a time.

---

## 4. Must-read resources (all verified live, 1–2 Nov 2026 IST; R8/R20/R21 re-checked 2 Nov)

**Foundations**

- [R1] Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- [R5] Hugging Face LLM Course — https://huggingface.co/learn/llm-course/chapter1/1
- [R6] Transformers: generation with LLMs — https://huggingface.co/docs/transformers/llm_tutorial
- [R7] Transformers: chat templates — https://huggingface.co/docs/transformers/main/en/chat_templating

**Fine-tuning methods & libraries**

- [R11] Hu et al., *LoRA* (2021) — https://arxiv.org/abs/2106.09685
- [R12] Dettmers et al., *QLoRA* (2023) — https://arxiv.org/abs/2305.14314
- [R13b] Rafailov et al., *Direct Preference Optimization* (2023) — https://arxiv.org/abs/2305.18290
- [R2] PEFT docs — https://huggingface.co/docs/peft/index · repo https://github.com/huggingface/peft
- [R3] bitsandbytes 4-bit quantisation in Transformers — https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes · HF blog on 4-bit + QLoRA https://huggingface.co/blog/4bit-transformers-bitsandbytes
- [R4] TRL docs — https://huggingface.co/docs/trl/index · SFTTrainer https://huggingface.co/docs/trl/sft_trainer · repo https://github.com/huggingface/trl
- [R4b] TRL DPOTrainer — https://huggingface.co/docs/trl/dpo_trainer
- [R8] Unsloth docs — https://docs.unsloth.ai/ · repo https://github.com/unslothai/unsloth

**Models**

- [R9] Qwen2.5-0.5B-Instruct card — https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- [R10] Qwen2.5-1.5B-Instruct card — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- [R13] Qwen2.5 blog — https://qwenlm.github.io/blog/qwen2.5/ · technical report https://arxiv.org/abs/2412.15115

**Closest prior work (study these, steal recipes)**

- [R14] TARizvi bank-txn-parser (Qwen2.5-1.5B QLoRA) — https://huggingface.co/TARizvi/bank-txn-parser-qwen2.5-1.5b
- [R15] TARizvi dataset — https://huggingface.co/datasets/TARizvi/indian-bank-txn-parsing
- [R16] tahleels Qwen2.5-0.5B JSON extraction QLoRA+DPO — https://huggingface.co/tahleels/qwen2.5-0.5b-json-extraction-qlora-dpo
- [R17] shreyashreddyk json-extraction-qlora-dpo (baseline/SFT/DPO metrics) — https://github.com/shreyashreddyk/json-extraction-qlora-dpo
- [R18] NuExtract model — https://huggingface.co/numind/NuExtract · [R19] repo https://github.com/numindai/nuextract
- [R30] deshpandetanmay qlora-structured-extraction (eval harness) — https://github.com/deshpandetanmay/qlora-structured-extraction
- [R31] CloudAura finetune (1.5B QLoRA+DPO Colab pipeline) — https://github.com/CloudAuraOfficial/cloudaura-finetune
- [R33] implicit-none blog, Qwen2.5-0.5B SFT on T4 for JSON — https://implicit-none.com/en/sft-lora-qwen-json-api/
- [R34] sbeconis structured-output lab (grammar vs field accuracy) — https://sbeconis.com/lab/structured-output/

**Inference & structured output**

- [R20] vLLM docs — https://docs.vllm.ai/en/latest/
- [R21] vLLM structured outputs — https://docs.vllm.ai/en/latest/features/structured_outputs.html
- [R22] llama.cpp — https://github.com/ggml-org/llama.cpp
- [R23] llama.cpp GBNF grammars — https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- [R24] Ollama — https://ollama.com · importing GGUF: https://docs.ollama.com/import
- [R26] Python `jsonschema` — https://python-jsonschema.readthedocs.io/en/stable/
- [R27] JSON Schema — https://json-schema.org/
- [R28] Outlines — https://github.com/dottxt-ai/outlines
- [R29] lm-format-enforcer — https://github.com/noamgat/lm-format-enforcer

**Platform & regulation**

- [R35] Google Colab — https://colab.research.google.com/ · FAQ (GPU limits are not guaranteed) https://research.google.com/colaboratory/faq.html
- [R40] MeitY Data Protection Framework (DPDP) — https://www.meity.gov.in/data-protection-framework
- [R41] RBI, *Storage of Payment System Data* (2018) — https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11244

*Benchmark numbers in third-party repos/blogs ([R14]–[R34]) are self-reported by their authors and not reproduced by us — treat as **Unverified** until we re-run them.*

---

## 5. Colab setup checklist + using our smoke notebook

1. Open `SLM_Narration_to_JSON_Colab_Smoke.ipynb` in Colab (it is also on Drive).
2. **Runtime → Change runtime type → T4 GPU.** Free GPU availability and session length vary and are not guaranteed [R35]; save outputs to Drive.
3. Run the setup cell (transformers, peft, trl, bitsandbytes, datasets, jsonschema; Unsloth is installed there when `USE_UNSLOTH=True` on Colab). **vLLM is not installed in setup** — Component E installs it only on Colab + GPU. Restart the runtime if prompted.
4. Optional: add a Hugging Face token as a Colab *Secret* (key icon) if you push models; never paste tokens into cells.
5. Flags: `USE_UNSLOTH=True` (train backend), `USE_VLLM=True` (serving), `SKIP_GPU=1` for a CPU-only smoke. Run all. Data is **bank-statement narrations only** (no SMS). What each section does:
   - **A — Data:** template generator for statement narrations (UPI/NACH/ACH/REV-style) → labelled 10-field JSON; 70/15/15 split.
   - **A2 — Humanize:** rewrites ~30% of lines with realistic noise (delimiters, case, truncation, NACH dialect) while keeping labels consistent.
   - **B — QLoRA SFT:** Qwen2.5-0.5B-Instruct in 4-bit, **1-step smoke** train. Backend: Unsloth if enabled and working, else PEFT + TRL; check the printed `TRAIN_BACKEND`. Raise steps/epochs for a real run.
   - **B2 — DPO:** builds chosen/rejected pairs (broken JSON, flipped EMI, wrong category, hallucinated merchant) and runs a 1-step `DPOTrainer` smoke on GPU.
   - **C — Eval:** valid JSON %, schema pass %, per-field exact match.
   - **D — Infer:** greedy decoding with Transformers (regex fallback without a model).
   - **E — vLLM serving:** merge adapter → vLLM generate on demo lines → fallback Transformers/regex; check the printed `INFER_BACKEND`.
6. On CPU (`SKIP_GPU=1`) the train cells and vLLM skip, and eval uses a regex fallback — so a 100% there tests the *plumbing*, not the model.
7. Known gap to fix before real training: humanize truncation can shorten a name while the label keeps the full string; tighten the consistency check.
8. To scale: swap the model id to `Qwen/Qwen2.5-1.5B-Instruct`, raise dataset size and epochs, and keep the same eval cell so results stay comparable.

---

## 6. Eval discipline

- **Freeze test sets early.** Keep three: in-distribution synthetic, **OOD synthetic** (held-out formats + merchants), and a **small hand-labelled real-like gold set**. Never train on them.
- **Report per field, per slice.** One headline accuracy hides that, e.g., NACH/e-mandate lines fail. TARizvi's model scored 0/28 on unseen NACH wording despite 72.2% overall OOD exact match [R14].
- **Always show the baseline** (regex + TF-IDF classifier) in the same table.
- **Three core metrics:** valid JSON %, schema-pass %, per-field exact match (plus all-fields exact match). Add latency/line and memory once serving.
- **No invented numbers.** If you didn't run it, write **To measure**. If someone else claims it, cite and write **Unverified**. Record seed, model id, dataset version and commit for every result.
- **Re-evaluate after every change of form**: after merging adapters, after GGUF quantisation, after switching to vLLM/constrained decoding.

---

## 7. Common pitfalls

1. **Synthetic → real gap.** All public bank-narration datasets we found are synthetic [see Repos Research]. Models learn your templates. Mitigate with humanize noise, OOD splits, and a real-like gold set; expect a drop on real data (**To measure**).
2. **Grammar ≠ field accuracy.** Constrained decoding makes JSON parse; it doesn't make labels correct [R34].
3. **DPO can hurt.** In shreyashreddyk's results, SFT *lowered* schema pass (0.9876 → 0.8492) while raising field F1, DPO restored schema pass to 0.9982, and latency rose from 10.8 s to 16.0 s [R17]. Measure recall per field before and after DPO; drop it if it doesn't pay.
4. **Training on the prompt.** Forgetting answer-only loss wastes capacity and can hurt quality [R4].
5. **Wrong chat template** between training and inference → silent quality loss [R7].
6. **Label leakage** between generator templates in train and test → inflated scores. Split by template/merchant, not by row.
7. **Asking the model for `confidence`.** Generated confidences are poorly calibrated; compute them from token probabilities or a calibrator (Deep Dive §2.5).
8. **Changing many things at once.** Model size, data, epochs, decoding — change one, re-run eval.
9. **Colab session loss.** Save checkpoints and adapters to Drive.

---

## 8. Glossary

- **Adapter** — small trainable weights added to a frozen model (LoRA).
- **Autoregressive** — generates one token at a time, each conditioned on previous ones.
- **Base vs Instruct model** — raw pretrained model vs one further tuned to follow instructions/chat.
- **Chat template** — the exact text format (roles, special tokens) a model expects.
- **Constrained decoding** — restricting allowed next tokens so output fits a grammar/schema.
- **DPO** — Direct Preference Optimization; learns from chosen vs rejected pairs.
- **Epoch** — one pass over the training data.
- **Exact match (EM)** — prediction equals gold exactly (per field, or all fields).
- **GGUF** — file format for quantised models used by llama.cpp/Ollama.
- **Greedy decoding** — always pick the most probable next token.
- **ID / OOD** — in-distribution vs out-of-distribution test data.
- **KV cache** — stored attention state that speeds up generation.
- **LoRA** — Low-Rank Adaptation; trains small low-rank matrices instead of all weights.
- **NF4** — 4-bit NormalFloat, the quantisation data type introduced by QLoRA.
- **QLoRA** — LoRA on top of a 4-bit-quantised frozen base model.
- **Schema pass** — output validates against the JSON Schema.
- **SFT** — supervised fine-tuning on input → target pairs.
- **SLM** — small language model (here, ~0.5–2B parameters).
- **T4** — NVIDIA 16 GB GPU commonly available on free Colab.
- **Token** — a word piece; the unit models read and write.
- **vLLM** — high-throughput GPU inference server.
