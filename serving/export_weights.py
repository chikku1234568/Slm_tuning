"""Optional standalone helper: merge a LoRA adapter into its base model and push to HF Hub.

Works with any PEFT/LoRA adapter folder (whatever your training notebook saved with
`trainer.save_model(...)` / `model.save_pretrained(...)`). Does not touch training code.

    export HF_TOKEN=hf_...            # a write token from https://huggingface.co/settings/tokens
    python serving/export_weights.py \
        --adapter /content/drive/MyDrive/finai/adapter \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --out /content/merged_model \
        --adapter-repo your-hf-user/finai-narration-0.5b-lora \
        --merged-repo  your-hf-user/finai-narration-0.5b-merged

Adapter vs merged — keep BOTH:
  * Adapter (LoRA weights only): a few MB to tens of MB. This is the real "training output".
    Cheap to store/version, easy to diff experiments, needs the exact base model to load.
    vLLM can serve it directly with --enable-lora (small runtime overhead; many adapters
    can share one base on one GPU).
  * Merged (base + LoRA folded in, fp16): about the size of the base model (~1 GB for 0.5B,
    ~3 GB for 1.5B in fp16). Plain model -> simplest and fastest to serve with vLLM,
    and the input for AWQ / GGUF conversion. Re-create it any time from base + adapter.
  Weights are saved as .safetensors (safe to load, no pickle code execution, memory-mapped/fast).

Train in 4-bit (QLoRA) but MERGE into an fp16 base: merging into a 4-bit-quantized base
degrades weights. So we reload the base in fp16 here.

W&B (optional, --wandb-project): logs the adapter folder as a W&B Artifact for lineage
(which run / data / config produced which weights). HF Hub stays the place you serve from.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="path to the saved LoRA adapter folder")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct", help="base model the adapter was trained on")
    ap.add_argument("--out", default="merged_model", help="local folder for the merged fp16 model")
    ap.add_argument("--adapter-repo", help="HF repo id for the adapter, e.g. user/finai-0.5b-lora")
    ap.add_argument("--merged-repo", help="HF repo id for the merged model, e.g. user/finai-0.5b-merged")
    ap.add_argument("--public", action="store_true", help="create public repos (default: private)")
    ap.add_argument("--wandb-project", help="optional: also log the adapter as a W&B Artifact")
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1) Load base in fp16 (not 4-bit!) and apply the adapter.
    print("loading base", a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, a.adapter)
    # 2) Fold W + (B @ A) * scale into the base weights; result is a normal HF model.
    merged = model.merge_and_unload()
    # 3) Save as safetensors + tokenizer (vLLM needs both in the same folder).
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    print("merged model saved to", a.out)

    # 4) Push to Hugging Face Hub (private by default).
    token = os.environ.get("HF_TOKEN")
    if a.adapter_repo or a.merged_repo:
        if not token:
            raise SystemExit("Set HF_TOKEN (write token) to push to the Hub.")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        for repo, folder in ((a.adapter_repo, a.adapter), (a.merged_repo, a.out)):
            if not repo:
                continue
            api.create_repo(repo, private=not a.public, exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=folder, commit_message=f"upload from {folder}")
            print("pushed", folder, "->", f"https://huggingface.co/{repo}")

    # 5) Optional lineage in W&B (adapter only; it's small).
    if a.wandb_project:
        import wandb
        run = wandb.init(project=a.wandb_project, job_type="export")
        art = wandb.Artifact("finai-lora-adapter", type="model",
                             metadata={"base": a.base, "hf_adapter_repo": a.adapter_repo, "hf_merged_repo": a.merged_repo})
        art.add_dir(a.adapter)
        run.log_artifact(art)
        run.finish()
        print("logged adapter to W&B project", a.wandb_project)


if __name__ == "__main__":
    main()
