# Qwen Intent Fine-Tuning (OUMI + LoRA)

This folder provides a complete starter pipeline for intent fine-tuning with labels:
- command
- memory_store
- memory_retrieve
- unclear

Target output JSON:
{"intent":"memory_store","is_fast":false,"needs_memory":true}

## 1) Install deps

From this folder:

python -m pip install -r requirements-training.txt

## 2) Build dataset (200 min, 600 default)

python build_intent_dataset.py --out_dir ./data --samples 600 --eval_ratio 0.1

Outputs:
- data/dataset_train.jsonl
- data/dataset_eval.jsonl

Each row is OUMI chat format:
{
  "messages": [
    {"role":"system","content":"You are an intent classifier. Output strict JSON."},
    {"role":"user","content":"I kept my keys in my bag"},
    {"role":"assistant","content":"{\"intent\":\"memory_store\",\"is_fast\":false,\"needs_memory\":true}"}
  ]
}

## 3) Train LoRA

oumi train ./oumi_lora_qwen25_3b.yaml

## 4) Merge adapter

oumi merge-lora --base Qwen/Qwen2.5-3B-Instruct --lora ./artifacts/qwen25-3b-intent-lora --output ./artifacts/qwen25-3b-intent-merged

## 5) Optional helper scripts

PowerShell:
- ./train.ps1
- ./merge_lora.ps1

## Notes

- LoRA config uses small memory settings (batch_size=2, grad_accum=4).
- Start with 600 samples; move to 1000+ after manual cleanup.
- For best results, add real utterances from your logs and rebalance ambiguous cases.
