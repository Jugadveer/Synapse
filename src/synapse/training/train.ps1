$ErrorActionPreference = 'Stop'

Write-Host "[1/3] Installing training dependencies..."
python -m pip install -r .\requirements-training.txt

Write-Host "[2/3] Building dataset (train/eval JSONL)..."
python .\build_intent_dataset.py --out_dir .\data --samples 600 --eval_ratio 0.1

Write-Host "[3/3] Starting OUMI LoRA training..."
oumi train .\oumi_lora_qwen25_3b.yaml
