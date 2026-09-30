"""Merge the LoRA adapter into the base weights.

Ollama dropped support for loading LoRA adapters ("LoRA adapters are no longer
supported" from 0.34), so the adapter has to be folded into the model before
it can be served. The result is a normal Qwen2 checkpoint.

    python train_router.py          # writes artifacts/router-lora
    python merge_router.py          # writes artifacts/router-merged
    python to_gguf.py               # converts and registers with Ollama
"""

import argparse
import os
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
ADAPTER = HERE / 'artifacts' / 'router-lora'
MERGED = HERE / 'artifacts' / 'router-merged'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--adapter', default=str(ADAPTER))
    parser.add_argument('--out', default=str(MERGED))
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(args.adapter)
    if not (adapter / 'adapter_model.safetensors').exists():
        raise SystemExit(f'No adapter at {adapter}. Run train_router.py first.')

    print(f'base    : {args.base}')
    print(f'adapter : {adapter}')

    # Merging on CPU in fp16 keeps it off a GPU that may be busy, and the
    # result is what gets quantised anyway.
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter))

    print('merging...')
    model = model.merge_and_unload()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(out))

    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'\nwrote {out} ({size / 1e9:.1f} GB)')
    print('next: python to_gguf.py')


if __name__ == '__main__':
    main()
