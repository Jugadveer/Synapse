"""Fine-tune the intent router with QLoRA.

The base model is not good enough on its own. Measured against the router's
own prompts, qwen2.5:1.5b-instruct gets 4 of 8 intents right and often omits
the confidence field entirely; qwen2.5:0.5b-instruct answers memory_store to
everything, including greetings, at confidence 1.0.

Training rows come from build_intent_dataset.py, which builds them from the
same prompt templates the router sends at runtime, so the model is trained on
exactly what it will be asked.

Loss is computed on the assistant turn only. Including the prompt would spend
most of the gradient teaching the model to reproduce a fixed instruction block
it is always given anyway.

Run from this directory:

    python build_intent_dataset.py --out_dir ./data --samples 1200
    python train_router.py
"""

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Before transformers is imported anywhere. With TensorFlow installed it probes
# for a TF backend and refuses to start under Keras 3.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
DEFAULT_OUT = HERE / 'artifacts' / 'router-lora'
MAX_LENGTH = 1024
IGNORE = -100


class ChatDataset(Dataset):
    """Prompt/response pairs with the prompt masked out of the loss."""

    def __init__(self, path, tokenizer, max_length=MAX_LENGTH):
        self.rows = []
        skipped = 0

        with open(path, encoding='utf-8') as handle:
            for line in handle:
                messages = json.loads(line)['messages']
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True,
                )
                full = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                )
                if len(full) > max_length:
                    skipped += 1
                    continue

                labels = [IGNORE] * len(prompt) + full[len(prompt):]
                self.rows.append({'input_ids': full, 'labels': labels})

        if skipped:
            print(f'  skipped {skipped} rows longer than {max_length} tokens')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch, pad_id):
    width = max(len(row['input_ids']) for row in batch)
    input_ids, labels, attention = [], [], []

    for row in batch:
        padding = width - len(row['input_ids'])
        input_ids.append(row['input_ids'] + [pad_id] * padding)
        labels.append(row['labels'] + [IGNORE] * padding)
        attention.append([1] * len(row['input_ids']) + [0] * padding)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention, dtype=torch.long),
    }


def build_model(base, four_bit):
    from transformers import AutoModelForCausalLM

    kwargs = {'dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32}

    if four_bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print('  loading in 4-bit')
        except ImportError:
            print('  bitsandbytes unavailable; loading in bf16')

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model.config.use_cache = False
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=DEFAULT_BASE)
    parser.add_argument('--data_dir', default=str(HERE / 'data'))
    parser.add_argument('--out_dir', default=str(DEFAULT_OUT))
    parser.add_argument('--epochs', type=float, default=3.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--no_4bit', action='store_true')
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer, Trainer, TrainingArguments,
    )

    print(f'base: {args.base}')
    print(f'cuda: {torch.cuda.is_available()}'
          f'{" (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else ""}')

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir = Path(args.data_dir)
    print('loading data...')
    train = ChatDataset(data_dir / 'dataset_train.jsonl', tokenizer)
    evaluation = ChatDataset(data_dir / 'dataset_eval.jsonl', tokenizer)
    print(f'  {len(train)} train, {len(evaluation)} eval')

    model = build_model(args.base, not args.no_4bit)
    if not args.no_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    ))
    model.print_trainable_parameters()

    out_dir = Path(args.out_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / 'checkpoints'),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            lr_scheduler_type='cosine',
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='no',
            bf16=torch.cuda.is_available(),
            report_to=[],
            optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
        ),
        train_dataset=train,
        eval_dataset=evaluation,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    print('training...')
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f'\nadapter written to {out_dir}')
    print('next: python merge_router.py, then register the result with Ollama')


if __name__ == '__main__':
    main()
