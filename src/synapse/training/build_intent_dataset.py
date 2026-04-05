import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = "You are an intent classifier. Output strict JSON."


def to_chat_record(user_text, intent, is_fast, needs_memory):
    assistant = {
        "intent": intent,
        "is_fast": is_fast,
        "needs_memory": needs_memory,
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": json.dumps(assistant, separators=(",", ":")),
            },
        ]
    }


def build_templates():
    commands = [
        "Set a reminder for {time}",
        "Call {name} now",
        "Turn on the {device}",
        "Stop the alarm",
        "Open the {place} app",
        "Play {music}",
    ]
    memory_store = [
        "I kept my {item} in the {place}",
        "I usually drink {drink} in the evening",
        "My favorite snack is {snack}",
        "I parked the car near {landmark}",
        "My doctor appointment is on {day}",
        "I store my medicine in the {place}",
    ]
    memory_retrieve = [
        "Where did I keep my {item}?",
        "What do I drink in the evening?",
        "What is my favorite snack?",
        "Where did I park the car?",
        "When is my doctor appointment?",
        "Where do I keep my medicine?",
    ]
    unclear = [
        "I kept it there",
        "Can you do that thing",
        "You know what I mean",
        "That one, not this",
        "I forgot what I wanted",
        "Maybe later, not sure",
    ]

    slots = {
        "time": ["7 PM", "8:30 AM", "tomorrow morning", "9 tonight"],
        "name": ["Sarah", "my daughter", "Dr. Khan", "John"],
        "device": ["kitchen lights", "fan", "heater", "TV"],
        "place": ["bag", "drawer", "cabinet", "top shelf"],
        "music": ["jazz", "old songs", "Quran recitation", "piano"],
        "item": ["keys", "wallet", "glasses", "phone"],
        "drink": ["tea", "coffee", "milk", "water"],
        "snack": ["biscuits", "fruit", "peanuts", "toast"],
        "landmark": ["the pharmacy", "the big tree", "gate B", "block C"],
        "day": ["Monday", "Friday", "next Tuesday", "the 15th"],
    }
    return commands, memory_store, memory_retrieve, unclear, slots


def fill(template, slots):
    text = template
    for key, values in slots.items():
        token = "{" + key + "}"
        if token in text:
            text = text.replace(token, random.choice(values))
    return text


def generate_records(n_samples):
    commands, memory_store, memory_retrieve, unclear, slots = build_templates()

    categories = [
        ("command", True, False, commands),
        ("memory_store", False, True, memory_store),
        ("memory_retrieve", False, False, memory_retrieve),
        ("unclear", False, False, unclear),
    ]

    records = []
    while len(records) < n_samples:
        intent, is_fast, needs_memory, templates = random.choice(categories)
        template = random.choice(templates)
        user_text = fill(template, slots)
        records.append(to_chat_record(user_text, intent, is_fast, needs_memory))

    random.shuffle(records)
    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./data")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = generate_records(args.samples)
    split = int(len(records) * (1.0 - args.eval_ratio))
    train_records = records[:split]
    eval_records = records[split:]

    write_jsonl(out_dir / "dataset_train.jsonl", train_records)
    write_jsonl(out_dir / "dataset_eval.jsonl", eval_records)

    print(f"Wrote {len(train_records)} train and {len(eval_records)} eval samples to {out_dir}")


if __name__ == "__main__":
    main()
