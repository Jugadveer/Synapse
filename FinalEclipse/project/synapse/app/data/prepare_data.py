"""Build train/validation splits for the audio classifier.

Splits are grouped by speaker. The previous version called train_test_split
stratified on the label alone, which put 68% of the validation speakers into
training as well - so the evaluation partly measured whether the model
recognised a familiar voice rather than whether it detected the condition.

Run from FinalEclipse/project:

    python synapse/app/data/prepare_data.py
"""

import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

SYNAPSE_DIR = Path(__file__).resolve().parents[2]
RESOURCES = SYNAPSE_DIR / 'resources'
OUT_DIR = Path(__file__).resolve().parent

CLASSES = {'Dementia': 'dementia', 'NoDementia': 'nodementia'}
VALIDATION_FRACTION = 0.2
SEED = 42


def collect():
    """Every clip, with the speaker it belongs to."""
    rows = []
    for folder, label in CLASSES.items():
        for path in sorted((RESOURCES / folder).glob('**/*.wav')):
            rows.append({
                'path': str(path.relative_to(SYNAPSE_DIR)),
                'label': label,
                'speaker': path.parent.name,
            })
    return rows


def split(rows):
    """Speaker-disjoint split, keeping the class balance in both halves."""
    labels = [r['label'] for r in rows]
    speakers = [r['speaker'] for r in rows]

    folds = max(int(round(1 / VALIDATION_FRACTION)), 2)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(splitter.split(rows, labels, speakers))

    train = [rows[i] for i in train_idx]
    valid = [rows[i] for i in valid_idx]

    overlap = {r['speaker'] for r in train} & {r['speaker'] for r in valid}
    assert not overlap, f'speaker leaked into both splits: {sorted(overlap)[:5]}'
    return train, valid


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'speaker'])
        writer.writeheader()
        writer.writerows(rows)


def describe(name, rows):
    labels = Counter(r['label'] for r in rows)
    speakers = {r['speaker'] for r in rows}
    print(f'  {name:6} {len(rows):4} clips  {len(speakers):4} speakers  {dict(labels)}')


def main():
    rows = collect()
    print(f'Total clips: {len(rows)}')
    print(f'Total speakers: {len({r["speaker"] for r in rows})}')
    print(Counter(r['label'] for r in rows))
    print()

    train, valid = split(rows)
    describe('train', train)
    describe('valid', valid)

    write(OUT_DIR / 'train_dm.csv', train)
    write(OUT_DIR / 'valid_dm.csv', valid)
    print(f'\nWrote {OUT_DIR / "train_dm.csv"} and {OUT_DIR / "valid_dm.csv"}')
    print('No speaker appears in both.')


if __name__ == '__main__':
    main()
