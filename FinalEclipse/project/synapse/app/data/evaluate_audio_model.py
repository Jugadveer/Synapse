"""Evaluate the audio dementia classifier on the held-out validation split.

Run from the repository root:

    python FinalEclipse/project/synapse/app/data/evaluate_audio_model.py

Accuracy on its own is misleading here because the classes are unbalanced.
For a screening tool the number that matters is sensitivity: of the people who
do have dementia, how many does it flag? A model that says "No Dementia" to
everybody scores well on accuracy and is worth nothing.

valid_dm.csv is a genuine holdout: prepare_data.py splits by speaker and
train_model.py fits on the training split only, so no voice here was seen
during training, model selection or threshold selection.
"""

import csv
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[5]
PROJECT_DIR = REPO_ROOT / 'FinalEclipse' / 'project'
SRC_APP_DIR = REPO_ROOT / 'src' / 'synapse'
for path in (str(PROJECT_DIR), str(SRC_APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

import django  # noqa: E402

django.setup()

from synapse.app.data.predict import predict_audio  # noqa: E402

SYNAPSE_DIR = PROJECT_DIR / 'synapse'
LABELS = {'dementia': 'Dementia', 'nodementia': 'No Dementia'}


def evaluate(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))

    tp = tn = fp = fn = skipped = 0
    for row in rows:
        path = SYNAPSE_DIR / row['path'].replace('\\', '/')
        if not path.exists():
            skipped += 1
            continue

        prediction, _ = predict_audio(str(path))
        if prediction is None:
            skipped += 1
            continue

        truth = LABELS[row['label'].strip().lower()]
        if truth == 'Dementia':
            tp += prediction == 'Dementia'
            fn += prediction != 'Dementia'
        else:
            tn += prediction == 'No Dementia'
            fp += prediction != 'No Dementia'

    return tp, tn, fp, fn, skipped


def report(tp, tn, fp, fn, skipped):
    total = tp + tn + fp + fn
    if not total:
        print('No usable rows; is resources/ present?')
        return 1

    positives, negatives = tp + fn, tn + fp
    accuracy = 100 * (tp + tn) / total
    baseline = 100 * max(positives, negatives) / total

    print(f'Held-out evaluation: {total} rows ({skipped} skipped)\n')
    print(f'  accuracy                       {accuracy:5.1f}%')
    print(f'  majority-class baseline        {baseline:5.1f}%')
    print(f'  sensitivity (dementia caught)  {100 * tp / positives:5.1f}%   [{tp}/{positives}]')
    print(f'  specificity (healthy cleared)  {100 * tn / negatives:5.1f}%   [{tn}/{negatives}]')
    print(f'\n  confusion: TP={tp} FN={fn} TN={tn} FP={fp}')

    balanced = 100 * ((tp / positives) + (tn / negatives)) / 2
    print(f'\n  balanced accuracy              {balanced:5.1f}%')
    print('  (raw accuracy sits below the majority baseline by design: the')
    print('   threshold is set for sensitivity, so healthy people are flagged')
    print('   more often. Balanced accuracy and AUC are the honest summaries.)')

    if balanced < 55:
        print(f'\n  WARNING: balanced accuracy {balanced:.1f}% is close to chance.')
    if 100 * tp / positives < 70:
        print(
            f'  WARNING: {fn} of {positives} people with dementia were told they were clear.'
        )
    return 0


if __name__ == '__main__':
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'valid_dm.csv'
    raise SystemExit(report(*evaluate(csv_path)))
