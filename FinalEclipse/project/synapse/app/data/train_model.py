"""Train the audio dementia classifier.

Three things are different from the previous version, and all three change
what the numbers mean:

1. Model selection and reporting use speaker-disjoint cross-validation. The
   old script trained on a split that shared 68% of its validation speakers
   with training, so it partly measured voice recognition.

2. Probabilities are calibrated, so the number shown to a person as a
   confidence is a probability rather than an arbitrary score.

3. The decision threshold is chosen for sensitivity, not left at 0.5. At 0.5
   this model answers "No Dementia" to almost everyone: it scored 7.7%
   sensitivity, telling 24 of 26 people who had dementia that they were clear.
   For screening, a missed case is worse than a false alarm.

Run from FinalEclipse/project:

    python synapse/app/data/prepare_data.py     # speaker-disjoint split
    python synapse/app/data/train_model.py
"""

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_features import extract_features  # noqa: E402

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
SYNAPSE_DIR = DATA_DIR.parents[1]
MODELS_DIR = SYNAPSE_DIR / 'app' / 'models'

#: Of the people who have dementia, the share we want flagged.
TARGET_SENSITIVITY = 0.80

#: Two cuts rather than one. Separation on this corpus is not good enough for
#: a binary verdict to be honest: forced to choose, the model either misses
#: cases or flags most healthy people. Below the low cut it reports no
#: indication, above the high cut some indication, and in between - about two
#: thirds of recordings - it says so instead of guessing.
BAND_MIN_SENSITIVITY = 0.95   # below the low cut, few cases should be missed
BAND_MIN_SPECIFICITY = 0.90   # above the high cut, a flag should mean something
LABELS = {'nodementia': 0, 'dementia': 1}
FOLDS = 5

# The feature vector is ~1600 wide against a few hundred clips, so every
# candidate is either strongly regularised or reduced first. Tried and
# rejected on this corpus: WavLM-base-plus embeddings (auc 0.711 against
# wav2vec2's 0.747) and transcript-derived linguistic features (auc 0.49,
# i.e. chance - see the README for why).
CANDIDATES = {
    'logistic_regression_c001': lambda: Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=0.01,
                                   class_weight='balanced', random_state=42)),
    ]),
    'logistic_regression_c0003': lambda: Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=0.003,
                                   class_weight='balanced', random_state=42)),
    ]),
    'pca32_logistic_regression': lambda: Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=32, random_state=42)),
        ('clf', LogisticRegression(max_iter=5000, C=0.5,
                                   class_weight='balanced', random_state=42)),
    ]),
    'pca64_svm_rbf': lambda: Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=64, random_state=42)),
        ('clf', SVC(C=1.0, gamma='scale', class_weight='balanced',
                    probability=True, random_state=42)),
    ]),
}


# --------------------------------------------------------------- data

def load_rows(name):
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(f'{name} not found. Run prepare_data.py first.')
    with open(path, encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def speaker_of(row):
    if row.get('speaker'):
        return row['speaker']
    return Path(row['path'].replace('\\', '/')).parent.name


def build_matrix(rows, cache_name=None):
    cache = DATA_DIR / cache_name if cache_name else None
    if cache and cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if len(stored['y']) == len(rows):
            print(f'  reusing {cache.name}')
            return stored['X'], stored['y'], stored['speaker']

    X, y, groups, kept = [], [], [], 0
    for i, row in enumerate(rows, 1):
        path = SYNAPSE_DIR / row['path'].replace('\\', '/')
        features = extract_features(str(path)) if path.exists() else None
        if features is None:
            continue
        X.append(features)
        y.append(LABELS[row['label'].strip().lower()])
        groups.append(speaker_of(row))
        kept += 1
        if i % 50 == 0:
            print(f'  featurised {i}/{len(rows)}', flush=True)

    print(f'  usable clips: {kept}/{len(rows)}')
    X, y, groups = np.vstack(X), np.array(y), np.array(groups)
    if cache:
        np.savez_compressed(cache, X=X, y=y, speaker=groups)
    return X, y, groups


# --------------------------------------------------- evaluation helpers

def oof_scores(factory, X, y, groups, calibrated=False):
    """Out-of-fold probabilities, no speaker on both sides of a split.

    With calibrated=True each fold fits the same CalibratedClassifierCV that
    ships, so the scores are on the scale the deployed model produces. The
    threshold has to be chosen on that scale: picking it from raw scores and
    applying it to isotonic-calibrated output is comparing two different
    distributions, and specificity on held-out speakers collapsed from 61%
    to 29% when that was done.
    """
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    scores = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in splitter.split(X, y, groups):
        assert not (set(groups[train_idx]) & set(groups[test_idx])), 'speaker leaked'
        if calibrated:
            model = calibrated_estimator(factory, X[train_idx], y[train_idx], groups[train_idx])
        else:
            model = factory()
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return scores


def calibrated_estimator(factory, X, y, groups):
    """The estimator that ships: calibrated over speaker-disjoint inner folds."""
    inner = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    folds = list(inner.split(X, y, groups))
    return CalibratedClassifierCV(factory(), method='isotonic', cv=folds)


def confusion(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def summarise(y, scores, threshold):
    tp, fn, tn, fp = confusion(y, scores, threshold)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        'accuracy': round((tp + tn) / len(y), 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round((sensitivity + specificity) / 2, 4),
        'roc_auc': round(float(roc_auc_score(y, scores)), 4),
        'true_positives': tp, 'false_negatives': fn,
        'true_negatives': tn, 'false_positives': fp,
    }


def band_thresholds(y, scores):
    """Cuts for the no-indication and some-indication bands."""
    low, high = 0.0, 1.0
    for candidate in np.unique(scores):
        flagged = scores >= candidate
        if (y == 1).any() and flagged[y == 1].mean() >= BAND_MIN_SENSITIVITY:
            low = max(low, float(candidate))
        if (y == 0).any() and (~flagged)[y == 0].mean() >= BAND_MIN_SPECIFICITY:
            high = min(high, float(candidate))
    if high <= low:  # degenerate separation; refuse to answer at all
        low, high = 0.0, 1.0
    return low, high


def band_summary(y, scores, low, high):
    below = scores < low
    middle = (scores >= low) & (scores < high)
    above = scores >= high
    decided = below | above
    correct = int(((above & (y == 1)) | (below & (y == 0))).sum())

    return {
        'share_no_indication': round(float(below.mean()), 4),
        'share_inconclusive': round(float(middle.mean()), 4),
        'share_some_indication': round(float(above.mean()), 4),
        'dementia_rate_no_indication': round(float(y[below].mean()), 4) if below.any() else None,
        'dementia_rate_inconclusive': round(float(y[middle].mean()), 4) if middle.any() else None,
        'dementia_rate_some_indication': round(float(y[above].mean()), 4) if above.any() else None,
        'share_answered': round(float(decided.mean()), 4),
        'accuracy_when_answered': round(correct / int(decided.sum()), 4) if decided.any() else None,
        'base_rate': round(float(y.mean()), 4),
    }


def threshold_for_sensitivity(y, scores, target):
    """Highest threshold that still reaches the target sensitivity."""
    best, best_spec = 0.5, -1.0
    for candidate in np.unique(scores):
        tp, fn, tn, fp = confusion(y, scores, candidate)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= target and specificity > best_spec:
            best, best_spec = float(candidate), specificity
    return best


# ---------------------------------------------------------------- main

def main():
    # Model choice, threshold and the fit itself all use the training split
    # only, so the validation split stays a genuine holdout. Fitting on every
    # clip would make any later evaluation on valid_dm.csv meaningless.
    train_rows = load_rows('train_dm.csv')
    valid_rows = load_rows('valid_dm.csv')

    print(f'Featurising {len(train_rows)} training clips...')
    X, y, groups = build_matrix(train_rows, 'features_train.npz')
    print(f'Featurising {len(valid_rows)} validation clips...')
    Xv, yv, groups_v = build_matrix(valid_rows, 'features_valid.npz')

    leaked = set(groups) & set(groups_v)
    assert not leaked, f'speaker in both splits: {sorted(leaked)[:5]}'

    print(f'\ntrain {X.shape[0]} clips / {len(set(groups))} speakers, '
          f'{int(y.sum())} dementia')
    print(f'valid {Xv.shape[0]} clips / {len(set(groups_v))} speakers, '
          f'{int(yv.sum())} dementia\n')

    print(f'Model selection, {FOLDS}-fold speaker-disjoint CV within train:')
    results = {}
    for name, factory in CANDIDATES.items():
        scores = oof_scores(factory, X, y, groups)
        auc = roc_auc_score(y, scores)
        results[name] = (scores, auc)
        print(f'  {name:24} roc_auc {auc:.3f}')

    best_name = max(results, key=lambda n: results[n][1])
    best_scores = results[best_name][0]
    print(f'\nSelected: {best_name} (roc_auc {results[best_name][1]:.3f})')

    # Re-score out-of-fold through the calibrated estimator, then choose the
    # threshold on those scores.
    print('\nScoring out-of-fold through the calibrated estimator...')
    best_scores = oof_scores(CANDIDATES[best_name], X, y, groups, calibrated=True)

    threshold = threshold_for_sensitivity(y, best_scores, TARGET_SENSITIVITY)
    at_default = summarise(y, best_scores, 0.5)
    at_chosen = summarise(y, best_scores, threshold)

    low, high = band_thresholds(y, best_scores)
    bands = band_summary(y, best_scores, low, high)
    print(f'\n  three bands, cuts at {low:.3f} and {high:.3f}:')
    print(f'    no indication   {bands["share_no_indication"]*100:5.1f}% of clips, '
          f'{(bands["dementia_rate_no_indication"] or 0)*100:5.1f}% of them dementia')
    print(f'    inconclusive    {bands["share_inconclusive"]*100:5.1f}%')
    print(f'    some indication {bands["share_some_indication"]*100:5.1f}% of clips, '
          f'{(bands["dementia_rate_some_indication"] or 0)*100:5.1f}% of them dementia')
    print(f'    answers {bands["share_answered"]*100:.0f}% of the time, '
          f'{bands["accuracy_when_answered"]*100:.1f}% correct when it does '
          f'(base rate {bands["base_rate"]*100:.0f}%)')

    print(f'\n  at the default 0.5 threshold:')
    print(f'    sensitivity {at_default["sensitivity"]*100:5.1f}%  '
          f'specificity {at_default["specificity"]*100:5.1f}%  '
          f'balanced {at_default["balanced_accuracy"]*100:5.1f}%')
    print(f'  at the chosen {threshold:.3f} threshold:')
    print(f'    sensitivity {at_chosen["sensitivity"]*100:5.1f}%  '
          f'specificity {at_chosen["specificity"]*100:5.1f}%  '
          f'balanced {at_chosen["balanced_accuracy"]*100:5.1f}%')

    # Final model on the training split, with calibrated probabilities.
    # Calibration folds are speaker-disjoint too, or calibration would leak.
    print('\nFitting final calibrated model on the training split...')
    model = calibrated_estimator(CANDIDATES[best_name], X, y, groups)
    model.fit(X, y)

    holdout_scores = model.predict_proba(Xv)[:, 1]
    holdout = summarise(yv, holdout_scores, threshold)
    holdout_band_summary = band_summary(yv, holdout_scores, low, high)
    print(f'\n  held-out speakers ({len(yv)} clips, never trained on):')
    print(f'    sensitivity {holdout["sensitivity"]*100:5.1f}%  '
          f'specificity {holdout["specificity"]*100:5.1f}%  '
          f'balanced {holdout["balanced_accuracy"]*100:5.1f}%  '
          f'auc {holdout["roc_auc"]:.3f}')

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / 'dementia_model.pkl')

    card = {
        'model': best_name,
        'calibration': 'isotonic',
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'n_clips': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_speakers': int(len(set(groups))),
        'n_dementia_clips': int(y.sum()),
        'evaluation': f'{FOLDS}-fold speaker-disjoint cross-validation',
        'decision_threshold': round(threshold, 4),
        'target_sensitivity': TARGET_SENSITIVITY,
        'band_low': round(low, 4),
        'band_high': round(high, 4),
        'bands': bands,
        'holdout_bands': holdout_band_summary,
        'metrics_at_threshold': at_chosen,
        'metrics_at_half': at_default,
        'holdout_metrics': holdout,
        'holdout_note': (
            'Held-out speakers, never seen in training or threshold selection.'
        ),
        'majority_class_baseline': round(float(max(y.mean(), 1 - y.mean())), 4),
    }
    (MODELS_DIR / 'audio_model_card.json').write_text(
        json.dumps(card, indent=2), encoding='utf-8'
    )

    # The scaler now lives inside the pipeline; the standalone file would be
    # stale and predict.py no longer reads it.
    stale = MODELS_DIR / 'scaler.pkl'
    if stale.exists():
        stale.unlink()
        print('  removed the standalone scaler (now inside the pipeline)')

    print(f'\nWrote {MODELS_DIR / "dementia_model.pkl"}')
    print(f'Wrote {MODELS_DIR / "audio_model_card.json"}')


if __name__ == '__main__':
    main()
