import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from extract_features import extract_features

# =========================
# LOAD DATA (IMPORTANT FIX)
# =========================

train_df = pd.read_csv("app/data/train_dm.csv")   # NO sep="\t"
valid_df = pd.read_csv("app/data/valid_dm.csv")

print("Train columns:", train_df.columns)
print("Valid columns:", valid_df.columns)

# =========================
# LABEL MAPPING
# =========================

label_map = {"nodementia": 0, "dementia": 1}

train_df["label"] = train_df["label"].map(label_map)
valid_df["label"] = valid_df["label"].map(label_map)

# =========================
# PROCESS DATA
# =========================

def process_dataframe(df):
    X, y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["path"]
        label = row["label"]

        feat = extract_features(path)

        if feat is not None:
            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = process_dataframe(train_df)
X_valid, y_valid = process_dataframe(valid_df)

print("\nTrain:", len(X_train))
print("Valid:", len(X_valid))

# =========================
# NORMALIZATION
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================

y_pred = model.predict(X_valid)

print("\n📊 Classification Report:")
print(classification_report(y_valid, y_pred))

# =========================
# SAVE
# =========================

import os
os.makedirs("app/models", exist_ok=True)

joblib.dump(model, "app/models/dementia_model.pkl")
joblib.dump(scaler, "app/models/scaler.pkl")

print("\n✅ Model saved!")