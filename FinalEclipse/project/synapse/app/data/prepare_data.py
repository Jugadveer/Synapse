import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# 📁 PATHS
# =========================

BASE_PATH = Path("resources")
DM_PATH = BASE_PATH / "Dementia"
ND_PATH = BASE_PATH / "NoDementia"

# =========================
# 📦 COLLECT DATA
# =========================

data = []

for path in DM_PATH.glob("**/*.wav"):
    data.append({
        "path": str(path),
        "label": "dementia"
    })

for path in ND_PATH.glob("**/*.wav"):
    data.append({
        "path": str(path),
        "label": "nodementia"
    })

df = pd.DataFrame(data)

print("Total samples:", len(df))
print(df["label"].value_counts())

# =========================
# 🔀 SPLIT
# =========================

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# =========================
# 💾 SAVE (IMPORTANT: comma separated)
# =========================

train_df.to_csv("app/data/train_dm.csv", index=False)
valid_df.to_csv("app/data/valid_dm.csv", index=False)

print("\n✅ Data prepared successfully!")