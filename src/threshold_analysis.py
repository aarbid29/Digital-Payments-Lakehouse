"""Precision/recall trade-off and top-k lift for the churn model."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data/serving/churn_features.parquet")
df["trend_30_vs_90"] = df["freq_30d"] / (df["freq_90d"] / 3 + 1e-9)
df["freq_per_active_day"] = df["freq_total"] / df["active_days"].clip(lower=1)

FEATURES = json.loads((ROOT / "models/churn_features.json").read_text())
X, y = df[FEATURES], df["churned"]
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = joblib.load(ROOT / "models/churn_model.joblib")
p = model.predict_proba(Xte)[:, 1]

print("threshold  precision  recall   f1")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (p >= t).astype(int)
    print(f"{t:>8.1f}  {precision_score(yte, pred, zero_division=0):>9.3f}  "
          f"{recall_score(yte, pred):>6.3f}  {f1_score(yte, pred):>5.3f}")

print("\nIf we contact only the top X% highest-risk users:")
order = np.argsort(-p)
yy = yte.to_numpy()
base = yy.mean()
for frac in [0.1, 0.2, 0.3, 0.5]:
    n = int(len(p) * frac)
    top = yy[order[:n]]
    print(f"top {frac:>4.0%}: captures {top.sum() / yy.sum():>5.1%} of all churners, "
          f"precision {top.mean():.1%} (vs {base:.1%} at random)")