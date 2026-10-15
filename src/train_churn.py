"""Train churn model, compare to baselines, save the best."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data/serving/churn_features.parquet")

# engineered trend features
df["trend_30_vs_90"] = df["freq_30d"] / (df["freq_90d"] / 3 + 1e-9)
df["freq_per_active_day"] = df["freq_total"] / df["active_days"].clip(lower=1)

FEATURES = ["recency_days", "tenure_days", "freq_total", "active_days",
            "freq_30d", "freq_60d", "freq_90d", "monetary_90d", "avg_amount",
            "n_merchants", "transfer_share", "kyc_level",
            "trend_30_vs_90", "freq_per_active_day"]
X, y = df[FEATURES], df["churned"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(Xtr):,}  test {len(Xte):,}  churn rate {y.mean():.1%}\n")

# baseline 1: recency only
print(f"Baseline (recency only)  ROC-AUC: {roc_auc_score(yte, Xte['recency_days']):.3f}")

# baseline 2: logistic regression
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
lr.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]
print(f"Logistic regression      ROC-AUC: {roc_auc_score(yte, p_lr):.3f}  "
      f"PR-AUC: {average_precision_score(yte, p_lr):.3f}")

# main model: gradient boosting
gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=300,
                                    random_state=42)
gb.fit(Xtr, ytr)
p_gb = gb.predict_proba(Xte)[:, 1]
print(f"Gradient boosting        ROC-AUC: {roc_auc_score(yte, p_gb):.3f}  "
      f"PR-AUC: {average_precision_score(yte, p_gb):.3f}")
print(f"(random PR-AUC would be {yte.mean():.3f})")

THRESH = 0.5
print(f"\nClassification report at threshold {THRESH}:")
print(classification_report(yte, (p_gb >= THRESH).astype(int), digits=3))

imp = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
imp_df = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
          .sort_values("importance", ascending=False))
print("Permutation importance (drop in ROC-AUC):")
print(imp_df.head(8).to_string(index=False))

joblib.dump(gb, ROOT / "models/churn_model.joblib")
(ROOT / "models/churn_features.json").write_text(json.dumps(FEATURES))
print("\nsaved models/churn_model.joblib")