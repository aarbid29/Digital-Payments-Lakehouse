"""K-Means customer segmentation on RFM features."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
rfm = pd.read_parquet(ROOT / "data/serving/rfm.parquet")

# log-transform skewed columns, then standardize
X = pd.DataFrame({
    "recency": np.log1p(rfm["recency_days"]),
    "frequency": np.log1p(rfm["frequency"]),
    "monetary": np.log1p(rfm["monetary"]),
})
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# choose k by silhouette score (on a sample, for speed)
rng = np.random.default_rng(42)
idx = rng.choice(len(Xs), size=min(10_000, len(Xs)), replace=False)
print("k   silhouette")
scores = {}
for k in range(3, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    scores[k] = silhouette_score(Xs[idx], labels[idx])
    print(f"{k}   {scores[k]:.3f}")
best_k = max(scores, key=scores.get)
print("chosen k =", best_k)

km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
rfm["segment"] = km.labels_

profile = (rfm.groupby("segment")
           .agg(users=("user_id", "count"),
                avg_recency_days=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"))
           .round(1))
profile["pct_of_users"] = (100 * profile["users"] / len(rfm)).round(1)
profile["pct_of_volume"] = (100 * rfm.groupby("segment")["monetary"].sum()
                            / rfm["monetary"].sum()).round(1)
print("\n", profile.to_string())

rfm.to_parquet(ROOT / "data/serving/segments.parquet", index=False)
joblib.dump({"scaler": scaler, "kmeans": km}, ROOT / "models/segmentation.joblib")
print("\nsaved segments.parquet and models/segmentation.joblib")