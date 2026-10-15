"""Generate synthetic digital-payments data (users, merchants, transactions)."""
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---- knobs: to scale up/down ----
N_USERS = 50_000
N_MERCHANTS = 2_000
N_TXN = 1_000_000         
CHUNK = 500_000
SEED = 42
# ----------------------------------------------
START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2026-09-30")
CHURN_CUTOFF = pd.Timestamp("2026-07-01")

rng = np.random.default_rng(SEED)
fake = Faker()
Faker.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
(OUT / "transactions").mkdir(parents=True, exist_ok=True)

CITIES = ["Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Biratnagar",
          "Birgunj", "Butwal", "Dharan", "Hetauda", "Nepalgunj"]
CATEGORIES = ["Grocery", "Restaurant", "Pharmacy", "Fuel", "Electronics",
              "Clothing", "Education", "Transport", "Utilities", "Entertainment"]
TXN_TYPES = np.array(["QR_PAYMENT", "BILL_PAY", "TOP_UP", "TRANSFER"])
TYPE_PROBS = [0.45, 0.15, 0.20, 0.20]
# typical amount (NPR) per type, as log-mean of a lognormal
TYPE_MU = {"QR_PAYMENT": 6.0, "BILL_PAY": 6.8, "TOP_UP": 5.5, "TRANSFER": 7.0}

EPOCH = pd.Timestamp("1970-01-01")


def to_epoch_s(idx):
    """Convert a DatetimeIndex or Timestamp to seconds since 1970 (float)."""
    return np.asarray((idx - EPOCH) / pd.Timedelta(seconds=1), dtype="float64")


def make_users():
    signup = START + pd.to_timedelta(rng.integers(0, 200, N_USERS), unit="D")
    churned = rng.random(N_USERS) < 0.30
    span_days = (CHURN_CUTOFF - signup).days.to_numpy()
    churn_end = signup + pd.to_timedelta(rng.uniform(30, span_days), unit="D")

    s_sec = to_epoch_s(signup)
    e_sec = np.where(churned, to_epoch_s(churn_end), float(to_epoch_s(END)))

    users = pd.DataFrame({
        "user_id": [f"U{i:06d}" for i in range(1, N_USERS + 1)],
        "name": [fake.name() for _ in range(N_USERS)],
        "city": rng.choice(CITIES, N_USERS),
        "kyc_level": rng.choice([1, 2, 3], N_USERS, p=[0.3, 0.5, 0.2]),
        "signup_date": signup.date,
    })
    # activity weights: a few heavy users, many light users
    w = rng.lognormal(0, 1, N_USERS)
    w = w / w.sum()
    return users, s_sec, e_sec, w

def make_merchants():
    return pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in range(1, N_MERCHANTS + 1)],
        "merchant_name": [fake.company() for _ in range(N_MERCHANTS)],
        "category": rng.choice(CATEGORIES, N_MERCHANTS),
        "city": rng.choice(CITIES, N_MERCHANTS),
    })


def make_chunk(n, start_idx, s_sec, e_sec, w):
    uidx = rng.choice(N_USERS, size=n, p=w)
    ts_sec = s_sec[uidx] + (e_sec[uidx] - s_sec[uidx]) * rng.random(n)

    types = rng.choice(TXN_TYPES, size=n, p=TYPE_PROBS)
    mu = pd.Series(types).map(TYPE_MU).to_numpy()
    amount = np.round(rng.lognormal(mu, 0.9), 2)

    has_merchant = np.isin(types, ["QR_PAYMENT", "BILL_PAY"])
    merch_ids = np.array([f"M{i:04d}" for i in range(1, N_MERCHANTS + 1)], dtype=object)
    merchant = np.where(has_merchant, rng.choice(merch_ids, n), None)

    df = pd.DataFrame({
        "txn_id": "T" + pd.Series(np.arange(start_idx, start_idx + n)).astype(str).str.zfill(9),
        "user_id": [f"U{i + 1:06d}" for i in uidx],
        "merchant_id": merchant,
        "txn_type": types,
        "amount": amount,
        "status": rng.choice(["SUCCESS", "FAILED", "PENDING"], n, p=[0.94, 0.05, 0.01]),
        "txn_ts": pd.to_datetime(ts_sec, unit="s"),
    })

    # ---- inject bad data (for the data-quality step) ----
    null_mask = rng.random(n) < 0.003
    df.loc[null_mask, "amount"] = np.nan
    neg_mask = rng.random(n) < 0.001
    df.loc[neg_mask, "amount"] = -df.loc[neg_mask, "amount"].abs()
    dups = df.sample(frac=0.01, random_state=int(start_idx % 1000))
    return pd.concat([df, dups], ignore_index=True)

if __name__ == "__main__":
    users, s_ns, e_ns, w = make_users()
    users.to_csv(OUT / "users.csv", index=False)
    make_merchants().to_csv(OUT / "merchants.csv", index=False)
    print("users + merchants written")

    written, part = 0, 0
    while written < N_TXN:
        n = min(CHUNK, N_TXN - written)
        chunk = make_chunk(n, written + 1, s_ns, e_ns, w)
        chunk.to_parquet(OUT / "transactions" / f"part_{part:03d}.parquet",
                         index=False, coerce_timestamps="us",
                         allow_truncated_timestamps=True)
        written += n
        part += 1
        print(f"wrote chunk {part}: {written:,}/{N_TXN:,}")
    print("done")