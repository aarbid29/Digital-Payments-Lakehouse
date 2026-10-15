from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

ACTIVE = {"recency_days": 3, "tenure_days": 150, "freq_total": 80, "active_days": 50,
          "freq_30d": 12, "freq_60d": 22, "freq_90d": 30, "monetary_90d": 40000,
          "avg_amount": 1200, "n_merchants": 15, "transfer_share": 0.2, "kyc_level": 2}
LAPSING = {"recency_days": 70, "tenure_days": 150, "freq_total": 20, "active_days": 15,
           "freq_30d": 0, "freq_60d": 0, "freq_90d": 3, "monetary_90d": 3000,
           "avg_amount": 900, "n_merchants": 3, "transfer_share": 0.3, "kyc_level": 1}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_lapsing_user_has_higher_churn_risk_than_active_user():
    p_active = client.post("/predict/churn", json=ACTIVE).json()["churn_probability"]
    p_lapsing = client.post("/predict/churn", json=LAPSING).json()["churn_probability"]
    assert 0 <= p_active <= 1 and 0 <= p_lapsing <= 1
    assert p_lapsing > p_active


def test_invalid_input_is_rejected():
    bad = {**ACTIVE, "recency_days": -5}
    assert client.post("/predict/churn", json=bad).status_code == 422


def test_segment_champion():
    r = client.post("/segment", json={"recency_days": 5, "frequency": 44, "monetary": 38000})
    assert r.status_code == 200
    assert r.json()["segment_name"] == "Champions"