from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

S = Path(__file__).resolve().parents[1] / "data" / "serving"
st.set_page_config(page_title="Payments Lakehouse", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(S / f"{name}.parquet")


daily = load("daily_metrics")
daily["txn_date"] = pd.to_datetime(daily["txn_date"])
by_type = load("daily_by_type")
by_type["txn_date"] = pd.to_datetime(by_type["txn_date"])
merch = load("merchant_stats")
ret = load("retention")
dq = load("dq_results")

st.title("Digital Payments Dashboard")
st.caption("Synthetic data. Source: Iceberg gold layer (Spark pipeline orchestrated by Airflow).")

# ---- sidebar filter ----
lo, hi = daily["txn_date"].min().date(), daily["txn_date"].max().date()
picked = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
if len(picked) == 2:
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
else:
    start, end = pd.Timestamp(lo), pd.Timestamp(hi)

d = daily[(daily.txn_date >= start) & (daily.txn_date <= end)]
bt = by_type[(by_type.txn_date >= start) & (by_type.txn_date <= end)]

# tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Merchants", "Retention", "Data quality"])
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Merchants", "Retention", "Data quality", "Segments"])

# ---- Overview ----
with tab1:
    fail = (d.txn_count * d.failure_rate).sum() / d.txn_count.sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{int(d.txn_count.sum()):,}")
    c2.metric("Successful volume (NPR)", f"{d.success_volume.sum() / 1e6:,.1f} M")
    c3.metric("Avg daily active users", f"{d.active_users.mean():,.0f}")
    c4.metric("Failure rate", f"{fail:.2%}")

    st.plotly_chart(px.line(d, x="txn_date", y="success_volume",
                            title="Daily successful volume (NPR)"),
                    width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(px.line(d, x="txn_date", y="active_users",
                              title="Daily active users"),
                      width="stretch")
    right.plotly_chart(px.line(d, x="txn_date", y="failure_rate",
                               title="Daily failure rate"),
                       width="stretch")

    mix = bt.groupby("txn_type", as_index=False)["volume"].sum()
    st.plotly_chart(px.pie(mix, names="txn_type", values="volume", hole=0.45,
                           title="Volume by transaction type"),
                    width="stretch")

# ---- Merchants ----
with tab2:
    cats = ["All"] + sorted(merch["category"].dropna().unique().tolist())
    cat = st.selectbox("Category", cats)
    m = merch if cat == "All" else merch[merch.category == cat]
    top = m.sort_values("volume", ascending=False).head(10)
    st.plotly_chart(px.bar(top, x="volume", y="merchant_name", orientation="h",
                           color="category", title="Top 10 merchants by volume (NPR)")
                    .update_layout(yaxis={"categoryorder": "total ascending"}),
                    width="stretch")
    st.dataframe(top[["merchant_name", "category", "txn_count",
                      "unique_customers", "volume"]], width="stretch")

# ---- Retention ----
with tab3:
    ret["cohort"] = pd.to_datetime(ret["cohort_month"]).dt.strftime("%Y-%m")
    heat = ret.pivot(index="cohort", columns="month_offset", values="retention")
    st.plotly_chart(px.imshow(heat, text_auto=".0%", aspect="auto",
                              color_continuous_scale="Blues",
                              labels=dict(x="Months since first transaction",
                                          y="Cohort", color="Retention"),
                              title="Monthly cohort retention"),
                    width="stretch")
    st.caption("A user's cohort is the month of their first successful transaction.")

# ---- Data quality ----
with tab4:
    latest = dq.sort_values("run_ts").groupby(["stage", "check_name"]).tail(1)
    st.metric("Checks passing (latest run)",
              f"{int(latest.passed.sum())} / {len(latest)}")
    st.dataframe(latest.sort_values(["stage", "check_name"]), width="stretch")
# ---- Segments ----
with tab5:
    seg = load("segments").copy()
    names = {0: "Regulars", 1: "Lapsed low-value", 2: "Champions", 3: "At-risk high-value"}
    seg["segment_name"] = seg["segment"].map(names)
    summary = (seg.groupby("segment_name")
               .agg(users=("user_id", "count"), volume=("monetary", "sum"),
                    avg_recency_days=("recency_days", "mean"),
                    avg_frequency=("frequency", "mean"),
                    avg_monetary=("monetary", "mean"))
               .reset_index())
    summary["pct_users"] = 100 * summary.users / summary.users.sum()
    summary["pct_volume"] = 100 * summary.volume / summary.volume.sum()
    long = summary.melt(id_vars="segment_name", value_vars=["pct_users", "pct_volume"],
                        var_name="metric", value_name="percent")
    st.plotly_chart(px.bar(long, x="segment_name", y="percent", color="metric",
                           barmode="group",
                           title="Share of users vs share of volume by segment"),
                    width="stretch")
    st.dataframe(summary.drop(columns="volume").round(1), width="stretch")