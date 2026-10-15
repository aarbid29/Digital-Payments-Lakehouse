"""Build churn training table. Features use data BEFORE the cutoff only."""
from pathlib import Path

from pyspark.sql import functions as F

from spark_session import get_spark

OUT = Path(__file__).resolve().parents[1] / "data" / "serving"
CUTOFF = "2026-07-31"


def run():
    spark = get_spark("churn_features")
    c = F.lit(CUTOFF).cast("date")
    ok = spark.table("lake.silver.transactions").filter("is_success = 1")

    pre = ok.filter(F.col("txn_date") < c)
    feats = pre.groupBy("user_id").agg(
        F.datediff(c, F.max("txn_date")).alias("recency_days"),
        F.datediff(c, F.min("txn_date")).alias("tenure_days"),
        F.count("*").alias("freq_total"),
        F.countDistinct("txn_date").alias("active_days"),
        F.sum(F.when(F.col("txn_date") >= F.date_sub(c, 30), 1).otherwise(0)).alias("freq_30d"),
        F.sum(F.when(F.col("txn_date") >= F.date_sub(c, 60), 1).otherwise(0)).alias("freq_60d"),
        F.sum(F.when(F.col("txn_date") >= F.date_sub(c, 90), 1).otherwise(0)).alias("freq_90d"),
        F.sum(F.when(F.col("txn_date") >= F.date_sub(c, 90), F.col("amount")).otherwise(0)).alias("monetary_90d"),
        F.avg("amount").alias("avg_amount"),
        F.countDistinct("merchant_id").alias("n_merchants"),
        F.avg(F.when(F.col("txn_type") == "TRANSFER", 1).otherwise(0)).alias("transfer_share"),
    ).filter("freq_90d > 0")          # only users active in last 90 days

    # label: no successful txn in the 60 days after the cutoff
    post = (ok.filter((F.col("txn_date") >= c) & (F.col("txn_date") < F.date_add(c, 60)))
            .groupBy("user_id").agg(F.count("*").alias("future_txns")))
    users = spark.table("lake.bronze.users").select("user_id", "kyc_level")

    df = (feats.join(post, "user_id", "left")
          .withColumn("churned", F.when(F.col("future_txns").isNull(), 1).otherwise(0))
          .drop("future_txns")
          .join(users, "user_id", "left"))

    pdf = df.toPandas()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(OUT / "churn_features.parquet", index=False)
    print(f"rows: {len(pdf):,}   churn rate: {pdf.churned.mean():.1%}")


if __name__ == "__main__":
    run()