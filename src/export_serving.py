"""Compute retention + RFM from silver and export gold tables to parquet."""
from pathlib import Path

from pyspark.sql import functions as F

from spark_session import get_spark

OUT = Path(__file__).resolve().parents[1] / "data" / "serving"


def save(df, name):
    pdf = df.toPandas()
    pdf.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name:<18} {len(pdf):>8,} rows")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    spark = get_spark("export_serving")

    # 1. existing gold tables
    for name in ["daily_metrics", "daily_by_type", "merchant_stats", "dq_results"]:
        save(spark.table(f"lake.gold.{name}"), name)

    ok = spark.table("lake.silver.transactions").filter("is_success = 1")

    # 2. monthly cohort retention
    t = ok.withColumn("month", F.date_trunc("month", "txn_ts").cast("date"))
    first = t.groupBy("user_id").agg(F.min("month").alias("cohort_month"))
    ret = (
        t.join(first, "user_id")
        .select("user_id", "cohort_month", "month").distinct()
        .withColumn("month_offset",
                    F.round(F.months_between("month", "cohort_month")).cast("int"))
        .groupBy("cohort_month", "month_offset")
        .agg(F.countDistinct("user_id").alias("active_users"))
    )
    sizes = (ret.filter("month_offset = 0")
             .select("cohort_month", F.col("active_users").alias("cohort_size")))
    ret = (ret.join(sizes, "cohort_month")
           .withColumn("retention", F.round(F.col("active_users") / F.col("cohort_size"), 4)))
    save(ret.orderBy("cohort_month", "month_offset"), "retention")

    # 3. RFM per user (as of the day after the last transaction)
    end = ok.agg(F.max("txn_date")).first()[0]
    rfm = (
        ok.groupBy("user_id")
        .agg(F.datediff(F.lit(end), F.max("txn_date")).alias("recency_days"),
             F.count("*").alias("frequency"),
             F.round(F.sum("amount"), 2).alias("monetary"),
             F.round(F.avg("amount"), 2).alias("avg_amount"))
    )
    users = spark.table("lake.bronze.users").select("user_id", "city", "kyc_level")
    save(rfm.join(users, "user_id", "left"), "rfm")

    print("export done")


if __name__ == "__main__":
    run()