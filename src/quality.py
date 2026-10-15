"""Data quality checks. Exits with code 1 if any check fails.

Usage:
    python src/quality.py --stage silver
    python src/quality.py --stage gold
"""
import argparse
import sys

from pyspark.sql import functions as F

from spark_session import get_spark


def run(stage: str, max_null_rate: float, max_dup_rate: float,
        max_reject_rate: float) -> bool:
    spark = get_spark(f"quality_{stage}")
    results = []

    def check(name, passed, detail):
        results.append((stage, name, bool(passed), str(detail)))

    if stage == "silver":
        bronze = spark.table("lake.bronze.transactions")
        silver = spark.table("lake.silver.transactions")
        rejected = spark.table("lake.silver.transactions_rejected")

        b = bronze.agg(
            F.count("*").alias("n"),
            F.countDistinct("txn_id").alias("distinct_ids"),
            F.sum(F.col("amount").isNull().cast("int")).alias("null_amount"),
        ).first()
        s = silver.agg(
            F.count("*").alias("n"),
            F.countDistinct("txn_id").alias("distinct_ids"),
            F.sum(F.col("txn_id").isNull().cast("int")).alias("null_ids"),
            F.sum((F.col("amount") <= 0).cast("int")).alias("bad_amount"),
            F.sum((~F.col("status").isin("SUCCESS", "FAILED", "PENDING"))
                  .cast("int")).alias("bad_status"),
            F.min("txn_ts").alias("min_ts"),
            F.max("txn_ts").alias("max_ts"),
        ).first()
        r_count = rejected.count()

        null_rate = b["null_amount"] / b["n"]
        dup_rate = 1 - b["distinct_ids"] / b["n"]
        reject_rate = r_count / b["n"]

        check("bronze_null_amount_rate", null_rate <= max_null_rate,
              f"{null_rate:.4%} (max {max_null_rate:.2%})")
        check("bronze_duplicate_rate", dup_rate <= max_dup_rate,
              f"{dup_rate:.4%} (max {max_dup_rate:.2%})")
        check("rejected_rate", reject_rate <= max_reject_rate,
              f"{reject_rate:.4%} (max {max_reject_rate:.2%})")
        check("silver_txn_id_unique", s["n"] == s["distinct_ids"],
              f"rows={s['n']} distinct={s['distinct_ids']}")
        check("silver_no_null_txn_id", s["null_ids"] == 0,
              f"null_ids={s['null_ids']}")
        check("silver_amount_positive", s["bad_amount"] == 0,
              f"bad_amount={s['bad_amount']}")
        check("silver_status_valid", s["bad_status"] == 0,
              f"bad_status={s['bad_status']}")
        check("silver_ts_in_range",
              str(s["min_ts"])[:4] >= "2025" and str(s["max_ts"])[:4] <= "2026",
              f"{s['min_ts']} -> {s['max_ts']}")
        check("silver_not_larger_than_bronze", s["n"] + r_count <= b["n"],
              f"silver={s['n']} rejected={r_count} bronze={b['n']}")

    elif stage == "gold":
        silver_n = spark.table("lake.silver.transactions").count()
        daily = spark.table("lake.gold.daily_metrics")
        merch = spark.table("lake.gold.merchant_stats")

        d = daily.agg(
            F.count("*").alias("days"),
            F.sum("txn_count").alias("total_txn"),
            F.min("failure_rate").alias("min_fr"),
            F.max("failure_rate").alias("max_fr"),
            F.sum("success_volume").alias("volume"),
        ).first()
        null_merchants = merch.filter("merchant_id IS NULL").count()

        check("gold_daily_has_rows", d["days"] > 0, f"days={d['days']}")
        check("gold_txn_count_matches_silver", d["total_txn"] == silver_n,
              f"gold={d['total_txn']} silver={silver_n}")
        check("gold_failure_rate_sane", 0 <= d["min_fr"] and d["max_fr"] <= 0.30,
              f"min={d['min_fr']} max={d['max_fr']}")
        check("gold_volume_positive", d["volume"] and d["volume"] > 0,
              f"volume={d['volume']}")
        check("gold_merchant_no_null_id", null_merchants == 0,
              f"null_merchant_ids={null_merchants}")

    # ---- print + persist results ----
    print(f"\n===== DATA QUALITY: {stage.upper()} =====")
    for _, name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<34} {detail}")

    df = (spark.createDataFrame(results, "stage string, check_name string, "
                                         "passed boolean, detail string")
          .withColumn("run_ts", F.current_timestamp()))
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.gold")
    tbl = "lake.gold.dq_results"
    writer = df.writeTo(tbl)
    if spark.catalog.tableExists(tbl):
        writer.append()
    else:
        writer.create()

    return all(p for _, _, p, _ in results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["silver", "gold"], required=True)
    ap.add_argument("--max-null-rate", type=float, default=0.01)
    ap.add_argument("--max-dup-rate", type=float, default=0.02)
    ap.add_argument("--max-reject-rate", type=float, default=0.01)
    a = ap.parse_args()

    ok = run(a.stage, a.max_null_rate, a.max_dup_rate, a.max_reject_rate)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    sys.exit(0 if ok else 1)