from spark_session import get_spark

spark = get_spark("check")

print("\n== Daily metrics (first 5) ==")
spark.sql("SELECT * FROM lake.gold.daily_metrics ORDER BY txn_date LIMIT 5").show()

print("== Top 5 merchants by volume ==")
spark.sql("""
    SELECT merchant_name, category, txn_count, ROUND(volume,0) AS volume
    FROM lake.gold.merchant_stats ORDER BY volume DESC LIMIT 5
""").show(truncate=False)

print("== Rejected rows by reason ==")
spark.sql("""
    SELECT reject_reason, COUNT(*) AS n
    FROM lake.silver.transactions_rejected GROUP BY 1
""").show()

print("== Iceberg snapshots (time travel metadata) ==")
spark.sql("SELECT snapshot_id, committed_at, operation FROM lake.silver.transactions.snapshots").show(truncate=False)