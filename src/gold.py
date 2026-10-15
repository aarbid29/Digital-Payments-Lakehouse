from pyspark.sql import functions as F

from spark_session import get_spark


def run():
    spark = get_spark("gold")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.gold")
    txn = spark.table("lake.silver.transactions")
    merchants = spark.table("lake.bronze.merchants").select(
        "merchant_id", "merchant_name", "category")

    # Daily KPIs
    daily = (
        txn.groupBy("txn_date")
        .agg(
            F.count("*").alias("txn_count"),
            F.sum(F.when(F.col("is_success") == 1, F.col("amount")).otherwise(0)).alias("success_volume"),
            F.countDistinct("user_id").alias("active_users"),
            F.round(1 - F.avg("is_success"), 4).alias("failure_rate"),
        )
        .orderBy("txn_date")
    )
    daily.writeTo("lake.gold.daily_metrics").createOrReplace()

    # Volume by transaction type per day
    by_type = (
        txn.filter("is_success = 1")
        .groupBy("txn_date", "txn_type")
        .agg(F.count("*").alias("txn_count"), F.sum("amount").alias("volume"))
    )
    by_type.writeTo("lake.gold.daily_by_type").createOrReplace()

    # Merchant leaderboard
    merch = (
        txn.filter("is_success = 1 AND merchant_id IS NOT NULL")
        .groupBy("merchant_id")
        .agg(F.count("*").alias("txn_count"),
             F.sum("amount").alias("volume"),
             F.countDistinct("user_id").alias("unique_customers"))
        .join(merchants, "merchant_id", "left")
    )
    merch.writeTo("lake.gold.merchant_stats").createOrReplace()

    print("gold tables built")


if __name__ == "__main__":
    run()