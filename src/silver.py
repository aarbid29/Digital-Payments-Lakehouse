from pyspark.sql import Window
from pyspark.sql import functions as F

from spark_session import get_spark


def run():
    spark = get_spark("silver")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.silver")
    bronze = spark.table("lake.bronze.transactions")

    # 1. split valid vs rejected
    is_valid = (
        F.col("txn_id").isNotNull() & F.col("user_id").isNotNull()
        & F.col("amount").isNotNull() & (F.col("amount") > 0)
    )
    rejected = (
        bronze.filter(~is_valid)
        .withColumn(
            "reject_reason",
            F.when(F.col("amount").isNull(), "null_amount")
             .when(F.col("amount") <= 0, "non_positive_amount")
             .otherwise("missing_key"),
        )
    )
    rejected.writeTo("lake.silver.transactions_rejected").createOrReplace()

    # 2. deduplicate on txn_id
    w = Window.partitionBy("txn_id").orderBy(F.col("ingested_at").desc())
    silver = (
        bronze.filter(is_valid)
        .withColumn("rn", F.row_number().over(w))
        .filter("rn = 1").drop("rn", "source_file")
        .withColumn("txn_date", F.to_date("txn_ts"))
        .withColumn("is_success", (F.col("status") == "SUCCESS").cast("int"))
    )
    (silver.writeTo("lake.silver.transactions")
     .partitionedBy(F.days("txn_ts"))
     .createOrReplace())

    print("bronze  :", bronze.count())
    print("rejected:", spark.table("lake.silver.transactions_rejected").count())
    print("silver  :", spark.table("lake.silver.transactions").count())


if __name__ == "__main__":
    run()