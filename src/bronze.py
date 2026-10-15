from pathlib import Path
from pyspark.sql import functions as F
from spark_session import get_spark

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def run():
    spark = get_spark("bronze")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")

    txn = (spark.read.parquet(str(RAW / "transactions"))
           .withColumn("ingested_at", F.current_timestamp())
           .withColumn("source_file", F.input_file_name()))
    txn.writeTo("lake.bronze.transactions").createOrReplace()

    for name in ["users", "merchants"]:
        df = (spark.read.option("header", True).option("inferSchema", True)
              .csv(str(RAW / f"{name}.csv"))
              .withColumn("ingested_at", F.current_timestamp()))
        df.writeTo(f"lake.bronze.{name}").createOrReplace()

    print("bronze transactions:", spark.table("lake.bronze.transactions").count())
    print("bronze users       :", spark.table("lake.bronze.users").count())
    print("bronze merchants   :", spark.table("lake.bronze.merchants").count())


if __name__ == "__main__":
    run()