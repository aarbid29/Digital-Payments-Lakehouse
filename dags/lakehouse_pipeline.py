from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = Path(__file__).resolve().parents[1]
PY = PROJECT / ".venv" / "bin" / "python"     


def job(task_id, script, args="", retries=0):
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT} && {PY} src/{script} {args}",
        retries=retries,
        retry_delay=timedelta(minutes=1),
    )


with DAG(
    dag_id="payments_lakehouse",
    description="bronze -> silver -> DQ -> gold -> DQ",
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "data-eng"},
    tags=["lakehouse", "iceberg", "spark"],
) as dag:

    bronze = job("bronze_ingest", "bronze.py", retries=1)
    silver = job("silver_clean", "silver.py", retries=1)
    dq_silver = job("dq_check_silver", "quality.py", "--stage silver")
    gold = job("gold_aggregate", "gold.py", retries=1)
    dq_gold = job("dq_check_gold", "quality.py", "--stage gold")

    bronze >> silver >> dq_silver >> gold >> dq_gold

    