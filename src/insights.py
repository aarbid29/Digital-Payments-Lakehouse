import duckdb

S = "data/serving"
con = duckdb.connect()


def q(title, sql):
    print(f"\n== {title} ==")
    print(con.sql(sql).df().to_string(index=False))


q("Monthly volume and active-user trend", f"""
    SELECT strftime(txn_date, '%Y-%m') AS month,
           SUM(txn_count)              AS txns,
           ROUND(SUM(success_volume)/1e6, 2) AS volume_million_npr,
           ROUND(SUM(txn_count * failure_rate) / SUM(txn_count) * 100, 2) AS failure_pct
    FROM '{S}/daily_metrics.parquet'
    GROUP BY 1 ORDER BY 1
""")

q("Share of volume by transaction type", f"""
    SELECT txn_type,
           ROUND(SUM(volume)/1e6, 2) AS volume_million_npr,
           ROUND(100 * SUM(volume) / SUM(SUM(volume)) OVER (), 1) AS pct_of_volume
    FROM '{S}/daily_by_type.parquet'
    GROUP BY 1 ORDER BY 2 DESC
""")

q("Top merchant categories", f"""
    SELECT category,
           SUM(txn_count) AS txns,
           ROUND(SUM(volume)/1e6, 2) AS volume_million_npr
    FROM '{S}/merchant_stats.parquet'
    GROUP BY 1 ORDER BY 3 DESC LIMIT 5
""")

q("Average retention by months since first transaction", f"""
    SELECT month_offset,
           ROUND(100 * SUM(active_users) / SUM(cohort_size), 1) AS retention_pct
    FROM '{S}/retention.parquet'
    GROUP BY 1 ORDER BY 1
""")

q("Heavy users: top 10% of users by spend", f"""
    WITH ranked AS (
        SELECT monetary, NTILE(10) OVER (ORDER BY monetary DESC) AS decile
        FROM '{S}/rfm.parquet'
    )
    SELECT ROUND(100 * SUM(CASE WHEN decile = 1 THEN monetary END) / SUM(monetary), 1)
           AS pct_of_volume_from_top_10pct_users
    FROM ranked
""")