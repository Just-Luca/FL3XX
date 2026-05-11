"""
Aviation BI — SQL Analytics Layer
===================================
Loads the synthetic CSVs into DuckDB, builds a clean view layer,
then runs analytical queries covering the Key Performance Indicators (KPIs) FL3XX's BRIGHT platform
exposes: fleet utilization, revenue, crew duty compliance, route performance.

Structure:
  1. Load raw CSVs as DuckDB tables
  2. Create analytical views (the "view layer" above raw tables)
  3. Run KPI queries — each one is a real BI deliverable
  4. Print results cleanly

Run:  python analytics.py
"""

import duckdb
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# 1. Connect and load raw tables
# ---------------------------------------------------------------------------

con = duckdb.connect()   # in-memory — no file needed

def load_tables(con):
    tables = [
        "fact_flights",
        "dim_aircraft",
        "dim_crew",
        "dim_client",
        "dim_route",
        "dim_operator",
    ]
    for t in tables:
        path = DATA_DIR / f"{t}.csv"
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{path}')")
    print("Tables loaded:", tables)

load_tables(con)


# ---------------------------------------------------------------------------
# 2. View layer — clean, deduplicated, enriched
#
#    These views sit between raw tables and analytical queries.
#    In a real Redshift setup, these would be CREATE VIEW statements
#    that your BI tool (QuickSight) reads from.
# ---------------------------------------------------------------------------

# v_flights: deduplicated fact table, joined to all dimensions
# This is the single source of truth for all downstream queries.
con.execute("""
CREATE VIEW v_flights AS
WITH deduped AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY aircraft_id, departure_date, route_id, crew_id
               ORDER BY flight_id
           ) AS rn
    FROM fact_flights
    WHERE block_hours < 20          -- exclude impossible outliers
      AND revenue_eur IS NOT NULL   -- exclude flights with missing revenue
)
SELECT
    f.flight_id,
    f.departure_date::DATE          AS departure_date,
    EXTRACT(YEAR  FROM f.departure_date::DATE) AS year,
    EXTRACT(MONTH FROM f.departure_date::DATE) AS month,
    EXTRACT(DOW   FROM f.departure_date::DATE) AS day_of_week,

    -- dimensions
    a.tail_number,
    a.aircraft_type,
    a.max_pax,
    a.max_range_nm,

    cr.crew_name,
    cr.role          AS crew_role,
    cr.ftl_compliant,
    cr.duty_hours_ytd,

    cl.client_name,
    cl.client_type,
    cl.country       AS client_country,

    r.origin_icao,
    r.dest_icao,
    r.distance_nm,
    r.region,

    op.operator_name,
    op.operator_type,
    f.operator_id,   -- kept for RLS filtering

    -- metrics
    f.block_hours,
    f.revenue_eur,
    f.delay_minutes,
    f.flight_status,

    -- derived fields
    ROUND(f.revenue_eur / NULLIF(f.block_hours, 0), 2)  AS yield_per_block_hour,
    CASE WHEN f.delay_minutes = 0 THEN 1 ELSE 0 END     AS is_on_time,
    CASE WHEN f.delay_minutes >= 60 THEN 1 ELSE 0 END   AS is_significantly_delayed,
    CASE WHEN f.flight_status = 'Cancelled' THEN 1 ELSE 0 END AS is_cancelled

FROM deduped f
JOIN dim_aircraft  a  ON f.aircraft_id  = a.aircraft_id
JOIN dim_crew      cr ON f.crew_id      = cr.crew_id
JOIN dim_client    cl ON f.client_id    = cl.client_id
JOIN dim_route     r  ON f.route_id     = r.route_id
JOIN dim_operator  op ON f.operator_id  = op.operator_id
WHERE rn = 1
""")

print("View v_flights created.\n")


# ---------------------------------------------------------------------------
# 3. KPI queries
# ---------------------------------------------------------------------------

def run(label, sql):
    """Run a query, print a header, return the result as a DataFrame."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    df = con.execute(sql).df()
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Q1. Fleet utilization — block hours per tail, per year
#     Utilization % = actual block hours / assumed available hours (1,200/yr)
#     Industry benchmark for a busy charter aircraft: ~600–900 hrs/yr
# ---------------------------------------------------------------------------

q1 = run("Q1 — Fleet utilization by tail number and year", """
SELECT
    tail_number,
    aircraft_type,
    year,
    COUNT(*)                                AS total_flights,
    ROUND(SUM(block_hours), 1)              AS total_block_hours,
    ROUND(SUM(block_hours) / 1200.0 * 100, 1) AS utilization_pct,
    ROUND(AVG(block_hours), 2)              AS avg_block_hours_per_flight,
    ROUND(SUM(revenue_eur) / 1000, 1)       AS revenue_k_eur
FROM v_flights
WHERE flight_status != 'Cancelled'
GROUP BY tail_number, aircraft_type, year
ORDER BY year, utilization_pct DESC
""")


# ---------------------------------------------------------------------------
# Q2. Monthly revenue trend with 3-month rolling average
#     Window function: AVG(...) OVER (ORDER BY ... ROWS BETWEEN 2 PRECEDING...)
# ---------------------------------------------------------------------------

q2 = run("Q2 — Monthly revenue with 3-month rolling average", """
WITH monthly AS (
    SELECT
        year,
        month,
        ROUND(SUM(revenue_eur), 2)        AS monthly_revenue,
        COUNT(*)                          AS flights,
        ROUND(AVG(yield_per_block_hour))  AS avg_yield
    FROM v_flights
    WHERE flight_status != 'Cancelled'
    GROUP BY year, month
)
SELECT
    year,
    month,
    monthly_revenue,
    flights,
    avg_yield,
    ROUND(
        AVG(monthly_revenue) OVER (
            ORDER BY year, month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_3m_revenue
FROM monthly
ORDER BY year, month
""")


# ---------------------------------------------------------------------------
# Q3. Revenue rank per operator — using RANK() window function
#     Simulates the multi-tenant view: each operator sees their own rank
# ---------------------------------------------------------------------------

q3 = run("Q3 — Revenue per operator with rank and share of total", """
WITH operator_rev AS (
    SELECT
        operator_name,
        operator_type,
        ROUND(SUM(revenue_eur))          AS total_revenue,
        COUNT(*)                         AS flights,
        ROUND(AVG(yield_per_block_hour)) AS avg_yield
    FROM v_flights
    WHERE flight_status != 'Cancelled'
    GROUP BY operator_name, operator_type
),
totals AS (
    SELECT SUM(total_revenue) AS grand_total FROM operator_rev
)
SELECT
    operator_name,
    operator_type,
    total_revenue,
    flights,
    avg_yield,
    RANK() OVER (ORDER BY total_revenue DESC)         AS revenue_rank,
    ROUND(total_revenue * 100.0 / totals.grand_total, 1) AS revenue_share_pct
FROM operator_rev
CROSS JOIN totals
ORDER BY revenue_rank
""")


# ---------------------------------------------------------------------------
# Q4. Delay analysis — on-time performance by route and region
#     Includes: on-time %, avg delay when delayed, % significantly delayed
# ---------------------------------------------------------------------------

q4 = run("Q4 — On-time performance by region and route", """
SELECT
    region,
    origin_icao || ' → ' || dest_icao   AS route,
    COUNT(*)                            AS total_flights,
    ROUND(AVG(is_on_time) * 100, 1)     AS on_time_pct,
    ROUND(AVG(CASE WHEN delay_minutes > 0
                   THEN delay_minutes END), 1) AS avg_delay_when_late,
    ROUND(AVG(is_significantly_delayed) * 100, 1) AS pct_60min_plus_delay,
    ROUND(AVG(is_cancelled) * 100, 1)   AS cancellation_rate_pct
FROM v_flights
GROUP BY region, origin_icao, dest_icao
ORDER BY on_time_pct ASC
""")


# ---------------------------------------------------------------------------
# Q5. Crew duty compliance — FTL flag + duty hours ranking
#     Identifies crew approaching or breaching legal limits
# ---------------------------------------------------------------------------

q5 = run("Q5 — Crew duty compliance and flight load", """
SELECT
    crew_name,
    crew_role,
    ftl_compliant,
    ROUND(duty_hours_ytd, 1)            AS duty_hours_ytd,
    COUNT(*)                            AS flights_assigned,
    ROUND(SUM(block_hours), 1)          AS block_hours_flown,
    ROUND(AVG(delay_minutes), 1)        AS avg_delay_on_their_flights,
    CASE
        WHEN duty_hours_ytd > 900 THEN 'BREACH — over 900h limit'
        WHEN duty_hours_ytd > 800 THEN 'WARNING — approaching limit'
        ELSE 'OK'
    END                                 AS duty_status
FROM v_flights
GROUP BY crew_name, crew_role, ftl_compliant, duty_hours_ytd
ORDER BY duty_hours_ytd DESC
""")


# ---------------------------------------------------------------------------
# Q6. Top clients by revenue with year-over-year growth
#     LAG() window function to compare this year vs last year
# ---------------------------------------------------------------------------

q6 = run("Q6 — Top clients YoY revenue growth (LAG window function)", """
WITH client_yearly AS (
    SELECT
        client_name,
        client_type,
        year,
        ROUND(SUM(revenue_eur))    AS revenue,
        COUNT(*)                   AS flights
    FROM v_flights
    WHERE flight_status != 'Cancelled'
    GROUP BY client_name, client_type, year
)
SELECT
    client_name,
    client_type,
    year,
    revenue,
    flights,
    LAG(revenue) OVER (
        PARTITION BY client_name
        ORDER BY year
    )                              AS prev_year_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (
            PARTITION BY client_name ORDER BY year
        )) * 100.0
        / NULLIF(LAG(revenue) OVER (
            PARTITION BY client_name ORDER BY year
        ), 0)
    , 1)                           AS yoy_growth_pct
FROM client_yearly
ORDER BY year, revenue DESC
""")


# ---------------------------------------------------------------------------
# Q7. Simulated Row-Level Security — operator-isolated view
#     In QuickSight this is a RLS dataset rule; here we show the SQL equivalent.
#     Every operator only sees their own flights.
# ---------------------------------------------------------------------------

run("Q7 — Operator-isolated view (RLS simulation) for operator_id = 1", """
SELECT
    departure_date,
    tail_number,
    origin_icao || ' → ' || dest_icao AS route,
    client_name,
    block_hours,
    revenue_eur,
    delay_minutes,
    flight_status
FROM v_flights
WHERE operator_id = 1          -- this WHERE clause IS the RLS rule
ORDER BY departure_date DESC
LIMIT 10
""")


# ---------------------------------------------------------------------------
# Q8. Data quality summary — what the view layer filtered out
#     Always document what you cleaned and why.
# ---------------------------------------------------------------------------

run("Q8 — Data quality report: what was excluded from v_flights", """
WITH raw_counts AS (
    SELECT
        COUNT(*) AS total_raw_rows,
        SUM(CASE WHEN block_hours >= 20 THEN 1 ELSE 0 END)    AS outlier_block_hours,
        SUM(CASE WHEN revenue_eur IS NULL THEN 1 ELSE 0 END)   AS null_revenue,
        COUNT(*) - COUNT(DISTINCT
            aircraft_id || '|' ||
            departure_date || '|' ||
            route_id || '|' ||
            crew_id
        )                                                       AS duplicate_rows
    FROM fact_flights
),
clean_count AS (
    SELECT COUNT(*) AS clean_rows FROM v_flights
)
SELECT
    raw_counts.total_raw_rows,
    raw_counts.duplicate_rows,
    raw_counts.null_revenue,
    raw_counts.outlier_block_hours,
    clean_count.clean_rows,
    raw_counts.total_raw_rows - clean_count.clean_rows AS total_excluded
FROM raw_counts, clean_count
""")

print("\n\nAll queries complete.")