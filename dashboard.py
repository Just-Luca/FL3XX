"""
Aviation BI Dashboard — FL3XX BRIGHT portfolio project
========================================================
A QuickSight-style analytics dashboard built in Streamlit + Plotly.
Covers: fleet utilization, revenue trends, delay analysis, crew duty,
client performance — with simulated Row-Level Security (operator filter).

Run:  streamlit run dashboard.py
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BRIGHT Aviation BI",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system — refined dark theme, aviation instrument aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0c0f;
    color: #c8cdd6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2530;
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; font-size: 12px; }

/* KPI cards */
.kpi-card {
    background: #111520;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: #4b5563;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    margin-top: 6px;
}
.delta-up   { color: #34d399; }
.delta-down { color: #f87171; }
.delta-neu  { color: #6b7280; }

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #3b82f6;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem 0;
}

/* RLS badge */
.rls-badge {
    display: inline-block;
    background: #0f2339;
    border: 1px solid #1e4976;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #60a5fa;
    letter-spacing: 0.05em;
}

/* Alert */
.alert-row {
    background: #1a0f0f;
    border: 1px solid #7f1d1d;
    border-left: 3px solid #ef4444;
    border-radius: 4px;
    padding: 8px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #fca5a5;
    margin-bottom: 6px;
}

/* Plotly container background */
.stPlotlyChart { background: transparent !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data layer — DuckDB connection, cached for performance
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

@st.cache_resource
def get_connection():
    """
    Load all CSVs into an in-memory DuckDB instance and create v_flights.
    @st.cache_resource means this runs once per session, not on every rerender.
    This is the Streamlit equivalent of a persistent database connection.
    """
    con = duckdb.connect()
    tables = ["fact_flights", "dim_aircraft", "dim_crew",
              "dim_client", "dim_route", "dim_operator"]
    for t in tables:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{DATA_DIR}/{t}.csv')")

    con.execute("""
    CREATE VIEW v_flights AS
    WITH deduped AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY aircraft_id, departure_date, route_id, crew_id
                   ORDER BY flight_id
               ) AS rn
        FROM fact_flights
        WHERE block_hours < 20
          AND revenue_eur IS NOT NULL
    )
    SELECT
        f.flight_id,
        f.departure_date::DATE                              AS departure_date,
        EXTRACT(YEAR  FROM f.departure_date::DATE)::INT    AS year,
        EXTRACT(MONTH FROM f.departure_date::DATE)::INT    AS month,
        a.tail_number, a.aircraft_type,
        cr.crew_name, cr.role AS crew_role,
        cr.ftl_compliant, cr.duty_hours_ytd,
        cl.client_name, cl.client_type,
        r.origin_icao, r.dest_icao, r.distance_nm, r.region,
        op.operator_name, op.operator_type,
        f.operator_id,
        f.block_hours, f.revenue_eur, f.delay_minutes, f.flight_status,
        ROUND(f.revenue_eur / NULLIF(f.block_hours,0), 2)  AS yield_per_block_hour,
        CASE WHEN f.delay_minutes = 0 THEN 1 ELSE 0 END    AS is_on_time,
        CASE WHEN f.flight_status = 'Cancelled' THEN 1 ELSE 0 END AS is_cancelled
    FROM deduped f
    JOIN dim_aircraft  a  ON f.aircraft_id  = a.aircraft_id
    JOIN dim_crew      cr ON f.crew_id      = cr.crew_id
    JOIN dim_client    cl ON f.client_id    = cl.client_id
    JOIN dim_route     r  ON f.route_id     = r.route_id
    JOIN dim_operator  op ON f.operator_id  = op.operator_id
    WHERE rn = 1
    """)
    return con

@st.cache_data
def query(_con, sql):
    """Run a SQL query and return a DataFrame. Cached by query string."""
    return _con.execute(sql).df()

con = get_connection()

# ---------------------------------------------------------------------------
# Plotly theme — shared across all charts
# ---------------------------------------------------------------------------

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e2530", linecolor="#1e2530", tickcolor="#1e2530"),
    yaxis=dict(gridcolor="#1e2530", linecolor="#1e2530", tickcolor="#1e2530"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
COLOR_SEQ   = ["#3b82f6", "#06b6d4", "#34d399", "#f59e0b", "#a78bfa", "#f472b6"]
COLOR_ALERT = "#ef4444"

# ---------------------------------------------------------------------------
# Sidebar — RLS operator filter + year filter
# ---------------------------------------------------------------------------

st.sidebar.markdown("### ✈ BRIGHT")
st.sidebar.markdown(
    "<span style='font-family:DM Mono;font-size:10px;color:#4b5563;letter-spacing:.1em'>"
    "AVIATION INTELLIGENCE PLATFORM</span>",
    unsafe_allow_html=True
)
st.sidebar.divider()

operators_df = query(con, "SELECT DISTINCT operator_id, operator_name FROM dim_operator ORDER BY operator_name")
operator_options = {"All operators": None} | dict(zip(operators_df["operator_name"], operators_df["operator_id"]))
selected_operator_name = st.sidebar.selectbox("Operator (RLS filter)", list(operator_options.keys()))
selected_operator_id   = operator_options[selected_operator_name]

years_df      = query(con, "SELECT DISTINCT year FROM v_flights ORDER BY year")
year_options  = ["All years"] + list(years_df["year"].astype(str))
selected_year = st.sidebar.selectbox("Year", year_options)

st.sidebar.divider()
st.sidebar.markdown(
    "<span style='font-family:DM Mono;font-size:10px;color:#374151'>"
    "Data: synthetic · 2023–2024<br>"
    "Engine: DuckDB · Streamlit<br>"
    "Project: FL3XX BRIGHT portfolio</span>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Build the WHERE clause from sidebar filters
# This is the RLS simulation: operator_id filter applied to every query.
# ---------------------------------------------------------------------------

def where_clause():
    parts = ["flight_status != 'Cancelled'"]
    if selected_operator_id is not None:
        parts.append(f"operator_id = {selected_operator_id}")
    if selected_year != "All years":
        parts.append(f"year = {selected_year}")
    return "WHERE " + " AND ".join(parts)

W = where_clause()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown(
        "<h1 style='font-family:Syne;font-size:22px;font-weight:800;"
        "color:#e2e8f0;margin:0;letter-spacing:0.02em'>BRIGHT — Aviation Intelligence</h1>",
        unsafe_allow_html=True
    )
with col_badge:
    if selected_operator_id:
        st.markdown(f"<div style='padding-top:6px'><span class='rls-badge'>RLS: {selected_operator_name}</span></div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1 — KPI cards
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>01 — KEY PERFORMANCE INDICATORS</div>", unsafe_allow_html=True)

kpi_df = query(con, f"""
SELECT
    COUNT(*)                                    AS total_flights,
    ROUND(SUM(block_hours), 0)                  AS total_block_hours,
    ROUND(SUM(revenue_eur) / 1e6, 2)            AS revenue_m_eur,
    ROUND(AVG(is_on_time) * 100, 1)             AS on_time_pct,
    ROUND(AVG(yield_per_block_hour), 0)         AS avg_yield,
    ROUND(AVG(delay_minutes), 1)                AS avg_delay
FROM v_flights {W}
""")

k = kpi_df.iloc[0]

def kpi_card(label, value, delta_html=""):
    return f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        {f"<div class='kpi-delta'>{delta_html}</div>" if delta_html else ""}
    </div>"""

c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, "Total Flights",       f"{int(k['total_flights']):,}",       ""),
    (c2, "Block Hours",         f"{int(k['total_block_hours']):,}",    ""),
    (c3, "Revenue (M EUR)",     f"€{k['revenue_m_eur']}M",            ""),
    (c4, "On-Time %",           f"{k['on_time_pct']}%",
         f"<span class='{'delta-up' if k['on_time_pct']>=75 else 'delta-down'}'>"
         f"{'▲ above' if k['on_time_pct']>=75 else '▼ below'} 75% target</span>"),
    (c5, "Avg Yield / Blk Hr",  f"€{int(k['avg_yield']):,}",          ""),
    (c6, "Avg Delay (min)",     f"{k['avg_delay']}",
         f"<span class='{'delta-up' if k['avg_delay']<15 else 'delta-down'}'>"
         f"{'▲ within' if k['avg_delay']<15 else '▼ over'} 15min target</span>"),
]
for col, label, value, delta in cards:
    with col:
        st.markdown(kpi_card(label, value, delta), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2 — Fleet utilization
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>02 — FLEET UTILIZATION</div>", unsafe_allow_html=True)

util_df = query(con, f"""
SELECT
    tail_number, aircraft_type, year,
    ROUND(SUM(block_hours), 1)              AS block_hours,
    ROUND(SUM(block_hours)/1200.0*100, 1)   AS utilization_pct,
    ROUND(SUM(revenue_eur)/1000, 0)         AS revenue_k
FROM v_flights {W}
GROUP BY tail_number, aircraft_type, year
ORDER BY utilization_pct DESC
""")

col_u1, col_u2 = st.columns([3, 2])

with col_u1:
    fig = px.bar(
        util_df, x="utilization_pct", y="tail_number",
        color="utilization_pct",
        color_continuous_scale=["#1e3a5f", "#3b82f6", "#06b6d4"],
        orientation="h",
        facet_col="year" if selected_year == "All years" else None,
        title="Fleet Utilization % (baseline: 1,200 hrs/yr)",
        labels={"utilization_pct": "Utilization %", "tail_number": ""},
        text="utilization_pct",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.add_vline(x=75, line_dash="dot", line_color="#f59e0b",
                  annotation_text="75% target", annotation_font_color="#f59e0b")
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=340)
    st.plotly_chart(fig, width='stretch')

with col_u2:
    fig2 = px.scatter(
        util_df, x="block_hours", y="revenue_k",
        size="utilization_pct", color="aircraft_type",
        color_discrete_sequence=COLOR_SEQ,
        title="Block Hours vs Revenue per Tail",
        labels={"block_hours": "Block Hours", "revenue_k": "Revenue (k EUR)"},
        hover_data=["tail_number", "utilization_pct"],
    )
    fig2.update_layout(**PLOT_LAYOUT, height=340)
    st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------------------
# Section 3 — Revenue trend
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>03 — REVENUE TREND</div>", unsafe_allow_html=True)

rev_df = query(con, f"""
WITH monthly AS (
    SELECT year, month,
           ROUND(SUM(revenue_eur)/1000, 1) AS revenue_k,
           COUNT(*)                         AS flights
    FROM v_flights {W}
    GROUP BY year, month
)
SELECT *,
       ROUND(AVG(revenue_k) OVER (
           ORDER BY year, month
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ), 1) AS rolling_3m
FROM monthly ORDER BY year, month
""")

rev_df["period"] = rev_df["year"].astype(str) + "-" + rev_df["month"].astype(str).str.zfill(2)

col_r1, col_r2 = st.columns([3, 1])

with col_r1:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=rev_df["period"], y=rev_df["revenue_k"],
        name="Monthly Revenue", marker_color="#1e3a5f",
        hovertemplate="<b>%{x}</b><br>Revenue: €%{y}k<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=rev_df["period"], y=rev_df["rolling_3m"],
        name="3-Month Rolling Avg", line=dict(color="#06b6d4", width=2),
        hovertemplate="<b>%{x}</b><br>3M Avg: €%{y}k<extra></extra>",
    ))
    fig3.update_layout(**PLOT_LAYOUT, title="Monthly Revenue (k EUR) + 3M Rolling Average",
                       height=300, barmode="overlay")
    st.plotly_chart(fig3, width='stretch')

with col_r2:
    client_df = query(con, f"""
    SELECT client_name,
           ROUND(SUM(revenue_eur)/1000, 0) AS revenue_k
    FROM v_flights {W}
    GROUP BY client_name
    ORDER BY revenue_k DESC LIMIT 6
    """)
    fig4 = px.pie(
        client_df, values="revenue_k", names="client_name",
        title="Revenue by Client (Top 6)",
        color_discrete_sequence=COLOR_SEQ, hole=0.5,
    )
    fig4.update_layout(**{**PLOT_LAYOUT,
                          "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))},
                       height=300)
    fig4.update_traces(textinfo="percent", textfont_size=10)
    st.plotly_chart(fig4, width='stretch')

# ---------------------------------------------------------------------------
# Section 4 — Delay & on-time performance
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>04 — DELAY & ON-TIME PERFORMANCE</div>", unsafe_allow_html=True)

delay_df = query(con, f"""
SELECT
    region,
    origin_icao || ' → ' || dest_icao       AS route,
    COUNT(*)                                 AS flights,
    ROUND(AVG(is_on_time)*100, 1)            AS on_time_pct,
    ROUND(AVG(CASE WHEN delay_minutes > 0
              THEN delay_minutes END), 1)    AS avg_delay_when_late
FROM v_flights {W}
GROUP BY region, origin_icao, dest_icao
ORDER BY on_time_pct ASC
""")

col_d1, col_d2 = st.columns(2)

with col_d1:
    fig5 = px.bar(
        delay_df, x="on_time_pct", y="route",
        color="on_time_pct",
        color_continuous_scale=["#7f1d1d", "#f59e0b", "#34d399"],
        orientation="h",
        title="On-Time Performance by Route (%)",
        labels={"on_time_pct": "On-Time %", "route": ""},
        text="on_time_pct",
    )
    fig5.update_traces(texttemplate="%{text}%", textposition="outside")
    fig5.add_vline(x=75, line_dash="dot", line_color="#6b7280")
    fig5.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=340)
    st.plotly_chart(fig5, width='stretch')

with col_d2:
    fig6 = px.scatter(
        delay_df, x="avg_delay_when_late", y="on_time_pct",
        size="flights", color="region",
        color_discrete_sequence=COLOR_SEQ,
        title="Avg Delay (when late) vs On-Time Rate",
        labels={"avg_delay_when_late": "Avg delay when late (min)",
                "on_time_pct": "On-Time %"},
        text="route",
    )
    fig6.update_traces(textposition="top center", textfont_size=9)
    fig6.update_layout(**PLOT_LAYOUT, height=340)
    st.plotly_chart(fig6, width='stretch')

# ---------------------------------------------------------------------------
# Section 5 — Crew duty & FTL compliance
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>05 — CREW DUTY & FTL COMPLIANCE</div>", unsafe_allow_html=True)

# FTL alerts — shown prominently above the chart
ftl_df = query(con, """
SELECT DISTINCT crew_name, crew_role, duty_hours_ytd, ftl_compliant
FROM v_flights
WHERE ftl_compliant = false
""")

if not ftl_df.empty:
    for _, row in ftl_df.iterrows():
        st.markdown(
            f"<div class='alert-row'>⚠ FTL BREACH — {row['crew_name']} "
            f"({row['crew_role']}) · {row['duty_hours_ytd']}h duty YTD · "
            f"Exceeds 900h annual limit</div>",
            unsafe_allow_html=True
        )

crew_df = query(con, f"""
SELECT
    crew_name, crew_role,
    ROUND(duty_hours_ytd, 0)        AS duty_hours_ytd,
    COUNT(*)                        AS flights,
    ROUND(SUM(block_hours), 1)      AS block_hours_flown,
    ftl_compliant
FROM v_flights {W}
GROUP BY crew_name, crew_role, duty_hours_ytd, ftl_compliant
ORDER BY duty_hours_ytd DESC
LIMIT 12
""")

col_c1, col_c2 = st.columns([2, 3])

with col_c1:
    fig7 = px.bar(
        crew_df, x="duty_hours_ytd", y="crew_name",
        orientation="h",
        color="ftl_compliant",
        color_discrete_map={True: "#3b82f6", False: "#ef4444"},
        title="Crew Duty Hours YTD (top 12)",
        labels={"duty_hours_ytd": "Duty Hours YTD", "crew_name": ""},
    )
    fig7.add_vline(x=900, line_dash="dot", line_color=COLOR_ALERT,
                   annotation_text="900h limit", annotation_font_color=COLOR_ALERT)
    fig7.add_vline(x=800, line_dash="dot", line_color="#f59e0b",
                   annotation_text="800h warning", annotation_font_color="#f59e0b")
    fig7.update_layout(**PLOT_LAYOUT, height=380)
    st.plotly_chart(fig7, width='stretch')

with col_c2:
    fig8 = px.scatter(
        crew_df, x="flights", y="block_hours_flown",
        color="crew_role", size="duty_hours_ytd",
        color_discrete_sequence=COLOR_SEQ,
        title="Flights Assigned vs Block Hours Flown",
        labels={"flights": "Flights Assigned", "block_hours_flown": "Block Hours Flown"},
        text="crew_name",
        hover_data=["duty_hours_ytd", "ftl_compliant"],
    )
    fig8.update_traces(textposition="top center", textfont_size=9)
    fig8.update_layout(**PLOT_LAYOUT, height=380)
    st.plotly_chart(fig8, width='stretch')

# ---------------------------------------------------------------------------
# Section 6 — Client YoY growth
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>06 — CLIENT YEAR-ON-YEAR PERFORMANCE</div>", unsafe_allow_html=True)

yoy_df = query(con, f"""
WITH cy AS (
    SELECT client_name, client_type,
           year, ROUND(SUM(revenue_eur)/1000, 0) AS revenue_k
    FROM v_flights {W}
    GROUP BY client_name, client_type, year
)
SELECT *,
       LAG(revenue_k) OVER (PARTITION BY client_name ORDER BY year) AS prev_k,
       ROUND((revenue_k - LAG(revenue_k) OVER (
           PARTITION BY client_name ORDER BY year)
       ) * 100.0 / NULLIF(
           LAG(revenue_k) OVER (PARTITION BY client_name ORDER BY year), 0
       ), 1) AS yoy_pct
FROM cy ORDER BY year, revenue_k DESC
""")

yoy_2024 = yoy_df[yoy_df["year"] == 2024].dropna(subset=["yoy_pct"])

if not yoy_2024.empty:
    fig9 = px.bar(
        yoy_2024.sort_values("yoy_pct"),
        x="yoy_pct", y="client_name",
        color="yoy_pct",
        color_continuous_scale=["#7f1d1d", "#374151", "#065f46"],
        orientation="h",
        title="Client Revenue YoY Growth % (2023 → 2024)",
        labels={"yoy_pct": "YoY Growth %", "client_name": ""},
        text="yoy_pct",
    )
    fig9.update_traces(texttemplate="%{text}%", textposition="outside")
    fig9.add_vline(x=0, line_color="#374151")
    fig9.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320)
    st.plotly_chart(fig9, width='stretch')