"""
Aviation BI — Data Quality & ETL Monitoring Module
====================================================
Runs a suite of data quality checks against the raw CSV layer,
classifies each issue by severity, and produces a one-page PDF
report — the kind you'd send to a product owner after each ETL run.

Checks performed:
  1.  Duplicate flights (same aircraft + date + route + crew)
  2.  NULL revenue on completed flights
  3.  Impossible block hours (outlier detection)
  4.  FTL compliance breaches in crew dimension
  5.  Orphaned foreign keys (flights referencing non-existent dimension rows)
  6.  Revenue outliers (statistical: beyond 3 standard deviations)
  7.  Date range validity (flights outside expected 2023-2024 window)
  8.  Cancelled flights with non-zero revenue (billing inconsistency)
  9.  Crew assigned to overlapping flights on the same date
  10. Data freshness (how recently was the fact table last updated)

Run:  python data_quality.py
Output: data_quality_report.pdf
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

DATA_DIR   = Path("data")
OUTPUT_PDF = Path("data_quality_report.pdf")

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
CRITICAL = "CRITICAL"
WARNING  = "WARNING"
INFO     = "INFO"
OK       = "OK"

SEV_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2, OK: 3}

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def load_db():
    con = duckdb.connect()
    for t in ["fact_flights", "dim_aircraft", "dim_crew",
              "dim_client", "dim_route", "dim_operator"]:
        con.execute(
            f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{DATA_DIR}/{t}.csv')"
        )
    return con


# ---------------------------------------------------------------------------
# Individual check functions
# Each returns a dict:
#   name, severity, passed (bool), count, details (str), recommendation (str)
# ---------------------------------------------------------------------------

def check_duplicates(con):
    df = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT
            aircraft_id || '|' || departure_date || '|' ||
            route_id    || '|' || crew_id
        ) AS dupes
        FROM fact_flights
    """).df()
    count = int(df["dupes"].iloc[0])
    passed = count == 0
    return {
        "name":           "Duplicate flight records",
        "severity":       OK if passed else WARNING,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} duplicate rows detected (same aircraft + date + route + crew)."
                          if not passed else "No duplicates found.",
        "recommendation": "Deduplicate using ROW_NUMBER() OVER (PARTITION BY aircraft_id, "
                          "departure_date, route_id, crew_id ORDER BY flight_id)."
                          if not passed else "—",
    }


def check_null_revenue(con):
    df = con.execute("""
        SELECT COUNT(*) AS n
        FROM fact_flights
        WHERE revenue_eur IS NULL
          AND flight_status != 'Cancelled'
    """).df()
    count = int(df["n"].iloc[0])
    passed = count == 0
    return {
        "name":           "NULL revenue on completed flights",
        "severity":       OK if passed else CRITICAL,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} completed/diverted flights have no revenue value."
                          if not passed else "All completed flights have revenue.",
        "recommendation": "Investigate invoicing pipeline — revenue may be missing due to "
                          "a failed billing sync or early flight record creation."
                          if not passed else "—",
    }


def check_block_hour_outliers(con):
    df = con.execute("SELECT block_hours FROM fact_flights").df()
    # Use IQR method: anything above Q3 + 3*IQR is a hard outlier
    q1, q3 = df["block_hours"].quantile([0.25, 0.75])
    iqr     = q3 - q1
    upper   = q3 + 3 * iqr
    outliers = df[df["block_hours"] > upper]
    count   = len(outliers)
    passed  = count == 0
    return {
        "name":           "Impossible block hours (outlier detection)",
        "severity":       OK if passed else CRITICAL,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} flights with block_hours > {upper:.1f}h "
                          f"(IQR upper fence). Max observed: {df['block_hours'].max():.1f}h."
                          if not passed else "All block hour values within expected range.",
        "recommendation": "Likely data-entry errors. Cross-reference with flight plan "
                          "records and set a hard cap at 16h in the ingestion layer."
                          if not passed else "—",
    }


def check_ftl_violations(con):
    df = con.execute("""
        SELECT crew_name, role, duty_hours_ytd, rest_hours_before
        FROM dim_crew
        WHERE ftl_compliant = false
           OR duty_hours_ytd > 900
           OR rest_hours_before < 8
    """).df()
    count  = len(df)
    passed = count == 0
    names  = ", ".join(df["crew_name"].tolist()) if not passed else ""
    return {
        "name":           "FTL compliance breaches (crew)",
        "severity":       OK if passed else CRITICAL,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} crew member(s) breaching FTL limits: {names}."
                          if not passed else "All crew within FTL duty limits.",
        "recommendation": "Ground affected crew immediately pending review. "
                          "Alert operations manager and compliance officer."
                          if not passed else "—",
    }


def check_orphaned_keys(con):
    checks = {
        "aircraft_id":  ("fact_flights", "dim_aircraft"),
        "crew_id":      ("fact_flights", "dim_crew"),
        "client_id":    ("fact_flights", "dim_client"),
        "route_id":     ("fact_flights", "dim_route"),
        "operator_id":  ("fact_flights", "dim_operator"),
    }
    total = 0
    broken = []
    for fk, (fact, dim) in checks.items():
        df = con.execute(f"""
            SELECT COUNT(*) AS n FROM {fact}
            WHERE {fk} NOT IN (SELECT {fk} FROM {dim})
        """).df()
        n = int(df["n"].iloc[0])
        if n > 0:
            total += n
            broken.append(f"{fk} ({n} rows)")
    passed = total == 0
    return {
        "name":           "Orphaned foreign keys",
        "severity":       OK if passed else CRITICAL,
        "passed":         passed,
        "count":          total,
        "details":        f"Referential integrity broken: {'; '.join(broken)}."
                          if not passed else "All foreign keys resolve correctly.",
        "recommendation": "Check dimension table load order in the ETL pipeline. "
                          "Dimensions must be fully loaded before the fact table."
                          if not passed else "—",
    }


def check_revenue_statistical_outliers(con):
    df = con.execute("""
        SELECT revenue_eur FROM fact_flights
        WHERE revenue_eur IS NOT NULL
    """).df()
    mean = df["revenue_eur"].mean()
    std  = df["revenue_eur"].std()
    upper = mean + 3 * std
    lower = mean - 3 * std
    outliers = df[(df["revenue_eur"] > upper) | (df["revenue_eur"] < lower)]
    count = len(outliers)
    passed = count == 0
    return {
        "name":           "Revenue statistical outliers (3-sigma rule)",
        "severity":       OK if passed else WARNING,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} flights with revenue outside "
                          f"[EUR {lower:,.0f} – EUR {upper:,.0f}] (mean ± 3 std)."
                          if not passed else "Revenue distribution within 3-sigma bounds.",
        "recommendation": "Manual review recommended — may be legitimate long-haul "
                          "charters or pricing errors."
                          if not passed else "—",
    }


def check_date_range(con):
    df = con.execute("""
        SELECT COUNT(*) AS n FROM fact_flights
        WHERE departure_date::DATE < '2023-01-01'
           OR departure_date::DATE > '2024-12-31'
    """).df()
    count  = int(df["n"].iloc[0])
    passed = count == 0
    return {
        "name":           "Flights outside expected date range (2023–2024)",
        "severity":       OK if passed else WARNING,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} flights outside the 2023-01-01 to 2024-12-31 window."
                          if not passed else "All flights within expected date range.",
        "recommendation": "Check ETL source filter and partition logic."
                          if not passed else "—",
    }


def check_cancelled_with_revenue(con):
    df = con.execute("""
        SELECT COUNT(*) AS n FROM fact_flights
        WHERE flight_status = 'Cancelled'
          AND revenue_eur > 0
    """).df()
    count  = int(df["n"].iloc[0])
    passed = count == 0
    return {
        "name":           "Cancelled flights with non-zero revenue",
        "severity":       OK if passed else WARNING,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} cancelled flights still carry a revenue value — "
                          "may indicate a missed credit note or premature invoicing."
                          if not passed else "No cancelled flights with revenue.",
        "recommendation": "Reconcile with finance system. Check whether cancellation "
                          "fees apply or whether invoices were issued in error."
                          if not passed else "—",
    }


def check_crew_double_booking(con):
    df = con.execute("""
        SELECT crew_id, departure_date, COUNT(*) AS n
        FROM fact_flights
        WHERE flight_status != 'Cancelled'
        GROUP BY crew_id, departure_date
        HAVING COUNT(*) > 1
    """).df()
    count  = len(df)
    passed = count == 0
    return {
        "name":           "Crew double-booked on same date",
        "severity":       OK if passed else WARNING,
        "passed":         passed,
        "count":          count,
        "details":        f"{count} date/crew combination(s) with more than one "
                          "active flight — possible scheduling conflict."
                          if not passed else "No crew double-booking detected.",
        "recommendation": "Cross-reference with actual block times to confirm overlap. "
                          "Update rostering logic to enforce single-assignment per date."
                          if not passed else "—",
    }


def check_data_freshness(con):
    df = con.execute("""
        SELECT MAX(departure_date::DATE) AS latest FROM fact_flights
    """).df()
    latest      = pd.to_datetime(df["latest"].iloc[0]).date()
    today       = date.today()
    days_behind = (today - latest).days
    # For synthetic data this will always be stale — that's the point.
    passed  = days_behind <= 3
    sev     = OK if passed else (WARNING if days_behind <= 30 else INFO)
    return {
        "name":           "Data freshness",
        "severity":       sev,
        "passed":         passed,
        "count":          days_behind,
        "details":        f"Most recent flight: {latest}. "
                          f"Data is {days_behind} days behind today ({today}).",
        "recommendation": "Verify ETL schedule. Expected refresh: daily by 06:00 UTC."
                          if not passed else "—",
    }


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def run_all_checks(con):
    checks = [
        check_duplicates,
        check_null_revenue,
        check_block_hour_outliers,
        check_ftl_violations,
        check_orphaned_keys,
        check_revenue_statistical_outliers,
        check_date_range,
        check_cancelled_with_revenue,
        check_crew_double_booking,
        check_data_freshness,
    ]
    results = []
    for fn in checks:
        r = fn(con)
        results.append(r)
        icon = "✓" if r["passed"] else "✗"
        print(f"  [{icon}] {r['severity']:<8}  {r['name']}")
    return results


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

# Color palette — dark aviation theme translated to reportlab RGB
C_BG       = colors.HexColor("#0a0c0f")
C_PANEL    = colors.HexColor("#111520")
C_BORDER   = colors.HexColor("#1e2530")
C_BLUE     = colors.HexColor("#3b82f6")
C_CYAN     = colors.HexColor("#06b6d4")
C_GREEN    = colors.HexColor("#34d399")
C_YELLOW   = colors.HexColor("#f59e0b")
C_RED      = colors.HexColor("#ef4444")
C_TEXT     = colors.HexColor("#c8cdd6")
C_SUBTEXT  = colors.HexColor("#6b7280")
C_WHITE    = colors.white

SEV_COLORS = {
    CRITICAL: C_RED,
    WARNING:  C_YELLOW,
    INFO:     C_CYAN,
    OK:       C_GREEN,
}


def make_styles():
    base = dict(fontName="Helvetica", textColor=C_TEXT, backColor=C_BG)
    return {
        "title": ParagraphStyle("title",
            fontSize=18, fontName="Helvetica-Bold",
            textColor=C_WHITE, spaceAfter=2, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle",
            fontSize=9, fontName="Helvetica",
            textColor=C_SUBTEXT, spaceAfter=12, alignment=TA_LEFT),
        "section": ParagraphStyle("section",
            fontSize=7, fontName="Helvetica-Bold",
            textColor=C_BLUE, spaceBefore=14, spaceAfter=6,
            alignment=TA_LEFT, leading=10),
        "body": ParagraphStyle("body",
            fontSize=8, fontName="Helvetica",
            textColor=C_TEXT, spaceAfter=2, leading=12),
        "small": ParagraphStyle("small",
            fontSize=7, fontName="Helvetica",
            textColor=C_SUBTEXT, spaceAfter=1, leading=10),
        "mono": ParagraphStyle("mono",
            fontSize=7, fontName="Courier",
            textColor=C_CYAN, spaceAfter=2, leading=10),
        "alert": ParagraphStyle("alert",
            fontSize=8, fontName="Helvetica-Bold",
            textColor=C_RED, spaceAfter=3),
    }


def severity_badge_color(sev):
    return SEV_COLORS.get(sev, C_SUBTEXT)


def build_summary_table(results, styles):
    """Top summary: counts by severity."""
    counts = {CRITICAL: 0, WARNING: 0, INFO: 0, OK: 0}
    for r in results:
        counts[r["severity"]] += 1

    data = [
        [Paragraph("CRITICAL", ParagraphStyle("x", fontSize=8,
            fontName="Helvetica-Bold", textColor=C_RED, alignment=TA_CENTER)),
         Paragraph("WARNING", ParagraphStyle("x", fontSize=8,
            fontName="Helvetica-Bold", textColor=C_YELLOW, alignment=TA_CENTER)),
         Paragraph("INFO", ParagraphStyle("x", fontSize=8,
            fontName="Helvetica-Bold", textColor=C_CYAN, alignment=TA_CENTER)),
         Paragraph("PASSED", ParagraphStyle("x", fontSize=8,
            fontName="Helvetica-Bold", textColor=C_GREEN, alignment=TA_CENTER))],
        [Paragraph(str(counts[CRITICAL]), ParagraphStyle("v", fontSize=22,
            fontName="Helvetica-Bold", textColor=C_RED, alignment=TA_CENTER)),
         Paragraph(str(counts[WARNING]),  ParagraphStyle("v", fontSize=22,
            fontName="Helvetica-Bold", textColor=C_YELLOW, alignment=TA_CENTER)),
         Paragraph(str(counts[INFO]),     ParagraphStyle("v", fontSize=22,
            fontName="Helvetica-Bold", textColor=C_CYAN, alignment=TA_CENTER)),
         Paragraph(str(counts[OK]),       ParagraphStyle("v", fontSize=22,
            fontName="Helvetica-Bold", textColor=C_GREEN, alignment=TA_CENTER))],
    ]
    t = Table(data, colWidths=[40*mm]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_PANEL),
        ("BOX",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LINEBELOW", (0,0), (-1,0), 0.5, C_BORDER),
    ]))
    return t


def build_checks_table(results, styles):
    """One row per check with severity badge, name, count, details."""
    header = [
        Paragraph("SEVERITY", ParagraphStyle("h", fontSize=7,
            fontName="Helvetica-Bold", textColor=C_SUBTEXT)),
        Paragraph("CHECK", ParagraphStyle("h", fontSize=7,
            fontName="Helvetica-Bold", textColor=C_SUBTEXT)),
        Paragraph("COUNT", ParagraphStyle("h", fontSize=7,
            fontName="Helvetica-Bold", textColor=C_SUBTEXT, alignment=TA_RIGHT)),
        Paragraph("DETAILS", ParagraphStyle("h", fontSize=7,
            fontName="Helvetica-Bold", textColor=C_SUBTEXT)),
    ]
    rows = [header]
    ts_cmds = [
        ("BACKGROUND", (0,0), (-1,0), C_PANEL),
        ("BOX",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ]

    sorted_results = sorted(results, key=lambda r: SEV_ORDER[r["severity"]])

    for i, r in enumerate(sorted_results):
        row_bg = C_PANEL if i % 2 == 0 else colors.HexColor("#0d1117")
        sev_color = severity_badge_color(r["severity"])

        sev_para = Paragraph(r["severity"], ParagraphStyle("s", fontSize=7,
            fontName="Helvetica-Bold", textColor=sev_color, alignment=TA_CENTER))
        name_para = Paragraph(r["name"], ParagraphStyle("n", fontSize=8,
            fontName="Helvetica", textColor=C_TEXT))
        count_para = Paragraph(str(r["count"]), ParagraphStyle("c", fontSize=8,
            fontName="Courier", textColor=sev_color, alignment=TA_RIGHT))
        detail_para = Paragraph(r["details"], ParagraphStyle("d", fontSize=7,
            fontName="Helvetica", textColor=C_SUBTEXT, leading=10))

        rows.append([sev_para, name_para, count_para, detail_para])
        row_i = len(rows) - 1
        ts_cmds.append(("BACKGROUND", (0, row_i), (-1, row_i), row_bg))

    t = Table(rows, colWidths=[20*mm, 52*mm, 14*mm, 74*mm])
    t.setStyle(TableStyle(ts_cmds))
    return t, sorted_results


def build_recommendations_table(sorted_results, styles):
    """Only non-passing checks, with recommendations."""
    failing = [r for r in sorted_results if not r["passed"]]
    if not failing:
        return Paragraph("All checks passed. No recommendations.", styles["body"])

    header = [
        Paragraph("CHECK", ParagraphStyle("h", fontSize=7,
            fontName="Helvetica-Bold", textColor=C_SUBTEXT)),
        Paragraph("RECOMMENDATION", ParagraphStyle("h", fontSize=7,
            fontName="Helvetica-Bold", textColor=C_SUBTEXT)),
    ]
    rows = [header]
    for r in failing:
        rows.append([
            Paragraph(r["name"], ParagraphStyle("n", fontSize=7,
                fontName="Helvetica-Bold", textColor=SEV_COLORS[r["severity"]])),
            Paragraph(r["recommendation"], ParagraphStyle("d", fontSize=7,
                fontName="Helvetica", textColor=C_SUBTEXT, leading=10)),
        ])

    t = Table(rows, colWidths=[60*mm, 100*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_PANEL),
        ("BOX",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ]))
    return t


def generate_pdf(results, output_path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )

    W, H = A4
    styles = make_styles()

    def on_page(canvas, doc):
        """Dark background + header bar on every page."""
        canvas.saveState()
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Top accent bar
        canvas.setFillColor(C_BLUE)
        canvas.rect(0, H - 4, W, 4, fill=1, stroke=0)
        # Footer
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_SUBTEXT)
        canvas.drawString(15*mm, 8*mm,
            f"BRIGHT Aviation BI · Data Quality Report · "
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
        canvas.drawRightString(W - 15*mm, 8*mm, f"Page {doc.page}")
        canvas.restoreState()

    story = []

    # Header
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("DATA QUALITY REPORT", styles["title"]))
    story.append(Paragraph(
        f"BRIGHT Aviation Intelligence Platform  ·  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        f"Source: fact_flights + dimension tables  ·  Engine: DuckDB",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=C_BORDER, spaceAfter=8))

    # Summary tiles
    story.append(Paragraph("── SUMMARY", styles["section"]))
    story.append(build_summary_table(results, styles))
    story.append(Spacer(1, 6*mm))

    # Check results table
    story.append(Paragraph("── CHECK RESULTS", styles["section"]))
    checks_table, sorted_results = build_checks_table(results, styles)
    story.append(checks_table)
    story.append(Spacer(1, 6*mm))

    # Recommendations
    story.append(Paragraph("── RECOMMENDATIONS", styles["section"]))
    story.append(build_recommendations_table(sorted_results, styles))
    story.append(Spacer(1, 6*mm))

    # Raw counts footer note
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=C_BORDER, spaceAfter=4))
    raw_df = con.execute("SELECT COUNT(*) AS n FROM fact_flights").df()
    clean_note = (
        f"Raw fact_flights rows: {int(raw_df['n'].iloc[0]):,}  ·  "
        f"Checks run: {len(results)}  ·  "
        f"This report is generated automatically after each ETL refresh."
    )
    story.append(Paragraph(clean_note, styles["small"]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"\nPDF saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    con = load_db()

    print("\nRunning data quality checks:")
    results = run_all_checks(con)

    n_fail = sum(1 for r in results if not r["passed"])
    n_crit = sum(1 for r in results if r["severity"] == CRITICAL)
    print(f"\nResult: {n_fail}/{len(results)} checks failed  ({n_crit} critical)")

    print("\nGenerating PDF report...")
    generate_pdf(results, OUTPUT_PDF)