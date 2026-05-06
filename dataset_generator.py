"""
Aviation BI Dataset Generator
==============================
Generates ~2 years of synthetic charter flight data for a mid-size operator.

Output: one CSV per dimension table + one fact table, saved to ./data/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)
N_FLIGHTS  = 3000   # realistic for a mid-size charter operator over 2 years


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------

def make_dim_aircraft():
    aircraft = [
        ("G-WAVE", "Gulfstream G550",   2015, 14, 6750),
        ("G-NOVA", "Gulfstream G650",   2018, 16, 7000),
        ("OE-LAX", "Bombardier Global 6000", 2017, 13, 6000),
        ("OE-LBX", "Bombardier Challenger 350", 2019, 9, 3200),
        ("HB-JKL", "Dassault Falcon 8X", 2020, 14, 6450),
        ("HB-JMN", "Embraer Praetor 600", 2021, 9,  4018),
        ("CS-DRX", "Pilatus PC-24",      2022, 8,  2000),
        ("D-ABCD", "Cessna Citation X+", 2016, 8,  3460),
    ]
    rows = []
    for i, (tail, atype, year, pax, rng) in enumerate(aircraft):
        rows.append({
            "aircraft_id":   i + 1,
            "tail_number":   tail,
            "aircraft_type": atype,
            "year_built":    year,
            "max_pax":       pax,
            "max_range_nm":  rng,
        })
    return pd.DataFrame(rows)


def make_dim_crew():
    first = ["James", "Sarah", "Marco", "Elena", "Luca", "Nina",
             "Tom", "Aisha", "Karl", "Mia", "Stefan", "Julia",
             "David", "Chloe", "Ravi", "Anna", "Felix", "Zara"]
    last  = ["Smith", "Müller", "Rossi", "Dubois", "Fischer",
             "García", "Patel", "Nguyen", "Weber", "Kowalski",
             "Johansson", "Santos", "Kim", "Okonkwo", "Berg"]
    roles = ["Captain", "Captain", "Captain", "First Officer",
             "First Officer", "Cabin Crew", "Cabin Crew"]
    rows = []
    for i in range(20):
        rows.append({
            "crew_id":           i + 1,
            "crew_name":         f"{random.choice(first)} {random.choice(last)}",
            "role":              random.choice(roles),
            "duty_hours_ytd":    round(np.random.uniform(200, 900), 1),
            "rest_hours_before": round(np.random.uniform(8, 36), 1),
            "ftl_compliant":     True,   # intentional: we'll corrupt some rows later
        })
    return pd.DataFrame(rows)


def make_dim_client():
    names = [
        ("Apogee Capital",       "Corporate"),
        ("Meridian Charter",     "Broker"),
        ("Vega Ventures",        "Corporate"),
        ("Blue Horizon Group",   "Corporate"),
        ("Skyline Charters",     "Broker"),
        ("Nexus Aviation",       "Broker"),
        ("Helix Industries",     "Corporate"),
        ("Aurora Flight Club",   "Leisure"),
        ("Pinnacle Partners",    "Corporate"),
        ("Orbit Consulting",     "Corporate"),
        ("Nimbus Logistics",     "Cargo"),
        ("Cascade Holdings",     "Corporate"),
    ]
    countries = ["Austria", "Germany", "Switzerland", "UK",
                 "France", "UAE", "USA", "Italy", "Spain"]
    rows = []
    for i, (name, ctype) in enumerate(names):
        rows.append({
            "client_id":   i + 1,
            "client_name": name,
            "client_type": ctype,
            "country":     random.choice(countries),
        })
    return pd.DataFrame(rows)


def make_dim_route():
    routes = [
        ("LOWW", "EGLL", 640,  "Europe"),   # Vienna → London
        ("LOWW", "LFPB", 820,  "Europe"),   # Vienna → Paris Le Bourget
        ("LOWW", "OMDB", 2800, "Middle East"),  # Vienna → Dubai
        ("EDDF", "KJFK", 3850, "Transatlantic"),  # Frankfurt → New York
        ("LSZH", "FACT", 5100, "Africa"),   # Zurich → Cape Town
        ("EGLL", "CYYZ", 3540, "Transatlantic"),  # London → Toronto
        ("LFPB", "UUWW", 1850, "Europe"),   # Paris → Moscow
        ("LOWW", "LIRF", 650,  "Europe"),   # Vienna → Rome
        ("EDDF", "LTBA", 1200, "Europe"),   # Frankfurt → Istanbul
        ("LSZH", "SBGR", 5900, "South America"),  # Zurich → São Paulo
        ("LOWW", "EHAM", 720,  "Europe"),   # Vienna → Amsterdam
        ("EGLL", "OMDB", 3400, "Middle East"),  # London → Dubai
    ]
    rows = []
    for i, (orig, dest, dist, reg) in enumerate(routes):
        rows.append({
            "route_id":    i + 1,
            "origin_icao": orig,
            "dest_icao":   dest,
            "distance_nm": dist,
            "region":      reg,
        })
    return pd.DataFrame(rows)


def make_dim_operator():
    operators = [
        ("AlphaJet Charters",  "Charter"),
        ("EuroWing Management","Aircraft Management"),
        ("SkyFrac Europe",     "Fractional"),
        ("MedAir Solutions",   "Medevac"),
    ]
    rows = []
    for i, (name, otype) in enumerate(operators):
        rows.append({
            "operator_id":   i + 1,
            "operator_name": name,
            "operator_type": otype,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fact table
# ---------------------------------------------------------------------------

def make_fact_flights(dim_aircraft, dim_crew, dim_client, dim_route, dim_operator):
    n = N_FLIGHTS

    # Random dates spread across 2 years, with seasonal bias
    # (more flights in summer and Christmas)
    date_range_days = (END_DATE - START_DATE).days
    raw_days = np.random.randint(0, date_range_days, size=n)
    dates = [START_DATE + timedelta(days=int(d)) for d in raw_days]

    # Sample dimension IDs with realistic weights
    aircraft_ids = np.random.choice(dim_aircraft["aircraft_id"], size=n,
                                    p=[0.15, 0.12, 0.12, 0.18, 0.12, 0.12, 0.10, 0.09])
    crew_ids      = np.random.choice(dim_crew["crew_id"], size=n)
    client_ids    = np.random.choice(dim_client["client_id"], size=n,
                                     p=[0.12, 0.10, 0.12, 0.10, 0.08, 0.08,
                                        0.10, 0.08, 0.08, 0.06, 0.04, 0.04])
    route_ids     = np.random.choice(dim_route["route_id"], size=n)
    operator_ids  = np.random.choice(dim_operator["operator_id"], size=n,
                                     p=[0.40, 0.30, 0.20, 0.10])

    # Block hours: correlated with route distance
    distances = dim_route.set_index("route_id")["distance_nm"]
    route_distances = np.array([distances[r] for r in route_ids])
    block_hours = (route_distances / 430) + np.random.normal(0, 0.3, size=n)
    block_hours = np.clip(block_hours, 0.5, 14.0).round(2)

    # Revenue: roughly €3,500–€8,000 per block hour + noise
    rate_per_hour = np.random.uniform(3500, 8000, size=n)
    revenue = (block_hours * rate_per_hour).round(2)

    # Delays: most flights on time, some delayed
    delay_distribution = np.random.choice(
        [0, 0, 0, 0, 0,                             # 50% on time
         np.random.randint(1, 15),                   # 10% minor delay
         np.random.randint(15, 60),                  # 10% moderate
         np.random.randint(60, 180)],                # 5% significant
        size=n
    )
    delay_minutes = np.where(
        np.random.random(n) < 0.75, 0,
        np.random.choice([5, 10, 15, 30, 45, 60, 90, 120, 180], size=n,
                         p=[0.20, 0.18, 0.15, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03])
    ).astype(int)

    # Flight status
    status_choices = np.where(
        delay_minutes == 0,
        np.random.choice(["Completed", "Completed", "Completed", "Cancelled"],
                         size=n, p=[0.95, 0.03, 0.01, 0.01]),
        np.random.choice(["Completed", "Diverted", "Cancelled"],
                         size=n, p=[0.90, 0.06, 0.04])
    )

    rows = {
        "flight_id":      np.arange(1, n + 1),
        "aircraft_id":    aircraft_ids,
        "crew_id":        crew_ids,
        "client_id":      client_ids,
        "route_id":       route_ids,
        "operator_id":    operator_ids,
        "departure_date": [d.strftime("%Y-%m-%d") for d in dates],
        "block_hours":    block_hours,
        "revenue_eur":    revenue,
        "delay_minutes":  delay_minutes,
        "flight_status":  status_choices,
    }

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Intentional data quality issues (for Phase 4 detection)
    # -----------------------------------------------------------------------

    # 1. Duplicate rows (~1%)
    dupes = df.sample(frac=0.01, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 2. Missing revenue (~0.5%)
    null_rev_idx = df.sample(frac=0.005, random_state=SEED).index
    df.loc[null_rev_idx, "revenue_eur"] = np.nan

    # 3. Impossible block hours outlier (data entry error)
    df.loc[df.sample(3, random_state=SEED).index, "block_hours"] = 99.0

    # 4. FTL violation: a few crew members flagged as non-compliant
    # (we'll inject this into the crew dim separately)

    return df


def inject_ftl_violations(dim_crew):
    """Mark 2 crew members as having borderline duty hours — triggers a flag."""
    violation_idx = dim_crew.sample(2, random_state=SEED).index
    dim_crew.loc[violation_idx, "duty_hours_ytd"]    = 999.0   # exceeds limit
    dim_crew.loc[violation_idx, "rest_hours_before"] = 7.5     # below 8h minimum
    dim_crew.loc[violation_idx, "ftl_compliant"]     = False
    return dim_crew


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating dimension tables...")
    dim_aircraft = make_dim_aircraft()
    dim_crew     = make_dim_crew()
    dim_client   = make_dim_client()
    dim_route    = make_dim_route()
    dim_operator = make_dim_operator()

    print("Injecting FTL violations into crew data...")
    dim_crew = inject_ftl_violations(dim_crew)

    print(f"Generating fact table ({N_FLIGHTS} flights + ~1% duplicates)...")
    fact_flights = make_fact_flights(
        dim_aircraft, dim_crew, dim_client, dim_route, dim_operator
    )

    print("Saving to ./data/ ...")
    dim_aircraft.to_csv(OUTPUT_DIR / "dim_aircraft.csv",  index=False)
    dim_crew.to_csv(    OUTPUT_DIR / "dim_crew.csv",      index=False)
    dim_client.to_csv(  OUTPUT_DIR / "dim_client.csv",    index=False)
    dim_route.to_csv(   OUTPUT_DIR / "dim_route.csv",     index=False)
    dim_operator.to_csv(OUTPUT_DIR / "dim_operator.csv",  index=False)
    fact_flights.to_csv(OUTPUT_DIR / "fact_flights.csv",  index=False)

    print("\nDone. Summary:")
    print(f"  dim_aircraft : {len(dim_aircraft):>5} rows")
    print(f"  dim_crew     : {len(dim_crew):>5} rows")
    print(f"  dim_client   : {len(dim_client):>5} rows")
    print(f"  dim_route    : {len(dim_route):>5} rows")
    print(f"  dim_operator : {len(dim_operator):>5} rows")
    print(f"  fact_flights : {len(fact_flights):>5} rows  (includes ~{int(N_FLIGHTS*0.01)} dupes, some nulls, some outliers)")
    print(f"\nData quality issues injected:")
    print(f"  - ~{int(N_FLIGHTS*0.01)} duplicate flight rows")
    print(f"  - ~{int(N_FLIGHTS*0.005)} flights with NULL revenue")
    print(f"  - 3 flights with impossible block_hours (99.0)")
    print(f"  - 2 crew members with FTL violations")