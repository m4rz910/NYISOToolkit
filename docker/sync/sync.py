#!/usr/bin/env python3
"""Sync NYISOToolkit datasets into Postgres/TimescaleDB for Grafana."""
import io
import os
import time
import logging
from datetime import datetime

import pandas as pd
import psycopg2

from nyisotoolkit import NYISOData, NYISOCapacity
from nyisotoolkit.nyisovis.nyisovis import CARBONFREE_SOURCES  # reuse canonical list

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nyiso-sync")

# ---- Config from env ---------------------------------------------------
PG_HOST = os.environ.get("POSTGRES_HOST", "timescaledb")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "nyiso")
PG_USER = os.environ.get("POSTGRES_USER", "nyiso")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "nyiso")

DATASETS = [d.strip() for d in os.environ.get(
    "SYNC_DATASETS", "fuel_mix_5m,load_5m,lbmp_dam_h").split(",") if d.strip()]
YEARS = [int(y.strip()) for y in os.environ.get(
    "SYNC_YEARS", "2025,2026").split(",") if y.strip()]
SYNC_INTERVAL_SECONDS = int(os.environ.get("SYNC_INTERVAL_SECONDS", "3600"))
SYNC_CAPACITY_PRICES = os.environ.get("SYNC_CAPACITY_PRICES", "true").lower() == "true"
CURRENT_YEAR = datetime.now().year

# ---- Per-dataset column -> (region, series) semantics -------------------
# "col"        -> use the (single-level) column value directly
# "const:X"    -> fixed literal value X
# "level:N"    -> use level N of a MultiIndex column tuple
# "levels:N,M" -> join levels N and M of a MultiIndex column tuple with " | "
MELT_PROFILES = {
    "load_5m":            {"region": "col",          "series": "const:load"},
    "load_h":             {"region": "col",          "series": "const:load"},
    "load_forecast_h":    {"region": "col",          "series": "const:load_forecast"},
    "fuel_mix_5m":        {"region": "const:NYCA",   "series": "col"},
    "interface_flows_5m": {"region": "level:1",      "series": "levels:0,2"},
    "lbmp_dam_h":         {"region": "level:1",      "series": "level:0"},
    "lbmp_rt_5m":         {"region": "level:1",      "series": "level:0"},
    "lbmp_dam_h_refbus":  {"region": "const:refbus", "series": "col"},
    "lbmp_rt_h_refbus":   {"region": "const:refbus", "series": "col"},
    "asp_dam":            {"region": "level:1",      "series": "level:0"},
    "asp_rt":             {"region": "level:1",      "series": "level:0"},
}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS timeseries (
    time    TIMESTAMPTZ       NOT NULL,
    dataset TEXT              NOT NULL,
    region  TEXT              NOT NULL DEFAULT '',
    series  TEXT              NOT NULL DEFAULT '',
    value   DOUBLE PRECISION,
    PRIMARY KEY (dataset, region, series, time)
);

SELECT create_hypertable(
    'timeseries', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_timeseries_dataset_time
    ON timeseries (dataset, time DESC);
CREATE INDEX IF NOT EXISTS idx_timeseries_region_series_time
    ON timeseries (region, series, time DESC);
"""


def get_conn():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                             user=PG_USER, password=PG_PASSWORD)


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def _resolve(spec, col):
    if spec == "col":
        return str(col)
    if spec.startswith("const:"):
        return spec.split(":", 1)[1]
    if spec.startswith("level:"):
        return str(col[int(spec.split(":")[1])])
    if spec.startswith("levels:"):
        idxs = [int(x) for x in spec.split(":")[1].split(",")]
        return " | ".join(str(col[i]) for i in idxs)
    raise ValueError(f"Unknown melt spec: {spec}")


COLUMN_BATCH_SIZE = 20  # bounds peak memory regardless of dataset width (e.g. interface_flows_5m)


def melt_dataframe_chunks(df: pd.DataFrame, dataset: str):
    """Turn a NYISOData wide dataframe (UTC tz-aware index) into
    long rows: time, dataset, region, series, value -- yielded in
    column batches so wide datasets don't require materializing the
    whole long-format frame (+ CSV buffer) in memory at once."""
    profile = MELT_PROFILES[dataset]
    df = df.copy()
    df.index.name = "time"
    cols = list(df.columns)
    for i in range(0, len(cols), COLUMN_BATCH_SIZE):
        batch_cols = cols[i:i + COLUMN_BATCH_SIZE]
        frames = []
        for col in batch_cols:
            s = df[col].rename("value").reset_index()
            s["region"] = _resolve(profile["region"], col)
            s["series"] = _resolve(profile["series"], col)
            frames.append(s)
        long_df = pd.concat(frames, ignore_index=True)
        long_df["dataset"] = dataset
        long_df = long_df.dropna(subset=["value"])
        yield long_df[["time", "dataset", "region", "series", "value"]]


def upsert_long_df(conn, long_df: pd.DataFrame):
    if long_df.empty:
        return
    buf = io.StringIO()
    long_df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE staging (
                time TIMESTAMPTZ, dataset TEXT, region TEXT,
                series TEXT, value DOUBLE PRECISION
            ) ON COMMIT DROP;
        """)
        cur.copy_expert(
            "COPY staging (time, dataset, region, series, value) FROM STDIN WITH CSV",
            buf,
        )
        cur.execute("""
            INSERT INTO timeseries (time, dataset, region, series, value)
            SELECT time, dataset, region, series, value FROM staging
            ON CONFLICT (dataset, region, series, time)
            DO UPDATE SET value = EXCLUDED.value;
        """)
    conn.commit()


def sync_dataset(conn, dataset: str, year: int):
    redownload = (year == CURRENT_YEAR)  # keep current year fresh every cycle
    log.info("Fetching %s %s (redownload=%s)", dataset, year, redownload)
    try:
        df = NYISOData(dataset=dataset, year=year, redownload=redownload).df
    except Exception:
        log.exception("Failed to fetch %s %s", dataset, year)
        return
    df = df.tz_convert("UTC")
    total = 0
    for chunk in melt_dataframe_chunks(df, dataset):
        upsert_long_df(conn, chunk)
        total += len(chunk)
    log.info("Upserted %d rows for %s %s", total, dataset, year)


def sync_capacity_prices(conn):
    log.info("Fetching capacity prices")
    try:
        prices = NYISOCapacity(date=pd.Timestamp.now()).prices()
    except Exception:
        log.exception("Failed to fetch capacity prices")
        return
    # NOTE: NYISOCapacity.prices()'s index tz/naivety and MultiIndex column
    # order were not verified against a live xlsx during implementation --
    # print prices.columns / prices.index once and adjust below if wrong.
    idx = pd.to_datetime(prices.index)
    if idx.tz is None:
        idx = idx.tz_localize("US/Eastern")
    prices.index = idx.tz_convert("UTC")
    prices.index.name = "time"

    frames = []
    for col in prices.columns:  # MultiIndex: (auction_type, locality)
        s = prices[col].rename("value").reset_index()
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        s["region"] = str(col[1])
        s["series"] = str(col[0])
        frames.append(s)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["dataset"] = "capacity_prices"
    long_df = long_df.dropna(subset=["value"])
    upsert_long_df(conn, long_df[["time", "dataset", "region", "series", "value"]])


def ensure_carbonfree_view(conn):
    """Push nyisovis.CARBONFREE_SOURCES into a SQL view so Grafana never
    hardcodes/duplicates the carbon-free source list."""
    sources_sql = ", ".join(f"'{s}'" for s in CARBONFREE_SOURCES)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE OR REPLACE VIEW carbonfree_pct AS
            WITH load AS (
                SELECT time, value AS load_mw
                FROM timeseries
                WHERE dataset = 'load_5m' AND region = 'NYCA' AND series = 'load'
            ),
            cf AS (
                SELECT time, SUM(value) AS cf_mw
                FROM timeseries
                WHERE dataset = 'fuel_mix_5m' AND series IN ({sources_sql})
                GROUP BY time
            )
            SELECT load.time, (cf.cf_mw / NULLIF(load.load_mw, 0)) * 100 AS carbonfree_pct
            FROM load JOIN cf ON load.time = cf.time;
        """)
    conn.commit()


def run_once():
    conn = get_conn()
    try:
        ensure_schema(conn)
        ensure_carbonfree_view(conn)
        for dataset in DATASETS:
            for year in YEARS:
                sync_dataset(conn, dataset, year)
        if SYNC_CAPACITY_PRICES:
            sync_capacity_prices(conn)
    finally:
        conn.close()


def main():
    log.info("Starting NYISO sync. datasets=%s years=%s interval=%ss",
              DATASETS, YEARS, SYNC_INTERVAL_SECONDS)
    while True:
        run_once()
        log.info("Sync cycle complete. Sleeping %ss", SYNC_INTERVAL_SECONDS)
        time.sleep(SYNC_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
