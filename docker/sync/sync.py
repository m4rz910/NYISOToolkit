#!/usr/bin/env python3
"""Sync NYISOToolkit datasets into Postgres/TimescaleDB for Grafana."""
import io
import os
import time
import threading
import zipfile
import logging
from datetime import datetime

import requests
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
SYNC_SYSTEM_EVENTS = os.environ.get("SYNC_SYSTEM_EVENTS", "true").lower() == "true"
# Separate, faster cadence for system_events so alerts show up on dashboards
# promptly instead of waiting for the (often hourly) SYNC_INTERVAL_SECONDS
# dataset cycle. 0 disables the fast path (events then only refresh via the
# slow full-backfill loop in run_once(), as before this was added).
SYNC_EVENTS_INTERVAL_SECONDS = int(os.environ.get("SYNC_EVENTS_INTERVAL_SECONDS", "120"))

# Historical (non-current-year) (dataset, year) pairs are fully static once
# synced once -- re-fetching + re-upserting them every cycle is pure waste.
# Track what's already been synced this process lifetime; reset on restart
# (self-healing, acceptable full resync once per container lifetime).
_historical_synced: set[tuple[str, int]] = set()

# NYISO's system-state/grid-alert log (thunderstorm alerts, reserve pick-ups,
# alert-state transitions), same mis.nyiso.com CSV infra NYISOData itself
# uses -- not part of NYISOData.SUPPORTED_DATASETS since it's text events,
# not a numeric time series.
EVENTS_URL = "http://mis.nyiso.com/public/csv/RealTimeEvents/{}01RealTimeEvents_csv.zip"

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

ALTER TABLE timeseries SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'dataset,region,series',
    timescaledb.compress_orderby = 'time DESC'
);

-- 90-day margin (not the more common 30) to keep a safety window for
-- late-arriving NYISO settlement corrections -- see sync_dataset's
-- psycopg2.Error handling for the fallback if a correction still lands
-- on an already-compressed chunk.
SELECT add_compression_policy('timeseries', INTERVAL '90 days', if_not_exists => true);

CREATE TABLE IF NOT EXISTS system_events (
    time    TIMESTAMPTZ NOT NULL,
    message TEXT        NOT NULL,
    PRIMARY KEY (time, message)
);

CREATE INDEX IF NOT EXISTS idx_system_events_time
    ON system_events (time DESC);
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


def _stage_and_insert(cur, long_df: pd.DataFrame):
    """Stage long_df via COPY into a TEMP TABLE and upsert into timeseries,
    skipping no-op writes when the value hasn't changed (avoids rewriting
    unchanged rows across the PK + 2 secondary indexes every cycle). Does
    NOT commit -- caller controls the transaction boundary so multiple
    chunks of one dataset-year can share a single, WAL-cheaper commit."""
    buf = io.StringIO()
    long_df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cur.execute("""
        CREATE TEMP TABLE IF NOT EXISTS staging (
            time TIMESTAMPTZ, dataset TEXT, region TEXT,
            series TEXT, value DOUBLE PRECISION
        ) ON COMMIT DROP;
    """)
    cur.execute("TRUNCATE staging;")
    cur.copy_expert(
        "COPY staging (time, dataset, region, series, value) FROM STDIN WITH CSV",
        buf,
    )
    cur.execute("""
        INSERT INTO timeseries (time, dataset, region, series, value)
        SELECT time, dataset, region, series, value FROM staging
        ON CONFLICT (dataset, region, series, time)
        DO UPDATE SET value = EXCLUDED.value
        WHERE timeseries.value IS DISTINCT FROM EXCLUDED.value;
    """)


def upsert_long_df(conn, long_df: pd.DataFrame, commit: bool = True):
    """High-level single-shot upsert -- used by callers that already have
    the whole frame in memory and want one commit per call (e.g. capacity
    prices). Wide, chunked datasets use _stage_and_insert directly via a
    shared cursor/transaction (see sync_dataset)."""
    if long_df.empty:
        return
    with conn.cursor() as cur:
        _stage_and_insert(cur, long_df)
    if commit:
        conn.commit()


def upsert_events(conn, events: pd.DataFrame):
    if events.empty:
        return
    buf = io.StringIO()
    events.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE staging_events (
                time TIMESTAMPTZ, message TEXT
            ) ON COMMIT DROP;
        """)
        cur.copy_expert(
            "COPY staging_events (time, message) FROM STDIN WITH CSV",
            buf,
        )
        cur.execute("""
            INSERT INTO system_events (time, message)
            SELECT time, message FROM staging_events
            ON CONFLICT (time, message) DO NOTHING;
        """)
    conn.commit()


def _fetch_events_month(year: int, month: int):
    """Fetch+parse a single month's RealTimeEvents zip. Returns a raw
    DataFrame with ['time', 'message'] columns (Eastern-naive), or None."""
    url = EVENTS_URL.format(f"{year}{month:02d}")
    try:
        r = requests.get(url, timeout=30)
        if not r.ok:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        frames = []
        for name in z.namelist():
            with z.open(name) as f:
                day_df = pd.read_csv(f)
            if day_df.empty:
                continue
            day_df.columns = ["time", "message"]
            frames.append(day_df)
        return pd.concat(frames, ignore_index=True) if frames else None
    except Exception:
        log.exception("Failed to fetch system events %s-%02d", year, month)
        return None


def _localize_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.tz_convert("UTC")
    return df


def sync_system_events(conn, year: int):
    """Full-month backfill for a year -- the slow, correctness-safety-net
    path, run from run_once() on the main SYNC_INTERVAL_SECONDS cadence.
    Deliberately excludes the current month when year is the current year:
    the fast run_events_loop thread already owns that month, and refetching
    it here every cycle would just duplicate that HTTP + COPY + upsert work."""
    log.info("Fetching system events %s", year)
    now = datetime.now()
    months = range(1, 13) if year < now.year else range(1, now.month)
    frames = [df for df in (_fetch_events_month(year, m) for m in months) if df is not None]
    if not frames:
        return
    events = _localize_events(pd.concat(frames, ignore_index=True))
    upsert_events(conn, events)
    log.info("Upserted %d system events for %s", len(events), year)


def sync_system_events_current_month(conn):
    """Cheap, high-frequency refresh: fetches only the current month's zip
    (one HTTP request) so dashboards see new alerts within
    SYNC_EVENTS_INTERVAL_SECONDS without redownloading the multi-year
    backfill every tick. Uses datetime.now() fresh each call so it stays
    correct across a Dec 31 -> Jan 1 rollover without a restart."""
    now = datetime.now()
    df = _fetch_events_month(now.year, now.month)
    if df is None:
        return
    events = _localize_events(df)
    upsert_events(conn, events)
    log.info("Upserted %d system events for %s-%02d (fast path)",
              len(events), now.year, now.month)


def run_events_loop():
    """Independent, faster-cadence loop for dashboard-visible events. Runs
    as a daemon thread started once from main(), after the one-time schema
    pre-flight -- it never calls ensure_schema itself, avoiding a race
    between two threads issuing CREATE EXTENSION / CREATE TABLE IF NOT
    EXISTS DDL concurrently."""
    log.info("Starting events fast-sync thread. interval=%ss", SYNC_EVENTS_INTERVAL_SECONDS)
    while True:
        try:
            conn = get_conn()
            try:
                sync_system_events_current_month(conn)
            finally:
                conn.close()
        except Exception:
            log.exception("Events fast-sync cycle failed")
        time.sleep(SYNC_EVENTS_INTERVAL_SECONDS)


def sync_dataset(conn, dataset: str, year: int):
    is_current = (year == datetime.now().year)  # computed fresh (not a
                                                  # module constant) so a
                                                  # long-lived container
                                                  # doesn't misclassify
                                                  # "current year" after a
                                                  # Dec 31 -> Jan 1 rollover
    if not is_current and (dataset, year) in _historical_synced:
        log.debug("Skipping static historical %s %s (already synced this run)",
                   dataset, year)
        return

    redownload = is_current  # keep current year fresh every cycle
    log.info("Fetching %s %s (redownload=%s)", dataset, year, redownload)
    try:
        df = NYISOData(dataset=dataset, year=year, redownload=redownload).df
    except Exception:
        log.exception("Failed to fetch %s %s", dataset, year)
        return
    df = df.tz_convert("UTC")

    total = 0
    with conn.cursor() as cur:
        cur.execute("SET LOCAL timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
        for chunk in melt_dataframe_chunks(df, dataset):
            if chunk.empty:
                continue
            try:
                _stage_and_insert(cur, chunk)
            except psycopg2.Error:
                # e.g. a late-arriving correction landing on an already
                # compressed TimescaleDB chunk -- log and move on rather
                # than aborting the whole sync cycle for every dataset.
                log.exception("Upsert failed for a chunk of %s %s "
                               "(possibly a compressed-chunk write)", dataset, year)
                conn.rollback()
                return
            total += len(chunk)
    conn.commit()
    log.info("Upserted %d rows for %s %s", total, dataset, year)

    if not is_current:
        _historical_synced.add((dataset, year))


def sync_capacity_prices(conn):
    log.info("Fetching capacity prices")
    try:
        prices = NYISOCapacity(date=pd.Timestamp.now()).prices()
    except Exception:
        log.exception("Failed to fetch capacity prices")
        return
    idx = pd.to_datetime(prices.index)
    if idx.tz is None:
        idx = idx.tz_localize("US/Eastern")
    prices.index = idx.tz_convert("UTC")
    prices.index.name = "time"

    frames = []
    for col in prices.columns:  # MultiIndex: (locality, auction_type)
        s = prices[col].rename("value").reset_index()
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        s["region"] = str(col[0])
        s["series"] = str(col[1])
        frames.append(s)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["dataset"] = "capacity_prices"
    long_df = long_df.dropna(subset=["value"])
    upsert_long_df(conn, long_df[["time", "dataset", "region", "series", "value"]])


def sync_capacity_mw(conn):
    """NYCA/GHIJ/NYC/LI monthly MW figures from the same ICAP-Market-Report
    xlsx as sync_capacity_prices -- MW Cleared (capacity actually procured)
    and Requirements (ICAP quota). Dropping the pct-of-requirement and Spot
    MCP columns from this sheet since MCP is already covered by
    capacity_prices."""
    log.info("Fetching capacity MW (cleared/requirements)")
    try:
        summary = NYISOCapacity(date=pd.Timestamp.now()).summary_table()
    except Exception:
        log.exception("Failed to fetch capacity MW summary")
        return
    idx = pd.to_datetime(summary.index)
    if idx.tz is None:
        idx = idx.tz_localize("US/Eastern")
    summary.index = idx.tz_convert("UTC")
    summary.index.name = "time"

    keep_metrics = {"MW Cleared", "Requirements"}
    frames = []
    for col in summary.columns:  # MultiIndex: (locality, metric)
        if col[1] not in keep_metrics:
            continue
        s = summary[col].rename("value").reset_index()
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        s["region"] = str(col[0])
        s["series"] = str(col[1])
        frames.append(s)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["dataset"] = "capacity_mw"
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
        for dataset in DATASETS:
            for year in YEARS:
                sync_dataset(conn, dataset, year)
        if SYNC_CAPACITY_PRICES:
            sync_capacity_prices(conn)
            sync_capacity_mw(conn)
        if SYNC_SYSTEM_EVENTS:
            for year in YEARS:
                sync_system_events(conn, year)
    finally:
        conn.close()


def main():
    log.info("Starting NYISO sync. datasets=%s years=%s interval=%ss events_interval=%ss",
              DATASETS, YEARS, SYNC_INTERVAL_SECONDS, SYNC_EVENTS_INTERVAL_SECONDS)

    # Schema/view setup runs exactly once here, before any concurrent loop
    # starts -- run_once() and run_events_loop() both assume it already
    # exists and never call ensure_schema themselves.
    conn = get_conn()
    try:
        ensure_schema(conn)
        ensure_carbonfree_view(conn)
    finally:
        conn.close()

    if SYNC_SYSTEM_EVENTS and SYNC_EVENTS_INTERVAL_SECONDS > 0:
        threading.Thread(target=run_events_loop, daemon=True, name="events-fast-sync").start()

    while True:
        run_once()
        log.info("Sync cycle complete. Sleeping %ss", SYNC_INTERVAL_SECONDS)
        time.sleep(SYNC_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
