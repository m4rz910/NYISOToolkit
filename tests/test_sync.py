import sys
import pathlib as pl

import numpy as np
import pandas as pd
import pytest

# docker/sync/sync.py is a standalone script, not part of the installed
# nyisotoolkit package -- import it directly by path.
sys.path.insert(0, str(pl.Path(__file__).resolve().parent.parent / "docker" / "sync"))
import sync  # noqa: E402


# ---- _resolve -------------------------------------------------------------

def test_resolve_col():
    assert sync._resolve("col", "NYCA") == "NYCA"


def test_resolve_const():
    assert sync._resolve("const:NYCA", "Wind") == "NYCA"


def test_resolve_level():
    col = ("Flow (MW)", "HQ CHPE", "External Flows")
    assert sync._resolve("level:1", col) == "HQ CHPE"


def test_resolve_levels():
    col = ("Flow (MW)", "HQ CHPE", "External Flows")
    assert sync._resolve("levels:0,2", col) == "Flow (MW) | External Flows"


def test_resolve_unknown_spec_raises():
    with pytest.raises(ValueError):
        sync._resolve("bogus:1", "NYCA")


# ---- melt_dataframe_chunks -------------------------------------------------

def _time_index(n=3):
    return pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")


def test_melt_dataframe_chunks_region_col():
    idx = _time_index()
    df = pd.DataFrame({"NYCA": [1.0, 2.0, 3.0], "LONGIL": [4.0, 5.0, 6.0]}, index=idx)
    chunks = list(sync.melt_dataframe_chunks(df, "load_5m"))
    long_df = pd.concat(chunks, ignore_index=True)
    assert set(long_df["region"]) == {"NYCA", "LONGIL"}
    assert set(long_df["series"]) == {"load"}
    assert list(long_df.columns) == ["time", "dataset", "region", "series", "value"]
    assert (long_df["dataset"] == "load_5m").all()
    assert len(long_df) == 6


def test_melt_dataframe_chunks_region_const():
    idx = _time_index()
    df = pd.DataFrame({"Wind": [1.0, 2.0, 3.0], "Nuclear": [4.0, 5.0, 6.0]}, index=idx)
    long_df = pd.concat(sync.melt_dataframe_chunks(df, "fuel_mix_5m"), ignore_index=True)
    assert set(long_df["region"]) == {"NYCA"}
    assert set(long_df["series"]) == {"Wind", "Nuclear"}


def test_melt_dataframe_chunks_region_level():
    idx = _time_index()
    columns = pd.MultiIndex.from_tuples(
        [("Flow (MW)", "HQ CHPE", "External Flows"),
         ("Flow (MW)", "PJM NEPTUNE", "External Flows")]
    )
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=idx, columns=columns)
    long_df = pd.concat(sync.melt_dataframe_chunks(df, "interface_flows_5m"), ignore_index=True)
    assert set(long_df["region"]) == {"HQ CHPE", "PJM NEPTUNE"}
    assert set(long_df["series"]) == {"Flow (MW) | External Flows"}


def test_melt_dataframe_chunks_drops_nan_values():
    idx = _time_index()
    df = pd.DataFrame({"NYCA": [1.0, np.nan, 3.0]}, index=idx)
    long_df = pd.concat(sync.melt_dataframe_chunks(df, "load_5m"), ignore_index=True)
    assert len(long_df) == 2
    assert not long_df["value"].isna().any()


def test_melt_dataframe_chunks_batches_wide_dataframe():
    idx = _time_index(n=2)
    ncols = 45  # > COLUMN_BATCH_SIZE (20), should split into 3 batches
    df = pd.DataFrame(
        np.ones((2, ncols)),
        index=idx,
        columns=[f"col{i}" for i in range(ncols)],
    )
    chunks = list(sync.melt_dataframe_chunks(df, "load_5m"))
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 2 * ncols


# ---- _localize_events -------------------------------------------------------

def test_localize_events_converts_to_utc():
    df = pd.DataFrame({
        "time": ["2023-06-15 10:00:00"],
        "message": ["normal"],
    })
    result = sync._localize_events(df)
    assert str(result["time"].dt.tz) == "UTC"
    # 2023-06-15 10:00 US/Eastern (EDT, UTC-4) -> 14:00 UTC
    assert result.iloc[0]["time"] == pd.Timestamp("2023-06-15 14:00:00", tz="UTC")


def test_localize_events_drops_ambiguous_dst_fallback_row():
    # 2023-11-05 01:30:00 occurred twice in US/Eastern (DST fall-back at 2am -> 1am)
    df = pd.DataFrame({
        "time": ["2023-06-15 10:00:00", "2023-11-05 01:30:00"],
        "message": ["normal", "ambiguous-fallback"],
    })
    result = sync._localize_events(df)
    assert len(result) == 1
    assert result.iloc[0]["message"] == "normal"
