import pathlib as pl

import pandas as pd

from nyisotoolkit.nyisodata.utils import check_and_interpolate_nans

FIXTURE = pl.Path(__file__).resolve().parent / "fixtures" / "fuel_mix_5m_sample.csv"


def test_fuel_mix_sample_parses_and_pivots():
    """Mirrors NYISOData.construct_database()'s "Time Zone in columns" branch
    (nyisotoolkit/nyisodata/nyisodata.py) against a small real-format sample,
    without going through NYISOData itself -- that class's completeness
    assert requires a full year of data, which isn't a "small" fixture."""
    df = pd.read_csv(FIXTURE, index_col=0)
    df.index = pd.to_datetime(df.index)

    df = df.tz_localize("US/Eastern", ambiguous=df["Time Zone"] == "EST")
    df = df.sort_index(axis="index").tz_convert("UTC")
    df = df.drop(columns=["Time Zone", "PTID"], errors="ignore")
    df = df.pivot(columns="Fuel Category", values="Gen MW")
    df = df.resample("5min").mean()
    df = check_and_interpolate_nans(df)

    assert set(df.columns) == {"Nuclear", "Hydro", "Wind"}
    assert len(df) == 12
    assert str(df.index.tz) == "UTC"
    assert not df.isnull().values.any()
    # spot-check the first row's known input values
    first = df.iloc[0]
    assert first["Nuclear"] == 3436.6
    assert first["Hydro"] == 1189.3
    assert first["Wind"] == 312.4


def test_check_and_interpolate_nans_fills_interior_and_edges():
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame({"NYCA": [1.0, None, 3.0, 4.0, None]}, index=idx)

    result = check_and_interpolate_nans(df)

    assert not result.isnull().values.any()
    assert result.loc[idx[1], "NYCA"] == 2.0  # interior: linear interpolation
    assert result.loc[idx[4], "NYCA"] == 4.0  # trailing: limit_direction="both"


def test_check_and_interpolate_nans_noop_when_no_nans():
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame({"NYCA": [1.0, 2.0, 3.0]}, index=idx)

    result = check_and_interpolate_nans(df.copy())

    pd.testing.assert_frame_equal(result, df)
