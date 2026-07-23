# CLAUDE.md — NYISOToolkit

Working notes for agent sessions in this repo. Read `README.md` first for the
user-facing docs (dataset tables, examples); this file is the cheat-sheet plus
the list of things that will trip you up.

## What this is

A pip-installable Python library (`nyisotoolkit`, version in `setup.py`) for
NYISO — New York grid — data. Three modules, all re-exported from the package
root (`nyisotoolkit/__init__.py`):

- **NYISOData** — downloads raw monthly CSV zips from `http://mis.nyiso.com/public/csv/`,
  resamples/interpolates to a uniform grid, and caches each `(year, dataset)`
  as a pickle. `.df` is the result: UTC-indexed, start-of-period convention.
  Years 2018+ only (`arg_validation`).
- **NYISOCapacity** (`nyisodata/capacity.py`) — separate path for the monthly
  ICAP Market Report xlsx (capacity prices / UCAP / summary tables). The
  README lists `capacity_prices` as a "dataset" but it is NOT in
  `SUPPORTED_DATASETS` — it goes through this class, not NYISOData.
- **NYISOStat / NYISOVis** — statistics tables and matplotlib/seaborn figures
  (CLCPA decarbonization tracking). Figures save to
  `nyisotoolkit/nyisovis/visualizations/` by default; the PNGs there are
  git-tracked (force-added past the `*.png` gitignore) because README embeds one.

No CLI, no service, no CI workflow (`.github/` is just issue templates).

## Layout

```
nyisotoolkit/
├── nyisodata/
│   ├── nyisodata.py        # NYISOData + construct_databases + SUPPORTED_DATASETS
│   ├── utils.py            # month-range, timestamp-range, NaN interpolation helpers
│   ├── capacity.py         # NYISOCapacity (ICAP xlsx reports)
│   ├── data_quality.py     # stubbed dataset fixups (TODO'd out of the main path)
│   ├── dataset_url_map.yml # per-dataset URL parts, frequency, pivot columns
│   └── storage/            # gitignored local cache: raw_datafiles/ + databases/*.pkl
├── nyisostat/nyisostat.py  # NYISOStat tables (annual energy, instate flows, ...)
└── nyisovis/
    ├── nyisovis.py         # NYISOVis figures + basic_plots/statistical_plots runners
    ├── legend_colors.yml   # fuel-category → color map (LEGEND_DEETS)
    └── visualizations/     # default fig output; committed PNGs used by README
tests/                      # pytest; tests.py is an old scratch script, not a test
```

## How to run & iterate

```bash
pip install -e .            # setup.py; package_data ships the *.yml files
python -m pytest tests/ -k utils     # fast, offline (date-range logic)
python -m pytest tests/              # SLOW + network: rebuilds every dataset
                                     # for 2023-2024 from live NYISO downloads
```

- Adding a dataset = new entry in `dataset_url_map.yml` (+ name in
  `SUPPORTED_DATASETS` and any special-casing in `dataset_adjustments`).
- `construct_database` ends in two asserts (no missing timestamps, no NaNs) —
  a new dataset that fails there usually needs resample/pivot config fixes in
  the yml, not assert removal.
- Capacity report URLs need a per-year magic code: `year_to_year_code` in
  `capacity.py:get_url` (a new NYISO documents ID every year — 2026 is the
  latest; each January this map needs a new entry).

## Known traps

- **The README's `NYISOData.construct_databases(...)` example is wrong twice**:
  `construct_databases` is a module-level function (import it from
  `nyisotoolkit`), not a classmethod, and the kwarg is `create_csv` not
  `create_csvs`. Follow the code, not the README.
- **`storage/` lives inside the package tree.** Downloads and pickle caches
  are written next to the installed code (site-packages for a non-editable
  install). Don't commit anything under it — `*.csv`/`*.pkl` are gitignored.
- **Cache invalidation is manual.** An existing pickle is returned as-is;
  current-year data goes stale unless callers pass `redownload=True` /
  `reconstruct=True`. Partial current-year builds end 2h before "now"
  (`fetch_ts_start_end`).
- **Everything indexes in UTC; `year` means the US/Eastern year.** Raw files
  are localized with DST disambiguation (`Time Zone` column or
  `ambiguous="infer"`) then converted. `tz_convert('US/Eastern')` before any
  by-local-day/hour math, as NYISOStat does.
- **Two long-lived branches**: `dev` (default working branch, currently ahead)
  and `master`. Dependabot PRs target `master`.

## Relationship to nyisotoolkit_website

Separate repo `~/Documents/dev/nyisotoolkit_website` (Django, powers
nyisotoolkit.com — has its own CLAUDE.md). It imports `NYISOCapacity` +
`current_year` from this package for its `ingest_capacity_reports` management
command only; the install line in its requirements.txt is commented out and
pinned to an old commit, so it's installed by hand from git. Renaming or
changing `NYISOCapacity`'s table methods breaks that ingest.
