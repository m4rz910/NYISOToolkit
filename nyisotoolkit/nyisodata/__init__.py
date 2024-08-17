import pathlib as pl

STORAGE_DIR = pl.Path(pl.Path(__file__).resolve().parent, 'storage')
DATABASE_DIR = pl.Path(STORAGE_DIR, 'databases')