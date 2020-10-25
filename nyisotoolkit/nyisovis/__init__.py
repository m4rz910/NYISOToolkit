import pathlib as pl

c_dir = pl.Path(__file__).resolve().parent
vis_dir = pl.Path(c_dir, "visualizations")
vis_dir.mkdir(parents=True, exist_ok=True)
