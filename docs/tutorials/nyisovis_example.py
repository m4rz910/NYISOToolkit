from nyisotoolkit import NYISOVis

if __name__ == "__main__":
    years = ['2019']
    for year in years:
        nv = NYISOVis(year=year)
        nv.fig_carbon_free_year()
        for f in ['D','M']:
            nv.fig_energy(f=f)
            nv.fig_clcpa_carbon_free(f=f)
            
    print(f"Figures saved by default to: {nv.out_dir} \nYou can change this by passing a pathlib object to the out_dir parameter to the NYISOVis object initialization.")