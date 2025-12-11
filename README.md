# ENSO-HighResMIP-TempAnoms

## Characterising Surface Discrepancies and Vertical Coherence of Ocean Temperature Anomalies in CMIP6 HighResMIP During ENSO Events

This repository contains all processing and plotting code used in the associated research article.

## Environment

```bash
dconda env create -f environment.yml
conda activate enso_highresmip_tempanoms
```

## Directory Structure

```
data/
    anomalies/          # Output from data_prep.ipynb
    processed/          # Output from enso_preprocessing.py
    thetao_con/         # Conservative temperature fields
    zon_wnds_850hpa/    # 850 hPa zonal wind + anomalies
    z20_depths/         # Depth of 20°C isotherm (Z20) timeseries
    griddes_025.grd     # CDO grid description for 0.25° grid
    pac_mask_025.npz    # Land–sea mask on 0.25° grid
    Centred_BS_bounds_CMIP6_HIGHRESMIP_1950-2050.csv
    Centred_BS_bounds_NPD_eORCA025_OBS_1976-2023.csv
figures/                # Figures generated from plotting.ipynb
data_prep.ipynb
plotting.ipynb
enso_preprocessing.py
utils.py
environment.yml
```

Large NetCDF and pickle files in `./data` are not included in the repository. Contact the author if needed.

## Data Sources

* CMIP6 HighResMIP simulations (CEDA Archive)
  [https://hrcm.ceda.ac.uk/research/cmip6-highresmip/](https://hrcm.ceda.ac.uk/research/cmip6-highresmip/)
* NPD eORCA025 simulations (NOC/MSM)
* ERA5, JRA55-do (Copernicus Climate Data Store)
* ORAS5 ocean reanalysis
* EN4 observational dataset

## Workflow

Run the following in order (requires data access):

1. `data_prep.ipynb` – prepares anomalies and conservative temperature fields.
2. `enso_preprocessing.py` – ENSO-focused preprocessing and event diagnostics.
3. `plotting.ipynb` – generates all figures.

## Contact

**Sreevathsa Golla** – [sg13n23@soton.ac.uk](mailto:sg13n23@soton.ac.uk)

### Acknowledgements

Portions of the repository (code formatting/tidying up and documentation cleanup) were assisted using generative AI tools (i.e., ChatGPT 5.1 and Claude Sonnet 4.5). All scientific content, analysis logic, and results were developed by the primary author. Please feel free to contact the author if you require any further clarification.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
