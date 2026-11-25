# ENSO-HighResMIP-TempAnoms 

## Surface and subsurface ENSO-related ocean temperature anomalies in CMIP6 HighResMIP: Model biases and their persistence under scenario forcing

This repository contains all processing and plotting code used in the associated research article.

## Environment

```bash
conda env create -f environment.yml
conda activate enso_highresmip_tempanoms
```

## Directory Structure

```
data/
    anomalies/                  # [*.nc gitignored] Output from data_prep.ipynb: temperature anomalies of each dataset
    processed/                  # Output from enso_preprocessing.py, appendix_A.ipynb and appendix_B.ipynb (see below)
                                # [gitignored: the three ENSO_EVENT_FLAVOURS_<period>.pkl files, and
                                # ENSO_SSTA_COMP_DIFF_RESULTS_WITH_SIG.nc and ENSO_N34_HOV_COMP_DIFF_RESULTS_WITH_SIG.nc]
        ENSO_TEMPANOM_COMPOSITES/   # [*.nc gitignored] Per-dataset ENSO composites (enso_preprocessing.py) and the NPD/OBS event-pairing CSVs (appendix_A.ipynb)
        EXT_TEMP_BOOTSTRAP/         # [*.nc gitignored] Bootstrapped extreme-temperature extent and intensity profiles (appendix_A.ipynb, see below)
    thetao_con/                 # [*.nc gitignored] Conservative temperature fields
    zon_wnds_850hpa/            # [*.nc gitignored] 850 hPa zonal wind + anomalies
    z20_depths/                 # [*.nc gitignored] Depth of 20°C isotherm (Z20) timeseries
    griddes_025.grd             # CDO grid description for 0.25° grid
    pac_mask_025.npz            # Land-sea mask on 0.25° grid (not read by the current notebooks)
    Centred_BS_bounds.pkl       # Climatological baseline information
figures/                        # Figures: figure_2-13 from plotting.ipynb, figure_B* and figure_C* from the appendix notebooks
                                # (files ending in _ptN are panels that are merged into the final figure)
appendices/
    appendix_A.ipynb            # Appendix A: statistical tests and uncertainty estimates
    appendix_B.ipynb            # Appendix B: sensitivity of SCEN - HIST comparisons to residual warming in 2041-2050
    appendix_C.ipynb            # Appendix C: supplementary figures
data_prep.ipynb
plotting.ipynb
enso_preprocessing.py
utils.py
environment.yml
CTRL_MODEL_DRIFT/               # [gitignored] CTRL sensitivity checks (not presented in the manuscript)
```

Items marked `[gitignored]` (or `[*.nc gitignored]` for the NetCDF files inside a folder) are excluded from the repository through `.gitignore`, as they are large. Contact the author if needed.

### Contents of `data/processed/`

Files are written for three period labels: `HIST_EVAL_1979-2014` (CTRL, HIST, NPD and OBS datasets), `FUTR_EVAL_2015-2050` (SCEN, the full scenario, called `SCEN_F` in the appendices) and `TRNC_SCEN_2015-2040` (SCEN, the truncated scenario, called `SCEN_T`).

| Files | Created by | Content |
|---|---|---|
| `N34_INDEX_<period>.csv`, `ONI_<period>.csv`, `ONI_STATS_<period>.csv` | `enso_preprocessing.py` | Niño 3.4 index, ONI and ONI statistics of each dataset |
| `ENSO_EVENT_TIMESTAMPS_<period>.pkl`, `ENSO_EVENT_FLAVOURS_<period>.pkl` | `enso_preprocessing.py` | El Niño / La Niña event months, and the event flavours (CP, EP, Mixed) |
| `ENSO_FLAVOUR_COUNTS_<period>.csv`, `ENSO_FLAVOUR_PROPORTIONS_<period>.csv` | `enso_preprocessing.py` | Number and proportion of events of each flavour |
| `GAUSS_KDE_5S5N_<period>.nc` | `enso_preprocessing.py` | Gaussian KDE of the location of peak \|SSTA\| within 5°S-5°N |
| `N34_HOV_ZVT_TOP300M_<period>.nc` | `enso_preprocessing.py` | Niño 3.4 depth-time (Hovmöller) anomalies of the top 300 m |
| `ENSO_TEMPANOM_COMPOSITES/` | `enso_preprocessing.py`, `appendix_A.ipynb` | ENSO composites of each dataset (`ENSO_TEMPANOM_COMPOSITES_<dataset>_<period>.nc`); event-pairing CSVs of the NPD and OBS groups |
| `ENSO_SSTA_COMP_DIFF_RESULTS_WITH_SIG.nc`, `ENSO_N34_HOV_COMP_DIFF_RESULTS_WITH_SIG.nc` | `appendix_A.ipynb` | Group-mean composites (CTRL, HIST, SCEN_F, SCEN_T, NPD, OBS) and group differences (HIST-OBS, CTRL-OBS, NPD-OBS, SCEN_F-HIST, SCEN_T-HIST, HIST-CTRL) of the SSTA maps and the Niño 3.4 Hovmöllers, with bootstrap significance masks |
| `EXT_TEMP_BOOTSTRAP/` | `appendix_A.ipynb` | See below |
| `PSD_PEAK_TEST_RED_NOISE.pkl`, `PSD_PEAK_SHIFT_TEST_PHASE_RANDOM.pkl` | `appendix_A.ipynb` | Results of the power-spectra tests (peak against red noise; shift of the dominant period between two runs) |
| `ENSO_STATS_DIFF_SCEN_F-T-HIST_2015-2050.csv` | `appendix_B.ipynb` | Table B1: SCEN - HIST changes in ENSO characteristics for the full and truncated scenarios |

### `EXT_TEMP_BOOTSTRAP/`

Bootstrapped profiles over depth (0-300 m, every 10 m) of the extent and intensity of the extreme temperature anomalies (beyond 2 standard deviations) of the ENSO composites, summarised in the Niño 3.4 region (5°S-5°N, 190-240°E). They are used for Figures 6, 11 and B2 (read by `plotting.ipynb` and `appendix_B.ipynb`). There is one file per group and ENSO phase:

```
ENSO_EXT_TEMP_[BIAS_]<label>_<phase>_b2000s<seed>.nc
```

* `<label>` is the group: `CTRL`, `HIST`, `SCEN_F`, `SCEN_T`, `NPD` or `OBS`; `<phase>` is `EL-NINO` or `LA-NINA`.
* Files without `BIAS_` hold the group mean (variable `M`); files with `BIAS_` hold the difference to a reference group (variable `D`): CTRL, HIST and NPD minus OBS, and SCEN_F and SCEN_T minus HIST (paired by model). OBS has no `BIAS_` file. This gives 22 files: 12 group means and 10 differences.
* `b2000` is the number of bootstrap draws and `s<seed>` the seed of the random number generator, which is set per group, phase and file type.
* Dimensions are `(type, depth, boot)`: `type` is `EXTENT` (latitudinal extent, in degrees of latitude) or `INTENSITY` (mean anomaly of the extremes). Variables are `M` or `D`, its p-value `p`, the boolean significance mask `sig`, the 95% confidence interval `lo` and `hi`, and all bootstrap draws `boot_var`.
* Each combination took too long to run in sequence, so the bootstrap was run as a SLURM array job (one task per file). The script itself is not part of this repository; a copy of the code is shown in Appendix A.2 of `appendices/appendix_A.ipynb`.

## Data Sources

* CMIP6 HighResMIP simulations (CEDA Archive)
  [https://hrcm.ceda.ac.uk/research/cmip6-highresmip/](https://hrcm.ceda.ac.uk/research/cmip6-highresmip/)
* NPD eORCA025 simulations (NOC/MSM)
* ERA5, JRA55-do (Copernicus Climate Data Store)
* ORAS5 ocean reanalysis
* EN4 observational dataset

## Workflow

Run the following in order (requires data access):

1. `data_prep.ipynb`: prepares anomalies, conservative temperature fields and other necessary fields.
2. `enso_preprocessing.py`: ENSO-focused preprocessing and event diagnostics.
3. `appendices/appendix_A.ipynb`: statistical tests and uncertainty estimates (Appendix A). It writes the composite differences with significance, the extreme-temperature bootstrap files and the power-spectra test results that `plotting.ipynb` and the other appendix notebooks read. The bootstrap and surrogate tests are slow, so the notebook is not meant to be re-run in full.
4. `plotting.ipynb`: generates the figures of the article (Figures 2-13).
5. `appendices/appendix_B.ipynb` and `appendices/appendix_C.ipynb`: generate the figures and table of Appendices B and C (figures B1-B2, Table B1, figures C1-C8).

Paths are relative: run `data_prep.ipynb`, `enso_preprocessing.py` and `plotting.ipynb` from the repository root, and the appendix notebooks from the `appendices/` folder.

## Contact

**Sreevathsa Golla**: [sg13n23@soton.ac.uk](mailto:sg13n23@soton.ac.uk)

### Acknowledgements

Portions of this repository underwent code refactoring and documentation cleanup assisted by generative AI tools (Claude Opus 5.5 and Sonnet 5.5). All scientific content, analysis logic, and results were developed by the primary author. The plotting and core analysis scripts were verified after refactoring by confirming that outputs matched those produced beforehand. The preprocessing scripts, which produce large datasets and could not feasibly be re-run in full, were manually reviewed to confirm that only formatting and documentation were changed, with no modification to the code logic. Please feel free to contact the author if you require any further clarification or would like to raise any issues.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
