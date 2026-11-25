"""
Code for ENSO-related preprocessing data from conservative temperature anomalies 
from CMIP6 HighResMIP model simulations, NPD_eORCA025 atmosphere-forced ocean-only 
model runs, and observations-based datasets
-------------------------------------------------------------------------------------------
Author: Sreevathsa G. (sg13n23@soton.ac.uk; ORCID ID: 0000-0003-4084-9677)
Last updated: 25 November 2026
"""

# Standard library
from itertools import product
import pickle
import warnings

# Third-party
import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

# Local modules
from utils import *

# Runtime configuration and Dask setup
dask.config.set({"distributed.admin.large-graph-warning-threshold": "1GiB"})
warnings.filterwarnings("ignore", message=r"Sending large graph of size", category=UserWarning, module=r"distributed\.client")

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Niño 3.4 lat/lon bounds
NINO34_LAT = slice(-5, 5)
NINO34_LON = slice(190, 240)

# Regularly spaced lat/lon grid for KDE calculations
KDE_LATS = np.arange(-5, 5.25, 0.25)
KDE_LONS = np.arange(140, 300.25, 0.25)

# Depth range for interpolation to standard vertical grid
HOV_DEPTH = np.arange(0, 300.1, 0.5)

# Power Spectral Density (PSD) settings using Nino 3.4 monthly SSTAs
PSD_SETTINGS = {'nperseg': 240, 'period_min': 0.5, 'period_max': 20.0, 'fs': 12.0}

# ENSO thresholds
ENSO_THRESHOLD = 0.5  # Temperature anomaly threshold for ENSO event determination
GAMMA_CP_EP = 0.2  # Gamma threshold for ENSO 'flavour' determination

# Mapping from El Niño/La Niña output keys to column groups
KEY_MAP = {"counts_elnino": ["EP-EN", "CP-EN", "MIXED-EN", "TOTAL-EN"],
           "counts_lanina": ["EP-LN", "CP-LN", "MIXED-LN", "TOTAL-LN"],}

# Dataset name lists
HIGHRESMIP_MODELS = ["HadGEM3-GC31-HH", "EC-Earth3P-HR", "CNRM-CM6-1-HR",
                     "CMCC-CM2-VHR4", "MPI-ESM1-2-HR", "MPI-ESM1-2-XR"]
# HighResMIP forcing groups evaluated in each period: CTRL (control-1950) and HIST (hist-1950)
# over the historical period; SCEN (highres-future) over the future period
HIST_EVAL_GROUPS = ["CTRL", "HIST"]
FUTR_EVAL_GROUPS = ["SCEN"]
HIGHRESMIP_HIST_EVAL_EXPS = [f"{model}_{group}" for model, group in product(HIGHRESMIP_MODELS, HIST_EVAL_GROUPS)]
HIGHRESMIP_FUTR_EVAL_EXPS = [f"{model}_{group}" for model, group in product(HIGHRESMIP_MODELS, FUTR_EVAL_GROUPS)]

NPD_EXPS = ["NPD_eORCA025_ERA5", "NPD_eORCA025_JRA55"]
OBS = ["EN4", "ORAS5"]

# Dataset lists for each evaluation period
DATASET_LIST_HIST_EVAL = HIGHRESMIP_HIST_EVAL_EXPS + NPD_EXPS + OBS  # 1979-2014
DATASET_LIST_FUTR_EVAL = HIGHRESMIP_FUTR_EVAL_EXPS  # 2015-2050

# ENSO flavour columns
ENSO_FLAVOUR_COLS = ["EP-EN", "CP-EN", "MIXED-EN", "TOTAL-EN", "EP-LN", "CP-LN", "MIXED-LN", "TOTAL-LN"]

# ONI statistics columns
ONI_STATS_COLS = ["DATASET", "PSD_PEAK_T", "EN_MEAN_DUR", "LN_MEAN_DUR", "EN_MEAN_MAX_INTENSITY", 
                  "LN_MEAN_MAX_INTENSITY", "STD_ONI", "ASYMMETRY_RATIO"]


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_data_containers(dataset_list, time_index):
    """Initialise all data containers for preprocessing."""
    # Niño3.4 index and ONI containers
    n34_data = pd.DataFrame(columns=dataset_list, index=time_index)
    oni_data = n34_data.copy(deep=True)
    
    # KDE container
    gaussian_kde_netcdf = xr.Dataset(coords={"dataset": dataset_list, "enso_phase": ['EL-NINO', 'LA-NINA'], 
                                            "lon": KDE_LONS, "lat": KDE_LATS})
    gaussian_kde_netcdf["GAUSS_KDE"] = xr.DataArray(np.full((len(dataset_list), 2, len(KDE_LATS), len(KDE_LONS)), np.nan),
                                                    dims=("dataset", "enso_phase", "lat", "lon"))
    
    # ONI stats container
    df_oni_stats = pd.DataFrame(columns=ONI_STATS_COLS)
    
    # ENSO flavour counts container
    enso_flavour_counts = pd.DataFrame(index=dataset_list, columns=ENSO_FLAVOUR_COLS)
    
    # Per-dataset ENSO event metadata
    enso_events = {}
    enso_event_flavour_eof_method = {}
    
    # Niño3.4 Z-T Hovmöller containers
    n34_hov_ds_list = []
    
    return (n34_data, oni_data, gaussian_kde_netcdf, df_oni_stats, enso_flavour_counts, 
            enso_events, enso_event_flavour_eof_method, n34_hov_ds_list)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_dataset(dataset, year_start, year_end, time_index):
    """
    Load and preprocess dataset for the specified time period.
    
    Reads interpolated conservative temperature anomaly files (Pacific domain: 
    120°E - 290°E; 30°S - 30°N) and slices to the time period of interest.
    Reindexes to a pre-defined time index to ensure all datasets share the 
    common time axis.
    """
    ds_pac = renamer(xr.open_mfdataset(f"./data/anomalies/{dataset}/BILINTERP_PAC_ANOM_*{dataset}_*.nc",
                                       combine="by_coords")).sel(time=slice(year_start, year_end)).reindex(time=time_index)
    return ds_pac


# =============================================================================
# NINO 3.4 INDEX CALCULATIONS
# =============================================================================

def compute_nino34_indices(ds_pac):
    """
    Calculate Niño 3.4 index and ONI.
    
    Returns:
    n34_series: Area-averaged SSTA in Niño 3.4 region
    oni_series: 3-month rolling mean of Niño 3.4 index
    """
    # Subset the Niño 3.4 box
    ds_nino34 = ds_pac.sel(lat=NINO34_LAT, lon=NINO34_LON)
    
    # Calculate area-weighted mean
    lat_weights = np.cos(np.deg2rad(ds_nino34["lat"]))
    anom_aw_mean = ds_nino34["anomaly"].weighted(lat_weights).mean(dim=["lat", "lon"])
    
    # Extract surface values
    n34_series = (anom_aw_mean.sel(depth=0, method="nearest").drop_vars("depth").sortby("time").values)
    
    return n34_series, anom_aw_mean


def calculate_oni_statistics(n34_series, oni_series, el_ts, ln_ts):
    """Calculate various statistics from ONI data."""
    # Dominant ENSO period from PSD
    _, _, psd_peak_t, _ = compute_psd(n34_series, **PSD_SETTINGS)
    
    # Mean ENSO durations
    en_mean_dur = np.nanmean([len(event_ts) for event_ts in el_ts])
    ln_mean_dur = np.nanmean([len(event_ts) for event_ts in ln_ts])
    
    # Mean of peak ENSO event intensity
    en_mean_max_intensity = np.nanmean([abs(oni_series.loc[event_ts]).max() for event_ts in el_ts])
    ln_mean_max_intensity = np.nanmean([abs(oni_series.loc[event_ts]).max() for event_ts in ln_ts])
    
    # Standard deviation and asymmetry
    std_oni = oni_series.std()
    asymmetry_ratio = en_mean_max_intensity / ln_mean_max_intensity
    
    return {'PSD_PEAK_T': psd_peak_t, 'EN_MEAN_DUR': en_mean_dur, 'LN_MEAN_DUR': ln_mean_dur,
            'EN_MEAN_MAX_INTENSITY': en_mean_max_intensity, 'LN_MEAN_MAX_INTENSITY': ln_mean_max_intensity,
            'STD_ONI': std_oni, 'ASYMMETRY_RATIO': asymmetry_ratio}


# =============================================================================
# ENSO EVENT PROCESSING
# =============================================================================

def build_event_timestamps(enso_event_flavour):
    """
    Build a flat list of ENSO events, El Niño first then La Niña (this order defines event_id), each item:
    (phase_type, flavour_type, event_timestamps, start, end, peak)
    """
    event_ts = []
    for phase, key in (('EL-NINO', 'events_elnino'), ('LA-NINA', 'events_lanina')):
        for event in enso_event_flavour[key]:
            event_ts.append((phase, event['label'],
                             list(pd.date_range(start=event['start'], end=event['end'], freq='1MS')),
                             pd.Timestamp(event['start']), pd.Timestamp(event['end']), pd.Timestamp(event['peak_time'])))
    return event_ts


def compute_enso_composites(ds_pac, event_ts, dataset):
    """Compute event-level ENSO composites and merge into a single dataset with per-event metadata coordinates."""
    composite_ds = []
    for event_id, (event_type, reg_type, event_timesteps, start, end, peak) in enumerate(event_ts):
        da = (ds_pac.sel(time=event_timesteps)
                    .mean(dim="time")                              # keeps (depth, lat, lon) or (lat, lon) depending on ds_pac
                    .rename({"anomaly": "event_level_composite"})
                    .expand_dims(event_id=[event_id], dataset=[dataset]))
        
        da = da.assign_coords(
            event_reg_type=("event_id", [reg_type]),
            event_type=("event_id", [event_type]),
            event_start=("event_id", np.array([start], dtype="datetime64[ns]")),   # first month of the event
            event_end=("event_id", np.array([end], dtype="datetime64[ns]")),       # last month of the event
            event_peak=("event_id", np.array([peak], dtype="datetime64[ns]")),     # month of peak |ONI|
        )
        composite_ds.append(da)
    return xr.merge(composite_ds).compute()

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_single_dataset(dataset, year_start, year_end, time_index,
                           n34_data, oni_data, gaussian_kde_netcdf, df_oni_stats,
                           enso_flavour_counts, enso_events, enso_event_flavour_eof_method,
                           n34_hov_ds_list, period_label):
    """Process a single dataset through the full preprocessing pipeline."""
    # Load dataset
    ds_pac = load_dataset(dataset, year_start, year_end, time_index)
    
    # Calculate Niño 3.4 index and ONI
    n34_series, anom_aw_mean = compute_nino34_indices(ds_pac)
    n34_data[dataset] = n34_series
    oni_data[dataset] = n34_data[dataset].rolling(window=3, center=True).mean()
    
    # Identify ENSO events
    el_ts, ln_ts = sequence_finder(n34_data[dataset].rename("SSTA").to_frame(),t=ENSO_THRESHOLD)
    enso_events[dataset] = {"EL-NINO": el_ts, "LA-NINA": ln_ts}
    
    # Calculate ONI statistics
    oni_stats = calculate_oni_statistics(n34_data[dataset], oni_data[dataset], el_ts, ln_ts)
    df_oni_stats.loc[len(df_oni_stats)] = [dataset] + list(oni_stats.values())
    
    # Niño 3.4 Z-T Hovmöller (top 300m)
    n34_hov_ds_list.append(anom_aw_mean.interp(depth=HOV_DEPTH).bfill(dim="depth").expand_dims(dataset=[dataset]))
    
    # Gaussian KDE for peak anomaly locations
    for enso_phase in ['EL-NINO', 'LA-NINA']:
        time_stamps = flatten(enso_events[dataset][enso_phase])
        kde_result = compute_enso_peak_kde(ds=ds_pac, event_timestamps=time_stamps,
                                        kde_lats=gaussian_kde_netcdf["lat"].values, 
                                        kde_lons=gaussian_kde_netcdf["lon"].values)
        gaussian_kde_netcdf["GAUSS_KDE"].loc[dict(dataset=dataset, enso_phase=enso_phase)] = kde_result
    
    # Determine ENSO 'flavour' (following Takahashi et al., 2011)
    enso_event_flavour_eof_method[dataset] = classify_cp_ep_from_ssta(ssta=ds_pac["anomaly"].sel(depth=0, method="nearest"),
                                                                      enso_events_dict=enso_events, dataset_name=dataset, gamma=GAMMA_CP_EP)
    
    # Count ENSO events by flavour
    for key, cols_out in KEY_MAP.items():
        counts = enso_event_flavour_eof_method[dataset][key]
        enso_flavour_counts.loc[dataset, cols_out] = [counts["EP"], counts["CP"], counts["Mixed"], counts["Total"]]
    
    # Build event timestamps by flavour
    event_ts = build_event_timestamps(enso_event_flavour_eof_method[dataset])
    
    # Compute and save composites
    composite_ds = compute_enso_composites(ds_pac, event_ts, dataset)
    composite_ds.to_netcdf(f'./data/processed/ENSO_TEMPANOM_COMPOSITES/ENSO_TEMPANOM_COMPOSITES_{dataset}_{period_label}.nc')


# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================

def compute_flavour_proportions(enso_flavour_counts):
    """Calculate ENSO flavour proportions from counts."""
    return pd.DataFrame([
        (enso_flavour_counts["EP-EN"] / enso_flavour_counts["TOTAL-EN"]).rename("EP-EN"),
        (enso_flavour_counts["CP-EN"] / enso_flavour_counts["TOTAL-EN"]).rename("CP-EN"),
        (enso_flavour_counts["MIXED-EN"] / enso_flavour_counts["TOTAL-EN"]).rename("MIXED-EN"),
        (enso_flavour_counts["EP-LN"] / enso_flavour_counts["TOTAL-LN"]).rename("EP-LN"),
        (enso_flavour_counts["CP-LN"] / enso_flavour_counts["TOTAL-LN"]).rename("CP-LN"),
        (enso_flavour_counts["MIXED-LN"] / enso_flavour_counts["TOTAL-LN"]).rename("MIXED-LN"),
    ]).transpose()


def save_processed_data(oni_data, n34_data, gaussian_kde_netcdf, enso_events, enso_event_flavour_eof_method, 
                        enso_flavour_counts, flavour_proportions, df_oni_stats, n34_hov_ds_list, period_label):
    """Save all processed datasets to disk."""
    # Save CSV files
    oni_data.to_csv(f"./data/processed/ONI_{period_label}.csv")
    n34_data.to_csv(f"./data/processed/N34_INDEX_{period_label}.csv")
    enso_flavour_counts.to_csv(f"./data/processed/ENSO_FLAVOUR_COUNTS_{period_label}.csv")
    flavour_proportions.to_csv(f"./data/processed/ENSO_FLAVOUR_PROPORTIONS_{period_label}.csv")
    
    df_oni_stats = df_oni_stats.round(2)
    df_oni_stats.set_index("DATASET", inplace=True)
    df_oni_stats.to_csv(f"./data/processed/ONI_STATS_{period_label}.csv")
    
    # Save NetCDF files
    gaussian_kde_netcdf.to_netcdf(f"./data/processed/GAUSS_KDE_5S5N_{period_label}.nc")
    
    n34_hov_ds = xr.concat(n34_hov_ds_list, dim="dataset").compute()
    n34_hov_ds.to_netcdf(f"./data/processed/N34_HOV_ZVT_TOP300M_{period_label}.nc")
    
    # Save pickle files
    with open(f"./data/processed/ENSO_EVENT_TIMESTAMPS_{period_label}.pkl", "wb") as f:
        pickle.dump(enso_events, f)
    
    with open(f"./data/processed/ENSO_EVENT_FLAVOURS_{period_label}.pkl", "wb") as f:
        pickle.dump(enso_event_flavour_eof_method, f)


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_period(dataset_list, start_date, end_date, period_label):
    """
    Run full preprocessing pipeline for a given time period.
    
    Parameters:
    -----------
    dataset_list : List of dataset names to process
    start_date : Start date in format 'YYYY-MM-DD'
    end_date : End date in format 'YYYY-MM-DD'
    period_label : Label for the time period (used in output filenames)
    """
    # Extract years and create time index
    year_start = start_date[:4]
    year_end = end_date[:4]
    time_index = pd.date_range(start=start_date, end=end_date, freq="1MS")
    
    # Initialise all data containers
    (n34_data, oni_data, gaussian_kde_netcdf, df_oni_stats, enso_flavour_counts, 
     enso_events, enso_event_flavour_eof_method, n34_hov_ds_list) = initialize_data_containers(dataset_list, time_index)
    
    # Process each dataset
    for dataset in tqdm(dataset_list, desc=f"Processing {period_label}"):
        process_single_dataset(dataset, year_start, year_end, time_index, n34_data, oni_data, gaussian_kde_netcdf, 
                               df_oni_stats, enso_flavour_counts, enso_events, enso_event_flavour_eof_method, 
                               n34_hov_ds_list, period_label)
    
    # Compute flavour proportions
    flavour_proportions = compute_flavour_proportions(enso_flavour_counts)
    
    # Save all processed data
    save_processed_data(oni_data, n34_data, gaussian_kde_netcdf, enso_events, enso_event_flavour_eof_method, 
                        enso_flavour_counts, flavour_proportions, df_oni_stats, n34_hov_ds_list, period_label)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Setup Dask cluster
    cluster, client = setup_dask_cluster()
    print('Dashboard URL:', client.dashboard_link)
    
    # Process historical evaluation period (1979-2014): CTRL, HIST, NPD and OBS
    run_period(dataset_list=DATASET_LIST_HIST_EVAL, start_date="1979-01-01", end_date="2014-12-31", period_label="HIST_EVAL_1979-2014")
    
    # Process future evaluation period (2015-2050): SCEN only
    run_period(dataset_list=DATASET_LIST_FUTR_EVAL, start_date="2015-01-01", end_date="2050-12-31", period_label="FUTR_EVAL_2015-2050")

    # Process truncated scenario period (2015-2040): SCEN only
    run_period(dataset_list=DATASET_LIST_FUTR_EVAL, start_date="2015-01-01", end_date="2040-12-31", period_label="TRNC_SCEN_2015-2040")
    
    # Cleanup
    client.shutdown()
    cluster.close()