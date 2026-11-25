"""
Code for storing utility functions for data pre-processing and plotting for analysis
-------------------------------------------------------------------------------------------
Author: Sreevathsa G. (sg13n23@soton.ac.uk; ORCID ID: 0000-0003-4084-9677)
Last updated: 25 November 2026
"""

# Standard library
import itertools
from typing import Dict, List, Tuple
import warnings

# Third-party
import cartopy.crs as ccrs
from cdo import Cdo
import dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.signal import detrend, welch
from scipy.stats import gaussian_kde
import xarray as xr

# Runtime configuration and clients
cdo = Cdo()
dask.config.set({"distributed.admin.large-graph-warning-threshold": "1GiB"})
warnings.filterwarnings("ignore", message=r"Sending large graph of size", category=UserWarning, module=r"distributed\.client")

# =============================================================================
# GENERIC HELPER FUNCTIONS THAT ARE BROADLY USED ACROSS THE REPOSITORY
# =============================================================================

def setup_dask_cluster(n_workers=6, threads_per_worker=3, memory_limit="48GB"):
    """
    Dask cluster setup (on ANEMONE HPC @ NOCS, UK):
    ----------------------------------------------
    Initialise and return a local Dask cluster and client. Temporary and local worker files are written
    to /dssgfs01/scratch/sg13n23/temp/.

    Parameters
    ----------
    n_workers (int)          : Number of workers. Default is 6.
    threads_per_worker (int) : Threads per worker. Default is 3.
    memory_limit (str)       : Memory limit of each worker. Default is "48GB".

    Returns
    -------
    cluster (dask.distributed.LocalCluster) : Local cluster (dashboard on port 8787).
    client (dask.distributed.Client)        : Client connected to the cluster.
    """
    dask.config.set({"temporary_directory": "/dssgfs01/scratch/sg13n23/temp/",
                     "local_directory": "/dssgfs01/scratch/sg13n23/temp/"})
    cluster = LocalCluster(n_workers=n_workers, 
                           threads_per_worker=threads_per_worker, 
                           memory_limit=memory_limit,
                           dashboard_address=":8787",)
    client = Client(cluster)
    return cluster, client

def renamer(ds_f):
    """
    Standardise coordinate and dimension names for various model outputs.

    This helper makes heterogeneous model files look consistent by
    renaming common variants of depth, lon/lat and time coordinates
    to a shared convention used everywhere else in the analysis.

    It performs the following normalisations if present:

    - 'deptht' or 'lev' -> 'depth'
    - 'longitude'/'latitude' -> 'lon'/'lat' and associated i/j index
    - 'x_2'/'y_2' -> 'x'/'y' (NEMO like staggered grids)
    - 'time_counter' -> 'time'

    Parameters
    ----------
    ds_f : Raw dataset as loaded from file.

    Returns
    -------
    xr.Dataset with harmonised coord and dimension names.
    """
    if 'deptht' in list(ds_f.coords):
        ds_f = ds_f.rename({'deptht': 'depth'})
    if 'lev' in list(ds_f.coords):
        ds_f = ds_f.rename({'lev': 'depth'})
    if 'longitude' in list(ds_f.coords):
        ds_f = ds_f.rename({'longitude': 'lon', 'latitude': 'lat'})
        ds_f = ds_f.rename({'i': 'x', 'j': 'y'})
    if 'y_2' in list(ds_f.dims):
        ds_f = ds_f.rename({'x_2': 'x', 'y_2': 'y'})
    if 'time_counter' in list(ds_f.coords):
        ds_f = ds_f.rename({'time_counter': 'time'})
    return ds_f


def flatten(list_of_lists):
    """
    Flatten a list of lists into a single list.

    This is a thin wrapper around itertools.chain used throughout the
    repository code when collecting lists of event timestamps.
    """
    return list(itertools.chain.from_iterable(list_of_lists))

# =============================================================================
# HELPER FUNCTIONS USED IN data_prep.ipynb NOTEBOOK
# =============================================================================
def cdo_remapbil(filename_list, var_name, plev=850.):
    """
    Remap files to a 0.25° grid using CDO bilinear interpolation.
    If var_name == 'ua', also select the specified pressure level (plev, in hPa).
    Returns the concatenated dataset along 'time'.
    
    NOTE: './data/griddes_025.grd' is a manually configured CDO grid description file.
    Its contents specify a regular lon-lat grid at 0.25° resolution.
    
    Parameters
    ----------
    filename_list (list of str) : List of input NetCDF file paths to process.
    var_name (str)              : Variable name to extract (e.g., 'thetao', 'so', 'thetao_con' or 'ua').
    plev (float)                : Pressure level in hPa (only used when var_name == 'ua').
                                  Converted internally to Pa for CDO ('sellevel'). Default is 850.

    Returns
    -------
    xr.Dataset : Dataset containing the selected variable remapped to a 0.25° grid,
                 concatenated along the 'time' dimension.
    """
    # Imported here so that 'from utils import *' does not replace the tqdm of the calling notebook or script
    from tqdm.auto import tqdm

    ds = []
    for f in tqdm(filename_list):
        if var_name == 'ua':  # For 850 hPa zonal wind: select level (CDO expects Pa)
            ds += [cdo.remapbil('./data/griddes_025.grd',
                                input=f'-selvar,{var_name} -sellevel,{plev*100} '+f,
                                returnXDataset=True)[['ua']]]
        else:  # Other vars: select var and remap, then standardise coords
            ds += [renamer(cdo.remapbil('./data/griddes_025.grd',
                                        input=f'-selvar,{var_name} '+f,
                                        returnXDataset=True))]
    return xr.concat(ds, dim='time')

def z20_calculator(ds):
    """
    Calculate the depth of the 20°C isotherm (Z20) by identifying, for each
    (time, lat, lon), the depth level whose conservative temperature is closest
    to 20°C. data_prep.ipynb crops the data to 5˚S-5˚N and 140˚E-280˚E before calling this function.

    Parameters
    ----------
    ds (xr.Dataset) : Dataset containing 'thetao_con' (conservative temperature)
                      and a 'depth' coordinate.

    Returns
    -------
    xr.DataArray : DataArray named 'z20' giving the depth of the 20°C isotherm
                   for each (time, lat, lon) where valid data exist.
    """
    # Absolute difference |T - 20°C| at all depths
    abs_diff = abs(ds['thetao_con'] - 20)
    # Identifying grid points that have at least one valid depth value
    valid_mask = abs_diff.notnull().any(dim='depth')
    # Masking out locations where all depths are NaN
    abs_diff = abs_diff.where(valid_mask)
    # Finding the depth index where |T - 20| is smallest (closest to 20°C)
    min_index = abs_diff.fillna(np.inf).argmin(dim='depth')
    # Extracting the actual depth values at those indices
    z20 = ds['depth'].isel(depth=min_index).where(valid_mask).rename('z20')
    return z20
    
# =============================================================================
# HELPER FUNCTIONS USED IN enso_preprocessing.py SCRIPT
# =============================================================================

def compute_psd(series, nperseg, period_min, period_max, fs,
                noverlap_frac=0.75, nfft=512):
    """
    Compute Welch power spectral density of a 1D time series.

    The PSD is estimated with scipy.signal.welch using a Hann window.
    Each segment is linearly detrended internally by welch before
    windowing, which removes local trends that a single global detrend
    would miss. The function returns periods rather than frequencies,
    and also identifies the dominant peak within a user supplied period
    band.

    Parameters
    ----------
    series (pd.Series) : Input time series (e.g. Niño 3.4 index) with
                         a datetime index.
    nperseg (int)      : Maximum segment length passed to 'welch'. The actual segment
                         length is 'min(nperseg, len(series))'.
    period_min (float) : Minimum period in years to keep for plotting and
                         peak detection. Outside this band PSD values are discarded.
    period_max (float) : Maximum period in years to keep for plotting and
                         peak detection. Outside this band PSD values are discarded.
    fs (float)         : Sampling frequency in samples per year. For monthly data this
                         is typically 12.0.
    noverlap_frac (float) : Fractional overlap between segments. 0.75 (75%) is
                            recommended for the Hann window to maximise data
                            utilisation from short records. Default is 0.75.
    nfft (int)         : FFT length. Values larger than nperseg zero-pad each segment,
                         producing a denser frequency grid without changing true
                         spectral resolution. Default is 512.

    Returns
    -------
    period (np.ndarray): 1D array of periods (years) retained for plotting. None if the
                         input length is too short or no values fall inside the band.
    Pxx (np.ndarray)   : Welch PSD estimate corresponding to period. Same shape as
                         "period" or None.
    peak_period (float): Period at which the PSD attains its maximum value inside the
                         requested band. None if not defined.
    peak_psd (float)   : PSD value at "peak_period". None if not defined.
    """
    x = series.astype(float).dropna().values
    seg = min(nperseg, len(x))
    if len(x) < max(32, seg):
        # Not enough data to compute a stable PSD
        return None, None, None, None

    # PSD via Welch's method with per-segment linear detrending
    f, Pxx = welch(x, fs=fs, window="hann",
                   nperseg=seg,
                   noverlap=int(seg * noverlap_frac),
                   nfft=nfft,
                   detrend="linear",
                   scaling="density")

    # Converting frequencies to periods in years, guard against divide by zero
    period = 1.0 / np.maximum(f, 1e-12)

    # Restricting to plotting band
    m = (period >= period_min) & (period <= period_max)
    if not np.any(m):
        return None, None, None, None

    # Peak in the plotted band
    idx = np.argmax(Pxx[m])
    return period[m], Pxx[m], period[m][idx], Pxx[m][idx]


def sequence_finder(df, t=0.5):
    """Identify El Niño and La Niña events from a monthly SSTA series.

    The algorithm uses a 3 month running mean of the SSTA column
    (ONI like) and applies a fixed threshold t in degrees Celsius.

    An event is defined as:
    - El Niño: 3 month mean > +t for at least 5 consecutive months
    - La Niña: 3 month mean < -t for at least 5 consecutive months

    Parameters
    ----------
    df (pd.DataFrame) : Must contain a column 'SSTA' and have a datetime index.
                        The function mutates df by adding several helper columns
                        (rolling means, boolean masks and group labels).
    t (float)         : Absolute threshold in degrees Celsius used to 
                        identify events.
    Returns
    -------
    el_nino_event_times : list of list of pandas.Timestamp
        Each inner list contains the timestamps belonging to one
        contiguous El Niño event.
    la_nina_event_times : list of list of pandas.Timestamp
        Each inner list contains the timestamps belonging to one
        contiguous La Niña event.
    """
    # Calculating 3 month rolling mean of SSTA
    df['rolling_mean'] = df['SSTA'].rolling(window=3, center=False).mean()

    # Fixed thresholds for El Niño and La Niña
    el_nino_threshold = t
    la_nina_threshold = -1 * t

    # Identifying where the rolling mean exceeds the El Niño threshold
    # or is below the La Niña threshold
    df['el_nino'] = df['rolling_mean'] > el_nino_threshold
    df['la_nina'] = df['rolling_mean'] < la_nina_threshold

    # Creating grouping IDs for consecutive True values of each condition
    df['el_nino_group'] = df['el_nino'].astype(int).diff().ne(0).cumsum()
    df['la_nina_group'] = df['la_nina'].astype(int).diff().ne(0).cumsum()

    # Filtering groups to those with at least five consecutive months
    el_nino_valid_groups = df[df['el_nino']].groupby('el_nino_group').filter(lambda x: len(x) >= 5)
    la_nina_valid_groups = df[df['la_nina']].groupby('la_nina_group').filter(lambda x: len(x) >= 5)

    # Gathering all timesteps for each valid event
    el_nino_event_times = [group.index.tolist() for _, group in el_nino_valid_groups.groupby('el_nino_group')]
    la_nina_event_times = [group.index.tolist() for _, group in la_nina_valid_groups.groupby('la_nina_group')]

    return el_nino_event_times, la_nina_event_times


def classify_cp_ep_from_ssta(ssta, enso_events_dict, dataset_name, lat_bounds = (-20.0, 20.0),
                             lon_bounds_eastdeg = (120.0, 280.0), gamma: float = 0.2,):
    """
    Classify ENSO events as EP, CP or Mixed using EOF-based indices.

    Given surface SSTA anomalies and a precomputed list of ENSO events
    windows, this function derives an E index and a C index from an EOF
    decomposition over the tropical Pacific and uses them to label
    events as Eastern Pacific (EP), Central Pacific (CP) or Mixed.

    The steps are:

    1. Normalise longitudes to a 0 to 360 grid and subset to a tropical
       domain (default 20S to 20N, 120E to 80W).
    2. Apply sqrt(cos(lat)) area weighting, linearly detrend, and compute the
       EOFs (via SVD) of the anomaly field.
    3. Take EOF1 as the E-like and EOF2 as the C-like pattern (the automatic
       assignment from the Niño 3 and Niño 4 loadings is disabled, see the
       COMMENT NOTE in the code).
    4. Flip signs so that positive E warms Niño 3 and positive C warms
       Niño 4.
    5. Standardise both principal components and rotate them by 45 degrees
       (Takahashi et al. 2011) to form the E_index and C_index time series.
       Compute the Niño 3, Niño 4 and Niño 3.4 indices from ssta.
    6. For each event window, pick the peak month (max Niño 3.4 for El
       Niño, min for La Niña) and compare |E| and |C| at that time:

       - if |E| - |C| >= gamma  -> EP
       - if |C| - |E| >= gamma  -> CP
       - otherwise              -> Mixed

    Parameters
    ----------
    ssta (xr.DataArray)     : Surface SST anomalies with dims (time, lat, lon). 
                              Time must be monthly. Lon can be 0 to 360 or -180 to 180 
                              and will be internally mapped to 0 to 360.
    enso_events_dict (dict) : Nested dictionary of event windows keyed by dataset 
                              name and by phase, for example:
                                {
                                  'HadGEM3-GC31-HH_CTRL': {
                                      'EL-NINO': [[ts1, ts2, ...], ...],
                                      'LA-NINA': [[ts1, ts2, ...], ...]
                                  },
                                  ...
                                }

    dataset_name (str)      : Key into enso_events_dict that selects the events to classify for this call.
    lat_bounds (tuple)      : Latitude bounds for the EOF domain. Default is 20S to 20N.
    lon_bounds_eastdeg (tuple) : Longitude bounds in degrees east on a 0 to 360 grid. Default is
                                120˚E to 80˚W, expressed as (120, 280).
    gamma (float)           : Dominance margin in standard deviation units used to separate
                              EP and CP from Mixed events.

    Returns
    -------
    result (dict)           : Dictionary with the following keys:
    
                              - 'counts_elnino' : mapping with keys 'EP', 'CP', 'Mixed', 'Total'
                              - 'counts_lanina' : mapping with keys 'EP', 'CP', 'Mixed', 'Total'
                              - 'events_elnino' : list of per event dicts for El Niño, with
                                                  start, end, peak_time, E, C, nino34 and label
                              - 'events_lanina' : list of per event dicts for La Niña
                              - 'indices'       : xr.Dataset containing E_index, C_index, Niño 3,
                                                  Niño 4, Niño 3.4 and the first two principal components
                              - 'eof_patterns'  : xr.Dataset with the raw EOF1 and EOF2 patterns (EOF1_raw,
                                                  EOF2_raw) and the rotated E and C patterns (EOF_E, EOF_C)
                                                  over the EOF domain
                              - 'flags'         : dict of booleans: 'EOF_FLIP' (always False, as the EOFs are
                                                  no longer swapped) and 'E_PATTERN_SIGN_FLIP' /
                                                  'C_PATTERN_SIGN_FLIP' (whether the sign of the pattern was flipped)
    """
    # Basic checks and converting longitude to 0 to 360 if needed
    if not set(['time', 'lat', 'lon']).issubset(set(ssta.dims) | set(ssta.coords)):
        raise ValueError("ssta must have dims/coords: time, lat, lon")

    def _to0360(lonvals):
        return ((lonvals % 360) + 360) % 360

    if float(ssta.lon.min()) < 0 or float(ssta.lon.max()) <= 180:
        ssta = ssta.assign_coords(lon=_to0360(ssta.lon)).sortby('lon')

    # 1. Subsetting to EOF domain
    lat_lo, lat_hi = lat_bounds
    lon_lo, lon_hi = lon_bounds_eastdeg

    sub = ssta.sel(lat=slice(lat_lo, lat_hi))
    if lon_lo <= lon_hi:
        sub = sub.sel(lon=slice(lon_lo, lon_hi))
    else:
        # Domain crosses the dateline, so stitch two slices together
        sub = xr.concat([sub.sel(lon=slice(0, lon_hi)), sub.sel(lon=slice(lon_lo, 360))], dim='lon')

    # 2. Area weighting, linear detrending and SVD EOF
    W = np.sqrt(np.cos(np.deg2rad(sub.lat)))
    A_loaded = (sub * W).transpose('time', 'lat', 'lon').load()  # SVD is not lazy

    nt, nlat, nlon = A_loaded.shape
    X = A_loaded.values.reshape(nt, nlat * nlon)

    # Masking all NaN columns
    good = ~np.all(np.isnan(X), axis=0)
    Xg = np.nan_to_num(X[:, good], nan=0.0)

    # Linear detrending along time axis (column-wise), also removes mean
    Xg = detrend(Xg, axis=0, type='linear')

    U, S, Vt = np.linalg.svd(Xg, full_matrices=False)
    PCs  = U * S
    EOFs = Vt

    # Reshaping first two EOFs back to (lat, lon)
    eof1 = np.full(nlat * nlon, np.nan)
    eof2 = np.full(nlat * nlon, np.nan)
    eof1[good] = EOFs[0, :]
    eof2[good] = EOFs[1, :]

    eof1 = xr.DataArray(eof1.reshape(nlat, nlon), coords={'lat': sub.lat, 'lon': sub.lon}, dims=('lat', 'lon'))
    eof2 = xr.DataArray(eof2.reshape(nlat, nlon), coords={'lat': sub.lat, 'lon': sub.lon}, dims=('lat', 'lon'))

    pc1 = xr.DataArray(PCs[:, 0], coords={'time': sub.time}, dims=('time',)).rename('PC1')
    pc2 = xr.DataArray(PCs[:, 1], coords={'time': sub.time}, dims=('time',)).rename('PC2')

    # 3. Assigning EOF1 to E (EP-like) and EOF2 to C (CP-like). The automatic assignment from the Niño 3 vs Niño 4 loadings is disabled (see below)
    def box_mean(da2, lat_bnds, lon_bnds):
        (lat_lo2, lat_hi2), (lo, hi) = lat_bnds, lon_bnds
        tmp = da2.sel(lat=slice(lat_lo2, lat_hi2))
        if lo <= hi:
            tmp = tmp.sel(lon=slice(lo, hi))
        else:
            tmp = xr.concat([tmp.sel(lon=slice(0, hi)), tmp.sel(lon=slice(lo, 360))], dim='lon')
        w = np.cos(np.deg2rad(tmp.lat))
        w = w / w.mean()
        return (tmp * w.broadcast_like(tmp)).mean(('lat', 'lon'), skipna=True)

    N3_box = ((-5, 5), (210, 270))   # 150W to 90W
    N4_box = ((-5, 5), (160, 210))   # 160E to 150W
    N34_box = ((-5, 5), (190, 240))  # 170W to 120W

    eof1_n3 = float(box_mean(eof1, *N3_box).values)
    eof1_n4 = float(box_mean(eof1, *N4_box).values)
    eof2_n3 = float(box_mean(eof2, *N3_box).values)
    eof2_n4 = float(box_mean(eof2, *N4_box).values)

    flags = {'EOF_FLIP': False, 'E_PATTERN_SIGN_FLIP': False, 'C_PATTERN_SIGN_FLIP': False}
    
    # # Choosing EOF that has relatively stronger N3 loading as E like pattern
    # if (eof1_n3 - eof1_n4) >= (eof2_n3 - eof2_n4):
    #     E_pattern, C_pattern = eof1.copy(), eof2.copy()
    #     E_pc, C_pc = pc1.copy(), pc2.copy()
    # else:
    #     E_pattern, C_pattern = eof2.copy(), eof1.copy()
    #     E_pc, C_pc = pc2.copy(), pc1.copy()
    #     flags['EOF_FLIP'] = True
    
    # COMMENT NOTE: Flipping EOFs was causing incorrect reassignment for some simulations 
    # where EOF1 and EOF2 were not being cleanly separated in spatial variability signature 
    # (leading to noisy/incorrect Niño 3 minus Niño 4 differential). After visually checking 
    # raw EOFs, it happened to be the case that EOF1 was always corresponding to E-like pattern.

    E_pattern, C_pattern = eof1.copy(), eof2.copy()
    E_pc, C_pc = pc1.copy(), pc2.copy()
    
    # 4. Fixing signs so that positive E warms Niño 3 and positive C warms Niño 4
    if float(box_mean(E_pattern, *N3_box).values) < 0:
        E_pattern = -E_pattern
        E_pc = -E_pc
        flags['E_PATTERN_SIGN_FLIP'] = True
    if float(box_mean(C_pattern, *N4_box).values) < 0:
        C_pattern = -C_pattern
        C_pc = -C_pc
        flags['C_PATTERN_SIGN_FLIP'] = True
        
    # 5. Standardise PCs BEFORE rotation (ensures equal variance)
    E_pc_std = (E_pc - E_pc.mean()) / E_pc.std()
    C_pc_std = (C_pc - C_pc.mean()) / C_pc.std()

    # 45-degree rotation (Takahashi et al. 2011)
    E = ((E_pc_std - C_pc_std) / np.sqrt(2)).rename('E_index')
    C = ((E_pc_std + C_pc_std) / np.sqrt(2)).rename('C_index')

    # Rotate spatial patterns
    E_pat_orig = E_pattern.copy()
    C_pat_orig = C_pattern.copy()
    E_pattern = (E_pat_orig - C_pat_orig) / np.sqrt(2)
    C_pattern = (E_pat_orig + C_pat_orig) / np.sqrt(2)

    # Niño indices
    nino3  = box_mean(ssta, *N3_box).rename('nino3')
    nino4  = box_mean(ssta, *N4_box).rename('nino4')
    nino34 = box_mean(ssta, *N34_box).rename('nino34')

    indices = xr.Dataset({'E_index': E, 'C_index': C, 'PC1': pc1, 
                          'PC2': pc2, 'nino3': nino3, 
                          'nino4': nino4, 'nino34': nino34,})

    # EOF Patterns: Both 'raw' and rotated E/C spatial patterns
    eof_patterns = xr.Dataset({'EOF1_raw': eof1.rename('EOF1_raw'),
                               'EOF2_raw': eof2.rename('EOF2_raw'),
                               'EOF_E': E_pattern.rename('EOF_E'),
                               'EOF_C': C_pattern.rename('EOF_C'),})

    # 6. Helper to classify one list of events given a peak selector
    def classify_events(event_windows: List[List[pd.Timestamp]], peak_selector: str) -> Tuple[Dict[str, int], List[Dict]]:
        EP = CP = MIX = 0
        details: List[Dict] = []
        for ev in event_windows:
            times = pd.to_datetime(ev)
            # Intersect with the index of the indices dataset
            times = times.intersection(pd.to_datetime(indices.time.values))
            if times.empty:
                continue
            subI = indices.sel(time=times)
            if peak_selector == 'warm':
                # For El Niño picking time of maximum Niño 3.4
                ix = int(np.nanargmax(subI['nino34'].values))
            else:
                # For La Niña picking time of minimum Niño 3.4
                ix = int(np.nanargmin(subI['nino34'].values))
            tpeak = pd.to_datetime(times[ix])
            e_val = float(indices['E_index'].sel(time=tpeak).values)
            c_val = float(indices['C_index'].sel(time=tpeak).values)
            n34 = float(indices['nino34'].sel(time=tpeak).values)

            if (abs(e_val) - abs(c_val)) >= gamma:
                lab = 'EP'
                EP += 1
            elif (abs(c_val) - abs(e_val)) >= gamma:
                lab = 'CP'
                CP += 1
            else:
                lab = 'Mixed'
                MIX += 1

            details.append({'start': pd.to_datetime(times.min()), 'end': pd.to_datetime(times.max()),
                            'peak_time': tpeak, 'E': e_val, 'C': c_val, 'nino34': n34, 'label': lab,})
        return ({'EP': EP, 'CP': CP, 'Mixed': MIX, 'Total': EP + CP + MIX}, details)

    # 6. Applying to El Niño (warm) and La Niña (cold) lists
    el_list = enso_events_dict.get(dataset_name, {}).get('EL-NINO', [])
    ln_list = enso_events_dict.get(dataset_name, {}).get('LA-NINA', [])

    counts_el, events_el = classify_events(el_list, peak_selector='warm')
    counts_ln, events_ln = classify_events(ln_list, peak_selector='cold')

    return {'counts_elnino': counts_el, 'counts_lanina': counts_ln, 'events_elnino': events_el,
            'events_lanina': events_ln, 'indices': indices, 'eof_patterns': eof_patterns, 'flags': flags}


# LEGACY FUNCTION: No longer used in any script or notebook of this repository (kept for reference)
def calculate_cdam(ds, threshold, method="sign"):
    """
    Calculate Cross-Dataset Agreement Mask (CDAM) and group mean composites
    for both El Niño and La Niña phases.

    For each phase (EN, LN), this function computes:

    1. A group mean composite ({phase}_COMPOSITE_GRPM) by averaging the
       per-dataset composites across the 'dataset' dimension.
    2. A Cross-Dataset Agreement Mask ({phase}_COMPOSITE_CDAM) encoding where
       datasets agree, using one of two methods:

       - 'sign': CDAM = 1 where >= threshold datasets agree on the sign
         of the composite (all positive or all negative), else 0.
       - 'significance': A priority-coded mask:
             CDAM =  2  where >= threshold datasets agree the value IS significant (p < 0.05).
             CDAM =  1  where >= threshold datasets agree the value IS NOT significant,
                        but not unanimously (i.e., at least one dataset disagrees).
             CDAM = -1  where ALL datasets agree the value is not significant.
             CDAM =  0  where no agreement condition is met.
             Priority: significance (2) > partial insignificance (1) > full insignificance (-1).

    Both results are added in-place to ds, and a human-readable
    provenance string is appended to ds.attrs['history'].

    Parameters
    ----------
    ds (xr.Dataset)  : Dataset with dimensions (dataset, lat, lon). Must contain
                        EN_COMPOSITE, LN_COMPOSITE, EN_COMPOSITE_SIG, LN_COMPOSITE_SIG.
    threshold (int)  : Number of datasets required for agreement.
                        Must satisfy 1 <= threshold <= n_datasets.
    method (str)     : Agreement criterion — 'sign' or 'significance' (see above).

    Returns
    -------
    ds (xr.Dataset)  : The input dataset with the following variables added for
                        each phase in {EN, LN}:
                        - {phase}_COMPOSITE_GRPM : Group mean composite across datasets.
                        - {phase}_COMPOSITE_CDAM : Cross-Dataset Agreement Mask (integer coded).
    """
    # --- Input validation ---
    if not isinstance(threshold, (int, np.integer)):
        raise TypeError(f"'threshold' must be an integer, got {type(threshold).__name__}")

    n = ds.sizes["dataset"]
    if threshold < 1 or threshold > n:
        raise ValueError(f"'threshold' must be between 1 and {n}, got {threshold}")

    # Checking that all required composite and significance variables are present
    required = [f"{p}_{v}" for p in ("EN", "LN") for v in ("COMPOSITE", "COMPOSITE_SIG")]
    missing = [v for v in required if v not in ds]
    if missing:
        raise KeyError(f"Missing required variables: {missing}")

    history_parts = []

    for phase in ("EN", "LN"):
        comp_var, sig_var = f"{phase}_COMPOSITE", f"{phase}_COMPOSITE_SIG"

        # Computing group mean composite across all datasets
        ds[f"{phase}_COMPOSITE_GRPM"] = ds[comp_var].mean(dim="dataset")

        # Building the Cross-Dataset Agreement Mask
        if method == "sign":
            # Counting datasets with positive/negative composite values per gridcell
            n_pos = (ds[comp_var] > 0).sum(dim="dataset")
            n_neg = (ds[comp_var] < 0).sum(dim="dataset")
            # Agreement if either polarity reaches the threshold
            cdam = xr.where((n_pos >= threshold) | (n_neg >= threshold), 1, 0)

            method_desc = (
                f"Sign agreement: CDAM=1 where >= {threshold}/{n} datasets agree on sign, else 0."
            )

        elif method == "significance":
            # Converting significance mask to boolean and counting agreements
            sig = ds[sig_var].astype(bool)
            n_sig = sig.sum(dim="dataset")
            n_insig = (~sig).sum(dim="dataset")

            # Initialising CDAM as zeros, then applying conditions in priority order
            # (lowest priority first, highest last so it overwrites)
            cdam = xr.full_like(ds[f"{phase}_COMPOSITE_GRPM"], 0, dtype=int)
            # All datasets agree value is not significant
            cdam = xr.where(n_insig == n, -1, cdam)
            # Threshold datasets agree value is not significant, but not all
            cdam = xr.where((n_insig >= threshold) & (n_insig < n), 1, cdam)
            # Threshold datasets agree value IS significant (highest priority)
            cdam = xr.where(n_sig >= threshold, 2, cdam)

            method_desc = (
                f"Significance agreement: CDAM=2 (>= {threshold}/{n} significant), "
                f"1 (>= {threshold}/{n} insignificant, not all), -1 (all insignificant), 0 (no agreement)."
            )

        else:
            raise ValueError(f"method must be 'sign' or 'significance', got '{method}'")

        ds[f"{phase}_COMPOSITE_CDAM"] = cdam

        # Recording provenance for this phase
        history_parts.append(
            f"{comp_var}: Composite mean across all {phase} events per dataset "
            f"(each event = temporal mean over event months). "
            f"{sig_var}: Per-gridcell significance (one-sample t-test vs zero, p<0.05). "
            f"{phase}_COMPOSITE_GRPM: Group mean of {comp_var} across {n} datasets. "
            f"{phase}_COMPOSITE_CDAM: {method_desc}"
        )

    # Appending provenance to dataset history attribute
    history = " | ".join(history_parts) + f" [threshold={threshold}, n_datasets={n}, method='{method}']"
    existing = ds.attrs.get("history", "")
    ds.attrs["history"] = f"{existing}\n{history}" if existing else history

    return ds

def compute_enso_peak_kde(ds, event_timestamps, kde_lats, kde_lons,
                          lat_bounds=(-5, 5), lon_bounds=(140, 270),):
    """
    Compute Gaussian KDE of El Niño/La Niña peak locations for a given dataset.

    For each ENSO month this function finds the location of the maximum
    absolute surface anomaly within a specified equatorial Pacific
    window, collects all such peak coordinates and fits a 2D Gaussian
    kernel density estimate. The KDE is then evaluated on a target
    lat lon grid.

    Parameters
    ----------
    ds (xr.Dataset)          : Dataset containing variable 'anomaly' with dims
                               (time, depth, lat, lon).
    event_timestamps (list)  : Flattened list of El Niño or La Niña months 
                               for this dataset.
    kde_lats  (1D array like): Target latitude coordinates for 
                               the KDE evaluation grid.
    kde_lons (1D array like) : Target longitude coordinates for 
                               the KDE evaluation grid.
    lat_bounds, lon_bounds   : Spatial subset used when locating peak anomalies. 
                               Defaults to 5˚S to 5˚N and 140˚E to 270˚E.

    Returns
    -------
    kde_field (np.ndarray)   : 2D array shaped (len(kde_lats), len(kde_lons)) containing KDE
                               values on the target grid. If no valid peaks are found the
                               array is filled with zeros.
    """
    # Subsetting El Niño/La Niña months and region, take absolute anomaly at surface
    ds_pac_enso = np.abs(ds.sel(time=event_timestamps, depth=0, method="nearest")
                         .sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))["anomaly"])

    # Mask: 1 at spatial maximum per time, NaN elsewhere
    mask = xr.where(ds_pac_enso == ds_pac_enso.max(dim=("lat", "lon")), 1, np.nan,)

    # Extracting peak lat/lon points across all times
    lat2d, lon2d = xr.broadcast(mask["lat"], mask["lon"])
    peak_lats = lat2d.where(mask.notnull()).values.ravel()
    peak_lons = lon2d.where(mask.notnull()).values.ravel()

    valid = ~np.isnan(peak_lats)
    if not np.any(valid):
        # No valid peaks; return zeros on the target grid
        return np.zeros((len(kde_lats), len(kde_lons)))

    pts = np.vstack([peak_lons[valid], peak_lats[valid]])

    # KDE on target grid
    kernel = gaussian_kde(pts)
    lon_grid, lat_grid = np.meshgrid(kde_lons, kde_lats)
    kde_vals = kernel(np.vstack([lon_grid.ravel(), lat_grid.ravel()]))

    return kde_vals.reshape(lat_grid.shape)

# =============================================================================
# HELPER FUNCTIONS USED IN plotting.ipynb AND THE APPENDIX NOTEBOOKS
# =============================================================================

def format_axis(ax = None, kind = None, **kwargs):
    """
    Generic axis formatting helper for the plotting.ipynb and appendix notebooks

    Parameters
    ----------
    ax (matplotlib.axes.Axes) : Axis to format.
    kind (str)                : Type of plot:
                                - 'oni'         : ONI time series panels
                                - 'psd'         : Niño 3.4 PSD panels
                                - 'map'         : Cartopy map panels
                                - 'kde'         : ENSO peak KDE panels
                                - 'event_comp'  : Event-centred Hovmöller panels
                                - 'vert_prof'   : Vertical profile panels (i < 2: latitudinal extent, else anomaly)
                                - 'z20'         : Z20 depth panels (i = 0: climatology, else bias)
    kwargs                    : Extra options needed for some kinds:
                                - period            (str)   for 'oni' (e.g. 'HIST_EVAL_1979-2014', 'FUTR_EVAL_2015-2050')
                                - twin              (bool)  for 'oni': True for an overlaid .twiny() axis
                                - model_year_offset (int)   for 'oni': label the x ticks as model years (e.g. 30 for CTRL)
                                - ylim              (tuple) for 'oni': y-axis limits. Default is (-3.5, 3.5)
                                - panel_id          (str)   for 'psd' (panel label)
                                - land_ft, coast_ft         for 'map' (cartopy features)
                                - i                 (int)   for 'event_comp', 'vert_prof', 'z20' (panel index)
                                - last_row          (bool)  for 'event_comp'
    """
    # Normalise kind string
    kind = str(kind).lower()

    if kind == "oni":
        period = kwargs.get("period")
        if period is None:
            raise ValueError("format_axis(kind='oni') requires period=...")

        twin = kwargs.get("twin", False)             # True for an overlaid .twiny() axis
        offset = kwargs.get("model_year_offset")     # e.g. 30 -> label ticks as CTRL model years

        # 5-yearly x ticks across the period; calendar years, or model years if offset is given
        y0, y1 = int(period.split('_')[-1][:4]), int(period.split('_')[-1][-4:])
        ticks = pd.date_range(f'{y0}-01-01', f'{y1}-12-31', freq='5YE')
        ax.set_xticks(ticks)
        if offset is None:
            ax.set_xticklabels([t.year for t in ticks])
        else:
            ax.set_xticklabels([str(t.year - y0 + offset) for t in ticks])
        ax.tick_params(axis='both', which='major', labelsize=12)

        if twin:
            # Overlaid axis: only its own x ticks; leave the y-axis decorations to the primary
            ax.minorticks_off()
            ax.grid(False)
        else:
            ax.set_ylim(kwargs.get("ylim", (-3.5, 3.5)))
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
            ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            ax.minorticks_on()
            ax.axhline(0, ls='-', color='grey', linewidth=2.5)
            ax.set_xlabel('Year', fontsize=16)
            ax.set_ylabel('ONI (°C)', fontsize=16)

    elif kind == "psd":
        panel_id = kwargs.get("panel_id", "")
        ax.set_xscale('log')
        ax.set_xlim(1.5, 10)
        ax.set_xlabel('Period (years)', fontsize=14)
        ax.set_ylabel('Power spectral density', fontsize=14)
        if panel_id:
            ax.set_title(f'{panel_id}) Niño 3.4 Power Spectra (Welch, detrended monthly index)', fontsize=16,)
        ax.tick_params(axis='both', which='major', labelsize=12)

        xticks = list(range(2, 7))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])

        # Highlight canonical 2–7 year ENSO band
        for yr in (2, 7):
            ax.axvline(yr, color='k', linestyle='--', linewidth=1.5, alpha=0.75, zorder=1)

        ax.grid(color='gray', alpha=0.25)

    elif kind == "map":
        land_ft = kwargs.get("land_ft")
        coast_ft = kwargs.get("coast_ft")
        if land_ft is None or coast_ft is None:
            raise ValueError("format_axis(kind='map') requires land_ft=..., coast_ft=...")

        ax.set_extent([140, 290, -30, 30], crs=ccrs.PlateCarree())
        ax.add_feature(land_ft)
        ax.add_feature(coast_ft)
        
        # Lat/lon gridlines with labels on left/bottom only
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                          color='gray', alpha=0.5, linestyle='--',)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = plt.FixedLocator(range(-180, 181, 30))
        gl.ylocator = plt.FixedLocator(range(-90, 91, 10))
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}

    elif kind == "kde":
        ax.set_ylabel('Latitude', fontsize=14)
        ax.set_xlabel('Longitude', fontsize=14)
        lon_ticks = np.arange(150, 310, 20)
        ax.set_xticks(lon_ticks)
        lat_ticks = np.arange(-4, 6, 2)
        ax.set_yticks(lat_ticks)
        ax.set_yticklabels(lat_ticks, fontsize=10)
        ax.grid(visible = True, linestyle = '--', linewidth = 1, alpha = 0.65)

    elif kind == "event_comp":
        i = kwargs.get("i")
        last_row = kwargs.get("last_row")
        if i is None:
            raise ValueError("format_axis(kind='event_comp') requires i=...")

        ax.set_xlim([-12,8])
        ax.set_ylim([250,0])
        ax.set_xticks(np.arange(-14, 8.1, 2))
        ax.set_yticks(np.arange(0, 300.1, 50))
        ax.grid(True, ls='--', alpha=0.2, color='k')
        ax.axvline(0, ls='--', color='k', lw=2)
        ax.axhspan(100, 150, color='gray', edgecolor='k', alpha=0.25, linewidth=2)
        ax.tick_params(axis='both', labelsize=10)

        # y-labels only on left column (even indices 0,2,4 etc.)
        ax.set_ylabel('Depth (m)' if i % 2 == 0 else '', fontsize=12)
        # x-label only on the bottom row (last_row=True)
        ax.set_xlabel('Time (in months) relative to phase peak' if last_row == True else '', fontsize=12,)

    elif kind == "vert_prof":
        i = kwargs.get("i")
        if i is None:
            raise ValueError("format_axis(kind='vert_prof') requires i=...")

        ax.axhline(0, color='black', linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid()
        ax.set_ylabel('Depth (m)', fontsize=18)
        ax.invert_yaxis()

        if i < 2:
            # Left two panels: latitudinal extent (auto-scaled x-axis, no % formatting)
            ax.set_xlabel('Latitudinal extent within 5°S-5°N', fontsize=18)
        else:
            # Right two panels: anomaly profile with a simpler locator
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_xlabel('Anomaly (˚C)', fontsize=18)

    elif kind == "z20":
        i = kwargs.get("i")
        if i is None:
            raise ValueError("format_axis(kind='z20') requires i=...")

        ax.set_xticks(np.arange(140, 280.1, 20))
        ax.set_xlim([135, 285])
        ax.grid(visible=True, ls='--', lw=1, alpha=0.5, color='gray')
        ax.set_xlabel('Longitude', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        # Highlight Niño 3.4 longitude band
        ax.axvspan(-170 + 360, -120 + 360, color='gray', alpha=0.2, lw=2, ec='black')

        if i < 1:
            ax.set_ylabel('Depth (m)', fontsize=16)
            ax.set_ylim([0, 200])
            ax.invert_yaxis()
        else:
            ax.set_ylabel('Depth bias (m)', fontsize=16)
            ax.set_ylim([-24, 24])

    else:
        raise ValueError(f"Unknown format kind: {kind!r}")


def add_colorbar(fig, axes, mappable, label, ticks, side="right", 
                 width=0.010, height=0.018, pad=0.007, ticksize=10, 
                 labelsize=12, height_scale=0.90,):
    """
    Add a colourbar aligned with a block of axes.

    Parameters
    ----------
    fig (matplotlib.figure.Figure) : Figure object.
    axes (sequence of Axes)        : Axes that define the block to align the colourbar with.
    mappable (ScalarMappable)      : Object returned by contourf/pcolormesh/etc.
    label (str)                    : Colourbar label.
    ticks (sequence)               : Tick locations for the colourbar.
    side ({"right", "bottom"})     : Where to place the colourbar relative to the axes block.
                                     - "right": vertical bar to the right (uses `width`, `height_scale`).
                                     - "bottom": horizontal bar below (uses `height`).
    width (float)                  : Thickness of the bar when side="right" (in figure fraction).
    height (float)                 : Thickness of the bar when side="bottom" (in figure fraction).
    pad (float)                    : Padding between the axes block and the colourbar (figure fraction)
    ticksize (int)                 : Font size for ticks.
    labelsize (int)                : Font size for label.
    height_scale (float)           : Only used for side="right": fraction of axes block height to use.

    Returns
    -------
    cb (matplotlib.colorbar.Colorbar) : Colourbar object
    """
    # Compute bounding box enclosing all axes
    row_box = Bbox.union([ax.get_position(fig).frozen() for ax in axes])

    # Decide placement and create colourbar axis
    if side == "right":
        # Vertical bar placed to the right of the axes block
        cax = fig.add_axes([row_box.x1 + pad, row_box.y0, width, row_box.height * height_scale,])
        orientation = "vertical"
    elif side == "bottom":
        # Horizontal bar placed below the axes block
        cax = fig.add_axes([row_box.x0, row_box.y0 - pad, row_box.width, height,])
        orientation = "horizontal"
    else:
        raise ValueError(f"Unsupported side: {side!r}")
    # Create and format the colourbar
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks)
    cb.set_label(label, fontsize=labelsize)
    cb.ax.tick_params(labelsize=ticksize)

    if side == "right":
        # Ensure ticks appear on the right for vertical bars
        cax.yaxis.set_ticks_position("right")
        cax.xaxis.set_visible(False)

    return cb


def plot_with_spread(ax, da, color, label, coord_dim, horizontal=True, linestyle = '-'):
    """
    Plot group-mean profile with ±1 std shaded.

    Parameters
    ----------
    ax (matplotlib.axes.Axes) : Axis to plot on.
    da (xr.DataArray)         : DataArray with dims including 'dataset' and coord_dim.
    color (any)               : Line/patch colour.
    label (str)               : Legend label.
    coord_dim (str)           : Dimension to plot along the x-axis (if horizontal=True) 
                                or y-axis (if horizontal=False).
    horizontal (bool)         : If True, plot coord_dim on x-axis. 
                                If False, plot coord_dim on y-axis.
    linestyle (str)           : Line style of the group mean. Default is '-'.
    """
    # Mean and std across datasets
    mean_vals = da.mean(dim="dataset")
    std_vals = da.std(dim="dataset")

    # Dropping NaNs in the plotted coordinate to avoid gaps
    mean_vals = mean_vals.dropna(dim=coord_dim)
    std_vals = std_vals.sel({coord_dim: mean_vals[coord_dim]})

    coord = mean_vals[coord_dim].values

    if horizontal:
        # Standard x = coord_dim, y = value
        ax.plot(coord, mean_vals.values, color=color, ls=linestyle, label=label)
        ax.fill_between(coord, mean_vals.values - std_vals.values, 
                        mean_vals.values + std_vals.values, color=color, alpha=0.2, )
    else:
        # Flipped: y = coord_dim, x = value (e.g. depth profiles)
        ax.plot(mean_vals.values, coord, color=color, ls=linestyle, label=label)
        ax.fill_betweenx(coord, mean_vals.values - std_vals.values,
                         mean_vals.values + std_vals.values, color=color, alpha=0.2,)


def merge_pngs_vertical(paths, out_path):
    """
    Merge multiple PNG files vertically into one PNG.

    Images narrower than the widest one are rescaled (LANCZOS) to the same width. The merged image is RGB
    with a white background.
    
    Parameters
    ----------
    paths (list)     : List of figure paths to merge, from top to bottom.
    out_path (str)   : Output path for the merged figure.
    """
    # Loading all images
    imgs = [Image.open(p) for p in paths]
    # Determining max width to align all images
    max_w = max(im.width for im in imgs)

    # Resizing images to have a uniform width
    imgs_resized = []
    for im in imgs:
        if im.width != max_w:
            new_h = int(im.height * max_w / im.width)
            imgs_resized.append(im.resize((max_w, new_h), Image.LANCZOS))
        else:
            imgs_resized.append(im)
            
    # Total height for final merged canvas
    total_h = sum(im.height for im in imgs_resized)
    combined = Image.new("RGB", (max_w, total_h), color="white")

    # Pasting images one below another
    y = 0
    for im in imgs_resized:
        combined.paste(im, (0, y))
        y += im.height

    combined.save(out_path)
    
def plot_psd_series(series_name, n34_data, ax, peak_points, psd_kwargs, line_kw, peak_marker_kw, ns_lst):
    """
    Plot PSD for a single series and record its peak location.

    Parameters
    ----------
    series_name (str)         : Column name in n34_data.
    n34_data (pd.DataFrame)   : DataFrame with Niño 3.4 indices (columns = datasets).
    ax (matplotlib.axes.Axes) : Axis to plot on.
    peak_points (list)        : List that will be appended with (peak_T, peak_P, peak_marker_kw, significant),
                                where significant is False if series_name is in ns_lst.
    psd_kwargs (dict)         : Keyword args passed to compute_psd (nperseg, period_min, period_max, fs,
                                noverlap_frac, nfft).
    line_kw (dict)            : Keyword args for the PSD line (color, ls, lw, etc.).
    peak_marker_kw (dict)     : Marker styling stored for later plotting of peak points.
    ns_lst (list of str)      : Names of the series whose dominant peak is not significant against red noise
                                (p > 0.05), see appendix_A.ipynb.
    """
    # Computing PSD for this series
    period, P, peak_T, peak_P = compute_psd(n34_data[series_name].dropna(), **psd_kwargs)
    if period is None:
        return
    
    # Plotting PSD curve
    ax.plot(period, P, **line_kw)
    
    # Storing peak location for later marker plotting
    if peak_T is not None:
        peak_points.append((peak_T, peak_P, peak_marker_kw, series_name not in ns_lst))

        
def peak_lonlat(da2d):
    """
    Return (lon, lat) of the maximum of a 2D KDE DataArray 'da2d' with dims ('lat','lon').
    """
    # Flattening to 1D index
    st = da2d.stack(z=('lat', 'lon'))
    k = int(st.argmax('z').values)
    
    # Converting flat index back to (lat, lon)
    nlon = da2d.sizes['lon']
    lat_pk = float(da2d['lat'].values[k // nlon])
    lon_pk = float(da2d['lon'].values[k % nlon])
    return lon_pk, lat_pk

# LEGACY FUNCTION: No longer used in any script or notebook of this repository (kept for reference)
def jitter(x, y, k, r=0.35):
    """
    Function to give tiny visual jitter to the KDE points, so they don't overlap.

    Parameters
    ----------
    x (float) : Longitude value of peak of KDE distribution
    y (float) : Latitude value of peak of KDE distribution
    k (int)   : Index of the point. The offset direction cycles through 12 angles of 30 degrees, i.e. k % 12
    r (float) : Scale factor (offset distance in the units of x and y)
    """
    ang = (k % 12) * (np.pi/6.0)
    return x + r*np.cos(ang), y + r*np.sin(ang)

# LEGACY FUNCTION: No longer used in any script or notebook of this repository (kept for reference)
def event_hov_calculator(dataset=None, event_type=None, enso_event_flav_dict = None, n34_hov = None):
    """
    Calculate El Niño or La Niña composite Hovmöller (depth vs. time; averaged over Niño 3.4)
    for a given dataset by taking the -14 to +8 months window about each ENSO event's peak timestamp.
    
    Parameters
    ----------
    dataset (str)                : Dataset name as a string.
    event_type (str)             : ENSO event type, 'elnino' or 'lanina' (as in the 'events_<event_type>' keys).
    enso_event_flav_dict (dict)  : Nested dictionary of ENSO event flavours, containing event peak timestamps
    n34_hov (xr.DataArray)       : Niño 3.4 Hovmöller (depth vs. time) for the full-time period.

    Returns
    -------
    xr.Dataset : Variables EN_COMPOSITE and EN_COMPOSITE_SIG (El Niño) or LN_COMPOSITE and LN_COMPOSITE_SIG
                 (La Niña) with dims (dataset, time, depth): the mean across events, and a boolean mask that is
                 True where it differs from zero (one-sample t-test, p < 0.05).
    """
    event_peak_times = [_['peak_time'] for _ in enso_event_flav_dict[dataset][f'events_{event_type}']]
    
    base = n34_hov.sel(dataset=dataset)
    rel_time = np.arange(-14,9)
    
    hov_list = []
    for idx, peak_ts in enumerate(event_peak_times):
        # Build absolute monthly timestamps
        target_time = peak_ts + pd.DateOffset(months=0)  # ensure Timestamp        
        target_time = pd.date_range(start=peak_ts + pd.DateOffset(months=-14),periods=23, freq='MS')
        # Reindex instead of slicing
        hov = base.reindex(time=target_time)
        # Replace absolute time with relative time
        hov = hov.assign_coords(time=rel_time)
        hov = hov.expand_dims(event_id=[idx])
        hov_list.append(hov)
    # Stacking events and averaging
    hov_all = xr.concat(hov_list, dim="event_id").expand_dims(dataset = [dataset])
    t_stat, p_val = stats.ttest_1samp(hov_all, popmean = 0, axis = 1)
    sig_mask = p_val < 0.05
    
    if event_type == 'elnino':
        hov_all = hov_all.assign_coords(event_type = ('event_id', ['EL-NINO']*hov_all['event_id'].shape[0]))
        hov_all = hov_all.mean(dim = 'event_id').rename('EN_COMPOSITE').to_dataset()
        hov_all['EN_COMPOSITE_SIG'] = (('dataset', 'time', 'depth'), sig_mask)
    elif event_type == 'lanina':
        hov_all = hov_all.assign_coords(event_type = ('event_id', ['LA-NINA']*hov_all['event_id'].shape[0]))
        hov_all = hov_all.mean(dim = 'event_id').rename('LN_COMPOSITE').to_dataset()
        hov_all['LN_COMPOSITE_SIG'] = (('dataset', 'time', 'depth'), sig_mask)
    return hov_all

# LEGACY FUNCTION: No longer used in any script or notebook of this repository (kept for reference)
def gridcell_counter(da):
    """
    Calculate the latitudinal extent (in degrees, assuming a 20˚ latitude span) of the non-NaN gridcells of an
    xarray.DataArray, as a measure of the latitudinal coverage by extreme temperature anomalies (ref. Figures 6
    and 11, panel a). The extent is now computed within 5˚S-5˚N in appendices/appendix_A.ipynb.
    
    Parameters
    ----------
    da (xr.DataArray) : Masked anomalies with dims including 'lat' and 'lon'.

    Returns
    -------
    xr.DataArray : 20 x the fraction of non-NaN latitudes, averaged across longitude.
    """
    # Count valid (non-NaN) cells along latitude
    number_of_gridcells = (~np.isnan(da)).sum(dim='lat')
    # Mask locations with zero valid cells
    number_of_gridcells = number_of_gridcells.where(number_of_gridcells != 0)
    # Convert to number of latitudinal degrees and average across longitude and return
    return 20*(number_of_gridcells / da.sizes['lat']).mean(dim = ['lon'])
    
def z20_grp_calc(da_z20, enso_timestamps, axes, color, label, linestyle='-'):
    """
    Calculate and plot the group-mean depth of the 20˚C (Z20) isotherm for ENSO events and the
    climatological baseline mean, with the spread (±1 std) across the datasets of the group
    
    Parameters
    ----------
    da_z20 (xr.DataArray)       : Z20 depths with dims (dataset, time, lon), averaged over 5˚S-5˚N and
                                  interpolated to a 0.1˚ grid between 140˚E and 280˚E (see data_prep.ipynb)
    enso_timestamps (dict)      : Nested dictionary of event windows keyed by dataset.
                                  name and by phase, for example:
                                  {
                                    'HadGEM3-GC31-HH_CTRL': {
                                        'EL-NINO': [[ts1, ts2, ...], ...],
                                        'LA-NINA': [[ts1, ts2, ...], ...]
                                        },
                                    ...
                                    }
    axes (sequence of Axes)     : Three axes: the baseline Z20, and the El Niño and La Niña Z20 anomalies
                                  relative to the baseline.
    color (any)                 : Line/patch colour.
    label (str)                 : Legend label.
    linestyle (str)             : Line style of the group mean. Default is '-'.
    """
    # Climatological Z20 baseline for each dataset (mean over its full time span)
    baseline = da_z20.mean(dim = ['time'])
    en, ln = [], []
    
    # Compute dataset-wise mean Z20 for EN and LN events
    for dataset in da_z20.dataset.values:
        en += [da_z20.sel(dataset = dataset, time = flatten(enso_timestamps[dataset]['EL-NINO'])).mean(dim = ['time'])]
        ln += [da_z20.sel(dataset = dataset, time = flatten(enso_timestamps[dataset]['LA-NINA'])).mean(dim = ['time'])]

    # Combine event-mean Z20 profiles across datasets
    en, ln = xr.concat(en, dim = 'dataset'), xr.concat(ln, dim = 'dataset')
    
    # Panel 1: Baseline Z20 (BASELINE)
    plot_with_spread(ax=axes[0], da=baseline, color=color, label=label, coord_dim="lon", horizontal=True, linestyle=linestyle)
    # Panel 2: El Niño Z20 anomaly (EN - BASELINE)
    plot_with_spread(ax=axes[1], da=en - baseline, color=color, label=label, coord_dim="lon", horizontal=True, linestyle=linestyle)
    # Panel 3: La Niña Z20 anomaly (LN - BASELINE)
    plot_with_spread(ax=axes[2], da=ln - baseline, color=color, label=label, coord_dim="lon", horizontal=True, linestyle=linestyle)
