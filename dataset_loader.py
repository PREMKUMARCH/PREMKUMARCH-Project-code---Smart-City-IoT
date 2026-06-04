"""
dataset_loader.py
-----------------
Loads the Intel Berkeley Research Lab sensor dataset and the CRAWDAD
Zigbee smart-home dataset, extracts temporal fluctuation sequences, and
returns normalised zero-mean unit-variance templates that are later
injected into the simulated link-quality process via Eq. (trace-adapt).

Both datasets are used ONLY as temporal fluctuation templates; they are
not used as geometric models of the dense topology.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────────────────
# Intel Berkeley Research Lab  (54 sensors, Feb–Apr 2004)
# Columns: date  time  epoch  moteid  temperature  humidity  light  voltage
# ────────────────────────────────────────────────────────────────────────────

def load_intel_lab(path: str) -> pd.DataFrame:
    """
    Load intel_lab_data.csv.
    Accepts either the raw space-separated .txt / .txt.gz format or a
    pre-saved CSV with a header row.
    Returns a tidy DataFrame with a datetime index.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Intel Lab dataset not found at '{path}'.\n"
            "Download from http://db.csail.mit.edu/labdata/data.txt.gz ,\n"
            "decompress, and save as datasets/intel_lab_data.csv\n"
            "with header: date,time,epoch,moteid,temperature,humidity,light,voltage"
        )

    # ── try header-less space-separated first ────────────────────────────────
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            names=["date", "time", "epoch", "moteid",
                   "temperature", "humidity", "light", "voltage"],
            na_values=[""],
        )
        # drop rows that look like a header accidentally included
        df = df[pd.to_numeric(df["epoch"], errors="coerce").notna()]
    except Exception:
        df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    numeric_cols = ["temperature", "humidity", "light", "voltage"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[Intel Lab] loaded {len(df):,} rows, "
          f"{df['moteid'].nunique()} sensors, "
          f"range {df['datetime'].min()} → {df['datetime'].max()}")
    return df


def intel_fluctuation_templates(df: pd.DataFrame,
                                 n_templates: int = 200,
                                 epoch_rate_hz: float = 1.0,
                                 segment_len: int = 3600,
                                 seed: int = 0) -> np.ndarray:
    """
    Extract *n_templates* zero-mean unit-variance fluctuation sequences
    from the Intel Lab dataset.

    Strategy
    --------
    1. For each sensor, resample the 'light' channel (most dynamic) to
       *epoch_rate_hz* Hz using linear interpolation.
    2. Randomly draw contiguous segments of length *segment_len* epochs.
    3. Normalise each segment to zero mean and unit variance.

    Returns  shape (n_templates, segment_len)
    """
    rng = np.random.default_rng(seed)
    templates = []
    sensor_ids = df["moteid"].dropna().unique()

    for sid in sensor_ids:
        sub = df[df["moteid"] == sid].set_index("datetime")["light"].dropna()
        if len(sub) < segment_len * 2:
            continue
        # resample to uniform 1-s grid
        sub_resampled = (
            sub.resample(f"{int(1/epoch_rate_hz)}s")
               .mean()
               .interpolate("linear")
               .dropna()
        )
        arr = sub_resampled.values
        if len(arr) < segment_len:
            continue
        # draw multiple non-overlapping segments
        max_start = len(arr) - segment_len
        starts = rng.integers(0, max_start, size=min(5, max_start // segment_len + 1))
        for s in starts:
            seg = arr[s: s + segment_len].astype(float)
            mu, sigma = seg.mean(), seg.std()
            if sigma < 1e-9:
                continue
            templates.append((seg - mu) / sigma)
            if len(templates) >= n_templates:
                break
        if len(templates) >= n_templates:
            break

    if not templates:
        # fallback: synthetic white-noise templates
        print("[Intel Lab] WARNING: could not extract templates; using synthetic fallback.")
        templates = [rng.standard_normal(segment_len) for _ in range(n_templates)]

    out = np.array(templates[:n_templates])
    print(f"[Intel Lab] extracted {out.shape[0]} fluctuation templates "
          f"of length {out.shape[1]}")
    return out


# ────────────────────────────────────────────────────────────────────────────
# CRAWDAD cmu/zigbee-smarthome
# Expected CSV columns (flexible):  timestamp, src, dst, rssi, lqi, [prr]
# ────────────────────────────────────────────────────────────────────────────

def load_crawdad_zigbee(path: str) -> pd.DataFrame:
    """
    Load the CRAWDAD Zigbee smart-home trace CSV.
    The tool attempts to auto-detect columns for timestamp, RSSI, and LQI.
    Returns a tidy DataFrame sorted by time.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CRAWDAD dataset not found at '{path}'.\n"
            "Download from https://ieee-dataport.org/collections/crawdad\n"
            "(free IEEE account required) and save as datasets/crawdad_zigbee.csv"
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # ── flexible timestamp parsing ────────────────────────────────────────────
    ts_candidates = [c for c in df.columns if "time" in c or "ts" in c]
    if ts_candidates:
        df["datetime"] = pd.to_datetime(df[ts_candidates[0]], errors="coerce",
                                        unit="s" if df[ts_candidates[0]].dtype in
                                        [np.int64, np.float64] else None)
    else:
        df["datetime"] = pd.NaT

    # ── RSSI / LQI columns ───────────────────────────────────────────────────
    for col in ["rssi", "lqi", "prr", "pdr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[CRAWDAD] loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def crawdad_fluctuation_templates(df: pd.DataFrame,
                                   n_templates: int = 200,
                                   epoch_rate_hz: float = 1.0,
                                   segment_len: int = 3600,
                                   seed: int = 42) -> np.ndarray:
    """
    Extract zero-mean unit-variance fluctuation templates from CRAWDAD RSSI
    or PRR time series (whichever is available).

    Returns  shape (n_templates, segment_len)
    """
    rng = np.random.default_rng(seed)
    templates = []

    # pick the best available quality metric
    metric = None
    for candidate in ["prr", "pdr", "lqi", "rssi"]:
        if candidate in df.columns and df[candidate].notna().sum() > 1000:
            metric = candidate
            break

    if metric is None or "datetime" not in df.columns or df["datetime"].isna().all():
        print("[CRAWDAD] WARNING: no usable metric or timestamps; using synthetic fallback.")
        return np.array([rng.standard_normal(segment_len) for _ in range(n_templates)])

    # group by link (src→dst) if columns exist
    link_cols = [c for c in ["src", "dst", "source", "destination"] if c in df.columns]
    if link_cols:
        groups = df.groupby(link_cols)
    else:
        groups = [(("all",), df)]

    for _, grp in groups:
        series = (
            grp.set_index("datetime")[metric]
               .dropna()
               .resample(f"{int(1/epoch_rate_hz)}s")
               .mean()
               .interpolate("linear")
               .dropna()
        )
        arr = series.values
        if len(arr) < segment_len:
            continue
        max_start = len(arr) - segment_len
        starts = rng.integers(0, max_start, size=min(4, max_start // segment_len + 1))
        for s in starts:
            seg = arr[s: s + segment_len].astype(float)
            mu, sigma = seg.mean(), seg.std()
            if sigma < 1e-9:
                continue
            templates.append((seg - mu) / sigma)
            if len(templates) >= n_templates:
                break
        if len(templates) >= n_templates:
            break

    if not templates:
        print("[CRAWDAD] WARNING: could not extract templates; using synthetic fallback.")
        templates = [rng.standard_normal(segment_len) for _ in range(n_templates)]

    out = np.array(templates[:n_templates])
    print(f"[CRAWDAD] extracted {out.shape[0]} fluctuation templates "
          f"of length {out.shape[1]}")
    return out


# ────────────────────────────────────────────────────────────────────────────
# Combined loader
# ────────────────────────────────────────────────────────────────────────────

def load_all_templates(intel_path: str,
                       crawdad_path: str,
                       n_each: int = 200,
                       segment_len: int = 3600,
                       seed: int = 0):
    """
    Load both datasets and return a combined (2*n_each, segment_len) array
    of normalised fluctuation templates.  Handles missing files gracefully
    by substituting synthetic white-noise templates with a warning.
    """
    templates = []

    # ── Intel Lab ─────────────────────────────────────────────────────────────
    try:
        intel_df = load_intel_lab(intel_path)
        intel_t  = intel_fluctuation_templates(intel_df, n_each, segment_len=segment_len, seed=seed)
        templates.append(intel_t)
    except FileNotFoundError as e:
        print(f"[dataset_loader] {e}")
        print("[dataset_loader] Substituting synthetic templates for Intel Lab.")
        rng = np.random.default_rng(seed)
        templates.append(np.array([rng.standard_normal(segment_len) for _ in range(n_each)]))

    # ── CRAWDAD ───────────────────────────────────────────────────────────────
    try:
        crawdad_df = load_crawdad_zigbee(crawdad_path)
        crawdad_t  = crawdad_fluctuation_templates(crawdad_df, n_each, segment_len=segment_len, seed=seed+1)
        templates.append(crawdad_t)
    except FileNotFoundError as e:
        print(f"[dataset_loader] {e}")
        print("[dataset_loader] Substituting synthetic templates for CRAWDAD.")
        rng = np.random.default_rng(seed + 1)
        templates.append(np.array([rng.standard_normal(segment_len) for _ in range(n_each)]))

    combined = np.vstack(templates)
    print(f"[dataset_loader] Combined template bank: {combined.shape}")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bank = load_all_templates(
        intel_path="datasets/intel_lab_data.csv",
        crawdad_path="datasets/crawdad_zigbee.csv",
        n_each=10,
        segment_len=500,
    )
    print(f"Template bank shape : {bank.shape}")
    print(f"Mean across bank    : {bank.mean():.4f}  (should be ~0)")
    print(f"Std  across bank    : {bank.std():.4f}  (should be ~1)")
