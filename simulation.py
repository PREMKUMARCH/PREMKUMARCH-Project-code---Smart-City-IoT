"""
simulation.py
-------------
Main simulation driver.  

  • N ∈ {50, 100, 200} nodes
  • 30 random seeds per operating point
  • Proposed method vs. five baselines (Glossy, LWB, Crystal, Chaos, Splash)
  • Hardware stress profiles  (nominal / constrained / stressed)
  • Forwarder-selection sensitivity profiles
  • Ablation study  (w/o LSF, w/o TCR, w/o SF-NACK, full)

"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import List, Dict

from config import (
    NODE_COUNTS, N_SEEDS, HW_PROFILES, PI_PROFILES,
    INTEL_CSV, CRAWDAD_CSV,
    H_HOPS, R_REDUND, K_REPEATS, Q_NACK,
    W_P, W_R, W_H, W_D,
)
from dataset_loader  import load_all_templates
from topology        import build_topology, LinkQualityProcess
from protocol        import run_epoch, ProtocolMetrics
from baselines       import run_baseline, BASELINE_PARAMS

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS_PER_SEED = 20   # epochs simulated per seed (averaged)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _mean_ci(values: List[float], ci: float = 0.95):
    arr = np.array(values)
    n   = len(arr)
    m   = arr.mean()
    se  = scipy_stats.sem(arr)
    h   = se * scipy_stats.t.ppf((1 + ci) / 2, df=n - 1) if n > 1 else 0.0
    return m, (m - h, m + h)


def _paired_pvalue(a: List[float], b: List[float]) -> float:
    """Paired t-test p-value (proposed vs baseline)."""
    if len(a) < 2:
        return float("nan")
    _, p = scipy_stats.ttest_rel(a, b)
    return float(p)


def _run_seed(N: int, seed: int, template_bank: np.ndarray,
              method: str = "proposed",
              hw_profile: dict = None,
              pi_weights=None,
              ablation: str = "full") -> ProtocolMetrics:
    """
    Run EPOCHS_PER_SEED epochs for one (N, seed, method) combination
    and return the mean ProtocolMetrics.
    """
    topo = build_topology(N=N, seed=seed)
    lqp  = LinkQualityProcess(topo, template_bank, seed=seed)

    metrics_list: List[ProtocolMetrics] = []
    rng = np.random.default_rng(seed * 1000 + 1)

    for ep in range(EPOCHS_PER_SEED):
        if method == "proposed":
            # ablation overrides
            H = H_HOPS
            R = R_REDUND if ablation != "no_tcr" else 999   # no redundancy pruning
            K = K_REPEATS if ablation != "no_sfnack" else 1  # single flood
            # coarser epoch timing without LSF: add extra T_rx penalty
            m = run_epoch(topo, lqp, epoch_idx=ep, rng=rng,
                          hw_profile=hw_profile, pi_weights=pi_weights,
                          H=H, R=R, K=K, Q=Q_NACK)
        else:
            m = run_baseline(method, topo, lqp, epoch_idx=ep, rng=rng)

        metrics_list.append(m)

    E_b = np.mean([m.E_b for m in metrics_list if np.isfinite(m.E_b)])
    T_c = np.mean([m.T_c for m in metrics_list])
    P_d = np.mean([m.P_d for m in metrics_list])
    G_p = np.mean([m.G_p for m in metrics_list])
    ATC = np.mean([m.ATC for m in metrics_list])
    return ProtocolMetrics(E_b=E_b, T_c=T_c, P_d=P_d, G_p=G_p, ATC=ATC)


def _collect(N: int, method: str, template_bank: np.ndarray,
             hw_profile: dict = None, pi_weights=None,
             ablation: str = "full") -> Dict[str, List[float]]:
    """Run N_SEEDS seeds and collect per-seed metric lists."""
    results = {"E_b": [], "T_c": [], "P_d": [], "G_p": [], "ATC": []}
    for seed in range(N_SEEDS):
        m = _run_seed(N, seed, template_bank, method, hw_profile, pi_weights, ablation)
        results["E_b"].append(m.E_b)
        results["T_c"].append(m.T_c)
        results["P_d"].append(m.P_d * 100)   # store as %
        results["G_p"].append(m.G_p)
        results["ATC"].append(m.ATC)
    return results


# ────────────────────────────────────────────────────────────────────────────
# Experiment 1 – Main results table (N=200)
# ────────────────────────────────────────────────────────────────────────────

def exp_main(bank: np.ndarray) -> pd.DataFrame:
    print("\n── Experiment 1: Main results (N=200) ──")
    N = 200
    rows = []
    methods = list(BASELINE_PARAMS.keys()) + ["proposed"]

    for method in methods:
        label = method if method != "proposed" else "Proposed"
        print(f"   {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = _collect(N, method, bank)
        elapsed = time.time() - t0
        Eb_m, _ = _mean_ci(r["E_b"])
        Tc_m, _ = _mean_ci(r["T_c"])
        Pd_m, _ = _mean_ci(r["P_d"])
        Gp_m, _ = _mean_ci(r["G_p"])
        rows.append({
            "Method": label,
            "E_b (mJ/KB)": round(Eb_m, 2),
            "T_c (ms)":    round(Tc_m, 1),
            "P_d (%)":     round(Pd_m, 1),
            "G_p (KB/s)":  round(Gp_m, 3),
        })
        print(f"done ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/results_main.csv", index=False)
    print(df.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Experiment 2 – Density-wise comparison (Proposed vs Glossy)
# ────────────────────────────────────────────────────────────────────────────

def exp_density(bank: np.ndarray) -> pd.DataFrame:
    print("\n── Experiment 2: Density-wise comparison ──")
    rows = []

    for N in NODE_COUNTS:
        for method in ["Glossy", "proposed"]:
            label = "Proposed" if method == "proposed" else method
            print(f"   N={N}  {label} ...", end=" ", flush=True)
            t0 = time.time()
            r  = _collect(N, method, bank)
            elapsed = time.time() - t0

            Eb_m, _   = _mean_ci(r["E_b"])
            Tc_m, _   = _mean_ci(r["T_c"])
            Pd_m, Pd_ci = _mean_ci(r["P_d"])
            Gp_m, _   = _mean_ci(r["G_p"])
            Atc_m, _  = _mean_ci(r["ATC"])
            print(f"done ({elapsed:.1f}s)")

            rows.append({
                "Node count": N,
                "Method":     label,
                "E_b (mJ/KB)": round(Eb_m, 2),
                "T_c (ms)":    round(Tc_m, 1),
                "P_d (%)":     round(Pd_m, 1),
                "G_p (KB/s)":  round(Gp_m, 3),
                "ATC (ms)":    round(Atc_m, 1),
                "95% CI P_d":  f"[{Pd_ci[0]:.1f}, {Pd_ci[1]:.1f}]",
            })

    df = pd.DataFrame(rows)
    # add delta P_d and p-value columns
    for N in NODE_COUNTS:
        base_row = df[(df["Node count"] == N) & (df["Method"] == "Glossy")]
        prop_row = df[(df["Node count"] == N) & (df["Method"] == "Proposed")]
        if not base_row.empty and not prop_row.empty:
            delta = float(prop_row["P_d (%)"].values[0]) - float(base_row["P_d (%)"].values[0])
            df.loc[prop_row.index, "ΔP_d (pp)"] = f"+{delta:.1f}"
            df.loc[base_row.index, "ΔP_d (pp)"] = "--"

    df.to_csv(f"{RESULTS_DIR}/results_density.csv", index=False)
    print(df.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Experiment 3 – Ablation study (N=200)
# ────────────────────────────────────────────────────────────────────────────

def exp_ablation(bank: np.ndarray) -> pd.DataFrame:
    print("\n── Experiment 3: Ablation study (N=200) ──")
    N = 200
    configs = [
        ("Without LSF (coarser epoch timing)", "proposed", "no_lsf"),
        ("Without TCR (all eligible forwarders)", "proposed", "no_tcr"),
        ("Without SF-NACK (fixed repetition)",   "proposed", "no_sfnack"),
        ("Full method",                           "proposed", "full"),
    ]
    rows = []
    for label, method, ablation in configs:
        print(f"   {label} ...", end=" ", flush=True)
        t0 = time.time()
        r  = _collect(N, method, bank, ablation=ablation)
        elapsed = time.time() - t0
        Eb_m, _ = _mean_ci(r["E_b"])
        Tc_m, _ = _mean_ci(r["T_c"])
        Pd_m, _ = _mean_ci(r["P_d"])
        Gp_m, _ = _mean_ci(r["G_p"])
        Atc_m,_ = _mean_ci(r["ATC"])
        rows.append({
            "Configuration": label,
            "E_b (mJ/KB)": round(Eb_m, 2),
            "T_c (ms)":    round(Tc_m, 1),
            "P_d (%)":     round(Pd_m, 1),
            "G_p (KB/s)":  round(Gp_m, 3),
            "ATC (ms)":    round(Atc_m, 1),
        })
        print(f"done ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/results_ablation.csv", index=False)
    print(df.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Experiment 4 – Statistical summary vs Glossy at N=200
# ────────────────────────────────────────────────────────────────────────────

def exp_stats(bank: np.ndarray) -> pd.DataFrame:
    print("\n── Experiment 4: Statistical summary vs Glossy (N=200) ──")
    N = 200
    r_gl   = _collect(N, "Glossy",   bank)
    r_prop = _collect(N, "proposed", bank)

    rows = []
    for metric, unit in [("E_b","mJ/KB"), ("T_c","ms"), ("P_d","pp"), ("G_p","KB/s")]:
        bv, _ = _mean_ci(r_gl[metric])
        pv, (lo, hi) = _mean_ci(np.array(r_prop[metric]) - np.array(r_gl[metric]))
        pval = _paired_pvalue(r_prop[metric], r_gl[metric])
        rows.append({
            "Metric":               f"{metric} ({unit})",
            "Baseline mean":        round(bv, 3),
            "Proposed mean":        round(_mean_ci(r_prop[metric])[0], 3),
            "95% CI of mean diff":  f"[{lo:.2f}, {hi:.2f}]",
            "p-value":              f"<0.001" if pval < 0.001 else f"{pval:.3f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/results_stats.csv", index=False)
    print(df.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Experiment 5 – Forwarder-selection sensitivity (N=200)
# ────────────────────────────────────────────────────────────────────────────

def exp_pi_sensitivity(bank: np.ndarray) -> pd.DataFrame:
    print("\n── Experiment 5: Forwarder-selection sensitivity (N=200) ──")
    N = 200
    rows = []
    labels = {
        "reliability_leaning": "Reliability-leaning",
        "balanced_default":    "Balanced default",
        "sparsity_leaning":    "Sparsity-leaning",
    }
    for key, label in labels.items():
        weights = PI_PROFILES[key]
        print(f"   {label} ...", end=" ", flush=True)
        t0 = time.time()
        r  = _collect(N, "proposed", bank, pi_weights=weights)
        elapsed = time.time() - t0
        Eb_m, _ = _mean_ci(r["E_b"])
        Pd_m, _ = _mean_ci(r["P_d"])
        Gp_m, _ = _mean_ci(r["G_p"])
        wp, wr, wh, wd = weights
        rows.append({
            "Profile":           label,
            "(w_p,w_r,w_h,w_d)": f"({wp},{wr},{wh},{wd})",
            "E_b (mJ/KB)":       round(Eb_m, 2),
            "P_d (%)":           round(Pd_m, 1),
            "G_p (KB/s)":        round(Gp_m, 3),
        })
        print(f"done ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/results_pi_sensitivity.csv", index=False)
    print(df.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Experiment 6 – Hardware stress (N=200)
# ────────────────────────────────────────────────────────────────────────────

def exp_hw_stress(bank: np.ndarray) -> pd.DataFrame:
    print("\n── Experiment 6: Hardware stress (N=200) ──")
    N = 200
    rows = []
    profile_labels = {
        "nominal":     "Nominal node model",
        "constrained": "Constrained node model",
        "stressed":    "Stressed node model",
    }
    for key, label in profile_labels.items():
        hw = HW_PROFILES[key]
        print(f"   {label} ...", end=" ", flush=True)
        t0 = time.time()
        r  = _collect(N, "proposed", bank, hw_profile=hw)
        elapsed = time.time() - t0
        Eb_m, _ = _mean_ci(r["E_b"])
        Tc_m, _ = _mean_ci(r["T_c"])
        Pd_m, _ = _mean_ci(r["P_d"])
        Gp_m, _ = _mean_ci(r["G_p"])
        rows.append({
            "Profile":            label,
            "Timing uncertainty": f"±{hw['sync_uncert_us']} µs",
            "Extra turnaround":   f"{hw['extra_ta_us']} µs",
            "Buffer budget":      f"{hw['buffer']} packets",
            "E_b (mJ/KB)":        round(Eb_m, 2),
            "T_c (ms)":           round(Tc_m, 1),
            "P_d (%)":            round(Pd_m, 1),
            "G_p (KB/s)":         round(Gp_m, 3),
        })
        print(f"done ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/results_hw_stress.csv", index=False)
    print(df.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 65)
    print("Dense IoT Synchronous Flooding — Simulation")
    print("=" * 65)

    # Load / generate fluctuation template bank
    print("\nLoading dataset templates …")
    bank = load_all_templates(
        intel_path=INTEL_CSV,
        crawdad_path=CRAWDAD_CSV,
        n_each=200,
        segment_len=3600,
    )

    exp_main(bank)
    exp_density(bank)
    exp_ablation(bank)
    exp_stats(bank)
    exp_pi_sensitivity(bank)
    exp_hw_stress(bank)

    print(f"\n✓  All experiments complete in {(time.time()-t_start)/60:.1f} min")
    print(f"   Results saved to: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
