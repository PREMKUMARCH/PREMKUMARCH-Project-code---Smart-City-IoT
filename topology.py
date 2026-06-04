"""
topology.py
-----------
Generates a random dense 2-D deployment of N static nodes and assigns
time-varying link-reception probabilities p_ij(t) following Eq. (trace-adapt):

    p_ij(t) = clip( p̄_ij + σ_ij · u_ij(t),  p_min, p_max )

where
  p̄_ij  – mean PRR derived from inter-node distance + log-normal shadowing
  u_ij(t) – zero-mean unit-variance fluctuation drawn from the template bank
  σ_ij   – perturbation strength chosen from {stable, moderate, interference}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from config import (
    AREA_SIDE_M, H_MAX_DEPTH, P_MIN, P_MAX, SIGMA_CLASSES
)


# ────────────────────────────────────────────────────────────────────────────
# Radio-range & path-loss parameters
# ────────────────────────────────────────────────────────────────────────────
RADIO_RANGE_M  = 15.0   # nominal TX range (m)
PATH_LOSS_EXP  = 2.5    # path-loss exponent
SHADOWING_STD  = 4.0    # log-normal shadowing σ_s (dB)
RX_THRESH_DBM  = -85.0  # minimum receive sensitivity (dBm)
TX_POWER_DBM   = 0.0    # transmit power (dBm)
FREQ_GHZ       = 2.4


@dataclass
class Link:
    i: int
    j: int
    distance_m: float
    mean_prr: float          # p̄_ij
    sigma: float             # σ_ij  (perturbation strength)
    link_class: str          # stable / moderate / interference


@dataclass
class Topology:
    N: int
    positions: np.ndarray                # shape (N, 2)
    links: Dict[Tuple[int,int], Link]    # (i,j) → Link  (directed)
    gateway: int = 0
    hop_distances: np.ndarray = field(default_factory=lambda: np.array([]))


# ────────────────────────────────────────────────────────────────────────────
# Distance → mean PRR  (log-normal path-loss + coverage model)
# ────────────────────────────────────────────────────────────────────────────

def _path_loss_db(dist_m: float, freq_ghz: float = FREQ_GHZ,
                  n: float = PATH_LOSS_EXP) -> float:
    if dist_m < 0.1:
        dist_m = 0.1
    # Friis free-space at 1 m, then n-exponent roll-off
    fspl_1m = 20 * np.log10(freq_ghz * 1e9) - 147.55   # dB
    return fspl_1m + 10 * n * np.log10(dist_m)


def _mean_prr(dist_m: float, rng: np.random.Generator) -> float:
    """
    Compute mean packet reception probability for a link of length dist_m.
    Uses a sigmoid-like mapping of the receive-power margin.
    """
    pl_db   = _path_loss_db(dist_m)
    shadow  = rng.normal(0, SHADOWING_STD)
    rx_dbm  = TX_POWER_DBM - pl_db + shadow
    margin  = rx_dbm - RX_THRESH_DBM        # positive ↔ good link
    # logistic mapping: margin 0 dB → PRR≈0.5, margin +15 dB → PRR≈0.97
    prr = 1.0 / (1.0 + np.exp(-margin / 7.0))
    return float(np.clip(prr, P_MIN, P_MAX))


# ────────────────────────────────────────────────────────────────────────────
# BFS hop distances from gateway
# ────────────────────────────────────────────────────────────────────────────

def _bfs_hop_distances(N: int, adj: Dict[int, List[int]], gateway: int) -> np.ndarray:
    dist = np.full(N, np.inf)
    dist[gateway] = 0
    queue = [gateway]
    while queue:
        cur = queue.pop(0)
        for nb in adj.get(cur, []):
            if dist[nb] == np.inf:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


# ────────────────────────────────────────────────────────────────────────────
# Main topology builder
# ────────────────────────────────────────────────────────────────────────────

def build_topology(N: int,
                   seed: int = 0,
                   area_side: float = AREA_SIDE_M,
                   radio_range: float = RADIO_RANGE_M,
                   sigma_classes: Dict[str, float] = SIGMA_CLASSES) -> Topology:
    """
    Place N nodes uniformly in [0, area_side]^2, create directed links for
    all pairs within radio_range, assign mean PRR and perturbation class.
    Guarantees the graph is connected (retries if needed).
    """
    rng = np.random.default_rng(seed)
    class_names  = list(sigma_classes.keys())
    class_sigmas = list(sigma_classes.values())
    # distribution: 50% stable, 30% moderate, 20% interference
    class_probs  = [0.50, 0.30, 0.20]

    for attempt in range(200):
        positions = rng.uniform(0, area_side, size=(N, 2))
        links: Dict[Tuple[int,int], Link] = {}
        adj:   Dict[int, List[int]]       = {i: [] for i in range(N)}

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d > radio_range:
                    continue
                prr   = _mean_prr(d, rng)
                cls   = rng.choice(class_names, p=class_probs)
                sigma = sigma_classes[cls]
                links[(i, j)] = Link(i, j, d, prr, sigma, cls)
                adj[i].append(j)

        # check connectivity (undirected reachability from node 0)
        hop_dist = _bfs_hop_distances(N, adj, gateway=0)
        if np.all(np.isfinite(hop_dist)) and hop_dist.max() <= H_MAX_DEPTH:
            topo = Topology(
                N=N,
                positions=positions,
                links=links,
                gateway=0,
                hop_distances=hop_dist,
            )
            return topo

    # Last resort: increase range and try once more
    return build_topology(N, seed, area_side, radio_range * 1.3, sigma_classes)


# ────────────────────────────────────────────────────────────────────────────
# Link-quality time series generator  (Eq. trace-adapt)
# ────────────────────────────────────────────────────────────────────────────

class LinkQualityProcess:
    """
    Wraps a Topology and a fluctuation template bank to produce p_ij(t)
    for every link at each epoch step.
    """

    def __init__(self, topo: Topology, template_bank: np.ndarray,
                 seed: int = 0):
        self.topo          = topo
        self.template_bank = template_bank          # (M, T)
        self.rng           = np.random.default_rng(seed)
        self._n_templates, self._tlen = template_bank.shape

        # assign a template index to each link
        self._link_template: Dict[Tuple[int,int], int] = {}
        for key in topo.links:
            self._link_template[key] = int(
                self.rng.integers(0, self._n_templates)
            )

    def get_prr(self, link_key: Tuple[int,int], t: int) -> float:
        """Return p_ij at epoch index t (wraps around template length)."""
        lnk = self.topo.links[link_key]
        tmpl_idx = self._link_template[link_key]
        u = self.template_bank[tmpl_idx, t % self._tlen]
        p = lnk.mean_prr + lnk.sigma * u
        return float(np.clip(p, P_MIN, P_MAX))

    def get_all_prr(self, t: int) -> Dict[Tuple[int,int], float]:
        """Return PRR for every link at epoch t."""
        return {k: self.get_prr(k, t) for k in self.topo.links}


# ────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    topo = build_topology(N=50, seed=7)
    print(f"Nodes: {topo.N}")
    print(f"Links: {len(topo.links)}")
    print(f"Max hop distance from GW: {topo.hop_distances[np.isfinite(topo.hop_distances)].max():.0f}")
    print(f"Mean hop distance       : {topo.hop_distances[np.isfinite(topo.hop_distances)].mean():.2f}")

    dummy_bank = np.random.default_rng(0).standard_normal((50, 500))
    lqp = LinkQualityProcess(topo, dummy_bank, seed=0)
    sample_key = list(topo.links.keys())[0]
    print(f"Sample p_ij(t=0) for link {sample_key}: {lqp.get_prr(sample_key, 0):.4f}")
