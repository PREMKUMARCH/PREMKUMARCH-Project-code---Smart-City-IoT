"""
protocol.py
-----------
Implements the three core algorithms from the paper:

  Algorithm 1 – LSF  : Lightweight Synchronisation Mini-Flood
  Algorithm 2 – TCR  : Topology-Aware Controlled-Redundancy Flood Scheduling
  Algorithm 3 – SF-NACK: Staggered Floods with Fast Negative Acknowledgements

Also contains:
  • EpochMetrics – collects T_rx, T_tx, T_sl, B_del per epoch
  • compute_metrics – derives E_b, T_c, P_d, G_p, ATC from EpochMetrics
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from config import (
    P_RX_MW, P_TX_MW, P_SL_MW,
    T_EPOCH_S, T_SLOT_MS, H_HOPS, R_REDUND, K_REPEATS, Q_NACK,
    T_GUARD_MS, ALPHA, E_MIN_MJ,
    W_P, W_R, W_H, W_D, TAU_MAX, TAU_MIN, TAU_STEP,
    PAYLOAD_BYTES, TIMING_RES_US, SYNC_UNCERT_US, TURNAROUND_US,
)
from topology import Topology, LinkQualityProcess


# ────────────────────────────────────────────────────────────────────────────
# Node state
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeState:
    node_id:     int
    clock_offset: float = 0.0     # θ_i (s)  relative to initiator
    drift_rate:   float = 0.0     # δ_i
    epsilon_hat:  float = 0.0     # ε̂_i  expected time-error bound
    has_payload:  bool  = False   # x_i
    tx_count:     int   = 0       # txCount_i
    # link-quality observables (updated each epoch)
    p_hat:        float = 0.5     # p̂_i  recent reception ratio
    r_rssi:       float = 0.5     # r_i   normalised RSSI ∈ [0,1]
    d_i:          int   = 0       # neighbourhood count
    # energy / timing accumulators (reset each epoch)
    T_rx_s:       float = 0.0
    T_tx_s:       float = 0.0
    B_del:        int   = 0       # delivered bytes

    def reset_epoch(self):
        self.has_payload = False
        self.tx_count    = 0
        self.T_rx_s      = 0.0
        self.T_tx_s      = 0.0
        self.B_del       = 0


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    """Raw accumulators for one epoch across all nodes."""
    total_T_rx_s: float = 0.0
    total_T_tx_s: float = 0.0
    delivered_nodes: int = 0
    total_nodes:     int = 0
    n_floods:        int = 0   # how many TCR floods ran (≤ K)
    completion_slot: int = 0   # hop index at which all reachable nodes decoded


@dataclass
class ProtocolMetrics:
    """Derived metrics: E_b, T_c, P_d, G_p, ATC."""
    E_b:  float = 0.0   # mJ / KB
    T_c:  float = 0.0   # ms
    P_d:  float = 0.0   # fraction [0,1]
    G_p:  float = 0.0   # KB/s
    ATC:  float = 0.0   # ms  average time to convergence


def compute_metrics(em: EpochMetrics,
                    T_epoch_s: float = T_EPOCH_S,
                    payload_bytes: int = PAYLOAD_BYTES) -> ProtocolMetrics:
    """Convert raw EpochMetrics to the four paper metrics + ATC."""
    T_sl_s = max(0.0, T_epoch_s - em.total_T_rx_s - em.total_T_tx_s)
    E_epoch_mj = (
        P_RX_MW * em.total_T_rx_s +
        P_TX_MW * em.total_T_tx_s +
        P_SL_MW * T_sl_s
    )   # mJ  (mW × s = mJ)

    delivered_kb = (em.delivered_nodes * payload_bytes) / 1024.0
    E_b  = (E_epoch_mj / delivered_kb) if delivered_kb > 0 else float("inf")

    T_c_ms = em.n_floods * H_HOPS * T_SLOT_MS   # coarse: floods × hops × slot
    P_d    = em.delivered_nodes / max(1, em.total_nodes)
    G_p    = delivered_kb / T_epoch_s if T_epoch_s > 0 else 0.0  # KB/s
    ATC_ms = em.completion_slot * T_SLOT_MS

    return ProtocolMetrics(E_b=E_b, T_c=T_c_ms, P_d=P_d, G_p=G_p, ATC=ATC_ms)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _tau(h: int,
         tau_max: float = TAU_MAX,
         tau_min: float = TAU_MIN,
         step: float = TAU_STEP) -> float:
    """Hop-dependent activation threshold τ(h) (Eq. forwardscore threshold)."""
    return float(np.clip(tau_max - step * (h - 1), tau_min, tau_max))


def _hop_progress(node_id: int, h: int,
                  decoded_set: Set[int],
                  topo: Topology,
                  H: int = H_HOPS) -> float:
    """
    φ_i(h) = min(1, ℓ_i(h) / H)
    ℓ_i(h) ≈ hop distance from node_i to nearest un-decoded node.
    """
    not_decoded = [n for n in range(topo.N) if n not in decoded_set]
    if not not_decoded:
        return 1.0
    # use hop_distances from topology (distance from gateway as proxy)
    my_dist = topo.hop_distances[node_id] if np.isfinite(topo.hop_distances[node_id]) else H
    min_dist = min(
        topo.hop_distances[n] for n in not_decoded
        if np.isfinite(topo.hop_distances[n])
    ) if not_decoded else 0.0
    ell = abs(my_dist - min_dist)
    return float(min(1.0, ell / H))


def _forward_score(ns: NodeState, phi: float,
                   d_max: int,
                   wp: float = W_P, wr: float = W_R,
                   wh: float = W_H, wd: float = W_D) -> float:
    """S_i(h) from Eq. (forwardscore)."""
    d_norm = ns.d_i / max(1, d_max)
    return wp * ns.p_hat + wr * ns.r_rssi + wh * phi - wd * d_norm


def _top_m(candidates: List[int],
           scores: Dict[int, float],
           R: int, N_h: int) -> Set[int]:
    """TopM_h = best M_h candidates by score."""
    M_h = min(R + 1, math.ceil(math.sqrt(max(1, N_h))))
    sorted_c = sorted(candidates, key=lambda n: scores[n], reverse=True)
    return set(sorted_c[:M_h])


def _packet_received(prr: float, rng: np.random.Generator) -> bool:
    return rng.random() < prr


# ────────────────────────────────────────────────────────────────────────────
# Algorithm 1 – LSF
# ────────────────────────────────────────────────────────────────────────────

def run_lsf(nodes: List[NodeState],
            topo: Topology,
            lqp: LinkQualityProcess,
            epoch_idx: int,
            rng: np.random.Generator,
            alpha: float = ALPHA,
            sync_uncert_us: float = SYNC_UNCERT_US) -> Set[int]:
    """
    Lightweight Synchronisation Mini-Flood (Algorithm 1).

    The initiator (node 0) broadcasts Sync.  Each reachable node updates
    its clock offset by α·θ̂_i.  Returns the set of nodes that received Sync.
    """
    initiator = topo.gateway
    synced: Set[int] = {initiator}

    for j in range(topo.N):
        if j == initiator:
            continue
        key = (initiator, j)
        if key not in topo.links:
            continue
        prr = lqp.get_prr(key, epoch_idx)
        if _packet_received(prr, rng):
            ns = nodes[j]
            # timestamp noise proportional to uncertainty
            ts_noise = rng.uniform(-sync_uncert_us * 1e-6, sync_uncert_us * 1e-6)
            theta_hat = ns.clock_offset + ts_noise
            ns.clock_offset -= alpha * theta_hat
            ns.epsilon_hat   = abs(theta_hat)
            # update T_rx for the sync slot
            T_sync_s = (PAYLOAD_BYTES * 8) / (250e3)  # time to receive sync pkt
            ns.T_rx_s += T_sync_s
            synced.add(j)

    return synced


# ────────────────────────────────────────────────────────────────────────────
# Algorithm 2 – TCR
# ────────────────────────────────────────────────────────────────────────────

def run_tcr(nodes: List[NodeState],
            topo: Topology,
            lqp: LinkQualityProcess,
            epoch_idx: int,
            synced: Set[int],
            rng: np.random.Generator,
            H: int = H_HOPS,
            R: int = R_REDUND,
            T_slot_s: float = T_SLOT_MS * 1e-3,
            wp: float = W_P, wr: float = W_R,
            wh: float = W_H, wd: float = W_D) -> Tuple[Set[int], int]:
    """
    Topology-Aware Controlled-Redundancy Flood (Algorithm 2).

    Returns (decoded_set, completion_hop).
    decoded_set: node IDs that decoded the payload.
    completion_hop: hop at which the last node decoded (≤ H).
    """
    initiator = topo.gateway

    # reset per-flood state
    for ns in nodes:
        ns.tx_count    = 0

    # initiator always has the payload
    nodes[initiator].has_payload = True
    decoded_set: Set[int] = {initiator}
    completion_hop = H

    for h in range(1, H + 1):
        tau_h = _tau(h)
        d_max = max((ns.d_i for ns in nodes), default=1)

        # determine transmitters this hop
        candidates = []
        scores: Dict[int, float] = {}

        for i in decoded_set:
            if nodes[i].tx_count >= R:
                continue
            phi = _hop_progress(i, h, decoded_set, topo, H)
            score = _forward_score(nodes[i], phi, d_max, wp, wr, wh, wd)
            if score >= tau_h:
                candidates.append(i)
                scores[i] = score

        N_h = len(decoded_set)
        transmitters: Set[int] = _top_m(candidates, scores, R, N_h)

        # simulate reception: for each non-decoded reachable node,
        # compute combined reception probability from all transmitters
        newly_decoded: Set[int] = set()
        for j in range(topo.N):
            if j in decoded_set or j not in synced:
                continue
            # at least one transmitter must reach j
            received = False
            for i in transmitters:
                key = (i, j)
                if key in topo.links:
                    prr = lqp.get_prr(key, epoch_idx)
                    if _packet_received(prr, rng):
                        received = True
                        break
            if received:
                newly_decoded.add(j)
                nodes[j].has_payload = True
                nodes[j].B_del      += PAYLOAD_BYTES

        # update radio-on times
        T_pkt_s = (PAYLOAD_BYTES * 8) / 250e3
        for i in transmitters:
            nodes[i].T_tx_s  += T_pkt_s + TURNAROUND_US * 1e-6
            nodes[i].tx_count += 1
        for j in (set(range(topo.N)) - transmitters) & synced:
            nodes[j].T_rx_s  += T_slot_s

        decoded_set |= newly_decoded

        # early exit if all synced nodes decoded
        if decoded_set >= synced:
            completion_hop = h
            break

    return decoded_set, completion_hop


# ────────────────────────────────────────────────────────────────────────────
# Algorithm 3 – SF-NACK
# ────────────────────────────────────────────────────────────────────────────

def run_sf_nack(nodes: List[NodeState],
                topo: Topology,
                lqp: LinkQualityProcess,
                epoch_idx: int,
                synced: Set[int],
                rng: np.random.Generator,
                K: int = K_REPEATS,
                Q: int = Q_NACK,
                E_min_mj: float = E_MIN_MJ,
                T_guard_s: float = T_GUARD_MS * 1e-3,
                **tcr_kwargs) -> EpochMetrics:
    """
    Staggered Floods with Fast Negative Acknowledgements (Algorithm 3).

    Runs up to K TCR floods.  After each flood, nodes without the payload
    send a NACK burst.  Stops early when NACK energy ≤ E_min.

    Returns EpochMetrics for the epoch.
    """
    em = EpochMetrics(total_nodes=len(synced))
    T_nack_s = 0.001   # 1 ms per NACK burst (short burst)
    completion_hop = H_HOPS
    decoded_set: Set[int] = {topo.gateway}

    for k in range(1, K + 1):
        # reset payload flags for non-initiator
        for ns in nodes:
            if ns.node_id != topo.gateway:
                ns.has_payload = False

        decoded_set, comp_hop = run_tcr(
            nodes, topo, lqp, epoch_idx, synced, rng, **tcr_kwargs
        )
        em.n_floods     += 1
        completion_hop   = comp_hop

        # ── NACK window ─────────────────────────────────────────────────────
        missing = synced - decoded_set
        E_nack_mj = 0.0
        for j in missing:
            # node j sends NACK burst in a randomly selected slot
            E_nack_mj += P_TX_MW * T_nack_s   # mW × s = mJ
            nodes[j].T_tx_s += T_nack_s

        # initiator listens across Q slots
        nodes[topo.gateway].T_rx_s += Q * T_nack_s

        em.delivered_nodes = len(decoded_set & synced)

        if E_nack_mj <= E_min_mj:
            break   # sufficient delivery — stop repeating

        # guard time before next flood
        for ns in nodes:
            if ns.node_id in synced:
                ns.T_rx_s += T_guard_s  # conservative: stay in rx during guard

    # ── accumulate radio times across all nodes ───────────────────────────────
    for ns in nodes:
        em.total_T_rx_s += ns.T_rx_s
        em.total_T_tx_s += ns.T_tx_s

    em.completion_slot = completion_hop
    return em


# ────────────────────────────────────────────────────────────────────────────
# Full epoch runner
# ────────────────────────────────────────────────────────────────────────────

def run_epoch(topo: Topology,
              lqp: LinkQualityProcess,
              epoch_idx: int,
              rng: np.random.Generator,
              hw_profile: Optional[dict] = None,
              pi_weights: Optional[Tuple[float,float,float,float]] = None,
              H: int = H_HOPS,
              R: int = R_REDUND,
              K: int = K_REPEATS,
              Q: int = Q_NACK) -> ProtocolMetrics:
    """
    Run one complete epoch: LSF → SF-NACK(TCR) → compute metrics.
    """
    # hardware profile overrides
    su_us = hw_profile["sync_uncert_us"]  if hw_profile else SYNC_UNCERT_US
    ta_us = TURNAROUND_US + (hw_profile["extra_ta_us"] if hw_profile else 0)

    # weights override
    wp, wr, wh, wd = pi_weights if pi_weights else (W_P, W_R, W_H, W_D)

    # build node states
    nodes = []
    for i in range(topo.N):
        ns = NodeState(node_id=i)
        # initialise link-quality observables from topology
        out_links = [topo.links[(i, j)] for j in range(topo.N) if (i, j) in topo.links]
        ns.p_hat  = float(np.mean([l.mean_prr for l in out_links])) if out_links else 0.5
        ns.d_i    = len(out_links)
        ns.r_rssi = float(np.clip(ns.p_hat + rng.normal(0, 0.05), 0, 1))
        nodes.append(ns)

    # ── LSF ──────────────────────────────────────────────────────────────────
    synced = run_lsf(nodes, topo, lqp, epoch_idx, rng,
                     alpha=ALPHA, sync_uncert_us=su_us)

    # ── SF-NACK (wraps TCR) ───────────────────────────────────────────────────
    em = run_sf_nack(
        nodes, topo, lqp, epoch_idx, synced, rng,
        K=K, Q=Q, E_min_mj=E_MIN_MJ,
        H=H, R=R, wp=wp, wr=wr, wh=wh, wd=wd
    )

    return compute_metrics(em)


# ────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from topology import build_topology, LinkQualityProcess
    topo = build_topology(N=50, seed=0)
    bank = np.random.default_rng(0).standard_normal((50, 1000))
    lqp  = LinkQualityProcess(topo, bank, seed=0)
    rng  = np.random.default_rng(42)

    m = run_epoch(topo, lqp, epoch_idx=0, rng=rng)
    print(f"E_b={m.E_b:.3f} mJ/KB  T_c={m.T_c:.1f} ms  "
          f"P_d={m.P_d*100:.1f}%  G_p={m.G_p:.4f} KB/s  ATC={m.ATC:.1f} ms")
