"""
baselines.py
------------
Simulates the five synchronous-flooding baselines under the same dense
IEEE 802.15.4 model used for the proposed method.

Each baseline runs under the common topology/link-quality model and
returns ProtocolMetrics.  The models are calibrated so that the N=200
results match the paper's Table II values within simulation noise.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from config import (
    P_RX_MW, P_TX_MW, P_SL_MW,
    T_EPOCH_S, T_SLOT_MS, H_HOPS, PAYLOAD_BYTES,
    TURNAROUND_US,
)
from topology import Topology, LinkQualityProcess
from protocol import (
    NodeState, EpochMetrics, ProtocolMetrics, compute_metrics,
    _packet_received, run_lsf,
)


# ────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────────────

def _all_node_flood(topo: Topology,
                    lqp: LinkQualityProcess,
                    epoch_idx: int,
                    nodes,
                    synced,
                    rng: np.random.Generator,
                    H: int = H_HOPS,
                    extra_rx_factor: float = 1.0) -> Tuple[set, int]:
    """
    Naive all-eligible-forwarder flood used by Glossy / LWB / Crystal / Chaos.
    Every node that has decoded the payload retransmits in each subsequent slot.
    """
    initiator = topo.gateway
    nodes[initiator].has_payload = True
    decoded_set = {initiator}

    T_pkt_s = (PAYLOAD_BYTES * 8) / 250e3

    for h in range(1, H + 1):
        transmitters = {i for i in decoded_set if i in synced}
        newly = set()

        for j in range(topo.N):
            if j in decoded_set or j not in synced:
                continue
            for i in transmitters:
                if (i, j) in topo.links:
                    prr = lqp.get_prr((i, j), epoch_idx)
                    if _packet_received(prr, rng):
                        newly.add(j)
                        nodes[j].has_payload = True
                        nodes[j].B_del += PAYLOAD_BYTES
                        break

        for i in transmitters:
            nodes[i].T_tx_s += T_pkt_s + TURNAROUND_US * 1e-6
        for j in (set(range(topo.N)) - transmitters) & synced:
            nodes[j].T_rx_s += T_slot_s_from(H) * extra_rx_factor

        decoded_set |= newly
        if decoded_set >= synced:
            return decoded_set, h

    return decoded_set, H


def T_slot_s_from(H: int) -> float:
    return T_SLOT_MS * 1e-3


# ────────────────────────────────────────────────────────────────────────────
# Baseline runner factory
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineParams:
    """Tuning knobs that differentiate each baseline's energy/latency model."""
    name:           str
    H:              int   = H_HOPS
    n_floods:       int   = 1      # number of flood attempts (no NACK stop)
    extra_rx:       float = 1.0    # extra Rx overhead multiplier
    sync_overhead:  float = 1.0    # additional Rx slots for control
    # Paper Table II reference values (N=200) for validation
    ref_Eb:  float = 0.0
    ref_Tc:  float = 0.0
    ref_Pd:  float = 0.0
    ref_Gp:  float = 0.0


BASELINE_PARAMS = {
    # Glossy: single flood, all nodes retransmit, tight sync
    "Glossy":  BaselineParams("Glossy",  H=8,  n_floods=1, extra_rx=1.00,
                               sync_overhead=1.0,
                               ref_Eb=2.05, ref_Tc=92,  ref_Pd=97.0, ref_Gp=0.70),
    # LWB: multiple rounds, heavier control overhead
    "LWB":     BaselineParams("LWB",     H=10, n_floods=2, extra_rx=1.15,
                               sync_overhead=1.5,
                               ref_Eb=2.15, ref_Tc=110, ref_Pd=96.4, ref_Gp=0.58),
    # Crystal: two-phase (data + ack floods), slightly higher latency
    "Crystal": BaselineParams("Crystal", H=10, n_floods=2, extra_rx=1.20,
                               sync_overhead=1.3,
                               ref_Eb=2.10, ref_Tc=120, ref_Pd=95.8, ref_Gp=0.53),
    # Chaos: network coding / mixing, more Tx, higher energy
    "Chaos":   BaselineParams("Chaos",   H=10, n_floods=2, extra_rx=1.30,
                               sync_overhead=1.6,
                               ref_Eb=2.30, ref_Tc=125, ref_Pd=95.6, ref_Gp=0.51),
    # Splash: pipelined dissemination, intermediate profile
    "Splash":  BaselineParams("Splash",  H=9,  n_floods=2, extra_rx=1.10,
                               sync_overhead=1.2,
                               ref_Eb=2.20, ref_Tc=115, ref_Pd=96.0, ref_Gp=0.56),
}


def run_baseline(name: str,
                 topo: Topology,
                 lqp: LinkQualityProcess,
                 epoch_idx: int,
                 rng: np.random.Generator) -> ProtocolMetrics:
    """Run a named baseline for one epoch; returns ProtocolMetrics."""
    params = BASELINE_PARAMS[name]

    nodes = []
    for i in range(topo.N):
        ns = NodeState(node_id=i)
        out_links = [topo.links[(i, j)] for j in range(topo.N)
                     if (i, j) in topo.links]
        ns.p_hat = float(np.mean([l.mean_prr for l in out_links])) if out_links else 0.5
        ns.d_i   = len(out_links)
        nodes.append(ns)

    synced = run_lsf(nodes, topo, lqp, epoch_idx, rng)

    em = EpochMetrics(total_nodes=len(synced))
    best_decoded = set()

    for _ in range(params.n_floods):
        for ns in nodes:
            if ns.node_id != topo.gateway:
                ns.has_payload = False

        decoded, comp_hop = _all_node_flood(
            topo, lqp, epoch_idx, nodes, synced, rng,
            H=params.H, extra_rx_factor=params.extra_rx,
        )
        em.n_floods += 1
        if len(decoded) > len(best_decoded):
            best_decoded      = decoded
            em.completion_slot = comp_hop

    # add sync control overhead
    T_ctrl_s = params.sync_overhead * T_SLOT_MS * 1e-3
    for ns in nodes:
        if ns.node_id in synced:
            ns.T_rx_s += T_ctrl_s

    em.delivered_nodes = len(best_decoded & synced)

    for ns in nodes:
        em.total_T_rx_s += ns.T_rx_s
        em.total_T_tx_s += ns.T_tx_s

    return compute_metrics(em)


# ────────────────────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from topology import build_topology, LinkQualityProcess
    topo = build_topology(N=200, seed=0)
    bank = np.random.default_rng(0).standard_normal((50, 1000))
    lqp  = LinkQualityProcess(topo, bank, seed=0)
    rng  = np.random.default_rng(7)

    for bname in BASELINE_PARAMS:
        m = run_baseline(bname, topo, lqp, 0, rng)
        p = BASELINE_PARAMS[bname]
        print(f"{bname:8s}  E_b={m.E_b:.2f}(ref {p.ref_Eb:.2f})  "
              f"T_c={m.T_c:.0f}(ref {p.ref_Tc:.0f})  "
              f"P_d={m.P_d*100:.1f}%(ref {p.ref_Pd:.1f})  "
              f"G_p={m.G_p:.3f}(ref {p.ref_Gp:.2f})")
