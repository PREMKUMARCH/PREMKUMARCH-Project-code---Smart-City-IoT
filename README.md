# Dense IoT Synchronous Flooding — Simulation Code

## Overview

This codebase implements and evaluates the proposed dense-IoT low-power synchronous-flooding protocol, along with five standard baselines (Glossy, LWB, Crystal, Chaos, Splash).

## File Structure

iot_sim/
├── config.py           # All hyperparameters & hardware constants
├── dataset_loader.py   # Intel Lab + CRAWDAD dataset loaders & template extraction
├── topology.py         # Random 2-D deployment & trace-driven link quality (Eq. trace-adapt)
├── protocol.py         # Algorithms 1–3: LSF, TCR, SF-NACK + metric computation
├── baselines.py        # Glossy / LWB / Crystal / Chaos / Splash models
├── simulation.py       # Main experiment runner → writes results/*.csv

## Dataset Setup

### 1. Intel Berkeley Research Lab Sensor Data

# Download and decompress
wget http://db.csail.mit.edu/labdata/data.txt.gz
gunzip data.txt.gz

# Add CSV header and move to datasets/
echo "date, time, epoch,moteid, temperature, humidity, light, voltage" \
  > datasets/intel_lab_data.csv
cat data.txt >> datasets/intel_lab_data.csv

### 2. CRAWDAD Zigbee Smart-Home

1. Visit https://ieee-dataport.org/open-access/crawdad-cmuzigbee-smarthome
2. Create a free IEEE account and download the dataset
3. Save as `datasets/crawdad_zigbee.csv`

> Note: If either dataset file is absent, the simulator automatically
> substitutes synthetic white-noise fluctuation templates with a warning,
> so all experiments still run correctly.

## Installation

pip install numpy pandas scipy matplotlib

## Running

# Run all 6 experiments (30 seeds × 20 epochs each, ~5–15 min)
python simulation.py

## Key Parameters (config.py)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `T_EPOCH_S` | 1.0 s | Epoch duration |
| `T_SLOT_MS` | 5 ms | Slot duration T_s |
| `H_HOPS` | 8 | Hop budget |
| `R_REDUND` | 2 | Retransmission budget per node |
| `K_REPEATS` | 3 | Max staggered repetitions |
| `Q_NACK` | 8 | NACK window slots |
| `ALPHA` | 0.5 | Clock correction gain |
| `E_MIN_MJ` | 0.02 mJ | NACK energy stop threshold |
| `W_P/W_R/W_H/W_D` | 0.35/0.20/0.30/0.15 | Forwarder score weights |

## Simulation Architecture

dataset_loader  →  fluctuation templates u_ij(t)
                              ↓
topology.build_topology  →  positions, links, mean PRR p̄_ij
                              ↓
topology.LinkQualityProcess  →  p_ij(t) = clip(p̄_ij + σ_ij·u_ij(t))
                              ↓
For each epoch:
  protocol.run_lsf      (Algorithm 1 — clock sync)
      ↓
  protocol.run_sf_nack  (Algorithm 3 — outer loop)
      ↓  calls
  protocol.run_tcr      (Algorithm 2 — data flood)
      ↓
  protocol.compute_metrics  →  E_b, T_c, P_d, G_p, ATC

