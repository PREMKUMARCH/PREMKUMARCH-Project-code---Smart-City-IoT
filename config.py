"""
Simulation Configuration
IEEE 802.15.4-class dense IoT synchronous flooding protocol
"""

# ── Hardware Power Model (CC2420-class) ──────────────────────────────────────
P_RX_MW      = 56.0      # mW  receive power
P_TX_MW      = 52.0      # mW  transmit power
P_SL_MW      = 0.06      # mW  sleep power

# ── Radio Configuration ──────────────────────────────────────────────────────
FREQ_GHZ     = 2.4
DATA_RATE_KBPS = 250
PAYLOAD_BYTES  = 64       # L  bytes per packet

# ── Modeled Implementation Settings ─────────────────────────────────────────
TIMING_RES_US    = 1.0    # µs timing resolution
SYNC_UNCERT_US   = 0.5    # ±µs synchronisation uncertainty  (nominal)
TURNAROUND_US    = 192.0  # µs CC2420-class Rx→Tx turnaround (nominal)
BUFFER_BUDGET    = 8      # packets

# ── Proposed Hyperparameters ─────────────────────────────────────────────────
T_EPOCH_S   = 1.0         # epoch duration  (s)
T_SLOT_MS   = 5.0         # T_s  slot duration (ms)
H_HOPS      = 8           # hop budget
R_REDUND    = 2           # retransmission budget per node
K_REPEATS   = 3           # max staggered repetitions
Q_NACK      = 8           # NACK window slots
T_GUARD_MS  = 10.0        # T_g guard time between repetitions (ms)
ALPHA       = 0.5         # clock-correction gain
E_MIN_MJ    = 0.02        # NACK energy stop threshold (mJ)

# ── Forwarder-selection weights ──────────────────────────────────────────────
W_P = 0.35   # link quality weight
W_R = 0.20   # RSSI weight
W_H = 0.30   # hop-progress weight
W_D = 0.15   # density penalty weight

TAU_MAX = 0.55   # activation threshold at hop 1
TAU_MIN = 0.35   # floor for threshold
TAU_STEP = 0.04  # per-hop decrement

# ── Topology & Density ───────────────────────────────────────────────────────
NODE_COUNTS  = [50, 100, 200]
AREA_SIDE_M  = 50.0        # side of square deployment area (m)
H_MAX_DEPTH  = 10          # max multi-hop depth in topology

# ── Link-quality clipping ────────────────────────────────────────────────────
P_MIN = 0.05
P_MAX = 0.99
SIGMA_CLASSES = {"stable": 0.05, "moderate": 0.12, "interference": 0.22}

# ── Statistical Protocol ─────────────────────────────────────────────────────
N_SEEDS      = 30          # random seeds per operating point
CI_LEVEL     = 0.95

# ── Hardware stress profiles ─────────────────────────────────────────────────
HW_PROFILES = {
    "nominal":      {"sync_uncert_us": 0.5,  "extra_ta_us":   0, "buffer": 8},
    "constrained":  {"sync_uncert_us": 1.0,  "extra_ta_us":  64, "buffer": 6},
    "stressed":     {"sync_uncert_us": 2.0,  "extra_ta_us": 128, "buffer": 4},
}

# ── Sensitivity profiles ─────────────────────────────────────────────────────
PI_PROFILES = {
    "reliability_leaning": (0.45, 0.20, 0.25, 0.10),
    "balanced_default":    (0.35, 0.20, 0.30, 0.15),
    "sparsity_leaning":    (0.30, 0.15, 0.25, 0.30),
}

# ── Dataset paths ────────────────────────────────────────────────────────────
INTEL_CSV   = "datasets/intel_lab_data.csv"
CRAWDAD_CSV = "datasets/crawdad_zigbee.csv"
