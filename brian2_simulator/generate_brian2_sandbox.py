from brian2 import *
import pandas as pd
import numpy as np
import os

# ==============================================================================
# DESIGN PRINCIPLES FOR ATTENTION-RECONSTRUCTABLE TOPOLOGIES
# ==============================================================================
# [P1] UNIQUE UPSTREAM SIGNATURES per node.
# [P2] STRONG, ASYMMETRIC DRIVE: drivers (col=0) high mean current, downstream
#      nodes low baseline. Maximises driven-vs-spontaneous contrast.
# [P3] LOW SYNAPTIC WEIGHTS (0.02–0.05) to avoid post-synaptic saturation.
# [P4] NARROW SMOOTHING WINDOW (5 ms) to preserve temporal lag structure.
# [P5] SPARSE TOPOLOGY: each downstream node has 1–3 upstream parents.
# [P6] ALL NODES REACHABLE from a col=0 driver.
# [P7] INDEPENDENT TIME-VARYING DRIVER INPUTS  ← UPDATED
#      Each driver row receives a UNIQUE time-varying current — a sum of
#      random sinusoids in the 2–10 Hz band with row-specific frequencies,
#      phases, and amplitudes. This breaks the statistical similarity that
#      previously made all drivers look like each other to the iTransformer
#      attention mechanism after StandardScaler removed their mean differences.
# ==============================================================================


class TopologyBuilder:
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.edges = {}

    def add_edge(self, src: tuple, tgt: tuple, weight=0.02, prob=0.7):
        self.edges[(src, tgt)] = (weight, prob)

    def remove_edge(self, src: tuple, tgt: tuple):
        if (src, tgt) in self.edges:
            del self.edges[(src, tgt)]
            print(f"   \u2702\ufe0f  Connection cut: {src} -> {tgt}")
        else:
            print(f"   \u26a0\ufe0f  Attempted to cut non-existent edge: {src} -> {tgt}")

    def add_connections_from_list(self, edge_list, default_weight=0.02, default_prob=0.7):
        for edge in edge_list:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1], default_weight, default_prob)
            elif len(edge) == 4:
                self.add_edge(edge[0], edge[1], edge[2], edge[3])
            else:
                print(f"   \u26a0\ufe0f  Ignoring malformed edge: {edge}")

    # ------------------------------------------------------------------
    # Built-in topologies
    # ------------------------------------------------------------------

    def build_standard_grid(self, weight=0.02, prob=0.7):
        """Standard right+down grid."""
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols:
                    self.add_edge((r, c), (r, c+1), weight, prob)
                if r + 1 < self.rows:
                    self.add_edge((r, c), (r+1, c), weight, prob)

    def build_chain_of_chains(self, weight=0.04, prob=0.7):
        """5 independent horizontal chains, no cross-row connections."""
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.add_edge((r, c), (r, c+1), weight, prob)

    def build_funnel(self, weight=0.04, prob=0.7):
        """All drivers fan in to hub (2,2); hub fans out to col=3, then col=4."""
        for r in range(self.rows):
            self.add_edge((r, 0), (2, 2), 0.015, prob)
        for r in range(self.rows):
            self.add_edge((2, 2), (r, 3), weight, prob)
            self.add_edge((r, 3), (r, 4), weight, prob)

    def build_highway(self, weight=0.02, prob=0.7, offramp_weight=0.04, offramp_prob=0.8):
        """Trunk row 2, on-ramps from other rows, off-ramps from (2,4)."""
        for c in range(self.cols - 1):
            self.add_edge((2, c), (2, c+1), weight, prob)
        on_ramp_targets = {0: (2, 1), 1: (2, 2), 3: (2, 3), 4: (2, 4)}
        for r, tgt in on_ramp_targets.items():
            self.add_edge((r, 0), tgt, weight, prob)
        for r in range(self.rows):
            if r != 2:
                self.add_edge((2, 4), (r, 4), offramp_weight, offramp_prob)

    def build_binary_tree(self, weight=0.04, prob=0.7):
        """Binary tree rooted at (2,0)."""
        # Root chain
        for c in range(self.cols - 1):
            self.add_edge((2, c), (2, c+1), weight, prob)
        # Upper subtree
        self.add_edge((2, 0), (1, 1), weight, prob)
        self.add_edge((1, 1), (0, 2), weight, prob)
        self.add_edge((1, 1), (1, 2), weight, prob)
        self.add_edge((0, 2), (0, 3), weight, prob)
        self.add_edge((0, 3), (0, 4), weight, prob)
        self.add_edge((1, 2), (1, 3), weight, prob)
        self.add_edge((1, 3), (1, 4), weight, prob)
        # Lower subtree
        self.add_edge((2, 0), (3, 1), weight, prob)
        self.add_edge((3, 1), (3, 2), weight, prob)
        self.add_edge((3, 1), (4, 2), weight, prob)
        self.add_edge((3, 2), (3, 3), weight, prob)
        self.add_edge((3, 3), (3, 4), weight, prob)
        self.add_edge((4, 2), (4, 3), weight, prob)
        self.add_edge((4, 3), (4, 4), weight, prob)

    def build_hourglass(self, weight=0.04, prob=0.7, bridge_weight=0.05, bridge_prob=0.8, fanin_weight=0.015):
        """Hourglass with single bottleneck bridge (2,2) -> (2,3)."""
        for r in range(self.rows):
            self.add_edge((r, 0), (r, 1), weight, prob)
            self.add_edge((r, 1), (2, 2), fanin_weight, prob)
        self.add_edge((2, 2), (2, 3), bridge_weight, bridge_prob)
        for r in range(self.rows):
            self.add_edge((2, 3), (r, 3), weight, prob)
            self.add_edge((r, 3), (r, 4), weight, prob)

    def export_config(self):
        return [(src, tgt, w, p) for (src, tgt), (w, p) in self.edges.items()]

    def print_summary(self):
        print(f"\n\U0001f4d0 Topology summary: {len(self.edges)} edges")
        for (src, tgt), (w, p) in sorted(self.edges.items()):
            print(f"   {src} --[w={w:.2f}, p={p:.1f}]--> {tgt}")


# ==============================================================================
# [P7] Independent driver-input waveform generator
# ==============================================================================
def _generate_row_driver_waveforms(
    n_rows: int,
    duration_sec: float,
    dt_ms: float = 1.0,
    base_current: float = 1.2,
    row_offset_step: float = 0.1,
    n_sinusoids: int = 3,
    freq_range=(2.0, 10.0),
    amp_range=(0.10, 0.20),
) -> np.ndarray:
    """
    Generate one independent time-varying current waveform per row.

    waveform[r, t] = (base + r * offset_step) + sum_k a_k * sin(2*pi*f_k*t + p_k)

    Each row's (a_k, f_k, p_k) are independently sampled, so the resulting
    waveforms are uncorrelated and carry row-specific temporal fingerprints
    that propagate downstream via synapses.
    """
    n_steps = int(duration_sec * 1000.0 / dt_ms)
    t_axis = np.arange(n_steps) * (dt_ms / 1000.0)  # seconds

    waveforms = np.zeros((n_rows, n_steps), dtype=np.float64)
    for r in range(n_rows):
        dc = base_current + r * row_offset_step
        freqs  = np.random.uniform(freq_range[0], freq_range[1], size=n_sinusoids)
        phases = np.random.uniform(0.0, 2*np.pi,            size=n_sinusoids)
        amps   = np.random.uniform(amp_range[0],  amp_range[1], size=n_sinusoids)
        modulation = np.zeros_like(t_axis)
        for a, f, p in zip(amps, freqs, phases):
            modulation += a * np.sin(2*np.pi*f*t_axis + p)
        waveforms[r] = dc + modulation
    return waveforms


# ==============================================================================
# Core simulation function
# ==============================================================================
def generate_simulation(
    output_path="./data/raw/brian2/brian2_data.csv",
    duration_sec=60,
    topology_type="chain_of_chains",
    is_faulty=False,
    custom_edges=None,
    fault_target=None,
    smooth_width_ms=5,
    input_current_base=1.2,
    baseline_current=0.3,
):
    print(f"\n\U0001f680 Brian2 5x5 | Topology: [{topology_type.upper()}] | "
          f"State: [{'Fault' if is_faulty else 'Healthy'}]")
    start_scope()
    defaultclock.dt = 1 * ms

    # Equation for downstream neurons — constant baseline current
    eqs_baseline = '''
    dv/dt = (I - v) / tau + sigma * xi * tau**-0.5 : 1 (unless refractory)
    I : 1
    tau : second
    sigma : 1
    '''

    # Equation for driver neurons — time-varying I from per-group TimedArray 'drv'
    eqs_driver = '''
    dv/dt = (I - v) / tau + sigma * xi * tau**-0.5 : 1 (unless refractory)
    I = drv(t) : 1
    tau : second
    sigma : 1
    '''

    rows, cols = 5, 5
    neurons = {}
    monitors = {}

    # ------------------------------------------------------------------
    # [P7] Generate independent driver waveforms — one TimedArray per row
    # ------------------------------------------------------------------
    row_waveforms = _generate_row_driver_waveforms(
        n_rows=rows,
        duration_sec=duration_sec,
        dt_ms=1.0,
        base_current=input_current_base,
        row_offset_step=0.1,
    )
    row_timed_arrays = [TimedArray(row_waveforms[r], dt=1*ms) for r in range(rows)]

    print(f"\U0001f39b\ufe0f  Driver waveforms: {rows} independent sinusoidal inputs "
          f"(mean I = {input_current_base:.2f} .. {input_current_base + (rows-1)*0.1:.2f})")

    # ------------------------------------------------------------------
    # Build neuron groups
    # ------------------------------------------------------------------
    for r in range(rows):
        for c in range(cols):
            if c == 0:
                # Driver group — references its row's TimedArray as 'drv'
                G = NeuronGroup(
                    50, eqs_driver,
                    threshold='v>1', reset='v=0', refractory=2*ms,
                    method='euler',
                    namespace={'drv': row_timed_arrays[r]},
                )
            else:
                # Downstream group — constant baseline current
                G = NeuronGroup(
                    50, eqs_baseline,
                    threshold='v>1', reset='v=0', refractory=2*ms,
                    method='euler',
                )
                G.I = baseline_current

            G.v = '0.0'
            G.tau = 10 * ms
            G.sigma = 0.15

            neurons[(r, c)] = G
            monitors[(r, c)] = PopulationRateMonitor(G)

    # ------------------------------------------------------------------
    # Build topology
    # ------------------------------------------------------------------
    builder = TopologyBuilder(rows, cols)
    if topology_type == "custom":
        if not custom_edges:
            raise ValueError("Must pass custom_edges when topology_type='custom'.")
        builder.add_connections_from_list(custom_edges)
    elif topology_type == "chain_of_chains":
        builder.build_chain_of_chains()
    elif topology_type == "funnel":
        builder.build_funnel()
    elif topology_type == "highway":
        builder.build_highway()
    elif topology_type == "binary_tree":
        builder.build_binary_tree()
    elif topology_type == "hourglass":
        builder.build_hourglass()
    elif topology_type == "grid":
        builder.build_standard_grid()
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    # Fault injection
    if is_faulty and fault_target:
        print("\U0001f6a8 Injecting fault...")
        if isinstance(fault_target, list):
            for src, tgt in fault_target:
                builder.remove_edge(src, tgt)
        else:
            builder.remove_edge(fault_target[0], fault_target[1])

    connection_config = builder.export_config()
    print(f"\U0001f517 Topology built: {len(connection_config)} edges")

    synapses_list = []
    for src_coord, tgt_coord, weight, prob in connection_config:
        S = Synapses(neurons[src_coord], neurons[tgt_coord], on_pre=f'v_post += {weight}')
        S.connect(p=prob)
        synapses_list.append(S)

    print(f"\u23f3 Running simulation for {duration_sec}s...")
    net = Network(*neurons.values(), *monitors.values(), *synapses_list)
    net.run(duration_sec * second)

    print("\U0001f504 Extracting data...")
    smooth_width = smooth_width_ms * ms
    data_dict = {'time_sec': monitors[(0, 0)].t / second}

    for r in range(rows):
        for c in range(cols):
            rate = monitors[(r, c)].smooth_rate(window='gaussian', width=smooth_width) / Hz
            data_dict[f'brian2_sensor_{r}_{c}'] = rate

    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\u2705 Saved to: {output_path}")
    print(f"\U0001f4ca Shape: {df.shape} | Columns: {list(df.columns[:4])} ...")
    return df