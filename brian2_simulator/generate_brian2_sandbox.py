from brian2 import *
import pandas as pd
import numpy as np
import os

# Topology Builder
class TopologyBuilder:
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.edges = {}

    def add_edge(self, src: tuple, tgt: tuple, weight=0.25, prob=0.6):
        self.edges[(src, tgt)] = (weight, prob)

    def remove_edge(self, src: tuple, tgt: tuple):
        if (src, tgt) in self.edges:
            del self.edges[(src, tgt)]
            print(f"   ✂️ The connection has been cut: {src} -> {tgt}")
        else:
            print(f"   ⚠️ Attempting to cut a non-existent connection: {src} -> {tgt}")

    def add_connections_from_list(self, edge_list, default_weight=0.3, default_prob=0.7):
        """Supported edge_list parameter formats: [(src, tgt)] or [(src, tgt, weight, prob)]"""
        for edge in edge_list:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1], default_weight, default_prob)
            elif len(edge) == 4:
                self.add_edge(edge[0], edge[1], edge[2], edge[3])
            else:
                print(f"   ⚠️ Ignore incorrectly formatted connection: {edge}")

    def build_standard_grid(self, weight=0.25, prob=0.6):
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols: self.add_edge((r, c), (r, c+1), weight, prob)
                if r + 1 < self.rows: self.add_edge((r, c), (r+1, c), weight, prob)

    def build_hourglass(self, weight=0.25, prob=0.6, bridge_weight=0.4, bridge_prob=0.8):
        for r in range(self.rows):
            self.add_edge((r, 0), (r, 1), weight, prob)
            if r != 1: self.add_edge((r, 1), (1, 1), weight, prob)

        self.add_edge((1, 1), (1, 2), bridge_weight, bridge_prob)

        for r in range(self.rows):
            if r != 1: self.add_edge((1, 2), (r, 2), weight, prob)
            for c in range(2, self.cols):
                if c + 1 < self.cols: self.add_edge((r, c), (r, c+1), weight, prob)
                if r + 1 < self.rows: self.add_edge((r, c), (r+1, c), weight, prob)

    def export_config(self):
        return [(src, tgt, w, p) for (src, tgt), (w, p) in self.edges.items()]


# ==============================================================================
# 🚀 Core simulation main function
# ==============================================================================
def generate_simulation(
    output_path="./data/raw/brian2/brian2_data.csv",
    duration_sec=60,
    topology_type="grid",
    is_faulty=False,
    custom_edges=None,
    fault_target=None
):
    print(f"\n🚀 Initialize 5x5 Brian 2 network | Topology: [{topology_type.upper()}] | State: [{'Fault 🚨' if is_faulty else 'Healthy ✅'}]")
    start_scope()
    defaultclock.dt = 1 * ms

    eqs = '''
    dv/dt = (I - v) / tau + sigma * xi * tau**-0.5 : 1 (unless refractory)
    I : 1
    tau : second
    sigma : 1
    '''

    rows, cols = 5, 5
    neurons = {}
    monitors = {}

    for r in range(rows):
        for c in range(cols):
            G = NeuronGroup(50, eqs, threshold='v>1', reset='v=0', refractory=2*ms, method='euler')
            G.v = 'rand()'
            G.tau = 10*ms
            G.sigma = 0.2
            if c == 0:
                G.I = 1.0 + 0.2 * rand()
            else:
                G.I = 0.5
            neurons[(r, c)] = G
            monitors[(r, c)] = PopulationRateMonitor(G)

    builder = TopologyBuilder(rows, cols)

    if topology_type == "custom":
        if not custom_edges:
            raise ValueError("When using the 'custom' topology, you must pass in the custom_edges list!")
        builder.add_connections_from_list(custom_edges)

    elif topology_type == "grid":
        builder.build_standard_grid()
    elif topology_type == "hourglass":
        builder.build_hourglass()
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    # Fault inject
    if is_faulty and fault_target:
        print("🚨 Injecting a fault...")
        builder.remove_edge(fault_target[0], fault_target[1])

    # --------------------------------------------------------------------------
    connection_config = builder.export_config()
    print(f"🔗 The topology has been constructed, generating a total of {len(connection_config)} valid edges!")

    synapses_list = []
    for src_coord, tgt_coord, weight, prob in connection_config:
        S = Synapses(neurons[src_coord], neurons[tgt_coord], on_pre=f'v_post += {weight}')
        S.connect(p=prob)
        synapses_list.append(S)

    print(f"⏳ The simulation is running for {duration_sec} seconds...")
    net = Network(*neurons.values(), *monitors.values(), *synapses_list)
    net.run(duration_sec * second)

    print("🔄 Extracting and saving data (this might take a few seconds)...")
    data_dict = {'time_sec': monitors[(0,0)].t / second}
    for r in range(rows):
        for c in range(cols):
            rate = monitors[(r, c)].smooth_rate(window='gaussian', width=20*ms) / Hz
            data_dict[f'brian2_sensor_{r}_{c}'] = rate

    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Done! Data saved to: {output_path}")