from brian2 import *
import pandas as pd
import numpy as np
import os

def generate_grid_simulation(output_path="./data/raw/brian2/brian2_data.csv", duration_sec=60):
    print("🚀 Initializing 5x5 Brian 2 simulation network...")
    start_scope()
    defaultclock.dt = 1 * ms

     # Basic neuron
    eqs = '''
    dv/dt = (I - v) / tau + sigma * xi * tau**-0.5 : 1 (unless refractory)
    I : 1       # Input current
    tau : second # Time constant
    sigma : 1   # Noise intensity
    '''

    # --- 1. Create 5 x 5 neuron network ---
    rows, cols = 5, 5
    neurons = {}
    monitors = {}

    for r in range(rows):
        for c in range(cols):
            # Each node is a cluster containing 50 neurons.
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

    # --- 2. Generate connections ---
    connection_config = []

    # 2.1 Standard topology (each node links to the right and downwards).
    for r in range(rows):
        for c in range(cols):
            # to right
            if c + 1 < cols:
                connection_config.append(((r, c), (r, c+1), 0.25, 0.6))
            # to downwards
            if r + 1 < rows:
                connection_config.append(((r, c), (r+1, c), 0.25, 0.6))

    # 2.2 Add several long-lived connections across regions
    connection_config.extend([
        ((0, 0), (2, 2), 0.20, 0.3),
        ((2, 2), (4, 4), 0.20, 0.3),
        ((1, 3), (3, 1), 0.15, 0.2),
    ])

    # --- 3. 3. Create synapses ---
    synapses_list = []
    for source_coord, target_coord, weight, prob in connection_config:
        G_src = neurons[source_coord]
        G_tgt = neurons[target_coord]

        S = Synapses(G_src, G_tgt, on_pre=f'v_post += {weight}')
        S.connect(p=prob)
        synapses_list.append(S)

    print(f"🔗 A total of {len(connection_config)} directed edges were generated!")

    # --- 4. Build the network and run the simulation ---
    print(f"⏳ The simulation is running for {duration_sec} seconds...")
    net = Network(
        *neurons.values(),
        *monitors.values(),
        *synapses_list
    )
    net.run(duration_sec * second)

    # --- 5. Extract and save data ---
    print("🔄 Processing smoothed discharge rate data, this may take a few seconds...")
    window_width = 20 * ms
    data_dict = {}

    data_dict['time_sec'] = monitors[(0,0)].t / second

    for r in range(rows):
        for c in range(cols):
            rate = monitors[(r, c)].smooth_rate(window='gaussian', width=window_width) / Hz
            col_name = f'brian2_sensor_{r}_{c}'
            data_dict[col_name] = rate

    df = pd.DataFrame(data_dict)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Done! Data has been saved to: {output_path}")
    print(f"📊 Data Preview:\n{df.iloc[:3, :5]} ...")

if __name__ == "__main__":
    generate_grid_simulation(output_path="./data/raw/brian2/brian2_data.csv", duration_sec=20)