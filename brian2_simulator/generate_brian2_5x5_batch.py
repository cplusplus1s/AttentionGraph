"""
generate_brian2_5x5_batch.py
────────────────────────────
Generates healthy baseline runs + one faulty run for each topology.
Run this script to produce a full dataset for all 5 topologies.

Directory layout produced:
  data/raw/brian2/
  ├── chain_of_chains/
  │   ├── healthy_1/brian2_data.csv  ...  healthy_30/brian2_data.csv
  │   └── unhealthy_1/brian2_data.csv
  ├── funnel/
  │   └── ...
  ├── highway/
  │   └── ...
  ├── binary_tree/
  │   └── ...
  └── hourglass/
      └── ...
"""

import os
from generate_brian2_sandbox import generate_simulation

# ==============================================================================
# Topology registry
# Each entry:
#   topology_type : str         — matches builder method name
#   fault_target  : tuple       — edge(s) to cut for the unhealthy run
#   description   : str         — why this fault is interesting
# ==============================================================================
TOPOLOGIES = [
    {
        "topology_type": "chain_of_chains",
        "fault_target": ((2, 1), (2, 2)),
        # Cuts row-2 chain in the middle. Nodes (2,2),(2,3),(2,4) go silent.
        # Attention from (2,1) to {(2,2),(2,3),(2,4)} should drop to baseline.
        "description": "Cut middle of row-2 chain — silences right half of row 2 only.",
    },
    {
        "topology_type": "funnel",
        "fault_target": ((2, 2), (2, 3)),
        # Hub→output edge cut. Only (2,3) and (2,4) lose drive.
        # All other output arms unaffected — localised, precise fault.
        "description": "Cut one hub output arm — (2,3)&(2,4) lose drive, others intact.",
    },
    # {
    #     "topology_type": "highway",
    #     "fault_target": ((2, 2), (2, 3)),
    #     # Cuts highway at mid-point. Everything downstream of (2,3) loses all
    #     # highway-routed signal. On-ramp from rows 3,4 also lost → dramatic change.
    #     "description": "Cut highway mid-point — entire right half loses highway signal.",
    # },
    {
        "topology_type": "binary_tree",
        "fault_target": ((2, 0), (3, 1)),
        # Prunes entire lower subtree: (3,1),(3,2),(3,3),(3,4),(4,2),(4,3),(4,4)
        # go silent. Upper subtree unaffected. Sharp half-tree disappearance.
        "description": "Prune lower subtree from root — 7 nodes go silent.",
    },
    {
        "topology_type": "hourglass",
        "fault_target": ((2, 2), (2, 3)),
        # THE most dramatic fault: single bridge edge cut silences ALL 10 right-half
        # nodes simultaneously. Healthy attention: strong hub→right. Fault: hub→right
        # attention collapses to zero. Unmissable in attention map.
        "description": "Cut bottleneck bridge — ALL right-half nodes (10/25) go silent.",
    },
]

DURATION = 30       # seconds — use 60s for better temporal statistics
N_HEALTHY = 10      # healthy baseline runs per topology


def generate_topology_batch(topo_cfg, n_healthy=N_HEALTHY, duration=DURATION):
    topo = topo_cfg["topology_type"]
    fault = topo_cfg["fault_target"]
    base_dir = f"./data/raw/brian2/{topo}"

    print(f"\n{'='*60}")
    print(f"📦 Topology: {topo.upper()}")
    print(f"   Fault: {fault}")
    print(f"   Note: {topo_cfg['description']}")
    print(f"{'='*60}")

    # Healthy runs
    for i in range(1, n_healthy + 1):
        out = f"{base_dir}/healthy_{i}/brian2_data.csv"
        generate_simulation(
            output_path=out,
            duration_sec=duration,
            topology_type=topo,
            is_faulty=False,
        )

    # Faulty run
    out_fault = f"{base_dir}/unhealthy_1/brian2_data.csv"
    generate_simulation(
        output_path=out_fault,
        duration_sec=duration,
        topology_type=topo,
        is_faulty=True,
        fault_target=fault,
    )

    print(f"✅ {topo} complete: {n_healthy} healthy + 1 faulty runs saved to {base_dir}/")


def generate_all():
    print("🚀 Starting full Brian2 batch generation")
    print(f"   Topologies : {len(TOPOLOGIES)}")
    print(f"   Healthy runs/topology: {N_HEALTHY}")
    print(f"   Duration/run: {DURATION}s")
    print(f"   Total runs: {len(TOPOLOGIES) * (N_HEALTHY + 1)}")

    for topo_cfg in TOPOLOGIES:
        generate_topology_batch(topo_cfg)

    print("\n🎉 All batch data generated!")
    print("\nRecommended evaluation order (easiest → hardest to reconstruct):")
    print("  1. hourglass       — single bridge, all-or-nothing signal")
    print("  2. chain_of_chains — independent chains, zero cross-row attention expected")
    print("  3. binary_tree     — clean single-parent structure")
    print("  4. funnel          — high fan-in hub, easy to spot in attention")
    print("  5. highway         — mixed fan-in/fan-out, most complex")


if __name__ == "__main__":
    # To run a single topology for quick testing:
    # generate_topology_batch(TOPOLOGIES[0], n_healthy=3, duration=10)

    generate_all()
