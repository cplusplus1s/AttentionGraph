"""
generate_brian2_chains_funnel_binarytree_fault.py
──────────────────────────────────────────────────
Standalone fault-data generator for three topologies:
  • chain_of_chains  — cuts synapse (2,1) → (2,2)
  • funnel           — cuts synapse (2,2) → (2,3)
  • binary_tree      — cuts synapse (2,0) → (3,1)

Workflow:
    1. Adjust N_UNHEALTHY and DURATION below if needed.
    2. Run: python generate_brian2_chains_funnel_binarytree_fault.py
    3. Output:
         data/raw/brian2/chain_of_chains/unhealthy_1/ .. unhealthy_N/
         data/raw/brian2/funnel/unhealthy_1/          .. unhealthy_N/
         data/raw/brian2/binary_tree/unhealthy_1/     .. unhealthy_N/

Notes:
- Each unhealthy run gets a unique random driver-input waveform, so all runs
  are statistically independent samples of their fault condition.
- Healthy runs in the corresponding healthy_* folders are NOT touched. Use
  generate_brian2_5x5_batch.py to (re)generate those.

CAUTION on directory cleanup:
- Existing unhealthy_N folders will be OVERWRITTEN if you re-run this script.
- Set CLEAN_OLD_UNHEALTHY = True (default) to wipe stale folders first,
  which is recommended whenever you reduce N_UNHEALTHY.
"""

import os
import shutil
from generate_brian2_sandbox import generate_simulation


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Number of unhealthy runs to generate per topology
N_UNHEALTHY = 10

# Duration of each simulation (seconds)
DURATION = 30

# Wipe stale unhealthy_N folders before generating?
CLEAN_OLD_UNHEALTHY = True

# Fault definitions — (topology_name, fault_edge)
FAULT_CONFIGS = [
    ("chain_of_chains", ((2, 1), (2, 2))),   # severs row-2 chain at mid-point
    ("funnel",          ((2, 2), (2, 3))),   # cuts hub output, isolates col≥3
    ("binary_tree",     ((2, 0), (3, 1))),   # removes lower subtree root branch
]

BASE_DIR_TEMPLATE = "./data/raw/brian2/{topology}"


# ==============================================================================
# Helpers
# ==============================================================================

def _format_fault(fault) -> str:
    """Pretty-print a single fault edge."""
    return f"{fault[0]}→{fault[1]}"


def _clean_unhealthy_dirs(base_dir: str) -> None:
    """Remove all existing unhealthy_N subdirectories under base_dir."""
    if not os.path.isdir(base_dir):
        return
    removed = 0
    for name in os.listdir(base_dir):
        if name.startswith("unhealthy_"):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
                removed += 1
    if removed:
        print(f"🧹 Cleaned {removed} stale unhealthy_N folder(s) from {base_dir}/")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    print(f"\n{'='*60}")
    print(f"🚨 Multi-Topology Fault Generator")
    print(f"   Topologies   : {', '.join(t for t, _ in FAULT_CONFIGS)}")
    print(f"   Unhealthy runs: {N_UNHEALTHY} per topology")
    print(f"   Duration/run : {DURATION}s")
    print(f"{'='*60}")

    for topology, fault_target in FAULT_CONFIGS:
        base_dir = BASE_DIR_TEMPLATE.format(topology=topology)

        print(f"\n{'─'*60}")
        print(f"📌 Topology     : {topology.upper()}")
        print(f"   Fault target : {_format_fault(fault_target)}")
        print(f"   Output dir   : {base_dir}/unhealthy_1 .. unhealthy_{N_UNHEALTHY}")
        print(f"{'─'*60}")

        if CLEAN_OLD_UNHEALTHY:
            _clean_unhealthy_dirs(base_dir)

        for i in range(1, N_UNHEALTHY + 1):
            out_path = f"{base_dir}/unhealthy_{i}/brian2_data.csv"
            print(f"\n  [{i}/{N_UNHEALTHY}] Generating unhealthy run — "
                  f"{topology} | fault {_format_fault(fault_target)} ...")
            generate_simulation(
                output_path=out_path,
                duration_sec=DURATION,
                topology_type=topology,
                is_faulty=True,
                fault_target=fault_target,
            )

        print(f"\n  ✅ {N_UNHEALTHY} unhealthy runs done for [{topology.upper()}]")

    print(f"\n{'='*60}")
    print(f"🎉 All topologies complete — {N_UNHEALTHY * len(FAULT_CONFIGS)} total runs generated")
    print(f"   Next steps:")
    print(f"     1. python main_pipeline.py  (process raw → CSV)")
    print(f"     2. Update run_brian2.ps1 Phase 3 to loop unhealthy_1..unhealthy_{N_UNHEALTHY}")
    print(f"     3. Run the pipeline → train, infer, visualize")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
