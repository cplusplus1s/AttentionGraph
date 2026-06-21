"""
generate_brian2_highway_fault.py
─────────────────────────────────
Standalone fault-data generator for the HIGHWAY topology.

Use this script to generate N unhealthy runs with a configurable fault edge,
without re-running the full multi-topology batch.

Workflow:
    1. Set FAULT_TARGET, N_UNHEALTHY, DURATION below
    2. Run: python generate_brian2_highway_fault.py
    3. Output: data/raw/brian2/highway/unhealthy_1/, unhealthy_2/, ...

Notes:
- Each unhealthy run gets a unique random driver-input waveform (Brian2 noise +
  TimedArray randomness vary per call to generate_simulation()), so the 10 runs
  are statistically independent samples of the fault condition.
- The healthy runs in data/raw/brian2/highway/healthy_* are NOT touched. Use
  generate_brian2_5x5_batch.py to (re)generate those.

CAUTION on directory cleanup:
- This script will OVERWRITE existing unhealthy_N folders if you re-run it.
- If you reduce N_UNHEALTHY (say from 10 to 5), the older unhealthy_6..10
  folders will remain on disk. Set CLEAN_OLD_UNHEALTHY = True below to wipe
  them before generating.
"""

import os
import shutil
from generate_brian2_sandbox import generate_simulation


# ==============================================================================
# CONFIGURATION — edit these to control the fault experiment
# ==============================================================================

# The edge to cut. Format: ((src_row, src_col), (tgt_row, tgt_col))
# Highway topology candidate faults (uncomment one):

FAULT_TARGET = ((2, 2), (2, 3))   # trunk mid-cut (default)

# Other interesting cuts you can try:
# FAULT_TARGET = ((2, 0), (2, 1))   # cut at trunk start — kills (2,1) on-ramp aggregation
# FAULT_TARGET = ((2, 3), (2, 4))   # cut just before highway end — isolates (2,4) and off-ramps
# FAULT_TARGET = ((0, 0), (2, 1))   # cut an on-ramp — (2,1) loses its lateral input
# FAULT_TARGET = ((3, 0), (2, 3))   # cut another on-ramp
# FAULT_TARGET = ((2, 4), (0, 4))   # cut a single off-ramp — only (0,4) loses drive

# Multiple simultaneous cuts (pass a list of tuples):
# FAULT_TARGET = [((2, 2), (2, 3)), ((2, 3), (2, 4))]   # double cut, isolates (2,3) entirely

# Number of unhealthy runs to generate
N_UNHEALTHY = 10

# Duration of each simulation (seconds)
DURATION = 30

# Wipe stale unhealthy_N folders before generating? (recommended when reducing N_UNHEALTHY)
CLEAN_OLD_UNHEALTHY = True

# Topology — fixed to highway for this script
TOPOLOGY = "highway"
BASE_DIR = f"./data/raw/brian2/{TOPOLOGY}"


# ==============================================================================
# Helpers
# ==============================================================================

def _format_fault(fault) -> str:
    """Pretty-print the fault target for logging."""
    if isinstance(fault, list):
        return " + ".join(f"{src}→{tgt}" for src, tgt in fault)
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
    print(f"🚨 Highway Fault Generator")
    print(f"   Fault target : {_format_fault(FAULT_TARGET)}")
    print(f"   Unhealthy runs: {N_UNHEALTHY}")
    print(f"   Duration/run : {DURATION}s")
    print(f"   Output dir   : {BASE_DIR}/unhealthy_1 .. unhealthy_{N_UNHEALTHY}")
    print(f"{'='*60}")

    if CLEAN_OLD_UNHEALTHY:
        _clean_unhealthy_dirs(BASE_DIR)

    for i in range(1, N_UNHEALTHY + 1):
        out_path = f"{BASE_DIR}/unhealthy_{i}/brian2_data.csv"
        print(f"\n[{i}/{N_UNHEALTHY}] Generating unhealthy run...")
        generate_simulation(
            output_path=out_path,
            duration_sec=DURATION,
            topology_type=TOPOLOGY,
            is_faulty=True,
            fault_target=FAULT_TARGET,
        )

    print(f"\n🎉 Done — {N_UNHEALTHY} unhealthy runs generated for fault "
          f"{_format_fault(FAULT_TARGET)}")
    print(f"   Next steps:")
    print(f"     1. python main_pipeline.py  (process raw → CSV)")
    print(f"     2. Update run_brian2.ps1 Phase 3 to loop unhealthy_1..unhealthy_{N_UNHEALTHY}")
    print(f"     3. Run the pipeline → train, infer, visualize")


if __name__ == "__main__":
    main()
