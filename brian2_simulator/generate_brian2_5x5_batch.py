import os
from generate_brian2_sandbox import generate_simulation

def generate_all():
    DURATION = 30

    # Create custom connections
    my_custom_edges = [
        ((0,0), (0,1)), ((0,2), (0,3)), ((0,3), (0,4)),
        ((0,1), (1,1)), ((1,1), (0,2)), ((1,2), (0,3)), ((0,4), (1,4)),
        ((1,0), (1,1)), ((1,1), (1,2)), ((1,2), (1,3)),
        ((1,1), (2,2)), ((2,1), (1,2)), ((1,3), (2,4)), ((2,3), (1,4)),
        ((2,0), (2,1)), ((2,2), (2,3)),
        ((2,2), (3,3)),
        ((3,0), (3,1)), ((3,1), (3,2)), ((3,2), (3,3)), ((3,3), (3,4)),
        ((4,0), (4,1)), ((4,1), (4,2)), ((4,2), (4,3)), ((4,3), (4,4)),
    ]

    CURRENT_TOPOLOGY = "custom"
    EDGES_TO_USE = my_custom_edges
    TARGET_TO_CUT = ((1,1), (2,2))

    print(f"🚀 Start generating Brian2 batch data (topology: {CURRENT_TOPOLOGY}, 30 healty baseline, 1 unhealthy)...")

    # 1. 30 healthy baseline
    for i in range(1, 31):
        out_path = f"./data/raw/brian2/healthy_{i}/brian2_data.csv"
        generate_simulation(
            output_path=out_path,
            duration_sec=DURATION,
            topology_type=CURRENT_TOPOLOGY,
            is_faulty=False,
            custom_edges=EDGES_TO_USE
        )

    # 2. 1 unhealthy
    out_path_fault = f"./data/raw/brian2/unhealthy_1/brian2_data.csv"
    generate_simulation(
        output_path=out_path_fault,
        duration_sec=DURATION,
        topology_type=CURRENT_TOPOLOGY,
        is_faulty=True,
        custom_edges=EDGES_TO_USE,
        fault_target=TARGET_TO_CUT   # cut here
    )

    print("\n✅ All batch data has been generated!")

if __name__ == "__main__":
    generate_all()