import numpy as np
from .data_structures import Waypoint, Trajectory
from .deconfliction_service import DeconflictionServer
from .visualisation import create_4d_plot  # Make sure this matches your filename


def run_sequential_scenario(scenario_name: str, all_drone_trajectories: list[Trajectory], safety_buffer: float = 10.0, visualize: bool = False):
    """
    Run a scenario where each drone is submitted sequentially for approval.
    Each drone is checked against all previously approved drones.
    """
    print(f"\n{'='*60}")
    print(f"🚁 SEQUENTIAL DRONE APPROVAL: {scenario_name}")
    print(f"{'='*60}")

    # Initialize server
    server = DeconflictionServer()
    server.clear_all_trajectories()

    print(f"📝 Submitting {len(all_drone_trajectories)} drones for sequential approval...")
    print(f"🛡️ Safety buffer: {safety_buffer}m\n")

    # Submit each drone sequentially
    for i, trajectory in enumerate(all_drone_trajectories, 1):
        print(f"--- DRONE {i}: {trajectory.drone_id} ---")

        # Submit for approval
        result = server.submit_mission_for_approval(trajectory, safety_buffer)

        # Report results
        if result.approved:
            print(f"✅ APPROVED - No conflicts detected")
            print(f"   Total approved drones now: {len(server.approved_trajectories)}")
        else:
            print(f"❌ REJECTED - {len(result.conflicts)} conflict(s) detected:")

            # Group conflicts by conflicting drone
            conflicts_by_drone = {}
            for conflict in result.conflicts:
                other_drone = [d for d in conflict.conflicting_drone_ids if d != trajectory.drone_id][0]
                if other_drone not in conflicts_by_drone:
                    conflicts_by_drone[other_drone] = []
                conflicts_by_drone[other_drone].append(conflict)

            for conflicting_drone, conflicts in conflicts_by_drone.items():
                print(f"   🚫 Conflicts with {conflicting_drone}:")
                for conflict in conflicts:
                    print(f"      • Time: {conflict.time_of_conflict:.2f}s")
                    print(f"      • Location: {np.round(conflict.location_of_conflict, 2)}")
                    print(f"      • Min separation: {conflict.minimum_separation:.2f}m")

            print(f"   Conflicting with drones: {result.conflicting_drones}")

        print()

    # Print final summary
    summary = server.get_approval_summary()
    print(f"🎯 FINAL SUMMARY:")
    print(f"   📊 Total submitted: {summary['total_submitted']}")
    print(f"   ✅ Approved: {summary['approved']}")
    print(f"   ❌ Rejected: {summary['rejected']}")
    print(f"   ⚠️ Total conflicts found: {summary['total_conflicts']}")

    # List approved and rejected drones
    if server.approved_trajectories:
        approved_list = ", ".join(server.approved_trajectories.keys())
        print(f"   ✅ Approved drones: {approved_list}")

    if server.rejected_trajectories:
        rejected_list = ", ".join(server.rejected_trajectories.keys())
        print(f"   ❌ Rejected drones: {rejected_list}")

    print()

    # Generate visualization if requested
    if visualize:
        all_trajectories = server.get_all_trajectories()
        all_conflicts = server.get_all_conflicts()
        create_4d_plot(all_trajectories, all_conflicts, scenario_name)

    return server


def main():
    # --- SCENARIO 1: GUARANTEED COLLISION ---
    print("🎯 Testing sequential approval with guaranteed collisions...")

    scenario1_drones = [
        Trajectory(
            drone_id="drone_1_first",
            waypoints=[
                Waypoint(position=np.array([0, 0, 10]), timestamp=0),
                Waypoint(position=np.array([100, 0, 10]), timestamp=10)
            ]
        ),
        Trajectory(
            drone_id="drone_2_crossing",
            waypoints=[
                Waypoint(position=np.array([50, -25, 10]), timestamp=0),
                Waypoint(position=np.array([50, 25, 10]), timestamp=10)
            ]
        ),
        Trajectory(
            drone_id="drone_3_parallel",
            waypoints=[
                Waypoint(position=np.array([0, 50, 10]), timestamp=0),
                Waypoint(position=np.array([100, 50, 10]), timestamp=10)
            ]
        )
    ]

    run_sequential_scenario("Sequential Collisions", scenario1_drones, visualize=True)

    # --- SCENARIO 2: MIXED SAFE AND UNSAFE ---
    print("🎯 Testing mixed safe and unsafe trajectories...")

    scenario2_drones = [
        Trajectory(
            drone_id="drone_A_safe",
            waypoints=[
                Waypoint(position=np.array([0, 0, 10]), timestamp=0),
                Waypoint(position=np.array([50, 0, 10]), timestamp=5)
            ]
        ),
        Trajectory(
            drone_id="drone_B_safe",
            waypoints=[
                Waypoint(position=np.array([0, 20, 10]), timestamp=0),  # 20m Y separation - SAFE
                Waypoint(position=np.array([50, 20, 10]), timestamp=5)
            ]
        ),
        Trajectory(
            drone_id="drone_C_unsafe",
            waypoints=[
                Waypoint(position=np.array([0, 5, 10]), timestamp=0),   # 5m from drone_A - UNSAFE
                Waypoint(position=np.array([50, 5, 10]), timestamp=5)
            ]
        ),
        Trajectory(
            drone_id="drone_D_multi_conflict",
            waypoints=[
                Waypoint(position=np.array([25, 10, 10]), timestamp=0), # Conflicts with A and B
                Waypoint(position=np.array([25, 10, 10]), timestamp=10) # Stationary
            ]
        )
    ]

    run_sequential_scenario("Mixed Safe Unsafe", scenario2_drones, visualize=True)


if __name__ == "__main__":
    main()
