import numpy as np
from .data_structures import Waypoint, Trajectory
from .deconfliction_service import DeconflictionServer
from .visualization import create_4d_plot  # Make sure this matches your filename


def run_sequential_scenario(scenario_name: str, all_drone_trajectories: list[Trajectory], safety_buffer: float = 10.0, visualize: bool = False):
    """
    Run a scenario where each drone is submitted sequentially for approval.
    Each drone is checked against all previously approved drones.
    Enhanced with spawn conflict detection and visual rejection highlighting.
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
            # Check if this is a spawn conflict
            if result.spawn_conflict:
                print(f"🚫 REJECTED - SPAWN CONFLICT detected!")
                print(f"   Drone spawns too close to existing approved drone(s)")
            else:
                print(f"❌ REJECTED - {len(result.conflicts)} trajectory conflict(s) detected:")

            # Group conflicts by conflicting drone
            conflicts_by_drone = {}
            for conflict in result.conflicts:
                other_drone = [d for d in conflict.conflicting_drone_ids if d != trajectory.drone_id][0]
                if other_drone not in conflicts_by_drone:
                    conflicts_by_drone[other_drone] = []
                conflicts_by_drone[other_drone].append(conflict)

            for conflicting_drone, conflicts in conflicts_by_drone.items():
                print(f"   {'🚫' if result.spawn_conflict else '⚠️'} Conflicts with {conflicting_drone}:")
                for conflict in conflicts:
                    if result.spawn_conflict:
                        print(f"      • SPAWN CONFLICT at t={conflict.time_of_conflict:.2f}s")
                        print(f"      • Spawn location: {np.round(conflict.location_of_conflict, 2)}")
                        print(f"      • Distance at spawn: {conflict.minimum_separation:.2f}m (< {safety_buffer}m buffer)")
                    else:
                        print(f"      • Path conflict at t={conflict.time_of_conflict:.2f}s")
                        print(f"      • Conflict location: {np.round(conflict.location_of_conflict, 2)}")
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
    if 'spawn_conflicts' in summary and summary['spawn_conflicts'] > 0:
        print(f"   🚫 Spawn conflicts: {summary['spawn_conflicts']}")

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
        rejected_drone_ids = server.get_rejected_drone_ids()  # NEW: Pass rejected IDs

        create_4d_plot(all_trajectories, all_conflicts, scenario_name, rejected_drone_ids)

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

    # --- SCENARIO 2: SPAWN CONFLICTS ---
    print("🎯 Testing spawn conflicts...")

    scenario2_drones = [
        Trajectory(
            drone_id="drone_A_first",
            waypoints=[
                Waypoint(position=np.array([0, 0, 10]), timestamp=0),
                Waypoint(position=np.array([50, 0, 10]), timestamp=5)
            ]
        ),
        Trajectory(
            drone_id="drone_B_spawn_conflict",
            waypoints=[
                Waypoint(position=np.array([15, 15, 10]), timestamp=0),   # Only 5m from drone_A spawn - SPAWN CONFLICT
                Waypoint(position=np.array([45, 10, 10]), timestamp=5)
            ]
        ),
        Trajectory(
            drone_id="drone_C_safe_spawn",
            waypoints=[
                Waypoint(position=np.array([0, 20, 10]), timestamp=0),  # 20m separation - SAFE
                Waypoint(position=np.array([50, 20, 10]), timestamp=5)
            ]
        ),
        Trajectory(
            drone_id="drone_D_path_conflict",
            waypoints=[
                Waypoint(position=np.array([100, 0, 10]), timestamp=0), # Safe spawn, but crosses path
                Waypoint(position=np.array([50, 30, 0]), timestamp=5)    # Path conflict with drone_A
            ]
        )
    ]

    run_sequential_scenario("Spawn and Path Conflicts", scenario2_drones, visualize=True)


if __name__ == "__main__":
    main()
