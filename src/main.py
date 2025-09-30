import numpy as np
from .data_structures import Waypoint, Trajectory
from .deconfliction_service import DeconflictionServer

def run_scenario(scenario_name: str, primary_mission: Trajectory, simulated_missions: list):
    """Helper function to run and print a single scenario."""
    print(f"--- Running Scenario: {scenario_name} ---")
    
    # 1. Initialize Server
    server = DeconflictionServer()
    
    # 2. Populate server with simulated drone trajectories
    for traj in simulated_missions:
        server.add_simulated_trajectory(traj)
    
    # 3. Define safety buffer and validate the primary mission
    safety_buffer_meters = 10.0
    conflicts = server.validate_mission(primary_mission, safety_buffer_meters)
    
    # 4. Report results
    if not conflicts:
        print("Mission is clear for execution. No conflicts detected.\n")
    else:
        print(f"Conflict Alert! {len(conflicts)} potential conflict(s) detected:")
        for conflict in conflicts:
            print(f" - With Drone: {conflict.conflicting_drone_ids}")
            print(f" - Time (approx): {conflict.time_of_conflict:.2f} seconds")
            print(f" - Location (approx): {np.round(conflict.location_of_conflict, 2)}")
            print(f" - Minimum Separation: {conflict.minimum_separation:.2f}m (Buffer: {safety_buffer_meters}m)\n")

def main():
    # --- SCENARIO 1: GUARANTEED COLLISION ---
    primary_mission_collision = Trajectory(
        drone_id="primary_drone",
        waypoints=[
            Waypoint(position=np.array([0, 0, 10]), timestamp=0),
            Waypoint(position=np.array([100, 0, 10]), timestamp=10)
        ]  # ← FIXED: Added closing bracket
    )  # ← FIXED: Added closing parenthesis

    simulated_missions_collision = [
        Trajectory(
            drone_id="sim_drone_1_colliding",
            waypoints=[
                Waypoint(position=np.array([50, 0, 10]), timestamp=0),
                Waypoint(position=np.array([50, 0, 10]), timestamp=10)
            ]  # ← FIXED: Added closing bracket
        ),  # ← FIXED: Added closing parenthesis
        Trajectory(
            drone_id="sim_drone_2_colliding",
            waypoints=[
                Waypoint(position=np.array([0, 50, 10]), timestamp=0),
                Waypoint(position=np.array([100, 50, 10]), timestamp=10)
            ]  # ← FIXED: Added closing bracket
        )   # ← FIXED: Added closing parenthesis
    ]  # ← FIXED: Added closing bracket for the list

    run_scenario("Guaranteed Collision", primary_mission_collision, simulated_missions_collision)

    # --- SCENARIO 2: NEAR MISS (SAFE) ---
    primary_mission_safe = Trajectory(
        drone_id="primary_drone",
        waypoints=[
            Waypoint(position=np.array([0, 0, 10]), timestamp=0),
            Waypoint(position=np.array([100, 0, 10]), timestamp=10)
        ]  # ← FIXED: Added closing bracket
    )  # ← FIXED: Added closing parenthesis

    simulated_missions_safe = [
        Trajectory(
            drone_id="sim_drone_2_safe",
            waypoints=[
                Waypoint(position=np.array([0, 20, 10]), timestamp=0),
                Waypoint(position=np.array([100, 20, 10]), timestamp=10)
            ]  # ← FIXED: Added closing bracket
        ),  # ← FIXED: Added closing parenthesis
        Trajectory(
            drone_id="sim_drone_3_near_miss",
            waypoints=[
                Waypoint(position=np.array([0, 0, 21]), timestamp=0), # Altitude is 21m (11m separation)
                Waypoint(position=np.array([100, 0, 21]), timestamp=10)
            ]  # ← FIXED: Added closing bracket
        )   # ← FIXED: Added closing parenthesis  
    ]  # ← FIXED: Added closing bracket for the list

    run_scenario("Near-Miss (Safe)", primary_mission_safe, simulated_missions_safe)

if __name__ == "__main__":
    main()
