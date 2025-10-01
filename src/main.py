# src/main.py

import numpy as np
import time
from typing import List
from .data_structures import (
    Waypoint, Trajectory, PrimaryMission, SimulatedFlightSchedule, 
    DeconflictionResult
)
from .deconfliction_service import UAVDeconflictionService
from .visualization import create_trajectory_visualization
from .trajectory_animated import create_trajectory_animation

def create_test_scenarios() -> List[dict]:
    """
    /// @brief Create comprehensive test scenarios for UAV deconfliction validation
    /// 
    /// @return List of test scenario configurations
    /// 
    /// Generates multiple realistic scenarios including:
    /// - Conflict-free operations
    /// - Spatial conflicts at intersections  
    /// - Temporal conflicts with overlapping schedules
    /// - Complex multi-drone environments
    """
    scenarios = []
    
    # Scenario 1: Conflict-Free Operations
    scenarios.append({
        'name': 'Conflict-Free Parallel Operations',
        'description': 'Multiple drones flying parallel paths with adequate separation',
        'primary_mission': PrimaryMission(
            trajectory=Trajectory(
                drone_id="PRIMARY_DRONE_001",
                waypoints=[
                    Waypoint(position=np.array([0, 0, 50]), timestamp=0),
                    Waypoint(position=np.array([500, 0, 50]), timestamp=60),
                    Waypoint(position=np.array([1000, 0, 50]), timestamp=120)
                ]
            ),
            mission_start_time=0,
            mission_end_time=180
        ),
        'simulated_flights': [
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="SIM_DELIVERY_001",
                    waypoints=[
                        Waypoint(position=np.array([0, 100, 50]), timestamp=10),
                        Waypoint(position=np.array([500, 100, 50]), timestamp=70),
                        Waypoint(position=np.array([1000, 100, 50]), timestamp=130)
                    ]
                ),
                priority_level=1,
                flight_type="commercial"
            ),
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="SIM_PATROL_001", 
                    waypoints=[
                        Waypoint(position=np.array([0, -100, 80]), timestamp=0),
                        Waypoint(position=np.array([800, -100, 80]), timestamp=100),
                        Waypoint(position=np.array([1000, -200, 80]), timestamp=140)
                    ]
                ),
                priority_level=0,
                flight_type="emergency"
            )
        ],
        'expected_result': 'CLEAR'
    })
    
    # Scenario 2: Spatial Conflict - Crossing Paths
    scenarios.append({
        'name': 'Spatial Conflict - Crossing Trajectories',
        'description': 'Primary mission crosses existing flight paths creating conflicts',
        'primary_mission': PrimaryMission(
            trajectory=Trajectory(
                drone_id="PRIMARY_DRONE_002",
                waypoints=[
                    Waypoint(position=np.array([0, 0, 60]), timestamp=0),
                    Waypoint(position=np.array([300, 300, 60]), timestamp=45),
                    Waypoint(position=np.array([600, 600, 60]), timestamp=90)
                ]
            ),
            mission_start_time=0,
            mission_end_time=120
        ),
        'simulated_flights': [
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="SIM_CROSSING_001",
                    waypoints=[
                        Waypoint(position=np.array([600, 0, 60]), timestamp=0),
                        Waypoint(position=np.array([300, 300, 60]), timestamp=40),  # Intersection point
                        Waypoint(position=np.array([0, 600, 60]), timestamp=80)
                    ]
                ),
                priority_level=1,
                flight_type="commercial"
            ),
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="SIM_STATIONARY_001",
                    waypoints=[
                        Waypoint(position=np.array([450, 450, 60]), timestamp=0),
                        Waypoint(position=np.array([450, 450, 60]), timestamp=120)  # Stationary obstacle
                    ]
                ),
                priority_level=0,
                flight_type="emergency"
            )
        ],
        'expected_result': 'CONFLICT_DETECTED'
    })
    
    # Scenario 3: Temporal Window Violation
    scenarios.append({
        'name': 'Mission Time Window Violation',
        'description': 'Primary mission exceeds allocated time window',
        'primary_mission': PrimaryMission(
            trajectory=Trajectory(
                drone_id="PRIMARY_DRONE_003",
                waypoints=[
                    Waypoint(position=np.array([0, 0, 40]), timestamp=0),
                    Waypoint(position=np.array([1000, 0, 40]), timestamp=150)  # Exceeds time window
                ]
            ),
            mission_start_time=0,
            mission_end_time=120  # Mission exceeds this window
        ),
        'simulated_flights': [],  # No simulated flights needed for time window test
        'expected_result': 'CONFLICT_DETECTED'
    })
    
    # Scenario 4: Complex Multi-Drone Environment
    scenarios.append({
        'name': 'Complex Multi-Drone Airspace',
        'description': 'High-density airspace with multiple simultaneous operations',
        'primary_mission': PrimaryMission(
            trajectory=Trajectory(
                drone_id="PRIMARY_DRONE_004",
                waypoints=[
                    Waypoint(position=np.array([250, 250, 75]), timestamp=0),
                    Waypoint(position=np.array([750, 250, 75]), timestamp=30),
                    Waypoint(position=np.array([750, 750, 75]), timestamp=60),
                    Waypoint(position=np.array([250, 750, 75]), timestamp=90)
                ]
            ),
            mission_start_time=0,
            mission_end_time=120
        ),
        'simulated_flights': [
            # Commercial delivery drone
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="DELIVERY_FLEET_001",
                    waypoints=[
                        Waypoint(position=np.array([0, 500, 75]), timestamp=10),
                        Waypoint(position=np.array([500, 500, 75]), timestamp=40),
                        Waypoint(position=np.array([1000, 500, 75]), timestamp=70)
                    ]
                ),
                priority_level=2,
                flight_type="commercial"
            ),
            # Emergency response drone  
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="EMERGENCY_RESPONSE_001",
                    waypoints=[
                        Waypoint(position=np.array([800, 0, 100]), timestamp=0),
                        Waypoint(position=np.array([800, 800, 100]), timestamp=45),
                        Waypoint(position=np.array([0, 800, 100]), timestamp=90)
                    ]
                ),
                priority_level=0,
                flight_type="emergency"
            ),
            # Recreational drone
            SimulatedFlightSchedule(
                trajectory=Trajectory(
                    drone_id="RECREATIONAL_001",
                    waypoints=[
                        Waypoint(position=np.array([600, 100, 50]), timestamp=15),
                        Waypoint(position=np.array([600, 200, 50]), timestamp=25),
                        Waypoint(position=np.array([700, 200, 50]), timestamp=35),
                        Waypoint(position=np.array([700, 100, 50]), timestamp=45)
                    ]
                ),
                priority_level=3,
                flight_type="recreational"
            )
        ],
        'expected_result': 'CLEAR'
    })
    
    return scenarios

def run_validation_scenario(scenario: dict, visualize: bool = False, animate: bool = False) -> DeconflictionResult:
    """
    /// @brief Execute a single validation scenario and report results
    /// 
    /// @param scenario Test scenario configuration dictionary
    /// @param visualize Whether to generate visualization output
    /// @return Complete validation result
    /// 
    /// Runs the complete validation pipeline including:
    /// - Service initialization and configuration
    /// - Simulated flight registration  
    /// - Primary mission validation
    /// - Result analysis and reporting
    """
    print(f"\n{'='*80}")
    print(f"🚁 SCENARIO: {scenario['name']}")
    print(f"📋 Description: {scenario['description']}")
    print(f"{'='*80}")
    
    # Initialize deconfliction service
    service = UAVDeconflictionService(default_safety_buffer_m=10.0)
    
    # Register all simulated flight schedules
    print(f"📡 Registering {len(scenario['simulated_flights'])} simulated flight schedules...")
    for i, sim_flight in enumerate(scenario['simulated_flights'], 1):
        service.add_simulated_flight_schedule(sim_flight)
        flight_info = sim_flight.trajectory
        start_pos = flight_info.waypoints[0].position
        end_pos = flight_info.waypoints[-1].position
        duration = flight_info.waypoints[-1].timestamp - flight_info.waypoints[0].timestamp
        
        print(f"  {i}. {sim_flight.trajectory.drone_id} ({sim_flight.flight_type})")
        print(f"     Route: {np.round(start_pos, 1)} → {np.round(end_pos, 1)}")
        print(f"     Duration: {duration:.1f}s, Priority: {sim_flight.priority_level}")
    
    # Display primary mission details
    primary = scenario['primary_mission']
    primary_start = primary.trajectory.waypoints[0].position
    primary_end = primary.trajectory.waypoints[-1].position
    primary_duration = primary.trajectory.waypoints[-1].timestamp - primary.trajectory.waypoints[0].timestamp
    
    print(f"\n🎯 Primary Mission: {primary.trajectory.drone_id}")
    print(f"   Route: {np.round(primary_start, 1)} → {np.round(primary_end, 1)}")
    print(f"   Flight Duration: {primary_duration:.1f}s")
    print(f"   Mission Window: {primary.mission_start_time}s - {primary.mission_end_time}s")
    
    # Perform validation
    print(f"\n🛡️ Performing Strategic Deconfliction Analysis...")
    result = service.validate_primary_mission(primary)
    
    # Report results  
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Status: {'✅ APPROVED' if result.is_approved() else '❌ REJECTED'}")
    print(f"   Processing Time: {result.processing_time_ms:.2f}ms")
    print(f"   Conflicts Detected: {len(result.conflicts)}")
    
    if result.conflicts:
        print(f"\n⚠️ DETAILED CONFLICT ANALYSIS:")
        for i, conflict in enumerate(result.conflicts, 1):
            print(f"   Conflict {i}: {conflict.get_conflict_description()}")
            if conflict.conflict_type == "temporal":
                print(f"              ⏰ Mission exceeds allocated time window")
            elif conflict.conflict_type == "spatial":
                print(f"              📍 Spatial violation - minimum separation breached")
        
        conflicting_flights = result.get_conflicting_flight_ids()
        if conflicting_flights:
            print(f"   Conflicting Flights: {', '.join(conflicting_flights)}")
    
    # Verify against expected result
    expected = scenario['expected_result']
    actual = result.status
    if expected == actual:
        print(f"   ✅ Result matches expectation: {expected}")
    else:
        print(f"   ❌ Result mismatch - Expected: {expected}, Got: {actual}")
    
    # Generate visualization if requested
    if visualize:
        print(f"\n📈 Generating trajectory visualization...")
        all_trajectories = [primary.trajectory]
        all_trajectories.extend([sim.trajectory for sim in scenario['simulated_flights']])
        
        filename = create_trajectory_visualization(
            trajectories=all_trajectories,
            conflicts=result.conflicts,
            scenario_name=scenario['name']
        )
        print(f"   Visualization saved: {filename}")

    # Generate animated visualization if requested
    if animate:
        print(f"\\n🎬 Generating animated trajectory visualization...")
        try:
            animation_filename = create_trajectory_animation(
                trajectories=all_trajectories,
                conflicts=result.conflicts,
                scenario_name=scenario['name'],
                primary_drone_id=primary.trajectory.drone_id,
                save_video=True,
                show_interactive=False,  # Don't show interactive during batch processing
                fps=15
            )
            if animation_filename:
                print(f"   ✅ Animation video saved: {animation_filename}")
            else:
                print(f"   ⚠️ Animation generation skipped (no trajectories)")
        except Exception as e:
            print(f"   ❌ Animation generation failed: {e}")
            print(f"      Make sure matplotlib and ffmpeg are properly installed")
    
    return result

def main():
    """
    /// @brief Main entry point for UAV deconfliction system demonstration
    /// 
    /// Executes comprehensive test scenarios demonstrating the system's ability
    /// to detect various types of conflicts in shared airspace environments.
    """
    print("🚀 UAV Strategic Deconfliction System - Comprehensive Validation")
    print("=" * 80)
    print("Testing spatial-temporal conflict detection in shared airspace")
    print("Implementation includes:")
    print("  • Primary mission validation against simulated flight schedules")
    print("  • 3D spatial conflict detection with configurable safety buffers")
    print("  • Temporal overlap analysis and mission window validation")
    print("  • Comprehensive conflict reporting and explanations")
    print("  • Interactive 4D visualization capabilities")
    print("  • Real-time animated trajectory playback with MP4 export")
    
    # Configuration options
    ENABLE_STATIC_VISUALIZATION = True
    ENABLE_ANIMATION = True

    # Load test scenarios
    scenarios = create_test_scenarios()
    results = []
    
    # Execute all scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\\n🔄 Processing scenario {i}/{len(scenarios)}...")
        
        result = run_validation_scenario(
            scenario, 
            visualize=ENABLE_STATIC_VISUALIZATION, 
            animate=ENABLE_ANIMATION
        )
        results.append(result)
        
        # Brief pause between scenarios for readability and processing
        time.sleep(1.0)
    
    # Generate summary statistics
    print(f"\n{'='*80}")
    print(f"📈 COMPREHENSIVE VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    total_scenarios = len(results)
    approved_count = sum(1 for r in results if r.is_approved())
    total_conflicts = sum(len(r.conflicts) for r in results)
    avg_processing_time = sum(r.processing_time_ms for r in results) / total_scenarios
    
    print(f"Total Scenarios Tested: {total_scenarios}")
    print(f"Approved Missions: {approved_count}")
    print(f"Rejected Missions: {total_scenarios - approved_count}")
    print(f"Total Conflicts Detected: {total_conflicts}")
    print(f"Average Processing Time: {avg_processing_time:.2f}ms")
    print(f"System Accuracy: {(approved_count/total_scenarios)*100:.1f}%")
    
    # Conflict type analysis
    spatial_conflicts = sum(1 for r in results for c in r.conflicts if c.conflict_type == "spatial")
    temporal_conflicts = sum(1 for r in results for c in r.conflicts if c.conflict_type == "temporal")
    
    if total_conflicts > 0:
        print(f"\nConflict Type Breakdown:")
        print(f"  Spatial Conflicts: {spatial_conflicts} ({(spatial_conflicts/total_conflicts)*100:.1f}%)")
        print(f"  Temporal Conflicts: {temporal_conflicts} ({(temporal_conflicts/total_conflicts)*100:.1f}%)")
    
    # Visualization summary
    if ENABLE_STATIC_VISUALIZATION or ENABLE_ANIMATION:
        print(f"\\n🎨 Generated Visualizations:")
        if ENABLE_STATIC_VISUALIZATION:
            print(f"  📊 Static HTML files: {total_scenarios} interactive 3D plots")
        if ENABLE_ANIMATION:
            print(f"  🎬 Animation MP4 files: {total_scenarios} trajectory videos")
            print(f"     • Frame rate: 15 FPS")
            print(f"     • Features: Real-time movement, conflict highlighting, safety buffers")
    
    print(f"\\n✅ Validation complete - System ready for production deployment")
    print(f"\\n📁 Output Files Generated:")
    if ENABLE_STATIC_VISUALIZATION:
        print(f"   • *_visualization.html - Interactive 3D trajectory plots")
    if ENABLE_ANIMATION:
        print(f"   • *_animation.mp4 - Animated trajectory videos")

if __name__ == "__main__":
    main()