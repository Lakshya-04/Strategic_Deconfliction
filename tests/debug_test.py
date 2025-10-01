#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import numpy as np

try:
    from src.data_structures import Waypoint, Trajectory
    print("✅ Successfully imported Waypoint and Trajectory")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test basic trajectory creation
print("\n🔧 TESTING TRAJECTORY CREATION:")
try:
    wp1 = Waypoint(position=np.array([0, 0, 10]), timestamp=0)
    wp2 = Waypoint(position=np.array([100, 0, 10]), timestamp=10)
    print(f"✅ Created waypoints: {wp1}, {wp2}")
    
    traj = Trajectory(drone_id="test_drone", waypoints=[wp1, wp2])
    print(f"✅ Created trajectory: {traj.drone_id} with {len(traj.waypoints)} waypoints")
    
except Exception as e:
    print(f"❌ Error creating trajectory: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test get_state_at_time method
print("\n🔍 TESTING get_state_at_time() METHOD:")
test_times = [0, 2.5, 5.0, 7.5, 10.0, 15.0]  # Include out-of-range time
for t in test_times:
    try:
        pos = traj.get_state_at_time(t)
        if pos is not None:
            print(f"✅ t={t}s -> position={pos}")
        else:
            print(f"⚠️  t={t}s -> position=None (out of range)")
    except Exception as e:
        print(f"❌ ERROR at t={t}s: {e}")
        import traceback
        traceback.print_exc()

# Test collision scenario manually
print("\n💥 TESTING COLLISION SCENARIO:")
try:
    # Primary drone: (0,0,10) -> (100,0,10) over 10s
    primary = Trajectory(
        drone_id="primary",
        waypoints=[
            Waypoint(position=np.array([0, 0, 10]), timestamp=0),
            Waypoint(position=np.array([100, 0, 10]), timestamp=10)
        ]
    )
    
    # Stationary drone at (50,0,10)
    stationary = Trajectory(
        drone_id="stationary", 
        waypoints=[
            Waypoint(position=np.array([50, 0, 10]), timestamp=0),
            Waypoint(position=np.array([50, 0, 10]), timestamp=10)
        ]
    )
    
    # Check positions at t=5s
    t_test = 5.0
    primary_pos = primary.get_state_at_time(t_test)
    stationary_pos = stationary.get_state_at_time(t_test)
    
    print(f"At t={t_test}s:")
    print(f"  Primary position: {primary_pos}")
    print(f"  Stationary position: {stationary_pos}")
    
    if primary_pos is not None and stationary_pos is not None:
        distance = np.linalg.norm(primary_pos - stationary_pos)
        print(f"  Distance: {distance:.3f}m")
        
        if distance < 10.0:
            print(f"  ❌ COLLISION! {distance:.3f}m < 10.0m safety buffer")
        else:
            print(f"  ✅ SAFE: {distance:.3f}m >= 10.0m safety buffer")
    else:
        print(f"  ❌ ERROR: One or both positions are None!")
        
except Exception as e:
    print(f"❌ Error in collision test: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 DEBUG TEST COMPLETE")   