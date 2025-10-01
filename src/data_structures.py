# src/data_structures.py

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class Waypoint:
    """
    /// @brief Represents a single waypoint in a drone's trajectory with position and timing
    /// 
    /// A waypoint defines a specific point in 3D space that a drone should reach at a 
    /// particular time. Used to construct complete flight paths through linear interpolation.
    """
    position: np.ndarray  # 3D position as NumPy array [x, y, z] in meters
    timestamp: float      # Time in seconds from mission start when drone should reach this point

@dataclass  
class Trajectory:
    """
    /// @brief Complete flight path for a single drone defined by waypoints
    /// 
    /// Represents the full trajectory of a drone from start to finish. Provides methods
    /// for calculating drone position at any time through linear interpolation between waypoints.
    """
    drone_id: str                    # Unique identifier for the drone
    waypoints: List[Waypoint]        # Ordered list of waypoints defining the flight path
    
    def get_state_at_time(self, t: float) -> np.ndarray:
        """
        /// @brief Calculate drone's 3D position at specific time using linear interpolation
        /// 
        /// @param t Target time in seconds from mission start
        /// @return 3D position array [x,y,z] or None if time is outside trajectory bounds
        /// 
        /// Uses linear interpolation between consecutive waypoints to determine exact
        /// position. Handles edge cases like identical timestamps and out-of-range queries.
        """
        # Ensure waypoints are chronologically ordered for proper interpolation
        self.waypoints.sort(key=lambda wp: wp.timestamp)
        
        # Check if query time is within trajectory bounds
        if not self.waypoints or t < self.waypoints[0].timestamp or t > self.waypoints[-1].timestamp:
            return None
            
        # Find the appropriate waypoint pair for interpolation
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]      # Start waypoint of segment
            wp2 = self.waypoints[i+1]    # End waypoint of segment
            
            if wp1.timestamp <= t <= wp2.timestamp:
                # Calculate linear interpolation between waypoints
                time_interval = wp2.timestamp - wp1.timestamp
                
                # Handle case where waypoints have identical timestamps
                if time_interval == 0:
                    return wp1.position
                    
                # Linear interpolation: P(t) = P1 + ratio * (P2 - P1)
                ratio = (t - wp1.timestamp) / time_interval
                position = wp1.position + ratio * (wp2.position - wp1.position)
                return position
        
        # Handle exact match with final waypoint timestamp
        if t == self.waypoints[-1].timestamp:
            return self.waypoints[-1].position
            
        return None

@dataclass
class PrimaryMission:
    """
    /// @brief Primary drone mission with overall time window constraints
    /// 
    /// Represents the main drone mission that needs approval. Contains the trajectory
    /// plus overall mission timing constraints that must be satisfied.
    """
    trajectory: Trajectory    # The flight path to be validated
    mission_start_time: float # Earliest allowable mission start time
    mission_end_time: float   # Latest allowable mission completion time
    
    def is_within_time_window(self) -> bool:
        """
        /// @brief Check if trajectory fits within overall mission time window
        /// 
        /// @return True if trajectory start/end times are within mission window
        """
        if not self.trajectory.waypoints:
            return False
            
        traj_start = self.trajectory.waypoints[0].timestamp
        traj_end = self.trajectory.waypoints[-1].timestamp
        
        return (self.mission_start_time <= traj_start and 
                traj_end <= self.mission_end_time)

@dataclass
class SimulatedFlightSchedule:
    """
    /// @brief Represents a simulated flight schedule of other drones in the airspace
    /// 
    /// Contains trajectory information for drones already operating in the shared
    /// airspace that the primary mission must avoid conflicting with.
    """
    trajectory: Trajectory    # Flight path of the simulated drone
    priority_level: int = 0   # Priority level (0 = highest, higher numbers = lower priority)
    flight_type: str = "commercial"  # Type of flight: "commercial", "emergency", "recreational"

@dataclass
class Conflict:
    """
    /// @brief Detailed information about a detected spatial-temporal conflict
    /// 
    /// Contains all relevant details about where, when, and between which drones
    /// a conflict occurs, along with severity metrics.
    """
    time_of_conflict: float           # Time when conflict occurs (seconds from mission start)
    location_of_conflict: np.ndarray  # 3D coordinates where conflict occurs [x,y,z]
    conflicting_drone_ids: Tuple[str, str]  # IDs of the two conflicting drones
    minimum_separation: float         # Minimum distance between drones (meters)
    conflict_type: str = "spatial"    # Type of conflict: "spatial", "temporal", "spawn"
    severity: str = "medium"          # Severity level: "low", "medium", "high", "critical"
    
    def get_conflict_description(self) -> str:
        """
        /// @brief Generate human-readable description of the conflict
        /// 
        /// @return Formatted string describing the conflict details
        """
        drone1, drone2 = self.conflicting_drone_ids
        return (f"Conflict between {drone1} and {drone2} at t={self.time_of_conflict:.2f}s "
               f"(location: {np.round(self.location_of_conflict, 1)}, "
               f"separation: {self.minimum_separation:.2f}m, severity: {self.severity})")

@dataclass
class DeconflictionResult:
    """
    /// @brief Result of primary mission validation against simulated flights
    /// 
    /// Contains the outcome of the deconfliction process including approval status,
    /// detected conflicts, and explanatory information.
    """
    primary_mission_id: str      # ID of the primary mission that was validated
    status: str                  # "CLEAR" or "CONFLICT_DETECTED" 
    conflicts: List[Conflict]    # List of all detected conflicts
    total_simulated_flights: int # Number of simulated flights checked against
    processing_time_ms: float    # Time taken for validation in milliseconds
    
    def is_approved(self) -> bool:
        """
        /// @brief Check if primary mission is approved for execution
        /// 
        /// @return True if status is "CLEAR", False if conflicts detected
        """
        return self.status == "CLEAR"
    
    def get_conflicting_flight_ids(self) -> List[str]:
        """
        /// @brief Get list of simulated flight IDs that cause conflicts
        /// 
        /// @return List of unique drone IDs that conflict with primary mission
        """
        conflicting_ids = set()
        for conflict in self.conflicts:
            # Add the simulated flight ID (not the primary mission ID)
            for drone_id in conflict.conflicting_drone_ids:
                if drone_id != self.primary_mission_id:
                    conflicting_ids.add(drone_id)
        return list(conflicting_ids)