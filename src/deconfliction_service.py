# src/deconfliction_service.py

from typing import List, Dict, Tuple, Optional
import numpy as np
import time
from .data_structures import (
    Trajectory, Conflict, PrimaryMission, SimulatedFlightSchedule, 
    DeconflictionResult
)

def _point_segment_distance(point: np.ndarray, seg_p1: np.ndarray, seg_p2: np.ndarray) -> float:
    """
    /// @brief Calculate shortest distance between a point and a line segment
    /// 
    /// @param point 3D coordinates of the query point
    /// @param seg_p1 Start point of the line segment  
    /// @param seg_p2 End point of the line segment
    /// @return Minimum distance in meters
    /// 
    /// Uses parametric line equation to find closest point on segment,
    /// handling degenerate cases where segment has zero length.
    """
    line_vec = seg_p2 - seg_p1  # Vector along the line segment
    point_vec = point - seg_p1  # Vector from segment start to query point
    
    # Handle degenerate case: zero-length segment (seg_p1 == seg_p2)
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0:
        return float(np.linalg.norm(point_vec))
    
    # Find projection parameter t: closest point = seg_p1 + t * line_vec
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)  # Clamp to segment bounds [0,1]
    
    # Calculate closest point on segment and return distance
    closest_point_on_segment = seg_p1 + t * line_vec
    return float(np.linalg.norm(point - closest_point_on_segment))

def min_distance_between_segments(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """
    /// @brief Calculate minimum distance between two 3D line segments
    /// 
    /// @param p1 Start point of first segment
    /// @param q1 End point of first segment
    /// @param p2 Start point of second segment  
    /// @param q2 End point of second segment
    /// @return Minimum distance between segments in meters
    /// 
    /// Implements robust 3D segment-to-segment distance algorithm that handles
    /// all degenerate cases including parallel, intersecting, and point segments.
    /// Uses parametric line equations and geometric optimization.
    """
    u = q1 - p1  # Direction vector of segment 1
    v = q2 - p2  # Direction vector of segment 2
    w = p1 - p2  # Vector between segment start points
    
    a = np.dot(u, u)  # Squared length of segment 1
    c = np.dot(v, v)  # Squared length of segment 2
    
    # Handle degenerate cases where one or both segments are points
    if a < 1e-7 and c < 1e-7:  # Both segments are points
        return float(np.linalg.norm(w))
    if a < 1e-7:  # Segment 1 is a point
        return _point_segment_distance(p1, p2, q2)
    if c < 1e-7:  # Segment 2 is a point
        return _point_segment_distance(p2, p1, q1)
    
    # Standard algorithm for non-degenerate segments
    b = np.dot(u, v)  # Dot product of direction vectors
    d = np.dot(u, w)  # Projection coefficients
    e = np.dot(v, w)
    D = a * c - b * b  # Determinant for parallel check
    
    # Initialize parameter variables for closest points
    sc, sN, sD = D, D, D  # Parameters for segment 1
    tc, tN, tD = D, D, D  # Parameters for segment 2
    
    # Check if segments are parallel (D ≈ 0)
    if D < 1e-7:
        sN = 0.0  # Use start of segment 1
        sD = 1.0
        tN = e    # Find closest point on segment 2
        tD = c
    else:
        # Calculate closest points on infinite lines
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        
        # Clamp segment 1 parameter to [0,1]
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    
    # Clamp segment 2 parameter to [0,1] and recompute segment 1 if needed
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
    
    # Calculate final parameters and closest points
    sc = 0.0 if abs(sN) < 1e-7 else sN / sD
    tc = 0.0 if abs(tN) < 1e-7 else tN / tD
    
    # Vector between closest points on the two segments
    dP = w + (sc * u) - (tc * v)
    return float(np.linalg.norm(dP))

class UAVDeconflictionService:
    """
    /// @brief Strategic deconfliction service for validating UAV missions in shared airspace
    /// 
    /// Serves as the authoritative system for validating primary drone missions against
    /// existing simulated flight schedules. Performs comprehensive spatial and temporal
    /// conflict detection with configurable safety parameters.
    """
    
    def __init__(self, default_safety_buffer_m: float = 10.0):
        """
        /// @brief Initialize the deconfliction service
        /// 
        /// @param default_safety_buffer_m Default minimum separation distance in meters
        """
        self.default_safety_buffer = default_safety_buffer_m
        self.simulated_flights: Dict[str, SimulatedFlightSchedule] = {}
        self.validation_history: List[DeconflictionResult] = []
        
    def add_simulated_flight_schedule(self, flight_schedule: SimulatedFlightSchedule) -> None:
        """
        /// @brief Add a simulated flight schedule to the airspace database
        /// 
        /// @param flight_schedule Flight schedule containing trajectory and metadata
        /// 
        /// Registers an existing flight schedule that primary missions will be
        /// validated against. Multiple schedules create a complex airspace environment.
        """
        self.simulated_flights[flight_schedule.trajectory.drone_id] = flight_schedule
    
    def clear_simulated_flights(self) -> None:
        """
        /// @brief Clear all simulated flight schedules
        /// 
        /// Removes all existing simulated flights, typically used when setting up
        /// new test scenarios or resetting the airspace environment.
        """
        self.simulated_flights.clear()
    
    def validate_primary_mission(self, 
                                primary_mission: PrimaryMission,
                                safety_buffer_m: Optional[float] = None) -> DeconflictionResult:
        """
        /// @brief Validate primary mission against all simulated flight schedules
        /// 
        /// @param primary_mission The mission requesting airspace clearance
        /// @param safety_buffer_m Override safety buffer (uses default if None)
        /// @return Complete validation result with conflicts and status
        /// 
        /// Core validation function that performs comprehensive conflict detection:
        /// 1. Validates mission timing against overall time window
        /// 2. Performs spatial-temporal conflict detection against all simulated flights
        /// 3. Returns detailed results with conflict explanations
        """
        start_time = time.time()
        
        # Use provided safety buffer or fall back to default
        safety_buffer = safety_buffer_m if safety_buffer_m is not None else self.default_safety_buffer
        
        # Validate mission time window constraints
        if not primary_mission.is_within_time_window():
            # Create a synthetic conflict for time window violation
            conflict = Conflict(
                time_of_conflict=primary_mission.trajectory.waypoints[0].timestamp,
                location_of_conflict=primary_mission.trajectory.waypoints[0].position,
                conflicting_drone_ids=(primary_mission.trajectory.drone_id, "SYSTEM_TIME_WINDOW"),
                minimum_separation=0.0,
                conflict_type="temporal",
                severity="critical"
            )
            
            # Return immediate rejection for time window violation
            processing_time = (time.time() - start_time) * 1000
            return DeconflictionResult(
                primary_mission_id=primary_mission.trajectory.drone_id,
                status="CONFLICT_DETECTED",
                conflicts=[conflict],
                total_simulated_flights=len(self.simulated_flights),
                processing_time_ms=processing_time
            )
        
        # Perform conflict detection against all simulated flights
        all_conflicts = []
        
        for sim_flight_id, sim_flight_schedule in self.simulated_flights.items():
            conflicts = self._detect_trajectory_conflicts(
                primary_mission.trajectory,
                sim_flight_schedule.trajectory,
                safety_buffer
            )
            all_conflicts.extend(conflicts)
        
        # Determine overall status
        status = "CLEAR" if len(all_conflicts) == 0 else "CONFLICT_DETECTED"
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create and store result
        result = DeconflictionResult(
            primary_mission_id=primary_mission.trajectory.drone_id,
            status=status,
            conflicts=all_conflicts,
            total_simulated_flights=len(self.simulated_flights),
            processing_time_ms=processing_time
        )
        
        self.validation_history.append(result)
        return result
    
    def _detect_trajectory_conflicts(self, 
                                   primary_trajectory: Trajectory,
                                   simulated_trajectory: Trajectory,
                                   safety_buffer: float) -> List[Conflict]:
        """
        /// @brief Detect conflicts between primary and simulated trajectory
        /// 
        /// @param primary_trajectory The mission trajectory to validate
        /// @param simulated_trajectory Existing flight to check against  
        /// @param safety_buffer Minimum required separation distance
        /// @return List of detected conflicts
        /// 
        /// Performs detailed segment-by-segment analysis using temporal overlap
        /// detection followed by 3D geometric distance calculations.
        """
        conflicts = []
        
        # Compare each segment of primary trajectory against simulated trajectory
        for i in range(len(primary_trajectory.waypoints) - 1):
            primary_wp1 = primary_trajectory.waypoints[i]
            primary_wp2 = primary_trajectory.waypoints[i + 1]
            
            for j in range(len(simulated_trajectory.waypoints) - 1):
                sim_wp1 = simulated_trajectory.waypoints[j]
                sim_wp2 = simulated_trajectory.waypoints[j + 1]
                
                # Check temporal overlap between segments
                temporal_overlap = (primary_wp1.timestamp <= sim_wp2.timestamp and
                                  sim_wp1.timestamp <= primary_wp2.timestamp)
                
                if not temporal_overlap:
                    continue  # No temporal overlap, skip spatial check
                
                # Calculate minimum spatial distance between trajectory segments
                min_distance = min_distance_between_segments(
                    primary_wp1.position, primary_wp2.position,
                    sim_wp1.position, sim_wp2.position
                )
                
                # Check if distance violates safety buffer
                if min_distance < safety_buffer:
                    # Calculate conflict time and location for reporting
                    overlap_start = max(primary_wp1.timestamp, sim_wp1.timestamp)
                    overlap_end = min(primary_wp2.timestamp, sim_wp2.timestamp)
                    conflict_time = (overlap_start + overlap_end) / 2.0
                    
                    # Get approximate positions at conflict time
                    primary_pos = primary_trajectory.get_state_at_time(conflict_time)
                    sim_pos = simulated_trajectory.get_state_at_time(conflict_time)
                    
                    if primary_pos is not None and sim_pos is not None:
                        conflict_location = (primary_pos + sim_pos) / 2.0
                        
                        # Determine conflict severity based on separation distance
                        if min_distance < safety_buffer * 0.25:
                            severity = "critical"
                        elif min_distance < safety_buffer * 0.5:
                            severity = "high"
                        elif min_distance < safety_buffer * 0.75:
                            severity = "medium"
                        else:
                            severity = "low"
                        
                        conflict = Conflict(
                            time_of_conflict=conflict_time,
                            location_of_conflict=conflict_location,
                            conflicting_drone_ids=(primary_trajectory.drone_id, simulated_trajectory.drone_id),
                            minimum_separation=min_distance,
                            conflict_type="spatial",
                            severity=severity
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def get_validation_statistics(self) -> Dict:
        """
        /// @brief Get statistics about validation history
        /// 
        /// @return Dictionary containing validation metrics and performance data
        """
        if not self.validation_history:
            return {"total_validations": 0, "approval_rate": 0.0, "avg_processing_time_ms": 0.0}
        
        total_validations = len(self.validation_history)
        approved_count = sum(1 for result in self.validation_history if result.is_approved())
        total_conflicts = sum(len(result.conflicts) for result in self.validation_history)
        avg_processing_time = sum(result.processing_time_ms for result in self.validation_history) / total_validations
        
        return {
            "total_validations": total_validations,
            "approved_missions": approved_count,
            "rejected_missions": total_validations - approved_count,
            "approval_rate": approved_count / total_validations,
            "total_conflicts_detected": total_conflicts,
            "avg_processing_time_ms": avg_processing_time,
            "current_simulated_flights": len(self.simulated_flights)
        }

# Legacy compatibility functions for existing code
def create_deconfliction_server(safety_buffer: float = 10.0) -> UAVDeconflictionService:
    """
    /// @brief Factory function to create deconfliction service instance
    /// 
    /// @param safety_buffer Default safety buffer in meters
    /// @return Configured deconfliction service ready for use
    """
    return UAVDeconflictionService(default_safety_buffer_m=safety_buffer)