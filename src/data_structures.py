import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

@dataclass
class Waypoint:
    position: np.ndarray
    timestamp: float

    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            raise TypeError("position must be a numpy ndarray")
        if self.position.shape != (3,):
            raise ValueError("position must be a 3-element vector")
        if not isinstance(self.timestamp, (float, int, np.floating)):
            raise TypeError("timestamp must be a float or int")
        self.timestamp = float(self.timestamp)

@dataclass
class Trajectory:
    drone_id: str
    waypoints: List[Waypoint]

    def __post_init__(self):
        if not isinstance(self.waypoints, list):
            raise TypeError("waypoints must be a list of Waypoint objects")
        if len(self.waypoints) == 0:
            raise ValueError("waypoints list must contain at least one waypoint")
        for wp in self.waypoints:
            if not isinstance(wp, Waypoint):
                raise TypeError("all items in waypoints must be Waypoint instances")

    def get_state_at_time(self, t: Union[float, np.floating]) -> Optional[np.ndarray]:
        t = float(t)
        self.waypoints.sort(key=lambda wp: wp.timestamp)
        if not self.waypoints or t < self.waypoints[0].timestamp or t > self.waypoints[-1].timestamp:
            return None
        for i in range(len(self.waypoints) - 1):
            wp1, wp2 = self.waypoints[i], self.waypoints[i+1]
            if wp1.timestamp <= t <= wp2.timestamp:
                interval = wp2.timestamp - wp1.timestamp
                if interval == 0:
                    return wp1.position
                ratio = (t - wp1.timestamp) / interval
                return wp1.position + ratio * (wp2.position - wp1.position)
        if t == self.waypoints[-1].timestamp:
            return self.waypoints[-1].position
        return None

@dataclass
class PrimaryMission:
    trajectory: Trajectory
    mission_start_time: float
    mission_end_time: float

    def __post_init__(self):
        if not isinstance(self.trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory instance")
        if not isinstance(self.mission_start_time, (float, int)):
            raise TypeError("mission_start must be a float or int")
        if not isinstance(self.mission_end_time, (float, int)):
            raise TypeError("mission_end must be a float or int")
        self.mission_start_time = float(self.mission_start_time)
        self.mission_end_time = float(self.mission_end_time)

    def is_within_time_window(self) -> bool:
        if not self.trajectory.waypoints:
            return False
        return self.mission_start_time <= self.trajectory.waypoints[0].timestamp and \
               self.mission_end_time >= self.trajectory.waypoints[-1].timestamp

@dataclass
class SimulatedFlightSchedule:
    trajectory: Trajectory
    priority_level: int = 0
    flight_type: str = "commercial"

    def __post_init__(self):
        if not isinstance(self.trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory instance")
        if not isinstance(self.priority_level, int):
            raise TypeError("priority_level must be an int")
        if not isinstance(self.flight_type, str):
            raise TypeError("flight_type must be a string")

@dataclass
class Conflict:
    time_of_conflict: float
    location_of_conflict: np.ndarray
    conflicting_drone_ids: Tuple[str, str]
    minimum_separation: float
    conflict_type: str = "spatial"
    severity: str = "medium"

    def __post_init__(self):
        if not isinstance(self.time_of_conflict, (float, int)):
            raise TypeError("time_of_conflict must be a float or int")
        self.time_of_conflict = float(self.time_of_conflict)
        if not isinstance(self.location_of_conflict, np.ndarray):
            raise TypeError("location_of_conflict must be a numpy ndarray")
        if not isinstance(self.conflicting_drone_ids, tuple):
            raise TypeError("conflicting_drone_ids must be a tuple of strings")
        if not isinstance(self.minimum_separation, (float, int)):
            raise TypeError("minimum_separation must be a float or int")
        self.minimum_separation = float(self.minimum_separation)
        if not isinstance(self.conflict_type, str):
            raise TypeError("conflict_type must be a string")
        if not isinstance(self.severity, str):
            raise TypeError("severity must be a string")

    def get_conflict_description(self) -> str:
        drone1, drone2 = self.conflicting_drone_ids
        return f"Conflict between {drone1} and {drone2} at t={self.time_of_conflict:.2f}s (location: {np.round(self.location_of_conflict,1)}, separation: {self.minimum_separation:.2f}m, severity: {self.severity})"

@dataclass
class DeconflictionResult:
    primary_mission_id: str
    status: str
    conflicts: List[Conflict]
    total_simulated_flights: int
    processing_time_ms: float

    def __post_init__(self):
        if not isinstance(self.primary_mission_id, str):
            raise TypeError("primary_id must be a string")
        if not isinstance(self.status, str):
            raise TypeError("status must be a string")
        if not isinstance(self.conflicts, list):
            raise TypeError("conflicts must be a list")
        if not isinstance(self.total_simulated_flights, int):
            raise TypeError("total_simulated must be an int")
        if not isinstance(self.processing_time_ms, (float,int)):
            raise TypeError("processing_time_ms must be a float")
        self.processing_time_ms = float(self.processing_time_ms)

    def is_approved(self) -> bool:
        return self.status == "CLEAR"

    def get_conflicting_flight_ids(self) -> List[str]:
        conflicting_ids = set()
        for conflict in self.conflicts:
            conflicting_ids.update([id for id in conflict.conflicting_drone_ids if id != self.primary_mission_id])
        return list(conflicting_ids)
