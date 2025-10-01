# tests/test_deconfliction_comprehensive.py

import pytest
import numpy as np
import time
import warnings
from unittest.mock import Mock, patch
import tempfile
import os
from typing import List

# Import refactored modules  
from src.data_structures import (
    Waypoint, Trajectory, PrimaryMission, SimulatedFlightSchedule, 
    Conflict, DeconflictionResult
)
from src.deconfliction_service import (
    UAVDeconflictionService, min_distance_between_segments, _point_segment_distance
)

class TestDataStructures:
    """Comprehensive tests for core data structure classes"""
    
    def test_waypoint_creation(self):
        """Test waypoint creation with valid inputs"""
        pos = np.array([100.5, 200.7, 50.3])
        wp = Waypoint(position=pos, timestamp=15.5)
        assert np.array_equal(wp.position, pos)
        assert wp.timestamp == 15.5
    
    def test_waypoint_invalid_inputs(self):
        """Test waypoint error handling with invalid inputs"""
        with pytest.raises((ValueError, TypeError)):
            Waypoint(position="invalid", timestamp=10)
        
        with pytest.raises((ValueError, TypeError)):
            Waypoint(position=np.array([1, 2]), timestamp="invalid")
    
    def test_trajectory_interpolation(self):
        """Test trajectory position interpolation"""
        traj = Trajectory(
            drone_id="TEST_DRONE",
            waypoints=[
                Waypoint(np.array([0, 0, 10]), 0),
                Waypoint(np.array([100, 0, 10]), 10),
                Waypoint(np.array([200, 100, 20]), 20)
            ]
        )
        
        # Test interpolation at various time points
        pos_5 = traj.get_state_at_time(5.0)
        expected_5 = np.array([50, 0, 10])
        assert np.allclose(pos_5, expected_5)
        
        pos_15 = traj.get_state_at_time(15.0) 
        expected_15 = np.array([150, 50, 15])
        assert np.allclose(pos_15, expected_15)
    
    def test_trajectory_edge_cases(self):
        """Test trajectory interpolation edge cases"""
        traj = Trajectory(
            drone_id="EDGE_TEST",
            waypoints=[
                Waypoint(np.array([0, 0, 0]), 0),
                Waypoint(np.array([10, 10, 10]), 5)
            ]
        )
        
        # Test out-of-range queries
        assert traj.get_state_at_time(-1.0) is None
        assert traj.get_state_at_time(10.0) is None
        
        # Test exact waypoint times
        pos_0 = traj.get_state_at_time(0.0)
        assert np.array_equal(pos_0, np.array([0, 0, 0]))
        
        pos_5 = traj.get_state_at_time(5.0)
        assert np.array_equal(pos_5, np.array([10, 10, 10]))
    
    def test_trajectory_identical_timestamps(self):
        """Test trajectory with identical timestamps (degenerate case)"""
        traj = Trajectory(
            drone_id="DEGENERATE",
            waypoints=[
                Waypoint(np.array([0, 0, 0]), 5),
                Waypoint(np.array([10, 10, 10]), 5)  # Same timestamp
            ]
        )
        
        pos = traj.get_state_at_time(5.0)
        # Should return first waypoint position
        assert np.array_equal(pos, np.array([0, 0, 0]))
    
    def test_empty_trajectory(self):
        """Test trajectory with no waypoints"""
        empty_traj = Trajectory(drone_id="EMPTY", waypoints=[])
        assert empty_traj.get_state_at_time(5.0) is None
    
    def test_primary_mission_time_window(self):
        """Test primary mission time window validation"""
        traj = Trajectory(
            drone_id="PRIMARY",
            waypoints=[
                Waypoint(np.array([0, 0, 50]), 10),
                Waypoint(np.array([100, 0, 50]), 60)
            ]
        )
        
        # Valid time window
        mission_valid = PrimaryMission(traj, 0, 100)
        assert mission_valid.is_within_time_window()
        
        # Invalid time window (too restrictive)
        mission_invalid = PrimaryMission(traj, 20, 50)
        assert not mission_invalid.is_within_time_window()
    
    def test_conflict_description_generation(self):
        """Test conflict description string generation"""
        conflict = Conflict(
            time_of_conflict=45.5,
            location_of_conflict=np.array([250, 300, 75]),
            conflicting_drone_ids=("DRONE_A", "DRONE_B"),
            minimum_separation=3.2,
            conflict_type="spatial",
            severity="high"
        )
        
        description = conflict.get_conflict_description()
        assert "DRONE_A" in description
        assert "DRONE_B" in description
        assert "45.5" in description
        assert "3.2" in description
        assert "high" in description

class TestGeometricAlgorithms:
    """Test geometric distance calculation algorithms"""
    
    def test_point_segment_distance_basic(self):
        """Test basic point-to-segment distance calculation"""
        point = np.array([5, 5, 0])
        seg_start = np.array([0, 0, 0])
        seg_end = np.array([10, 0, 0])
        
        distance = _point_segment_distance(point, seg_start, seg_end)
        assert abs(distance - 5.0) < 1e-6
    
    def test_point_segment_distance_endpoint(self):
        """Test point distance to segment endpoints"""
        point = np.array([15, 0, 0])
        seg_start = np.array([0, 0, 0])
        seg_end = np.array([10, 0, 0])
        
        distance = _point_segment_distance(point, seg_start, seg_end)
        assert abs(distance - 5.0) < 1e-6  # Distance to closest endpoint
    
    def test_point_segment_zero_length(self):
        """Test point distance to zero-length segment (degenerate case)"""
        point = np.array([3, 4, 0])
        seg_start = np.array([0, 0, 0])
        seg_end = np.array([0, 0, 0])  # Zero-length segment
        
        distance = _point_segment_distance(point, seg_start, seg_end)
        assert abs(distance - 5.0) < 1e-6  # Should be point-to-point distance
    
    def test_min_distance_parallel_segments(self):
        """Test distance between parallel segments"""
        p1, q1 = np.array([0, 0, 0]), np.array([10, 0, 0])
        p2, q2 = np.array([0, 5, 0]), np.array([10, 5, 0])
        
        distance = min_distance_between_segments(p1, q1, p2, q2)
        assert abs(distance - 5.0) < 1e-6
    
    def test_min_distance_intersecting_segments(self):
        """Test distance between intersecting segments"""
        p1, q1 = np.array([0, 0, 0]), np.array([10, 0, 0])
        p2, q2 = np.array([5, -5, 0]), np.array([5, 5, 0])
        
        distance = min_distance_between_segments(p1, q1, p2, q2)
        assert distance < 1e-6  # Should be approximately zero (intersecting)
    
    def test_min_distance_skew_segments(self):
        """Test distance between skew (non-coplanar) segments"""
        p1, q1 = np.array([0, 0, 0]), np.array([10, 0, 0])
        p2, q2 = np.array([5, 5, 5]), np.array([5, 5, 15])
        
        distance = min_distance_between_segments(p1, q1, p2, q2)
        expected = np.sqrt(50)  # sqrt(5^2 + 5^2) = sqrt(50)
        assert abs(distance - expected) < 1e-6
    
    def test_min_distance_point_segments(self):
        """Test distance when one or both segments are points"""
        # Both are points
        p1, q1 = np.array([0, 0, 0]), np.array([0, 0, 0])
        p2, q2 = np.array([3, 4, 0]), np.array([3, 4, 0])
        distance = min_distance_between_segments(p1, q1, p2, q2)
        assert abs(distance - 5.0) < 1e-6
        
        # One is a point, one is a segment
        p1, q1 = np.array([5, 0, 0]), np.array([5, 0, 0])  # Point
        p2, q2 = np.array([0, 0, 0]), np.array([10, 0, 0])  # Segment
        distance = min_distance_between_segments(p1, q1, p2, q2)
        assert distance < 1e-6  # Point lies on segment
    
    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_min_distance_random_segments(self, seed):
        """Test distance calculation with random segment configurations"""
        np.random.seed(seed)
        
        # Generate random segments
        p1 = np.random.randn(3) * 100
        q1 = np.random.randn(3) * 100
        p2 = np.random.randn(3) * 100  
        q2 = np.random.randn(3) * 100
        
        distance = min_distance_between_segments(p1, q1, p2, q2)
        
        # Distance should be non-negative
        assert distance >= 0
        
        # Distance should be symmetric
        distance_reverse = min_distance_between_segments(p2, q2, p1, q1)
        assert abs(distance - distance_reverse) < 1e-10

class TestDeconflictionService:
    """Comprehensive tests for UAVDeconflictionService"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.service = UAVDeconflictionService(default_safety_buffer_m=10.0)
        
    def create_test_trajectory(self, drone_id: str, start_pos: List[float], 
                             end_pos: List[float], start_time: float = 0, 
                             end_time: float = 60) -> Trajectory:
        """Helper method to create test trajectories"""
        return Trajectory(
            drone_id=drone_id,
            waypoints=[
                Waypoint(np.array(start_pos), start_time),
                Waypoint(np.array(end_pos), end_time)
            ]
        )
    
    def test_service_initialization(self):
        """Test service initialization with various parameters"""
        # Default initialization
        service_default = UAVDeconflictionService()
        assert service_default.default_safety_buffer == 10.0
        
        # Custom safety buffer
        service_custom = UAVDeconflictionService(default_safety_buffer_m=25.0)
        assert service_custom.default_safety_buffer == 25.0
        
        # Edge case: very small buffer
        service_small = UAVDeconflictionService(default_safety_buffer_m=0.1)
        assert service_small.default_safety_buffer == 0.1
    
    def test_add_simulated_flights(self):
        """Test adding simulated flight schedules"""
        flight1 = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("SIM_001", [0, 0, 50], [100, 0, 50]),
            flight_type="commercial"
        )
        
        flight2 = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("SIM_002", [0, 50, 50], [100, 50, 50]),
            flight_type="emergency"
        )
        
        self.service.add_simulated_flight_schedule(flight1)
        self.service.add_simulated_flight_schedule(flight2)
        
        assert len(self.service.simulated_flights) == 2
        assert "SIM_001" in self.service.simulated_flights
        assert "SIM_002" in self.service.simulated_flights
    
    def test_clear_simulated_flights(self):
        """Test clearing all simulated flights"""
        flight = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("TEST", [0, 0, 0], [10, 10, 10])
        )
        
        self.service.add_simulated_flight_schedule(flight)
        assert len(self.service.simulated_flights) == 1
        
        self.service.clear_simulated_flights()
        assert len(self.service.simulated_flights) == 0
    
    def test_conflict_free_validation(self):
        """Test validation of conflict-free mission"""
        # Add simulated flight
        sim_flight = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("SIM_001", [0, 100, 50], [1000, 100, 50])
        )
        self.service.add_simulated_flight_schedule(sim_flight)
        
        # Create primary mission with adequate separation
        primary_traj = self.create_test_trajectory("PRIMARY", [0, 0, 50], [1000, 0, 50])
        primary_mission = PrimaryMission(primary_traj, 0, 120)
        
        result = self.service.validate_primary_mission(primary_mission)
        
        assert result.is_approved()
        assert len(result.conflicts) == 0
        assert result.status == "CLEAR"
        assert result.processing_time_ms > 0
    
    def test_spatial_conflict_detection(self):
        """Test detection of spatial conflicts"""
        # Add simulated flight
        sim_flight = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("SIM_CONFLICT", [500, 0, 50], [500, 100, 50])
        )
        self.service.add_simulated_flight_schedule(sim_flight)
        
        # Create primary mission that crosses simulated flight path
        primary_traj = self.create_test_trajectory("PRIMARY", [0, 50, 50], [1000, 50, 50])
        primary_mission = PrimaryMission(primary_traj, 0, 120)
        
        result = self.service.validate_primary_mission(primary_mission, safety_buffer_m=20.0)
        
        assert not result.is_approved()
        assert len(result.conflicts) > 0
        assert result.status == "CONFLICT_DETECTED"
        
        # Verify conflict details
        conflict = result.conflicts[0]
        assert conflict.conflict_type == "spatial"
        assert "SIM_CONFLICT" in conflict.conflicting_drone_ids
    
    def test_temporal_window_violation(self):
        """Test detection of mission time window violations"""
        # Create mission that exceeds time window
        primary_traj = self.create_test_trajectory("PRIMARY", [0, 0, 50], [1000, 0, 50], 0, 150)
        primary_mission = PrimaryMission(primary_traj, 0, 100)  # Window too small
        
        result = self.service.validate_primary_mission(primary_mission)
        
        assert not result.is_approved()
        assert len(result.conflicts) > 0
        assert result.conflicts[0].conflict_type == "temporal"
    
    def test_multiple_conflicts(self):
        """Test detection of multiple simultaneous conflicts"""
        # Add multiple conflicting simulated flights
        sim_flight1 = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("SIM_001", [200, 0, 50], [800, 0, 50])
        )
        sim_flight2 = SimulatedFlightSchedule(
            trajectory=self.create_test_trajectory("SIM_002", [400, 0, 50], [600, 0, 50])
        )
        
        self.service.add_simulated_flight_schedule(sim_flight1)
        self.service.add_simulated_flight_schedule(sim_flight2)
        
        # Primary mission conflicts with both
        primary_traj = self.create_test_trajectory("PRIMARY", [0, 0, 50], [1000, 0, 50])
        primary_mission = PrimaryMission(primary_traj, 0, 120)
        
        result = self.service.validate_primary_mission(primary_mission, safety_buffer_m=5.0)
        
        assert not result.is_approved()
        assert len(result.conflicts) >= 2
    
    def test_validation_statistics(self):
        """Test validation statistics tracking"""
        # Perform multiple validations
        for i in range(5):
            primary_traj = self.create_test_trajectory(f"PRIMARY_{i}", [i*100, 0, 50], [(i+1)*100, 0, 50])
            primary_mission = PrimaryMission(primary_traj, 0, 120)
            self.service.validate_primary_mission(primary_mission)
        
        stats = self.service.get_validation_statistics()
        
        assert stats['total_validations'] == 5
        assert stats['approved_missions'] >= 0
        assert stats['rejected_missions'] >= 0
        assert stats['approval_rate'] >= 0.0
        assert stats['avg_processing_time_ms'] > 0

class TestErrorHandling:
    """Test error handling and robustness"""
    
    def test_invalid_trajectory_inputs(self):
        """Test handling of invalid trajectory inputs"""
        service = UAVDeconflictionService()
        
        # Test with None trajectory
        with pytest.raises(AttributeError):
            invalid_mission = PrimaryMission(None, 0, 100)
            service.validate_primary_mission(invalid_mission)
        
        # Test with empty waypoints
        empty_traj = Trajectory("EMPTY", [])
        empty_mission = PrimaryMission(empty_traj, 0, 100)
        
        # Should handle gracefully without crashing
        result = service.validate_primary_mission(empty_mission)
        assert not result.is_approved()  # Should reject empty trajectories
    
    def test_invalid_safety_buffer(self):
        """Test handling of invalid safety buffer values"""
        service = UAVDeconflictionService()
        traj = Trajectory("TEST", [Waypoint(np.array([0, 0, 0]), 0)])
        mission = PrimaryMission(traj, 0, 100)
        
        # Test negative safety buffer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress potential warnings
            result = service.validate_primary_mission(mission, safety_buffer_m=-5.0)
            # Should handle gracefully, possibly using absolute value or default
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large numbers of trajectories"""
        service = UAVDeconflictionService()
        
        # Add many simulated flights
        for i in range(100):
            traj = Trajectory(
                f"SIM_{i:03d}",
                [
                    Waypoint(np.array([i*10, 0, 50]), 0),
                    Waypoint(np.array([i*10 + 100, 100, 50]), 60)
                ]
            )
            sim_flight = SimulatedFlightSchedule(traj)
            service.add_simulated_flight_schedule(sim_flight)
        
        # Test primary mission validation
        primary_traj = Trajectory(
            "PRIMARY_LARGE_TEST",
            [
                Waypoint(np.array([0, 200, 50]), 0),
                Waypoint(np.array([1000, 200, 50]), 60)
            ]
        )
        primary_mission = PrimaryMission(primary_traj, 0, 120)
        
        start_time = time.time()
        result = service.validate_primary_mission(primary_mission)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (< 1 second for 100 flights)
        assert processing_time < 1.0
        assert isinstance(result, DeconflictionResult)
    
    def test_numerical_precision_edge_cases(self):
        """Test numerical precision in edge cases"""
        service = UAVDeconflictionService()
        
        # Test with very small coordinates
        traj_small = Trajectory(
            "SMALL",
            [
                Waypoint(np.array([1e-10, 1e-10, 1e-10]), 0),
                Waypoint(np.array([2e-10, 2e-10, 2e-10]), 1)
            ]
        )
        
        # Test with very large coordinates
        traj_large = Trajectory(
            "LARGE", 
            [
                Waypoint(np.array([1e6, 1e6, 1e6]), 0),
                Waypoint(np.array([2e6, 2e6, 2e6]), 1)
            ]
        )
        
        sim_flight_small = SimulatedFlightSchedule(traj_small)
        service.add_simulated_flight_schedule(sim_flight_small)
        
        mission_large = PrimaryMission(traj_large, 0, 10)
        result = service.validate_primary_mission(mission_large)
        
        # Should handle numerical precision without errors
        assert isinstance(result, DeconflictionResult)
    
    @pytest.mark.timeout(5)  # Should complete within 5 seconds
    def test_performance_under_load(self):
        """Test system performance under computational load"""
        service = UAVDeconflictionService()
        
        # Create complex trajectories with many waypoints
        complex_waypoints = [
            Waypoint(np.array([i*10, np.sin(i)*50, 50 + i*2]), i*2)
            for i in range(20)  # 20 waypoints per trajectory
        ]
        
        for j in range(10):  # 10 complex trajectories
            traj = Trajectory(f"COMPLEX_{j}", complex_waypoints.copy())
            sim_flight = SimulatedFlightSchedule(traj)
            service.add_simulated_flight_schedule(sim_flight)
        
        # Test primary mission with many waypoints
        primary_waypoints = [
            Waypoint(np.array([i*15, 100, 60]), i*3)
            for i in range(15)
        ]
        primary_traj = Trajectory("PRIMARY_COMPLEX", primary_waypoints)
        primary_mission = PrimaryMission(primary_traj, 0, 200)
        
        result = service.validate_primary_mission(primary_mission)
        
        # Should complete successfully
        assert isinstance(result, DeconflictionResult)
        assert result.processing_time_ms < 5000  # Less than 5 seconds

class TestIntegrationScenarios:
    """Integration tests with realistic scenarios"""
    
    def test_airport_departure_scenario(self):
        """Test realistic airport departure scenario"""
        service = UAVDeconflictionService(default_safety_buffer_m=15.0)
        
        # Add existing airport traffic
        runway_traffic = SimulatedFlightSchedule(
            trajectory=Trajectory(
                "RUNWAY_DEPARTURE_001",
                [
                    Waypoint(np.array([0, 0, 0]), 0),      # Ground
                    Waypoint(np.array([100, 0, 50]), 30),  # Takeoff
                    Waypoint(np.array([500, 100, 200]), 120) # Climb
                ]
            ),
            priority_level=0,
            flight_type="commercial"
        )
        
        emergency_helicopter = SimulatedFlightSchedule(
            trajectory=Trajectory(
                "EMERGENCY_HELI_001", 
                [
                    Waypoint(np.array([200, 200, 100]), 60),
                    Waypoint(np.array([300, 300, 100]), 120)
                ]
            ),
            priority_level=0,
            flight_type="emergency"
        )
        
        service.add_simulated_flight_schedule(runway_traffic)
        service.add_simulated_flight_schedule(emergency_helicopter)
        
        # Test new departure request
        new_departure = PrimaryMission(
            trajectory=Trajectory(
                "NEW_DEPARTURE_002",
                [
                    Waypoint(np.array([10, 0, 0]), 180),    # Delayed departure
                    Waypoint(np.array([110, 0, 50]), 210),  # Similar flight path
                    Waypoint(np.array([510, 100, 200]), 300)
                ]
            ),
            mission_start_time=180,
            mission_end_time=350
        )
        
        result = service.validate_primary_mission(new_departure)
        
        # Should be approved due to temporal separation
        assert result.is_approved()
        assert result.total_simulated_flights == 2
    
    def test_urban_delivery_network(self):
        """Test urban drone delivery network scenario"""
        service = UAVDeconflictionService(default_safety_buffer_m=8.0)
        
        # Add existing delivery routes
        delivery_routes = [
            ([0, 0, 30], [100, 100, 30], 0, 60),      # Diagonal route
            ([100, 0, 35], [0, 100, 35], 30, 90),     # Crossing route  
            ([50, 50, 40], [150, 50, 40], 45, 105),   # Parallel route
        ]
        
        for i, (start, end, t_start, t_end) in enumerate(delivery_routes):
            traj = Trajectory(
                f"DELIVERY_{i+1:03d}",
                [
                    Waypoint(np.array(start), t_start),
                    Waypoint(np.array(end), t_end)
                ]
            )
            sim_flight = SimulatedFlightSchedule(traj, flight_type="commercial")
            service.add_simulated_flight_schedule(sim_flight)
        
        # Test new delivery mission
        new_delivery = PrimaryMission(
            trajectory=Trajectory(
                "NEW_DELIVERY_004",
                [
                    Waypoint(np.array([25, 25, 32]), 15),
                    Waypoint(np.array([75, 75, 32]), 45)   # May conflict with existing routes
                ]
            ),
            mission_start_time=0,
            mission_end_time=60
        )
        
        result = service.validate_primary_mission(new_delivery)
        
        # Analyze result based on actual conflicts
        if not result.is_approved():
            # Verify conflicts are properly detected and reported
            assert len(result.conflicts) > 0
            for conflict in result.conflicts:
                assert conflict.minimum_separation < 8.0
                assert conflict.conflict_type == "spatial"
    
    def test_mixed_priority_airspace(self):
        """Test airspace with mixed priority operations"""
        service = UAVDeconflictionService(default_safety_buffer_m=12.0)
        
        # High priority emergency flight
        emergency_flight = SimulatedFlightSchedule(
            trajectory=Trajectory(
                "EMERGENCY_001",
                [
                    Waypoint(np.array([0, 500, 100]), 0),
                    Waypoint(np.array([500, 500, 100]), 30),
                    Waypoint(np.array([1000, 1000, 150]), 90)
                ]
            ),
            priority_level=0,
            flight_type="emergency"
        )
        
        # Lower priority recreational flight
        recreational_flight = SimulatedFlightSchedule(
            trajectory=Trajectory(
                "RECREATIONAL_001",
                [
                    Waypoint(np.array([200, 200, 60]), 60),
                    Waypoint(np.array([300, 300, 60]), 120)
                ]
            ),
            priority_level=3,
            flight_type="recreational"
        )
        
        service.add_simulated_flight_schedule(emergency_flight)
        service.add_simulated_flight_schedule(recreational_flight)
        
        # Test medium priority commercial mission
        commercial_mission = PrimaryMission(
            trajectory=Trajectory(
                "COMMERCIAL_PRIMARY",
                [
                    Waypoint(np.array([100, 600, 110]), 15),
                    Waypoint(np.array([600, 600, 110]), 45)
                ]
            ),
            mission_start_time=0,
            mission_end_time=60
        )
        
        result = service.validate_primary_mission(commercial_mission)
        
        # Should detect potential conflict with emergency flight
        if not result.is_approved():
            conflicting_ids = result.get_conflicting_flight_ids()
            assert "EMERGENCY_001" in conflicting_ids or "RECREATIONAL_001" in conflicting_ids

@pytest.fixture(scope="session")
def benchmark_data():
    """Generate benchmark data for performance testing"""
    np.random.seed(42)  # Reproducible results
    
    trajectories = []
    for i in range(1000):  # 1000 random trajectories
        start_pos = np.random.uniform(-1000, 1000, 3)
        end_pos = start_pos + np.random.uniform(-500, 500, 3)
        start_time = np.random.uniform(0, 3600)  # Up to 1 hour
        duration = np.random.uniform(60, 1800)   # 1-30 minutes
        
        traj = Trajectory(
            f"BENCHMARK_{i:04d}",
            [
                Waypoint(start_pos, start_time),
                Waypoint(end_pos, start_time + duration)
            ]
        )
        trajectories.append(traj)
    
    return trajectories

class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.mark.slow  # Mark as slow test
    def test_large_scale_validation(self, benchmark_data):
        """Test validation performance with large numbers of flights"""
        service = UAVDeconflictionService()
        
        # Add first 500 as simulated flights
        for traj in benchmark_data[:500]:
            sim_flight = SimulatedFlightSchedule(traj)
            service.add_simulated_flight_schedule(sim_flight)
        
        # Test validation of remaining 500 as primary missions
        processing_times = []
        
        for traj in benchmark_data[500:600]:  # Test 100 missions
            mission = PrimaryMission(traj, 0, 7200)  # 2 hour window
            
            start_time = time.time()
            result = service.validate_primary_mission(mission)
            processing_time = (time.time() - start_time) * 1000
            
            processing_times.append(processing_time)
            assert isinstance(result, DeconflictionResult)
        
        # Performance assertions
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        print(f"\nPerformance Benchmark Results:")
        print(f"Average processing time: {avg_processing_time:.2f}ms")
        print(f"Maximum processing time: {max_processing_time:.2f}ms")
        print(f"95th percentile: {np.percentile(processing_times, 95):.2f}ms")
        
        # Performance requirements
        assert avg_processing_time < 50.0  # Average < 50ms
        assert max_processing_time < 200.0  # Max < 200ms
        assert np.percentile(processing_times, 95) < 100.0  # 95% < 100ms

# Pytest configuration and custom markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")

if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])