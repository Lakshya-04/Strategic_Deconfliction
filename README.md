# UAV Strategic Deconfliction System

## Overview

A comprehensive system for validating UAV missions in shared airspace by detecting spatial-temporal conflicts against existing flight schedules. This service is the final authority for verifying whether a drone’s planned waypoint mission is safe to execute.

## Features

✈️ **Core Functionality**

- Spatial conflict detection with adjustable safety buffers
- Temporal overlap analysis across overlapping time segments
- Mission time-window validation
- Detailed conflict reporting (locations, times, explanations)

🎯 **Primary Mission Validation**

- Waypoint-based, multi-segment trajectories
- Full 3D (x, y, z) coordinates with altitude awareness
- Time-window constraints on mission start and end
- Simple API returning status and conflict details

📡 **Simulated Flight Integration**

- In-memory flight schedule database
- Support for different flight priorities (emergency, commercial, recreational)
- Detection of spatial, temporal, and window-violation conflicts
- Scalable design for high-density airspace

📊 **Visualization \& Reporting**

- 4D trajectory rendering with time-based color coding
- Conflict markers in interactive plots
- Exportable HTML visualizations and detailed reports


## Installation

1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the package (reads dependencies from `pyproject.toml`)

```bash
pip3 install .
```

3. Verify installation by running the demo script:

```bash
python3 -m src.main
```


## Quick Start

### Basic Usage

```python
import numpy as np
from data_structures import Waypoint, Trajectory, PrimaryMission, SimulatedFlightSchedule
from deconfliction_service import UAVDeconflictionService

# Initialize service with a 10 m safety buffer
service = UAVDeconflictionService(default_safety_buffer_m=10.0)

# Add an existing flight
existing = SimulatedFlightSchedule(
    trajectory=Trajectory(
        drone_id="DELIVERY_001",
        waypoints=[
            Waypoint(position=np.array([0, 100, 50]), timestamp=0),
            Waypoint(position=np.array([1000, 100, 50]), timestamp=60)
        ]
    ),
    flight_type="commercial"
)
service.add_simulated_flight_schedule(existing)

# Define primary mission
primary = PrimaryMission(
    trajectory=Trajectory(
        drone_id="PRIMARY_001",
        waypoints=[
            Waypoint(position=np.array([0, 0, 50]), timestamp=0),
            Waypoint(position=np.array([1000, 0, 50]), timestamp=60)
        ]
    ),
    mission_start_time=0,
    mission_end_time=120
)

# Validate
result = service.validate_primary_mission(primary)
if result.is_approved():
    print("✅ Mission APPROVED")
else:
    print("❌ Mission REJECTED:")
    for conflict in result.conflicts:
        print(f"  • {conflict.description}")
```


### Advanced Scenario Testing

```python
from main import create_test_scenarios, run_validation_scenario

# Load all scenarios
scenarios = create_test_scenarios()

# Run and visualize scenario #1 (spatial conflict)
result = run_validation_scenario(scenarios[1], visualize=True)
print(f"Status: {result.status}")
print(f"Conflicts: {len(result.conflicts)}")
print(f"Time: {result.processing_time_ms:.2f} ms")
```


## Architecture

```
.
├── data_structures.py       # Waypoint, Trajectory, PrimaryMission, Conflict, etc.
├── deconfliction_service.py # UAVDeconflictionService engine
├── trajectory_animated.py   # Animated 4D trajectory visualizations
├── visualization.py         # Static plot and dashboard utilities
└── main.py                  # Demo entry point & test scenarios
```


### Core Modules

- **data_structures.py**
Defines `Waypoint`, `Trajectory`, `PrimaryMission`, `SimulatedFlightSchedule`, `Conflict`, and `DeconflictionResult`.
- **deconfliction_service.py**
Implements `UAVDeconflictionService` with spatial-temporal algorithms, mission window checks, and performance metrics.
- **trajectory_animated.py**
Creates interactive, animated 4D trajectory plots.
- **visualization.py**
Generates static plots, conflict markers, and summary dashboards.
- **main.py**
Provides demonstration scenarios, scenario loader, and `run_validation_scenario()` with optional visualization.

## License

MIT License – see `LICENSE` for details.

