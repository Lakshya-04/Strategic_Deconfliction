# src/visualization.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List
from .data_structures import Trajectory, Conflict


def create_4d_plot(trajectories: List, conflicts: List[Conflict], scenario_name: str):
    """
    Generates an interactive 4D plot of drone trajectories and saves it as an HTML file.
    The 4th dimension (time) is represented by color gradient along continuous lines.
    """
    print(f"--- Generating 4D Visualization for: {scenario_name} ---")

    # Create the main figure
    fig = go.Figure()

    # Define a color palette for different drones
    drone_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']

    for idx, traj in enumerate(trajectories):
        if not traj.waypoints:
            continue

        print(f"Processing trajectory for drone: {traj.drone_id}")

        # Determine the full time range for this trajectory
        start_time = traj.waypoints[0].timestamp
        end_time = traj.waypoints[-1].timestamp

        # Create dense time points for smooth interpolation (every 0.05 seconds)
        time_step = 0.05
        time_points = np.arange(start_time, end_time + time_step, time_step)

        # Collect trajectory data
        trajectory_data = []
        for t in time_points:
            position = traj.get_state_at_time(t)
            if position is not None:
                trajectory_data.append({
                    'x': position[0],
                    'y': position[1], 
                    'z': position[2],
                    'time': t
                })

        if not trajectory_data:
            print(f"Warning: No interpolated points for {traj.drone_id}")
            continue

        # Convert to DataFrame for this trajectory
        traj_df = pd.DataFrame(trajectory_data)
        print(f"  Generated {len(traj_df)} points for {traj.drone_id}")

        # Create continuous line with time-based color gradient
        drone_color = drone_colors[idx % len(drone_colors)]

        # Add the main trajectory line with time coloring
        fig.add_trace(go.Scatter3d(
            x=traj_df['x'],
            y=traj_df['y'],
            z=traj_df['z'],
            mode='lines+markers',
            line=dict(
                color=traj_df['time'], 
                colorscale='Viridis',
                width=4,
                showscale=True,
                colorbar=dict(
                    title="Time (s)",
                    x=1.02
                )
            ),
            marker=dict(
                size=2,
                color=traj_df['time'],
                colorscale='Viridis',
                showscale=False
            ),
            name=f'{traj.drone_id} Trajectory',
            text=[f'{traj.drone_id}<br>t={t:.2f}s<br>pos=({x:.1f},{y:.1f},{z:.1f})' 
                  for t, x, y, z in zip(traj_df['time'], traj_df['x'], traj_df['y'], traj_df['z'])],
            hoverinfo='text'
        ))

        # Add prominent start and end markers
        start_point = traj_df.iloc[0]
        end_point = traj_df.iloc[-1]

        # Start marker (green sphere)
        fig.add_trace(go.Scatter3d(
            x=[start_point['x']],
            y=[start_point['y']],
            z=[start_point['z']],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='circle',
                line=dict(color='darkgreen', width=2)
            ),
            name=f'{traj.drone_id} Start',
            text=f'{traj.drone_id} START<br>t={start_point["time"]:.2f}s',
            hoverinfo='text',
            showlegend=True
        ))

        # End marker (red sphere) 
        fig.add_trace(go.Scatter3d(
            x=[end_point['x']],
            y=[end_point['y']],
            z=[end_point['z']],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                line=dict(color='darkred', width=2)
            ),
            name=f'{traj.drone_id} End',
            text=f'{traj.drone_id} END<br>t={end_point["time"]:.2f}s',
            hoverinfo='text',
            showlegend=True
        ))

    # Add conflict markers if any exist
    if conflicts:
        print(f"Adding {len(conflicts)} conflict markers")

        conflict_x = [c.location_of_conflict[0] for c in conflicts]
        conflict_y = [c.location_of_conflict[1] for c in conflicts]
        conflict_z = [c.location_of_conflict[2] for c in conflicts]
        conflict_text = [f"⚠️ CONFLICT ⚠️<br>Drones: {c.conflicting_drone_ids}<br>Time: {c.time_of_conflict:.2f}s<br>Separation: {c.minimum_separation:.2f}m" 
                        for c in conflicts]

        fig.add_trace(go.Scatter3d(
            x=conflict_x,
            y=conflict_y,
            z=conflict_z,
            mode='markers',
            marker=dict(
                symbol='x',
                color='red',
                size=15,
                line=dict(color='darkred', width=4)
            ),
            text=conflict_text,
            hoverinfo='text',
            name='⚠️ CONFLICTS ⚠️',
            showlegend=True
        ))

    # Improve layout and styling
    fig.update_layout(
        title={
            'text': f'4D Spatio-Temporal Visualization: {scenario_name}',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)', 
            zaxis_title='Z (meters)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.8)',
            borderwidth=1
        ),
        width=1200,
        height=800
    )

    # Save to HTML file
    filename = f"{scenario_name.replace(' ', '_').lower()}_4d_plot.html"
    fig.write_html(filename)
    print(f"✅ Enhanced 4D Visualization saved to '{filename}'")
    print(f"   - Continuous trajectory lines with time-based color gradients")
    print(f"   - {len(trajectories)} drone trajectories plotted")
    print(f"   - {len(conflicts)} conflicts marked")
    print()

    return filename
