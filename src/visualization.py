# src/visualization.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List
from .data_structures import Trajectory, Conflict


def create_4d_plot(trajectories: List, conflicts: List[Conflict], scenario_name: str, rejected_drone_ids: List[str] = None):
    """
    Generates an interactive 4D plot of drone trajectories and saves it as an HTML file.
    The 4th dimension (time) is represented by color gradient along continuous lines.
    Rejected trajectories are highlighted in RED.
    """
    print(f"--- Generating 4D Visualization for: {scenario_name} ---")

    if rejected_drone_ids is None:
        rejected_drone_ids = []

    # Create the main figure
    fig = go.Figure()

    # Define a color palette for approved drones
    approved_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']

    for idx, traj in enumerate(trajectories):
        if not traj.waypoints:
            continue

        print(f"Processing trajectory for drone: {traj.drone_id}")

        # Check if this drone is rejected
        is_rejected = traj.drone_id in rejected_drone_ids

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

        # Determine colors based on approval status
        if is_rejected:
            # REJECTED: Entire trajectory in RED
            line_color = 'red'
            colorscale = [[0, 'red'], [1, 'darkred']]  # Red gradient
            trajectory_name = f'❌ {traj.drone_id} (REJECTED)'
            line_width = 6  # Thicker for emphasis
            print(f"  🔴 Marking {traj.drone_id} as REJECTED (red trajectory)")
        else:
            # APPROVED: Normal time-based coloring
            line_color = traj_df['time']
            colorscale = 'Viridis'
            trajectory_name = f'✅ {traj.drone_id} (APPROVED)'
            line_width = 4

        # Add the main trajectory line
        fig.add_trace(go.Scatter3d(
            x=traj_df['x'],
            y=traj_df['y'],
            z=traj_df['z'],
            mode='lines+markers',
            line=dict(
                color=line_color,
                colorscale=colorscale,
                width=line_width,
                showscale=(not is_rejected),  # Only show colorbar for approved drones
                colorbar=dict(
                    title="Time (s)",
                    x=1.02
                ) if not is_rejected else None
            ),
            marker=dict(
                size=3 if is_rejected else 2,
                color=line_color,
                colorscale=colorscale,
                showscale=False
            ),
            name=trajectory_name,
            text=[f'{traj.drone_id}<br>{"❌ REJECTED" if is_rejected else "✅ APPROVED"}<br>t={t:.2f}s<br>pos=({x:.1f},{y:.1f},{z:.1f})' 
                  for t, x, y, z in zip(traj_df['time'], traj_df['x'], traj_df['y'], traj_df['z'])],
            hoverinfo='text'
        ))

        # Add start and end markers
        start_point = traj_df.iloc[0]
        end_point = traj_df.iloc[-1]

        # Start marker colors
        start_color = 'red' if is_rejected else 'green'
        start_border = 'darkred' if is_rejected else 'darkgreen'

        # Start marker
        fig.add_trace(go.Scatter3d(
            x=[start_point['x']],
            y=[start_point['y']],
            z=[start_point['z']],
            mode='markers',
            marker=dict(
                size=10 if is_rejected else 8,
                color=start_color,
                symbol='circle',
                line=dict(color=start_border, width=2)
            ),
            name=f'{trajectory_name} Start',
            text=f'{traj.drone_id} START<br>{"❌ REJECTED" if is_rejected else "✅ APPROVED"}<br>t={start_point["time"]:.2f}s',
            hoverinfo='text',
            showlegend=True
        ))

        # End marker  
        fig.add_trace(go.Scatter3d(
            x=[end_point['x']],
            y=[end_point['y']],
            z=[end_point['z']],
            mode='markers',
            marker=dict(
                size=10 if is_rejected else 8,
                color='red',
                symbol='circle',
                line=dict(color='darkred', width=2)
            ),
            name=f'{trajectory_name} End',
            text=f'{traj.drone_id} END<br>{"❌ REJECTED" if is_rejected else "✅ APPROVED"}<br>t={end_point["time"]:.2f}s',
            hoverinfo='text',
            showlegend=True
        ))

    # Add conflict markers if any exist
    if conflicts:
        print(f"Adding {len(conflicts)} conflict markers")

        # Separate spawn conflicts from trajectory conflicts
        spawn_conflicts = []
        trajectory_conflicts = []

        for conflict in conflicts:
            # Check if this is a spawn conflict (time very close to 0)
            if abs(conflict.time_of_conflict) < 0.1:  # Within 0.1s of spawn
                spawn_conflicts.append(conflict)
            else:
                trajectory_conflicts.append(conflict)

        # Add spawn conflict markers (different style)
        if spawn_conflicts:
            spawn_x = [c.location_of_conflict[0] for c in spawn_conflicts]
            spawn_y = [c.location_of_conflict[1] for c in spawn_conflicts]
            spawn_z = [c.location_of_conflict[2] for c in spawn_conflicts]
            spawn_text = [f"🚫 SPAWN CONFLICT 🚫<br>Drones: {c.conflicting_drone_ids}<br>Spawn Time: {c.time_of_conflict:.2f}s<br>Distance: {c.minimum_separation:.2f}m" 
                         for c in spawn_conflicts]

            fig.add_trace(go.Scatter3d(
                x=spawn_x,
                y=spawn_y,
                z=spawn_z,
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    color='orange',
                    size=20,
                    line=dict(color='red', width=4)
                ),
                text=spawn_text,
                hoverinfo='text',
                name='🚫 SPAWN CONFLICTS',
                showlegend=True
            ))

        # Add trajectory conflict markers
        if trajectory_conflicts:
            traj_conflict_x = [c.location_of_conflict[0] for c in trajectory_conflicts]
            traj_conflict_y = [c.location_of_conflict[1] for c in trajectory_conflicts]
            traj_conflict_z = [c.location_of_conflict[2] for c in trajectory_conflicts]
            traj_conflict_text = [f"⚠️ PATH CONFLICT ⚠️<br>Drones: {c.conflicting_drone_ids}<br>Time: {c.time_of_conflict:.2f}s<br>Separation: {c.minimum_separation:.2f}m" 
                                 for c in trajectory_conflicts]

            fig.add_trace(go.Scatter3d(
                x=traj_conflict_x,
                y=traj_conflict_y,
                z=traj_conflict_z,
                mode='markers',
                marker=dict(
                    symbol='x',
                    color='red',
                    size=15,
                    line=dict(color='darkred', width=4)
                ),
                text=traj_conflict_text,
                hoverinfo='text',
                name='⚠️ PATH CONFLICTS',
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
            bgcolor='rgba(255,255,255,0.9)',
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
    print(f"   - Approved trajectories: Normal time-based coloring")
    print(f"   - Rejected trajectories: RED highlighting")
    print(f"   - {len(trajectories)} drone trajectories plotted")
    print(f"   - {len(conflicts)} conflicts marked")
    if spawn_conflicts:
        print(f"   - {len(spawn_conflicts)} spawn conflicts (orange diamonds)")
    if trajectory_conflicts:
        print(f"   - {len(trajectory_conflicts)} path conflicts (red X)")
    print()

    return filename
