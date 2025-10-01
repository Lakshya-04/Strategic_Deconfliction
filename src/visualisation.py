# src/visualization.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List
from.data_structures import Trajectory, Conflict

def create_4d_plot(trajectories: List, conflicts: List[Conflict], scenario_name: str):
    """
    Generates an interactive 4D plot of drone trajectories and saves it as an HTML file.
    The 4th dimension (time) is represented by color.
    """
    print(f"--- Generating 4D Visualization for: {scenario_name} ---")

    plot_data = []
    time_step = 0.1  # Sample every 0.1 seconds for a smooth line

    for traj in trajectories:
        if not traj.waypoints:
            continue
            
        # Determine the full time range for this trajectory
        start_time = traj.waypoints[0].timestamp 
        end_time = traj.waypoints[-1].timestamp
        
        # Create a dense set of time points for interpolation
        time_points = np.arange(start_time, end_time + time_step, time_step)

        for t in time_points:
            position = traj.get_state_at_time(t)
            if position is not None:
                plot_data.append({
                    'drone_id': traj.drone_id,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'time': t
                })

    if not plot_data:
        print("Warning: No data to plot for visualization.")
        return

    df = pd.DataFrame(plot_data)

    # Create the 3D line plot with time as color
    fig = px.line_3d(df,
                     x='x',
                     y='y',
                     z='z',
                     color='time',
                     line_group='drone_id',
                     hover_name='drone_id',
                     labels={'time': 'Time (s)'},
                     title=f'4D Spatio-Temporal Visualization: {scenario_name}')

    # Add markers for the start and end of each trajectory
    for drone_id in df['drone_id'].unique():
        drone_df = df[df['drone_id'] == drone_id]
        start_point = drone_df.iloc[0]
        end_point = drone_df.iloc[-1]
        fig.add_trace(go.Scatter3d(
            x=[start_point['x'], end_point['x']],
            y=[start_point['y'], end_point['y']],
            z=[start_point['z'], end_point['z']],
            mode='markers',
            marker=dict(size=5, color='black', symbol='circle'),
            name=f'{drone_id} Start/End'
        ))

    # Add markers for detected conflicts
    if conflicts:
        conflict_x = [c.location_of_conflict[0] for c in conflicts]
        conflict_y = [c.location_of_conflict[1] for c in conflicts]
        conflict_z = [c.location_of_conflict[2] for c in conflicts]
        conflict_text = ""
        
        fig.add_trace(go.Scatter3d(
            x=conflict_x,
            y=conflict_y,
            z=conflict_z,
            mode='markers',
            marker=dict(
                symbol='x',
                color='red',
                size=10,
                line=dict(width=3)
            ),
            text=conflict_text,
            hoverinfo='text',
            name='Conflicts'
        ))

    # Improve layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)'
        ),
        legend_title="Legend"
    )

    # Save to an HTML file
    filename = f"{scenario_name.replace(' ', '_').lower()}_4d_plot.html"
    fig.write_html(filename)
    print(f"✅ Visualization saved to '{filename}'\n")