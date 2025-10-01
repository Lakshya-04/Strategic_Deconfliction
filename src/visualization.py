# src/visualization.py

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import List, Optional
from .data_structures import Trajectory, Conflict
from typing import Union, List, Any

def create_trajectory_visualization(trajectories: List[Trajectory], 
                                  conflicts: List[Conflict], 
                                  scenario_name: str,
                                  primary_drone_id: Optional[str] = None) -> str:
    """
    /// @brief Generate interactive 4D visualization of drone trajectories with conflicts
    /// 
    /// @param trajectories List of all drone trajectories to visualize
    /// @param conflicts List of detected conflicts to highlight
    /// @param scenario_name Name of the scenario for title and filename
    /// @param primary_drone_id ID of primary mission drone (highlighted differently)
    /// @return Filename of the generated HTML visualization
    /// 
    /// Creates comprehensive interactive visualization with:
    /// - 3D trajectory paths with time-based color coding
    /// - Conflict markers with detailed hover information
    /// - Primary mission highlighting in distinct colors
    /// - Start/end markers for all trajectories
    /// - Legend and interactive controls
    """
    print(f"📊 Generating 4D visualization for: {scenario_name}")
    
    # Create main 3D scatter plot
    fig = go.Figure()
    
    # Define color schemes
    primary_colors = ['red', 'darkred']  # Primary mission in red
    sim_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    
    # Process each trajectory
    for traj_idx, trajectory in enumerate(trajectories):
        if not trajectory.waypoints:
            continue
            
        print(f"  Processing trajectory: {trajectory.drone_id}")
        
        # Determine if this is the primary mission
        is_primary = (primary_drone_id and trajectory.drone_id == primary_drone_id) or \
                    (primary_drone_id is None and 'PRIMARY' in trajectory.drone_id.upper())
        
        # Generate dense interpolated points for smooth visualization
        time_start = trajectory.waypoints[0].timestamp
        time_end = trajectory.waypoints[-1].timestamp
        time_points = np.arange(time_start, time_end + 0.1, 0.1)  # 0.1s resolution
        
        # Collect trajectory data points
        traj_data = []
        for t in time_points:
            position = trajectory.get_state_at_time(t)
            if position is not None:
                traj_data.append({
                    'x': position[0], 'y': position[1], 'z': position[2], 'time': t
                })
        
        if not traj_data:
            continue
            
        traj_df = pd.DataFrame(traj_data)
        colorscale: Union[str, List[List[Any]]]
        
        # Configure colors and styling based on drone type
        if is_primary:
            line_color = traj_df['time']
            colorscale = [[0, 'red'], [1, 'darkred']]
            line_width = 8
            marker_size = 4
            name_prefix = "🎯 PRIMARY"
        else:
            line_color = traj_df['time']
            colorscale = 'Viridis'
            line_width = 5
            marker_size = 3
            name_prefix = "📡 SIMULATED"
        
        # Add main trajectory line
        fig.add_trace(go.Scatter3d(
            x=traj_df['x'], y=traj_df['y'], z=traj_df['z'],
            mode='lines+markers',
            line=dict(
                color=line_color,
                colorscale=colorscale,
                width=line_width,
                showscale=is_primary  # Only show colorbar for primary mission
            ),
            marker=dict(
                size=marker_size,
                color=line_color,
                colorscale=colorscale,
                showscale=False,
                opacity=0.8
            ),
            name=f"{name_prefix}: {trajectory.drone_id}",
            text=[f'{trajectory.drone_id}<br>t={t:.1f}s<br>pos=({x:.0f},{y:.0f},{z:.0f})' 
                  for t, x, y, z in zip(traj_df['time'], traj_df['x'], traj_df['y'], traj_df['z'])],
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
        
        # Add start marker
        start_point = traj_df.iloc[0]
        start_color = 'darkred' if is_primary else 'darkgreen'
        fig.add_trace(go.Scatter3d(
            x=[start_point['x']], y=[start_point['y']], z=[start_point['z']],
            mode='markers',
            marker=dict(
                size=12 if is_primary else 10,
                color=start_color,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            name=f"🚀 START: {trajectory.drone_id}",
            text=f'START: {trajectory.drone_id}<br>t={start_point["time"]:.1f}s',
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
        
        # Add end marker
        end_point = traj_df.iloc[-1]
        end_color = 'red' if is_primary else 'blue'
        fig.add_trace(go.Scatter3d(
            x=[end_point['x']], y=[end_point['y']], z=[end_point['z']],
            mode='markers',
            marker=dict(
                size=12 if is_primary else 10,
                color=end_color,
                symbol='square',
                line=dict(color='white', width=2)
            ),
            name=f"🏁 END: {trajectory.drone_id}",
            text=f'END: {trajectory.drone_id}<br>t={end_point["time"]:.1f}s',
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
    
    # Add conflict markers
    if conflicts:
        print(f"  Adding {len(conflicts)} conflict markers")
        
        # Separate conflicts by type for different visualization
        spatial_conflicts = [c for c in conflicts if c.conflict_type == "spatial"]
        temporal_conflicts = [c for c in conflicts if c.conflict_type == "temporal"]
        
        # Add spatial conflict markers
        if spatial_conflicts:
            spatial_x = [c.location_of_conflict[0] for c in spatial_conflicts]
            spatial_y = [c.location_of_conflict[1] for c in spatial_conflicts] 
            spatial_z = [c.location_of_conflict[2] for c in spatial_conflicts]
            spatial_text = [
                f"⚠️ SPATIAL CONFLICT<br>"
                f"Drones: {c.conflicting_drone_ids[0]} vs {c.conflicting_drone_ids[1]}<br>"
                f"Time: {c.time_of_conflict:.1f}s<br>"
                f"Separation: {c.minimum_separation:.1f}m<br>"
                f"Severity: {c.severity.upper()}"
                for c in spatial_conflicts
            ]
            
            fig.add_trace(go.Scatter3d(
                x=spatial_x, y=spatial_y, z=spatial_z,
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=20,
                    color='red',
                    line=dict(color='darkred', width=4)
                ),
                text=spatial_text,
                hovertemplate='%{text}<extra></extra>',
                name="⚠️ SPATIAL CONFLICTS",
                showlegend=True
            ))
        
        # Add temporal conflict markers
        if temporal_conflicts:
            temporal_x = [c.location_of_conflict[0] for c in temporal_conflicts]
            temporal_y = [c.location_of_conflict[1] for c in temporal_conflicts]
            temporal_z = [c.location_of_conflict[2] for c in temporal_conflicts]
            temporal_text = [
                f"⏰ TEMPORAL CONFLICT<br>"
                f"Mission: {c.conflicting_drone_ids[0]}<br>"
                f"Issue: {c.conflicting_drone_ids[1]}<br>"
                f"Time: {c.time_of_conflict:.1f}s<br>"
                f"Severity: {c.severity.upper()}"
                for c in temporal_conflicts
            ]
            
            fig.add_trace(go.Scatter3d(
                x=temporal_x, y=temporal_y, z=temporal_z,
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=18,
                    color='orange',
                    line=dict(color='red', width=3)
                ),
                text=temporal_text,
                hovertemplate='%{text}<extra></extra>',
                name="⏰ TEMPORAL CONFLICTS",
                showlegend=True
            ))
    
    # Configure layout and styling
    fig.update_layout(
        title={
            'text': f'UAV Strategic Deconfliction: {scenario_name}',
            'x': 0.5,
            'y': 0.95,
            'font': {'size': 18, 'color': 'darkblue'}
        },
        scene=dict(
            xaxis_title='X Coordinate (meters)',
            yaxis_title='Y Coordinate (meters)',
            zaxis_title='Altitude Z (meters)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8)  # Optimal viewing angle
            ),
            aspectmode='cube',
            bgcolor='lightgray'
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1,
            font=dict(size=10)
        ),
        width=1400,
        height=900,
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    # Add informational annotations
    annotation_text = f"Safety Buffer: 10m | Trajectories: {len(trajectories)} | Conflicts: {len(conflicts)}"
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.5, y=0.02,
        showarrow=False,
        font=dict(size=12, color="darkblue"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="darkblue",
        borderwidth=1
    )
    
    # Generate filename and save
    filename = f"{scenario_name.replace(' ', '_').replace('-', '_').lower()}_visualization.html"
    fig.write_html(filename)
    
    # Generate summary
    print(f"✅ Visualization saved: {filename}")
    print(f"   • Trajectories visualized: {len(trajectories)}")
    print(f"   • Conflicts highlighted: {len(conflicts)}")
    if conflicts:
        spatial_count = len([c for c in conflicts if c.conflict_type == "spatial"])
        temporal_count = len([c for c in conflicts if c.conflict_type == "temporal"])
        print(f"   • Spatial conflicts: {spatial_count}")
        print(f"   • Temporal conflicts: {temporal_count}")
    print()
    
    return filename

def create_summary_dashboard(validation_results: List, output_filename: str = "validation_dashboard.html"):
    """
    /// @brief Create comprehensive dashboard summarizing all validation results
    /// 
    /// @param validation_results List of DeconflictionResult objects from all scenarios
    /// @param output_filename Output filename for the dashboard HTML
    /// @return Generated dashboard filename
    /// 
    /// Generates multi-panel dashboard showing:
    /// - Approval rate statistics
    /// - Conflict type distribution  
    /// - Processing time analysis
    /// - Scenario comparison charts
    """
    print(f"📈 Creating validation summary dashboard...")
    
    # Extract data for dashboard
    scenario_names = [f"Scenario {i+1}" for i in range(len(validation_results))]
    approval_status = [r.is_approved() for r in validation_results]
    conflict_counts = [len(r.conflicts) for r in validation_results]
    processing_times = [r.processing_time_ms for r in validation_results]
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Approval Status by Scenario', 'Conflicts Detected per Scenario',
                       'Processing Time Analysis', 'Conflict Type Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Approval status chart
    approval_colors = ['green' if approved else 'red' for approved in approval_status]
    fig.add_trace(
        go.Bar(x=scenario_names, y=[1 if a else 0 for a in approval_status],
               marker_color=approval_colors, name="Approved"),
        row=1, col=1
    )
    
    # Conflict count chart  
    fig.add_trace(
        go.Bar(x=scenario_names, y=conflict_counts,
               marker_color='orange', name="Conflicts"),
        row=1, col=2
    )
    
    # Processing time scatter
    fig.add_trace(
        go.Scatter(x=scenario_names, y=processing_times,
                  mode='markers+lines', marker_size=8, name="Processing Time"),
        row=2, col=1
    )
    
    # Conflict type pie chart
    spatial_conflicts = sum(1 for r in validation_results for c in r.conflicts if c.conflict_type == "spatial")
    temporal_conflicts = sum(1 for r in validation_results for c in r.conflicts if c.conflict_type == "temporal") 
    
    if spatial_conflicts + temporal_conflicts > 0:
        fig.add_trace(
            go.Pie(labels=['Spatial', 'Temporal'], values=[spatial_conflicts, temporal_conflicts],
                  name="Conflict Types"),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="UAV Deconfliction System - Validation Dashboard",
        showlegend=False,
        height=800,
        width=1200
    )
    
    # Save dashboard
    fig.write_html(output_filename)
    print(f"✅ Dashboard saved: {output_filename}")
    
    return output_filename