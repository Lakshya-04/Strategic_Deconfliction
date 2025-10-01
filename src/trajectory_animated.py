# src/trajectory_animation.py

import numpy as np
import matplotlib.collections
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Dict, Tuple
import time
import os
from .data_structures import Trajectory, Conflict


class TrajectoryAnimator:
    """
    /// @brief 3D Trajectory Animation System for UAV Strategic Deconfliction
    /// 
    /// Creates real-time animated visualizations of multiple drone trajectories
    /// with conflict highlighting, safety buffers, and comprehensive visual effects.
    """
    
    def __init__(self, trajectories: List[Trajectory], conflicts: List[Conflict], 
                 scenario_name: str, primary_drone_id: Optional[str] = None):
        """
        Initialize the trajectory animator.
        
        Args:
            trajectories: List of all drone trajectories to animate
            conflicts: List of detected conflicts to highlight
            scenario_name: Name of the scenario for title and filename
            primary_drone_id: ID of primary mission drone (highlighted differently)
        """
        self.trajectories = trajectories
        self.conflicts = conflicts
        self.scenario_name = scenario_name
        self.primary_drone_id = primary_drone_id
        
        # Animation parameters
        self.time_step = 0.1  # seconds per frame
        self.animation_speed = 1.0  # playback speed multiplier
        self.trail_length = 50  # number of points in trail
        self.safety_buffer = 10.0  # meters
        
        # Color schemes
        self.primary_colors = ['red', 'darkred']
        self.sim_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta']
        self.conflict_color = 'red'
        self.trail_alpha = 0.6
        
        # Calculate time bounds
        self.start_time = min(traj.waypoints[0].timestamp for traj in trajectories if traj.waypoints)
        self.end_time = max(traj.waypoints[-1].timestamp for traj in trajectories if traj.waypoints)
        self.total_frames = int((self.end_time - self.start_time) / self.time_step) + 1
        
        # Initialize figure and axis
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Store plot objects for each drone
        self.drone_objects: Dict[str, matplotlib.collections.PathCollection] = {}
        self.trail_objects: Dict[str, matplotlib.lines.Line2D] = {}
        self.conflict_markers: List[matplotlib.collections.PathCollection] = []
        
        self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup the 3D visualization environment."""
        print(f"🎬 Setting up 3D trajectory animation for: {self.scenario_name}")
        
        # Calculate plot boundaries
        all_positions = []
        for traj in self.trajectories:
            for wp in traj.waypoints:
                all_positions.append(wp.position)
        
        if all_positions:
            all_pos_array = np.array(all_positions)
            x_min, x_max = all_pos_array[:, 0].min() - 50, all_pos_array[:, 0].max() + 50
            y_min, y_max = all_pos_array[:, 1].min() - 50, all_pos_array[:, 1].max() + 50
            z_min, z_max = all_pos_array[:, 2].min() - 20, all_pos_array[:, 2].max() + 20
            
            # Set axis limits
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
        
        # Configure axis properties
        self.ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        self.ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        self.ax.set_zlabel('Altitude Z (meters)', fontsize=12)
        self.ax.set_title(f'UAV Strategic Deconfliction: {self.scenario_name}\\nReal-time Trajectory Animation', 
                         fontsize=14, pad=20)
        
        # Set viewing angle for optimal perspective
        self.ax.view_init(elev=20, azim=45)
        
        # Initialize drone visualization objects
        for i, traj in enumerate(self.trajectories):
            if not traj.waypoints:
                continue
                
            is_primary = (self.primary_drone_id and traj.drone_id == self.primary_drone_id) or \
                        (self.primary_drone_id is None and 'PRIMARY' in traj.drone_id.upper())
            
            # Choose colors
            if is_primary:
                drone_color = 'red'
                trail_color = 'darkred'
                marker_size = 200
                marker_symbol = '^'  # Triangle for primary
            else:
                color_idx = i % len(self.sim_colors)
                drone_color = self.sim_colors[color_idx]
                trail_color = drone_color
                marker_size = 120
                marker_symbol = 'o'  # Circle for simulated
            
            # Create drone marker (current position)
            drone_scatter = self.ax.scatter([], [], [], 
                                          c=drone_color, s=marker_size, 
                                          marker=marker_symbol, alpha=0.9,
                                          edgecolors='black', linewidths=2)
            
            # Create trail line (path history)
            trail_line, = self.ax.plot([], [], [], 
                                     color=trail_color, alpha=self.trail_alpha, 
                                     linewidth=3 if is_primary else 2)
            
            # Store objects
            self.drone_objects[traj.drone_id] = drone_scatter
            self.trail_objects[traj.drone_id] = trail_line
        
        # Add legend
        self._create_legend()
        
        # Add time display
        self.time_text = self.fig.text(0.02, 0.98, '', fontsize=14, 
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                      transform=self.ax.transAxes)
        
        # Add scenario info
        info_text = f"Trajectories: {len(self.trajectories)} | Conflicts: {len(self.conflicts)} | Safety Buffer: {self.safety_buffer}m"
        self.fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _create_legend(self):
        """Create a comprehensive legend for the animation."""
        legend_elements = []
        
        # Add drone types to legend
        primary_found = False
        sim_count = 0
        
        for traj in self.trajectories:
            is_primary = (self.primary_drone_id and traj.drone_id == self.primary_drone_id) or \
                        (self.primary_drone_id is None and 'PRIMARY' in traj.drone_id.upper())
            
            if is_primary and not primary_found:
                legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                                markerfacecolor='red', markersize=12,
                                                label='🎯 Primary Mission', markeredgecolor='black'))
                primary_found = True
            elif not is_primary and sim_count < 3:  # Show max 3 simulated drone types in legend
                color = self.sim_colors[sim_count % len(self.sim_colors)]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, markersize=10,
                                                label=f'📡 Simulated Drone {sim_count+1}', markeredgecolor='black'))
                sim_count += 1
        
        # Add conflict indicator
        if self.conflicts:
            legend_elements.append(plt.Line2D([0], [0], marker='X', color='w',
                                            markerfacecolor='red', markersize=15,
                                            label='⚠️ Conflict Zone', markeredgecolor='darkred'))
        
        # Create legend
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    def _update_frame(self, frame: int) -> List:
        """
        Update animation frame.
        
        Args:
            frame: Current frame number
            
        Returns:
            List of updated plot objects
        """
        current_time = self.start_time + frame * self.time_step
        updated_objects = []
        
        # Update time display
        self.time_text.set_text(f'Time: {current_time:.1f}s / {self.end_time:.1f}s')
        
        # Update each drone
        for traj in self.trajectories:
            if not traj.waypoints or traj.drone_id not in self.drone_objects:
                continue
                
            # Get current position
            current_pos = traj.get_state_at_time(current_time)
            
            if current_pos is not None:
                # Update drone marker position
                drone_obj = self.drone_objects[traj.drone_id]
                drone_obj._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
                updated_objects.append(drone_obj)
                
                # Update trail (last N positions)
                trail_times = np.arange(max(self.start_time, current_time - self.trail_length * self.time_step), 
                                      current_time + self.time_step, self.time_step)
                trail_positions = []
                
                for t in trail_times:
                    pos = traj.get_state_at_time(t)
                    if pos is not None:
                        trail_positions.append(pos)
                
                if trail_positions:
                    trail_pos_array = np.array(trail_positions)
                    trail_obj = self.trail_objects[traj.drone_id]
                    trail_obj.set_data_3d(trail_pos_array[:, 0], 
                                         trail_pos_array[:, 1], 
                                         trail_pos_array[:, 2])
                    updated_objects.append(trail_obj)
        
        # Update conflict markers (show conflicts that are active at current time)
        # Clear previous conflict markers
        for marker in self.conflict_markers:
            marker.remove()
        self.conflict_markers.clear()
        
        # Add current active conflicts
        for conflict in self.conflicts:
            time_tolerance = 2.0  # Show conflict for ±2 seconds around conflict time
            if abs(current_time - conflict.time_of_conflict) <= time_tolerance:
                conflict_marker = self.ax.scatter([conflict.location_of_conflict[0]], 
                                                [conflict.location_of_conflict[1]], 
                                                [conflict.location_of_conflict[2]],
                                                c='red', s=400, marker='X', 
                                                alpha=0.8, edgecolors='darkred', linewidths=3)
                self.conflict_markers.append(conflict_marker)
                updated_objects.append(conflict_marker)
                
                # Add safety buffer visualization around conflict
                if conflict.conflict_type == "spatial":
                    self._add_safety_sphere(conflict.location_of_conflict, self.safety_buffer, current_time)
        
        return updated_objects
    
    def _add_safety_sphere(self, center: np.ndarray, radius: float, current_time: float):
        """Add a semi-transparent safety buffer sphere around conflict location."""
        # Create a wireframe sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        # Add wireframe sphere (only a few circles for performance)
        for i in range(0, 20, 4):  # Every 4th circle
            sphere_circle = self.ax.plot(x[i, :], y[i, :], z[i, :], 
                                       color='red', alpha=0.3, linewidth=1)
            self.conflict_markers.extend(sphere_circle)
    
    def create_animation(self, save_video: bool = True, fps: int = 10, 
                        video_format: str = 'mp4') -> animation.FuncAnimation:
        """
        Create and optionally save the trajectory animation.
        
        Args:
            save_video: Whether to save animation as video file
            fps: Frames per second for video output
            video_format: Video format ('mp4' or 'avi')
            
        Returns:
            Matplotlib FuncAnimation object
        """
        print(f"🎬 Creating trajectory animation...")
        print(f"   • Duration: {self.end_time - self.start_time:.1f} seconds")
        print(f"   • Total frames: {self.total_frames}")
        print(f"   • Frame rate: {fps} FPS")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self._update_frame, frames=self.total_frames,
            interval=int(1000/fps), blit=False, repeat=True
        )
        
        # Save video if requested
        if save_video:
            filename = f"{self.scenario_name.replace(' ', '_').replace('-', '_').lower()}_animation.{video_format}"
            print(f"💾 Saving animation video: {filename}")
            
            try:
                # Configure video writer
                if video_format.lower() == 'mp4':
                    writer = animation.FFMpegWriter(fps=fps, bitrate=5000, codec='h264')
                else:
                    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
                
                # Save with progress indication
                start_time = time.time()
                anim.save(filename, writer=writer, dpi=100)
                
                save_time = time.time() - start_time
                print(f"\\n✅ Animation video saved: {filename}")
                print(f"   • File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
                print(f"   • Encoding time: {save_time:.1f} seconds")
                
            except Exception as e:
                print(f"❌ Error saving video: {e}")
                print("   Tip: Make sure ffmpeg is installed and accessible")
        
        return anim
    
    def show_animation(self):
        """Display the animation in an interactive window."""
        print("🖥️ Displaying interactive animation...")
        print("   Controls:")
        print("     • Mouse: Rotate view")
        print("     • Mouse wheel: Zoom in/out") 
        print("     • Close window to stop animation")
        
        # Show the plot
        plt.tight_layout()
        plt.show()


def create_trajectory_animation(trajectories: List[Trajectory], 
                              conflicts: List[Conflict], 
                              scenario_name: str,
                              primary_drone_id: Optional[str] = None,
                              save_video: bool = True,
                              show_interactive: bool = True,
                              fps: int = 10) -> str:
    """
    /// @brief Main function to create animated 3D trajectory visualization
    /// 
    /// @param trajectories List of all drone trajectories to animate
    /// @param conflicts List of detected conflicts to highlight
    /// @param scenario_name Name of the scenario for title and filename
    /// @param primary_drone_id ID of primary mission drone (highlighted differently)
    /// @param save_video Whether to save animation as MP4 video
    /// @param show_interactive Whether to display interactive animation window
    /// @param fps Frames per second for video output
    /// @return Filename of the generated video (if saved)
    /// 
    /// Creates comprehensive 4D animated visualization with:
    /// - Real-time drone movement with position markers
    /// - Dynamic trail visualization showing flight history
    /// - Conflict highlighting with safety buffer zones
    /// - Interactive 3D viewing controls
    /// - High-quality MP4 video export
    """
    
    if not trajectories:
        print("⚠️ No trajectories provided for animation")
        return ""
    
    # Create animator
    animator = TrajectoryAnimator(trajectories, conflicts, scenario_name, primary_drone_id)
    
    # Create animation
    anim = animator.create_animation(save_video=save_video, fps=fps)
    
    # Show interactive animation if requested
    if show_interactive:
        animator.show_animation()
    
    # Return video filename
    if save_video:
        video_filename = f"{scenario_name.replace(' ', '_').replace('-', '_').lower()}_animation.mp4"
        return video_filename
    
    return ""


def create_comparison_animation(scenario_results: List[Dict], 
                              output_filename: str = "comparison_animation.mp4") -> str:
    """
    /// @brief Create side-by-side comparison animation of multiple scenarios
    /// 
    /// @param scenario_results List of scenario result dictionaries
    /// @param output_filename Output filename for comparison video
    /// @return Generated comparison video filename
    /// 
    /// Creates a multi-panel animation comparing different deconfliction scenarios.
    """
    print(f"🎬 Creating scenario comparison animation...")
    
    num_scenarios = len(scenario_results)
    if num_scenarios == 0:
        print("⚠️ No scenarios provided for comparison")
        return ""
    
    # Create subplot layout
    cols = min(2, num_scenarios)
    rows = (num_scenarios + 1) // 2
    
    fig = plt.figure(figsize=(8*cols, 6*rows))
    
    animators = []
    
    for i, scenario in enumerate(scenario_results, 1):
        ax = fig.add_subplot(rows, cols, i, projection='3d')
        
        # Create mini animator for this subplot
        mini_animator = TrajectoryAnimator(
            scenario['trajectories'], 
            scenario['conflicts'], 
            scenario['name']
        )
        mini_animator.ax = ax
        mini_animator._setup_visualization()
        animators.append(mini_animator)
    
    def update_comparison(frame):
        updated_objects = []
        for animator in animators:
            updated_objects.extend(animator._update_frame(frame))
        return updated_objects
    
    # Calculate total frames based on longest scenario
    total_frames = max(animator.total_frames for animator in animators)
    
    # Create comparison animation
    anim = animation.FuncAnimation(
        fig, update_comparison, frames=total_frames,
        interval=100, blit=False, repeat=True
    )
    
    # Save comparison video
    print(f"💾 Saving comparison animation: {output_filename}")
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=8000, codec='h264')
        anim.save(output_filename, writer=writer, dpi=100)
        print(f"✅ Comparison animation saved: {output_filename}")
    except Exception as e:
        print(f"❌ Error saving comparison video: {e}")
    
    return output_filename