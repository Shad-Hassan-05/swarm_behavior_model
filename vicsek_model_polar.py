import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio
import os
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from pathlib import Path

class VicsekModel:
    def __init__(self, n_particles=150, size=700, radius=50, noise=0.5, speed=3, 
                 inertia=0.3, dt=1, use_noise=True, use_inertia=True, 
                 save_animation=False, noise_type='discrete', model_name="Default",
                 use_external_force=False, use_wave=False, wave_period=200,
                 save_dir=None, n_ghost_particles=0, ghost_circle_radius=100, 
                 ghost_speed=2.0, ghost_color=(255, 0, 255)):
        """
        Vicsek Model with configurable noise and inertia, using polar plot for visualization
        
        Parameters:
        -----------
        n_particles : int
            Number of particles in the simulation
        size : int
            Size of the simulation area
        radius : int
            Radius of interaction between particles
        noise : float
            Noise level in the system
        speed : float
            Speed of particles
        inertia : float
            Inertia factor in the system
        dt : float
            Time step for the simulation
        use_noise : bool
            Whether to use noise in the system
        use_inertia : bool
            Whether to use inertia in the system
        save_animation : bool
            Whether to save frames of the simulation
        noise_type : str
            Type of noise to use ('discrete' or 'stochastic')
        model_name : str
            Name of the model
        use_external_force : bool
            Whether to use external force in the system
        use_wave : bool
            Whether to use wave alignment in the system
        wave_period : int
            Period of the wave in the system
        save_dir : str or None
            Directory where animations will be saved (default: Desktop/vicsek_animations)
        n_ghost_particles : int
            Number of ghost particles that move in circles
        ghost_circle_radius : float
            Radius of the circle that ghost particles move in
        ghost_speed : float
            Speed of ghost particles
        ghost_color : tuple
            RGB color for ghost particles
        """
        # Set default save directory to Desktop/vicsek_animations if not specified
        if save_dir is None:
            desktop_path = str(Path.home() / "Desktop" / "vicsek_animations")
            self.save_dir = desktop_path
        else:
            self.save_dir = save_dir
        
        # Simulation parameters
        self.n_particles = n_particles
        self.size = size
        self.radius = radius
        self.noise = noise
        self.speed = speed
        self.inertia = inertia
        self.dt = dt
        self.use_noise = use_noise
        self.use_inertia = use_inertia
        self.save_animation = save_animation
        self.noise_type = noise_type
        self.model_name = model_name
        self.use_external_force = use_external_force
        self.use_wave = use_wave
        self.wave_period = wave_period
        self.wave_amplitude = np.pi/2
        self.center = np.array([size/2, size/2])
        
        # Ghost particle parameters
        self.n_ghost_particles = n_ghost_particles
        self.ghost_circle_radius = ghost_circle_radius
        self.ghost_speed = ghost_speed
        self.ghost_color = ghost_color
        self.ghost_angles = np.linspace(0, 2*np.pi, n_ghost_particles, endpoint=False)
        self.ghost_angular_velocities = np.full(n_ghost_particles, ghost_speed / ghost_circle_radius)
        
        # Initialize particle states
        self.positions = np.random.uniform(0, size, (n_particles, 2))
        self.angles = np.random.uniform(0, 2*np.pi, n_particles)
        self.angular_velocities = np.zeros(n_particles)
        
        # Initialize ghost particle positions
        if n_ghost_particles > 0:
            self.ghost_positions = np.zeros((n_ghost_particles, 2))
            self.update_ghost_positions()
        else:
            self.ghost_positions = np.zeros((0, 2))
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((size * 2, size))
        title = "Vicsek Model with Polar Plot"
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        
        # Font for displaying state
        self.font = pygame.font.Font(None, 36)
        
        # Setup matplotlib figure for polar plotting
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.canvas = FigureCanvasAgg(self.fig)
        
        # Create a cached surface for the plot
        self.plot_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.last_plot_update = 0
        
        # Data recording for polar plot
        self.avg_velocities = []
        self.frame_count = 0
        
        # Setup for saving frames
        if self.save_animation:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.frame_dir = f"frames_{self.timestamp}"
            os.makedirs(self.frame_dir, exist_ok=True)
            self.frames = []
        
        # Dictionary of important parameters
        self.important_params = {
            "Edge of Chaos": ["noise", "inertia"],
            "Highly Aligned Swarm": ["radius", "noise", "inertia"],
            "Turbulent Dynamics": ["radius", "noise", "speed"],
            "Dense Clustering": ["n_particles", "speed"],
            "Chaotic Mixing": ["noise", "speed", "inertia"],
            "Spiral Swarm": ["speed", "inertia", "noise"],
            "Wave Swarm": ["speed", "wave_period", "inertia"],
            "Default": ["noise", "inertia", "noise_type"]
        }

    def apply_noise(self):
        """Apply noise based on selected type"""
        if not self.use_noise:
            return np.zeros(self.n_particles)
            
        if self.noise_type == 'discrete':
            return self.noise * np.random.uniform(-np.pi, np.pi, self.n_particles)
        else:  # stochastic
            return self.noise * np.sqrt(self.dt) * np.random.normal(0, 1, self.n_particles)
    
    def apply_external_force(self):
        """Apply external force for spiral motion"""
        if not self.use_external_force:
            return np.zeros_like(self.angles)
        
        rel_pos = self.positions - self.center
        distances = np.linalg.norm(rel_pos, axis=1)
        
        velocities = np.column_stack([
            np.cos(self.angles),
            np.sin(self.angles)
        ])
        
        tangential = np.column_stack([
            -rel_pos[:, 1],
            rel_pos[:, 0]
        ])
        tangential /= np.linalg.norm(tangential, axis=1)[:, np.newaxis]
        
        radial = -rel_pos / distances[:, np.newaxis]
        
        spiral_vel = tangential + 0.3 * radial
        spiral_vel /= np.linalg.norm(spiral_vel, axis=1)[:, np.newaxis]
        
        target_angles = np.arctan2(spiral_vel[:, 1], spiral_vel[:, 0])
        angle_diff = (target_angles - self.angles + np.pi) % (2 * np.pi) - np.pi
        
        return angle_diff * 0.1
    
    def apply_wave_alignment(self):
        """Apply sinusoidal wave pattern to particle alignment"""
        if not self.use_wave:
            return np.zeros_like(self.angles)
        
        phase = 2 * np.pi * self.frame_count / self.wave_period
        x_normalized = self.positions[:, 0] / self.size
        y_normalized = self.positions[:, 1] / self.size
        wave = np.sin(phase + 2*np.pi * (x_normalized + y_normalized))
        return self.wave_amplitude * wave

    def update(self):
        """Update particle states and record average velocity as complex number"""
        # Update ghost particles first (they move independently)
        self.update_ghost_positions()
        
        # Combine regular particles and ghost particles for interaction calculations
        all_positions = np.vstack([self.positions, self.ghost_positions]) if self.n_ghost_particles > 0 else self.positions
        all_angles = np.concatenate([self.angles, self.ghost_angles]) if self.n_ghost_particles > 0 else self.angles
        
        # Calculate interactions for regular particles only
        positions_i = self.positions[:, np.newaxis, :]
        positions_j = all_positions[np.newaxis, :, :]
        diff = positions_i - positions_j
        
        diff = np.where(diff > self.size/2, diff - self.size, diff)
        diff = np.where(diff < -self.size/2, diff + self.size, diff)
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        neighbors = distances < self.radius
        
        cos_angles = np.cos(all_angles)
        sin_angles = np.sin(all_angles)
        neighbor_count = np.sum(neighbors, axis=1)
        
        target_angles = np.zeros_like(self.angles)
        mask = neighbor_count > 0
        if np.any(mask):
            mean_cos = np.sum(neighbors * cos_angles[np.newaxis, :], axis=1)[mask]
            mean_sin = np.sum(neighbors * sin_angles[np.newaxis, :], axis=1)[mask]
            target_angles[mask] = np.arctan2(mean_sin, mean_cos)
            
            if self.use_noise:
                noise = self.apply_noise()
                target_angles += noise
        
        if self.use_external_force:
            external_force = self.apply_external_force()
            angle_diff = (target_angles - self.angles + np.pi) % (2 * np.pi) - np.pi
            angle_diff += external_force
        elif self.use_wave:
            wave_effect = self.apply_wave_alignment()
            angle_diff = (target_angles - self.angles + np.pi) % (2 * np.pi) - np.pi
            angle_diff += wave_effect
        else:
            angle_diff = (target_angles - self.angles + np.pi) % (2 * np.pi) - np.pi
        
        max_turn_rate = np.pi/2 * self.dt
        angle_diff = np.clip(angle_diff, -max_turn_rate, max_turn_rate)
        
        if self.use_inertia:
            self.angular_velocities = (self.inertia * self.angular_velocities + 
                                     (1 - self.inertia) * angle_diff / self.dt)
            self.angles += self.angular_velocities * self.dt
        else:
            self.angles += angle_diff
        
        self.angles = self.angles % (2 * np.pi)
        
        # Update positions and calculate average velocity (only for regular particles)
        velocities = self.speed * np.column_stack([
            np.cos(self.angles),
            np.sin(self.angles)
        ])
        self.positions += velocities * self.dt
        self.positions %= self.size
        
        # Calculate average velocity as complex number (only for regular particles)
        mean_velocity = np.mean(velocities, axis=0)
        velocity_magnitude = np.linalg.norm(mean_velocity)
        velocity_angle = np.arctan2(mean_velocity[1], mean_velocity[0])
        self.avg_velocities.append(velocity_magnitude * np.exp(1j * velocity_angle))

    def update_plot_surface(self):
        """Update the cached plot surface"""
        self.ax.clear()
        if self.avg_velocities:
            angles = np.angle(self.avg_velocities)
            magnitudes = np.abs(self.avg_velocities)
            
            points = np.column_stack((angles, magnitudes))
            
            # Plot the path using a single line collection
            segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
            # Use a fallback colormap that works across matplotlib versions
            try:
                colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
            except AttributeError:
                colors = plt.cm.jet(np.linspace(0, 1, len(segments)))
            
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.7)
            self.ax.add_collection(lc)
            
            # Plot the current point
            current_angle = angles[-1]
            current_magnitude = magnitudes[-1]
            self.ax.scatter(current_angle, current_magnitude, c='red', s=100)
            
            # Set plot limits and grid
            self.ax.set_rmax(self.speed)
            self.ax.set_rlim(0, self.speed)
            self.ax.grid(True)
            self.ax.set_title('Average Velocity (Polar)')
        
        # Render to the cached surface
        self.canvas.draw()
        plot_string = np.asarray(self.canvas.buffer_rgba())
        plot_surface = pygame.image.frombuffer(plot_string, 
                                             self.canvas.get_width_height(), 
                                             'RGBA')
        
        # Scale the surface
        scaled_surface = pygame.transform.scale(plot_surface, (self.size, self.size))
        self.plot_surface.blit(scaled_surface, (0, 0))

    def draw(self):
        """Draw particles and polar plot"""
        self.screen.fill((0, 0, 0))
        
        # Draw particles
        for pos, angle in zip(self.positions, self.angles):
            x, y = pos
            dx = self.speed * 3 * np.cos(angle)
            dy = self.speed * 3 * np.sin(angle)
            
            end_pos = (int(x + dx), int(y + dy))
            pygame.draw.line(self.screen, (255, 255, 255),
                           (int(x), int(y)), end_pos, 2)
            
            head_length = self.speed * 2
            head_angle1 = angle + np.pi*3/4
            head_angle2 = angle - np.pi*3/4
            
            head1 = (int(end_pos[0] + head_length * np.cos(head_angle1)),
                    int(end_pos[1] + head_length * np.sin(head_angle1)))
            head2 = (int(end_pos[0] + head_length * np.cos(head_angle2)),
                    int(end_pos[1] + head_length * np.sin(head_angle2)))
            
            pygame.draw.line(self.screen, (255, 255, 255), end_pos, head1, 2)
            pygame.draw.line(self.screen, (255, 255, 255), end_pos, head2, 2)
        
        # Draw ghost particles
        for pos in self.ghost_positions:
            pygame.draw.circle(self.screen, self.ghost_color, (int(pos[0]), int(pos[1])), 5)
        
        # Draw simulation state
        y_offset = 10
        text_color = (255, 255, 255)
        
        title_text = f"Model: {self.model_name}"
        title_surface = self.font.render(title_text, True, text_color)
        self.screen.blit(title_surface, (10, y_offset))
        y_offset += 40
        
        # Display ghost particle info
        ghost_text = f"Ghost Particles: {self.n_ghost_particles}"
        ghost_surface = self.font.render(ghost_text, True, self.ghost_color)
        self.screen.blit(ghost_surface, (10, y_offset))
        y_offset += 40
        
        important_params = self.important_params.get(self.model_name, [])
        param_values = {
            "n_particles": (self.n_particles, "Particles"),
            "radius": (self.radius, "Radius"),
            "noise": (self.noise, "Noise"),
            "speed": (self.speed, "Speed"),
            "inertia": (self.inertia, "Inertia"),
            "noise_type": (self.noise_type, "Noise Type"),
            "wave_period": (self.wave_period, "Wave Period")
        }
        
        for param in important_params:
            value, label = param_values[param]
            if isinstance(value, (int, float)):
                param_text = f"{label}: {value:.2f}"
            else:
                param_text = f"{label}: {value}"
            param_surface = self.font.render(param_text, True, text_color)
            self.screen.blit(param_surface, (10, y_offset))
            y_offset += 30
        
        # Display controls
        controls_y = self.size - 120
        controls_text = [
            "Controls: 1=+1 ghost, 2=+5 ghosts, 3=-1 ghost, 4=-5 ghosts, 0=clear all",
            "ESC to exit"
        ]
        for i, control_text in enumerate(controls_text):
            control_surface = self.font.render(control_text, True, text_color)
            self.screen.blit(control_surface, (10, controls_y + i * 25))
        
        # Update plot every 10 frames
        if self.frame_count % 10 == 0:
            self.update_plot_surface()
        
        # Always blit the cached surface
        self.screen.blit(self.plot_surface, (self.size, 0))
        
        pygame.display.flip()
        
        if self.save_animation:
            pygame.image.save(self.screen, 
                            f"{self.frame_dir}/frame_{self.frame_count:05d}.png")
    
    def create_gif(self):
        """Create GIF from saved frames"""
        if not self.save_animation:
            return
            
        frames = []
        frame_files = sorted(os.listdir(self.frame_dir))
        for frame_file in frame_files:
            if frame_file.endswith('.png'):
                frame_path = os.path.join(self.frame_dir, frame_file)
                frames.append(imageio.imread(frame_path))
                os.remove(frame_path)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save GIF in the specified directory
        gif_path = os.path.join(self.save_dir, f"vicsek_simulation_{self.timestamp}_{self.model_name}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Animation saved as: {gif_path}")
        
        os.rmdir(self.frame_dir)
    
    def run(self, duration=1000):
        """Run simulation
        
        Parameters:
        -----------
        duration : float
            Duration in seconds to run the simulation
        """
        running = True
        target_fps = 60
        elapsed_time = 0
        
        while running and elapsed_time < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        # Add 1 ghost particle
                        self.add_ghost_particles(1)
                    elif event.key == pygame.K_2:
                        # Add 5 ghost particles
                        self.add_ghost_particles(5)
                    elif event.key == pygame.K_3:
                        # Remove 1 ghost particle
                        self.remove_ghost_particles(1)
                    elif event.key == pygame.K_4:
                        # Remove 5 ghost particles
                        self.remove_ghost_particles(5)
                    elif event.key == pygame.K_0:
                        # Remove all ghost particles
                        self.remove_all_ghost_particles()
            
            self.update()
            self.draw()
            
            # Control frame rate and track time
            elapsed_time += 1/target_fps
            self.clock.tick(target_fps)
            self.frame_count += 1
        
        pygame.quit()
        
        if self.save_animation:
            self.create_gif()

    def add_ghost_particles(self, count):
        """Add ghost particles to the simulation"""
        if count <= 0:
            return
            
        # Calculate new angles for the new particles
        if self.n_ghost_particles == 0:
            new_angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        else:
            # Add new particles at evenly spaced angles
            existing_angles = self.ghost_angles
            new_angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Offset them from existing particles
            new_angles += np.pi / count
        
        # Extend arrays
        self.ghost_angles = np.concatenate([self.ghost_angles, new_angles])
        self.ghost_angular_velocities = np.concatenate([
            self.ghost_angular_velocities, 
            np.full(count, self.ghost_speed / self.ghost_circle_radius)
        ])
        
        # Add new positions
        new_positions = np.zeros((count, 2))
        self.ghost_positions = np.vstack([self.ghost_positions, new_positions])
        
        self.n_ghost_particles += count
        self.update_ghost_positions()
        print(f"Added {count} ghost particles. Total: {self.n_ghost_particles}")

    def remove_ghost_particles(self, count):
        """Remove ghost particles from the simulation"""
        if count <= 0 or self.n_ghost_particles == 0:
            return
            
        count = min(count, self.n_ghost_particles)
        
        # Remove from arrays
        self.ghost_angles = self.ghost_angles[:-count]
        self.ghost_angular_velocities = self.ghost_angular_velocities[:-count]
        self.ghost_positions = self.ghost_positions[:-count]
        
        self.n_ghost_particles -= count
        print(f"Removed {count} ghost particles. Total: {self.n_ghost_particles}")

    def remove_all_ghost_particles(self):
        """Remove all ghost particles from the simulation"""
        self.ghost_angles = np.array([])
        self.ghost_angular_velocities = np.array([])
        self.ghost_positions = np.zeros((0, 2))
        self.n_ghost_particles = 0
        print("Removed all ghost particles")

    def update_ghost_positions(self):
        """Update ghost particle positions based on circular motion"""
        if self.n_ghost_particles == 0:
            return
            
        # Update ghost angles
        self.ghost_angles += self.ghost_angular_velocities * self.dt
        
        # Calculate positions on circles
        self.ghost_positions[:, 0] = self.center[0] + self.ghost_circle_radius * np.cos(self.ghost_angles)
        self.ghost_positions[:, 1] = self.center[1] + self.ghost_circle_radius * np.sin(self.ghost_angles)
        
        # Wrap positions to stay within bounds
        self.ghost_positions %= self.size

if __name__ == "__main__":
    # Example configurations with ghost particles
    model_with_ghosts = VicsekModel(
        n_particles=150,
        size=700,
        radius=50,
        noise=0.1,
        speed=3,
        inertia=0.8,
        dt=2,
        use_noise=True,
        use_inertia=True,
        save_animation=True,
        noise_type='discrete',
        model_name="Ghost_Particles_Demo",
        n_ghost_particles=50,  # Start with 5 ghost particles
        ghost_circle_radius=150,  # Larger circle radius
        ghost_speed=1.75,  # Slower ghost speed
        ghost_color=(255, 0, 255),  # Magenta color
    )
    
    print("Ghost Particle Controls:")
    print("1: Add 1 ghost particle")
    print("2: Add 5 ghost particles") 
    print("3: Remove 1 ghost particle")
    print("4: Remove 5 ghost particles")
    print("0: Remove all ghost particles")
    print("ESC: Exit")
    
    # Run simulation for 30 seconds
    model_with_ghosts.run(duration=30)