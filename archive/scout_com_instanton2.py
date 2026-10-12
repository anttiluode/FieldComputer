import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from scipy.ndimage import center_of_mass
from scipy.fft import fft2, ifft2, fftfreq
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class InstantonScoutPair:
    """A bound instanton-scout pair forming a computational unit"""
    id: int
    
    # Instanton properties (wave center-of-mass)
    instanton_pos: np.ndarray  # [x, y]
    instanton_vel: np.ndarray  # [vx, vy] 
    instanton_amplitude: float
    instanton_phase: float
    
    # Scout properties (classical particle)
    scout_pos: np.ndarray     # [x, y]
    scout_vel: np.ndarray     # [vx, vy]
    scout_mass: float
    
    # Binding properties
    equilibrium_distance: float
    binding_strength: float
    coupling_phase: float
    
    # Information/memory
    information_mass: float
    memory_traces: List[np.ndarray]
    learning_rate: float
    
    # Topological properties
    topological_charge: float
    orbital_modes: List[float]
    
    def __post_init__(self):
        if len(self.memory_traces) == 0:
            self.memory_traces = []

class MultiScaleFieldComputer:
    """
    Physics-native computation through solitonic field dynamics
    Combines TADS field, COM instantons, and attractor scouts
    """
    
    def __init__(self, grid_size=128, dt=0.01):
        self.grid_size = grid_size
        self.dt = dt
        self.time = 0.0
        self.step_count = 0
        
        # Initialize all three field types
        self.init_tads_field()
        self.init_fractal_attractor_field()
        self.init_computational_units()
        
        # Coupling parameters
        self.wave_scout_coupling = 0.3
        self.scout_wave_coupling = 0.2
        self.field_feedback_strength = 0.1
        
        # Analysis storage
        self.topological_charges = []
        self.binding_energies = []
        self.orbital_frequencies = []
        self.collective_dynamics = []
        
    def init_tads_field(self):
        """Initialize TADS phi-field for emergent atoms"""
        # TADS parameters (optimized for atom formation)
        self.tads_params = {
            'base_wave_speed_sq': 2.0,
            'tension_factor': 16.279,
            'collision_threshold': 0.328,
            'collision_strength': 0.25,
            'damping': 0.001,
            'potential_a': 1.0,
            'potential_b': 0.2
        }
        
        # Field state
        self.phi = np.zeros((self.grid_size, self.grid_size), dtype=np.complex128)
        self.phi_prev = np.zeros_like(self.phi)
        
        # Laplacian kernel for efficient computation
        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
    def init_fractal_attractor_field(self):
        """Initialize fractal landscape for scout navigation"""
        # Generate multi-scale fractal field
        self.attractor_field = self.generate_fractal_noise(beta=0.75, seed=42)
        self.attractor_grad_x, self.attractor_grad_y = np.gradient(self.attractor_field)
        
        # Dynamic attractor modification
        self.attractor_memory = np.zeros_like(self.attractor_field)
        
    def init_computational_units(self):
        """Initialize instanton-scout pairs"""
        self.pairs: List[InstantonScoutPair] = []
        
    def generate_fractal_noise(self, beta=0.75, seed=None):
        """Generate fractal noise field with specified exponent"""
        if seed is not None:
            np.random.seed(seed)
            
        kx = fftfreq(self.grid_size).reshape(1, -1)
        ky = fftfreq(self.grid_size).reshape(-1, 1)
        k = np.sqrt(kx*kx + ky*ky)
        k[0,0] = 1e-6
        
        spectrum = (k**(-beta/2)) * np.exp(2j*np.pi*np.random.rand(self.grid_size, self.grid_size))
        noise = np.real(ifft2(spectrum))
        return (noise - noise.min()) / (noise.max() - noise.min())
    
    def create_instanton_scout_pair(self, position, amplitude=1.5, scout_mass=5.0):
        """Create a new bound instanton-scout pair"""
        pair_id = len(self.pairs)
        x, y = position
        
        # Create instanton (localized wave packet)
        self.plant_instanton(position, amplitude)
        
        # Create scout nearby
        scout_offset = np.random.uniform(-5, 5, 2)
        scout_pos = np.array([x + scout_offset[0], y + scout_offset[1]])
        scout_pos = np.clip(scout_pos, 0, self.grid_size-1)
        
        pair = InstantonScoutPair(
            id=pair_id,
            instanton_pos=np.array([x, y], dtype=float),
            instanton_vel=np.zeros(2),
            instanton_amplitude=amplitude,
            instanton_phase=0.0,
            scout_pos=scout_pos,
            scout_vel=np.zeros(2),
            scout_mass=scout_mass,
            equilibrium_distance=np.linalg.norm(scout_offset),
            binding_strength=0.1,
            coupling_phase=0.0,
            information_mass=scout_mass,
            memory_traces=[],
            learning_rate=0.01,
            topological_charge=0.0,
            orbital_modes=[]
        )
        
        self.pairs.append(pair)
        return pair
    
    def plant_instanton(self, position, amplitude=1.5, radius=8):
        """Plant a localized wave packet (instanton) in the field"""
        x0, y0 = position
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        
        # Gaussian profile with complex phase
        dist_sq = (x_coords - x0)**2 + (y_coords - y0)**2
        profile = amplitude * np.exp(-dist_sq / (2*radius**2))
        phase = np.random.uniform(0, 2*np.pi)
        
        self.phi += profile * np.exp(1j * phase)
        self.phi_prev = self.phi.copy()
    
    def evolve_tads_field(self):
        """Evolve the TADS phi-field with emergent atom formation"""
        # Numerical stability checks
        if np.any(~np.isfinite(self.phi)):
            print("⚠️ NaN/Inf detected in field, resetting...")
            self.phi = np.random.randn(*self.phi.shape) * 0.01
            self.phi_prev = self.phi.copy()
            return
        
        # Energy clamping to prevent blowup
        field_energy = np.sum(np.abs(self.phi)**2)
        if field_energy > 1e6:
            scale_factor = np.sqrt(1e6 / field_energy)
            self.phi *= scale_factor
            self.phi_prev *= scale_factor
            
        # Compute Laplacian
        laplacian_real = convolve2d(np.real(self.phi), self.laplacian_kernel, mode='same', boundary='wrap')
        laplacian_imag = convolve2d(np.imag(self.phi), self.laplacian_kernel, mode='same', boundary='wrap')
        laplacian_phi = laplacian_real + 1j * laplacian_imag
        
        # Variable wave speed (tension modulation) with saturation
        tension_map = np.abs(self.phi)**2
        tension_map = np.clip(tension_map, 0, 100)  # Prevent extreme values
        local_speed = self.tads_params['base_wave_speed_sq'] / (1.0 + self.tads_params['tension_factor'] * tension_map + 1e-6)
        
        # Nonlinear collision term with amplitude limiting
        collision_effect = np.zeros_like(self.phi)
        active_regions = tension_map > (self.tads_params['collision_threshold']**2)
        if np.any(active_regions):
            phi_limited = np.clip(np.abs(self.phi[active_regions]), 0, 5.0)  # Limit amplitude
            phase = np.angle(self.phi[active_regions])
            collision_effect[active_regions] = (self.tads_params['collision_strength'] * 
                                              phi_limited**3 * np.exp(1j * phase * 3))
        
        # Potential force with amplitude limiting
        phi_abs_limited = np.clip(np.abs(self.phi), 0, 5.0)
        potential_force = (self.tads_params['potential_a'] * self.phi - 
                          self.tads_params['potential_b'] * phi_abs_limited**2 * self.phi)
        
        # Scout feedback to field
        scout_feedback = self.compute_scout_field_feedback()
        
        # Field evolution with smaller timestep for stability
        effective_dt = min(self.dt, 0.005)  # Adaptive timestep
        phi_new = (2 * self.phi - self.phi_prev + 
                  effective_dt**2 * (local_speed * laplacian_phi + collision_effect + 
                                    potential_force + scout_feedback))
        
        # Final amplitude limiting
        phi_abs = np.abs(phi_new)
        max_amplitude = 10.0
        mask = phi_abs > max_amplitude
        if np.any(mask):
            phi_new[mask] = (phi_new[mask] / phi_abs[mask]) * max_amplitude
        
        # Update field
        self.phi_prev = self.phi.copy()
        self.phi = phi_new
    
    def compute_scout_field_feedback(self):
        """Compute how scouts modify the field"""
        feedback = np.zeros_like(self.phi)
        
        for pair in self.pairs:
            # Validate scout position
            if not np.isfinite(pair.scout_pos).all():
                # Reset to center if NaN
                pair.scout_pos = np.array([self.grid_size/2, self.grid_size/2])
                pair.scout_vel = np.zeros(2)
                continue
                
            x, y = pair.scout_pos
            x, y = float(x), float(y)  # Ensure float conversion
            
            # Clamp to valid range
            x = np.clip(x, 0, self.grid_size-1)
            y = np.clip(y, 0, self.grid_size-1)
            
            # Update clamped position
            pair.scout_pos = np.array([x, y])
            
            # Scout creates local field modification
            sigma = 3.0  # Gaussian width
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            
            gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2*sigma**2))
            strength = self.field_feedback_strength * np.clip(pair.binding_strength, 0, 1.0)
            
            # Complex coupling with phase
            phase_factor = np.exp(1j * pair.coupling_phase)
            feedback += strength * gaussian * phase_factor
        
        return feedback
    
    def track_instanton_centers(self):
        """Track center-of-mass of each instanton"""
        for pair in self.pairs:
            # Validate current position
            if not np.isfinite(pair.instanton_pos).all():
                pair.instanton_pos = np.array([self.grid_size/2, self.grid_size/2])
                pair.instanton_vel = np.zeros(2)
                continue
                
            # Find region around instanton
            x0, y0 = pair.instanton_pos
            radius = 15
            
            # Extract local field with bounds checking
            x_min = max(0, int(x0 - radius))
            x_max = min(self.grid_size, int(x0 + radius))
            y_min = max(0, int(y0 - radius))
            y_max = min(self.grid_size, int(y0 + radius))
            
            if x_max <= x_min or y_max <= y_min:
                continue  # Skip if invalid region
                
            local_field = self.phi[y_min:y_max, x_min:x_max]
            local_intensity = np.abs(local_field)**2
            
            # Check for finite values
            if not np.isfinite(local_intensity).all() or np.sum(local_intensity) < 1e-8:
                continue
            
            try:
                # Compute center of mass
                com_rel = center_of_mass(local_intensity)
                if not np.isfinite(com_rel).all():
                    continue
                    
                new_pos = np.array([x_min + com_rel[1], y_min + com_rel[0]])
                
                # Validate new position
                if not np.isfinite(new_pos).all():
                    continue
                    
                # Limit movement speed to prevent jumps
                max_movement = 2.0  # Maximum pixels per timestep
                displacement = new_pos - pair.instanton_pos
                if np.linalg.norm(displacement) > max_movement:
                    displacement = displacement / np.linalg.norm(displacement) * max_movement
                    new_pos = pair.instanton_pos + displacement
                
                # Update velocity
                pair.instanton_vel = (new_pos - pair.instanton_pos) / self.dt
                pair.instanton_pos = new_pos
                
                # Update amplitude and phase
                ix, iy = int(np.clip(new_pos[0], 0, self.grid_size-1)), int(np.clip(new_pos[1], 0, self.grid_size-1))
                complex_val = self.phi[iy, ix]
                if np.isfinite(complex_val):
                    pair.instanton_amplitude = min(10.0, np.abs(complex_val))  # Limit amplitude
                    pair.instanton_phase = np.angle(complex_val)
                        
            except Exception as e:
                # Reset on any error
                pair.instanton_pos = np.array([self.grid_size/2, self.grid_size/2])
                pair.instanton_vel = np.zeros(2)
    
    def evolve_scouts(self):
        """Evolve scout dynamics with instanton coupling"""
        for pair in self.pairs:
            # Validate scout state
            if not np.isfinite(pair.scout_pos).all() or not np.isfinite(pair.scout_vel).all():
                pair.scout_pos = np.array([self.grid_size/2, self.grid_size/2])
                pair.scout_vel = np.zeros(2)
                continue
                
            # Forces on scout
            total_force = np.zeros(2)
            
            # 1. Attractor field gradient force
            ix, iy = int(np.clip(pair.scout_pos[0], 0, self.grid_size-1)), int(np.clip(pair.scout_pos[1], 0, self.grid_size-1))
            
            attractor_force = np.array([
                self.attractor_grad_x[iy, ix],
                self.attractor_grad_y[iy, ix]
            ]) * np.clip(pair.learning_rate, 0, 0.1)  # Limit learning rate
            
            total_force += attractor_force
            
            # 2. Binding force from instanton
            if np.isfinite(pair.instanton_pos).all():
                displacement = pair.instanton_pos - pair.scout_pos
                distance = np.linalg.norm(displacement) + 1e-6
                
                # Harmonic binding potential around equilibrium distance
                binding_force = (np.clip(pair.binding_strength, 0, 1.0) * 
                               (distance - pair.equilibrium_distance) * 
                               displacement / distance)
                
                total_force += binding_force
            
            # 3. Field coupling force
            field_val = self.phi[iy, ix]
            if np.isfinite(field_val):
                field_intensity = min(100.0, np.abs(field_val)**2)  # Limit intensity
                
                # Compute gradient safely
                ix_next = min(ix+1, self.grid_size-1)
                ix_prev = max(ix-1, 0)
                iy_next = min(iy+1, self.grid_size-1)
                iy_prev = max(iy-1, 0)
                
                field_gradient = np.array([
                    np.real(self.phi[iy, ix_next] - self.phi[iy, ix_prev]) / 2,
                    np.real(self.phi[iy_next, ix] - self.phi[iy_prev, ix]) / 2
                ])
                
                if np.isfinite(field_gradient).all():
                    wave_force = self.wave_scout_coupling * field_intensity * field_gradient
                    total_force += wave_force
            
            # 4. Fractal noise (small)
            fractal_noise = 0.02 * np.random.randn(2)
            total_force += fractal_noise
            
            # Limit total force magnitude
            force_magnitude = np.linalg.norm(total_force)
            max_force = 0.5
            if force_magnitude > max_force:
                total_force = total_force / force_magnitude * max_force
            
            # Update scout dynamics with information mass
            effective_mass = np.clip(pair.information_mass, 1.0, 20.0)  # Limit mass range
            pair.scout_vel += (total_force / effective_mass) * self.dt
            
            # Velocity limiting and damping
            velocity_magnitude = np.linalg.norm(pair.scout_vel)
            max_velocity = 2.0
            if velocity_magnitude > max_velocity:
                pair.scout_vel = pair.scout_vel / velocity_magnitude * max_velocity
                
            pair.scout_vel *= 0.98  # Stronger damping for stability
            
            # Update position
            pair.scout_pos += pair.scout_vel * self.dt
            pair.scout_pos = np.clip(pair.scout_pos, 1, self.grid_size-2)  # Keep away from edges
            
            # Update information mass (learning makes scout "heavier")
            if np.isfinite(force_magnitude):
                pair.information_mass += 0.001 * force_magnitude
                pair.information_mass = np.clip(pair.information_mass, 1.0, 20.0)
            
            # Store memory trace
            if len(pair.memory_traces) < 50:  # Reduce memory to prevent overflow
                pair.memory_traces.append(pair.scout_pos.copy())
            else:
                pair.memory_traces.pop(0)
                pair.memory_traces.append(pair.scout_pos.copy())
    
    def compute_topological_charges(self):
        """Compute topological charge for each instanton"""
        for pair in self.pairs:
            x0, y0 = pair.instanton_pos
            radius = 10
            
            # Extract region around instanton
            x_min = max(0, int(x0 - radius))
            x_max = min(self.grid_size, int(x0 + radius))
            y_min = max(0, int(y0 - radius))
            y_max = min(self.grid_size, int(y0 + radius))
            
            if x_max > x_min and y_max > y_min:
                local_field = self.phi[y_min:y_max, x_min:x_max]
                
                # Compute topological charge density
                phi_x = np.gradient(local_field, axis=1)
                phi_y = np.gradient(local_field, axis=0)
                
                # Q = (1/2π) ∫ φ* (∂φ/∂x ∂φ*/∂y - ∂φ/∂y ∂φ*/∂x) dx dy
                charge_density = np.imag(np.conj(local_field) * (phi_x * np.conj(phi_y) - phi_y * np.conj(phi_x)))
                topological_charge = np.sum(charge_density) / (2 * np.pi)
                
                pair.topological_charge = topological_charge
    
    def analyze_orbital_modes(self):
        """Analyze orbital frequency modes around each instanton"""
        for pair in self.pairs:
            if len(pair.memory_traces) > 50:
                # Extract time series of distances from scout to instanton
                distances = []
                for trace_pos in pair.memory_traces[-50:]:
                    dist = np.linalg.norm(trace_pos - pair.instanton_pos)
                    distances.append(dist)
                
                # FFT to find orbital frequencies
                fft_result = np.fft.fft(distances - np.mean(distances))
                frequencies = np.fft.fftfreq(len(distances), self.dt)
                power_spectrum = np.abs(fft_result)**2
                
                # Find peaks (dominant orbital modes)
                peak_indices = []
                for i in range(1, len(power_spectrum)//2):
                    if (power_spectrum[i] > power_spectrum[i-1] and 
                        power_spectrum[i] > power_spectrum[i+1] and
                        power_spectrum[i] > 0.1 * np.max(power_spectrum)):
                        peak_indices.append(i)
                
                pair.orbital_modes = [frequencies[i] for i in peak_indices[:5]]
    
    def step(self):
        """Single evolution step of the complete system"""
        # 1. Evolve TADS field (creates emergent atoms)
        self.evolve_tads_field()
        
        # 2. Track instanton centers of mass
        self.track_instanton_centers()
        
        # 3. Evolve scout dynamics
        self.evolve_scouts()
        
        # 4. Update coupling between scouts and instantons
        self.update_coupling_dynamics()
        
        # 5. Analyze topological and orbital properties
        if self.step_count % 10 == 0:
            self.compute_topological_charges()
            self.analyze_orbital_modes()
        
        # 6. Store collective dynamics
        self.store_collective_state()
        
        self.time += self.dt
        self.step_count += 1
    
    def update_coupling_dynamics(self):
        """Update coupling between scouts and instantons"""
        for pair in self.pairs:
            # Distance between scout and instanton
            distance = np.linalg.norm(pair.scout_pos - pair.instanton_pos)
            
            # Adaptive equilibrium distance based on field strength
            field_strength = pair.instanton_amplitude
            pair.equilibrium_distance = 5.0 + 3.0 * field_strength
            
            # Adaptive binding strength
            pair.binding_strength = 0.1 * (1.0 + field_strength)
            
            # Phase coupling evolves based on relative motion
            relative_vel = pair.scout_vel - pair.instanton_vel
            pair.coupling_phase += 0.01 * np.linalg.norm(relative_vel) * self.dt
            pair.coupling_phase = pair.coupling_phase % (2 * np.pi)
    
    def store_collective_state(self):
        """Store state for analysis"""
        if len(self.pairs) > 0:
            # Average binding energy
            total_binding = sum(pair.binding_strength * 
                              (np.linalg.norm(pair.scout_pos - pair.instanton_pos) - pair.equilibrium_distance)**2 
                              for pair in self.pairs)
            self.binding_energies.append(total_binding)
            
            # Average topological charge
            avg_charge = np.mean([pair.topological_charge for pair in self.pairs])
            self.topological_charges.append(avg_charge)
            
            # Collective orbital frequency
            all_modes = []
            for pair in self.pairs:
                all_modes.extend(pair.orbital_modes)
            if all_modes:
                self.orbital_frequencies.append(np.mean(all_modes))
            else:
                self.orbital_frequencies.append(0.0)
    
    def get_visualization_data(self):
        """Get current state for visualization"""
        return {
            'phi_field': self.phi,
            'attractor_field': self.attractor_field,
            'pairs': self.pairs,
            'time': self.time,
            'step_count': self.step_count,
            'field_energy': np.sum(np.abs(self.phi)**2),
            'topological_charges': self.topological_charges,
            'binding_energies': self.binding_energies,
            'orbital_frequencies': self.orbital_frequencies
        }

# Visualization and demonstration
def create_multi_scale_demo():
    """Create and run a demonstration of the multi-scale field computer"""
    computer = MultiScaleFieldComputer(grid_size=128)
    
    # Create several instanton-scout pairs
    positions = [(32, 32), (96, 32), (64, 96), (32, 96)]
    for pos in positions:
        computer.create_instanton_scout_pair(pos, amplitude=1.2)
    
    # Set up visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Scale Field-Based Computer: Solitonic Computation', fontsize=16)
    
    # Field visualization
    ax_field = axes[0, 0]
    ax_scouts = axes[0, 1]
    ax_analysis = axes[0, 2]
    ax_topo = axes[1, 0]
    ax_binding = axes[1, 1]
    ax_orbital = axes[1, 2]
    
    def animate(frame):
        # Evolve system
        for _ in range(5):  # Multiple steps per frame
            computer.step()
        
        data = computer.get_visualization_data()
        
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # 1. TADS field (emergent atoms)
        phi_abs = np.abs(data['phi_field'])
        im1 = ax_field.imshow(phi_abs, cmap='plasma', origin='lower')
        ax_field.set_title(f'TADS Field (Emergent Atoms)\nt={data["time"]:.1f}')
        
        # Overlay instanton positions
        for pair in data['pairs']:
            x, y = pair.instanton_pos
            ax_field.plot(x, y, 'wo', markersize=8, markeredgecolor='black')
        
        # 2. Scout navigation with attractors
        im2 = ax_scouts.imshow(data['attractor_field'], cmap='terrain', alpha=0.5, origin='lower')
        ax_scouts.set_title('Scout Navigation & Binding')
        
        # Plot scouts and their trails
        colors = ['red', 'blue', 'green', 'orange']
        for i, pair in enumerate(data['pairs']):
            color = colors[i % len(colors)]
            
            # Scout position
            sx, sy = pair.scout_pos
            ax_scouts.plot(sx, sy, 'o', color=color, markersize=10, markeredgecolor='white')
            
            # Instanton position
            ix, iy = pair.instanton_pos
            ax_scouts.plot(ix, iy, 's', color=color, markersize=8, alpha=0.7)
            
            # Binding line
            ax_scouts.plot([sx, ix], [sy, iy], '--', color=color, alpha=0.7)
            
            # Memory trail
            if len(pair.memory_traces) > 1:
                trail = np.array(pair.memory_traces[-20:])  # Last 20 positions
                ax_scouts.plot(trail[:, 0], trail[:, 1], '-', color=color, alpha=0.3, linewidth=2)
        
        # 3. System analysis
        ax_analysis.text(0.05, 0.95, f'Multi-Scale Field Computer\nStep: {data["step_count"]:,}', 
                        transform=ax_analysis.transAxes, fontsize=12, verticalalignment='top')
        
        analysis_text = f"""
        Field Energy: {data['field_energy']:.1f}
        Active Pairs: {len(data['pairs'])}
        
        INSTANTON-SCOUT PAIRS:
        """
        
        for i, pair in enumerate(data['pairs']):
            distance = np.linalg.norm(pair.scout_pos - pair.instanton_pos)
            analysis_text += f"""
        Pair {i+1}:
          Separation: {distance:.1f}
          Binding: {pair.binding_strength:.3f}
          Info Mass: {pair.information_mass:.1f}
          Topo Charge: {pair.topological_charge:.3f}
          Orbital Modes: {len(pair.orbital_modes)}
        """
        
        ax_analysis.text(0.05, 0.85, analysis_text, transform=ax_analysis.transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax_analysis.set_xlim(0, 1)
        ax_analysis.set_ylim(0, 1)
        ax_analysis.axis('off')
        
        # 4. Topological charges over time
        if len(data['topological_charges']) > 1:
            ax_topo.plot(data['topological_charges'])
            ax_topo.set_title('Topological Charge Evolution')
            ax_topo.set_ylabel('Average Topological Charge')
            ax_topo.grid(True)
        
        # 5. Binding energies
        if len(data['binding_energies']) > 1:
            ax_binding.plot(data['binding_energies'])
            ax_binding.set_title('Binding Energy Dynamics')
            ax_binding.set_ylabel('Total Binding Energy')
            ax_binding.grid(True)
        
        # 6. Orbital frequencies
        if len(data['orbital_frequencies']) > 1:
            ax_orbital.plot(data['orbital_frequencies'])
            ax_orbital.set_title('Collective Orbital Frequencies')
            ax_orbital.set_ylabel('Average Frequency')
            ax_orbital.grid(True)
        
        plt.tight_layout()
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=100, repeat=True)
    
    return fig, ani, computer

if __name__ == "__main__":
    print("🌌 Launching Multi-Scale Field-Based Computer")
    print("🔬 Combining TADS field dynamics, COM instantons, and attractor scouts")
    print("⚛️  Watch emergent solitonic computation in real-time!")
    
    fig, ani, computer = create_multi_scale_demo()
    plt.show()
    
    # Analysis after running
    print(f"\n📊 Final Analysis:")
    print(f"Total steps: {computer.step_count:,}")
    print(f"Active computational units: {len(computer.pairs)}")
    
    if computer.pairs:
        avg_topo_charge = np.mean([pair.topological_charge for pair in computer.pairs])
        avg_binding = np.mean([pair.binding_strength for pair in computer.pairs])
        avg_info_mass = np.mean([pair.information_mass for pair in computer.pairs])
        
        print(f"Average topological charge: {avg_topo_charge:.4f}")
        print(f"Average binding strength: {avg_binding:.4f}")
        print(f"Average information mass: {avg_info_mass:.2f}")
        
        total_orbital_modes = sum(len(pair.orbital_modes) for pair in computer.pairs)
        print(f"Total orbital modes detected: {total_orbital_modes}")
        
        print("\n🧬 This demonstrates physics-native computation:")
        print("   • Thoughts as solitonic field configurations")
        print("   • Memory as attractor landscapes") 
        print("   • Learning as information mass accumulation")
        print("   • Binding as emergent computational relationships")

