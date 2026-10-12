"""Batched adaptation of the preserved 2025 field, with explicit input ports.

Local generators replace the archive's global RNG. Nonfinite states raise an
error instead of silently replacing the field. All finite-state equations,
including the archive's swapped attractor-gradient names, are preserved.
"""
import copy
import numpy as np


class FieldBatch:
    MODES = ('reciprocal', 'driven', 'uncoupled', 'linear')

    def __init__(self, seed, count, mode, grid_size=24):
        if mode not in self.MODES or count < 1 or grid_size < 20:
            raise ValueError('Invalid mode, episode count, or grid size')
        self.seed, self.count, self.mode, self.n = seed, count, mode, grid_size
        self.dt, self.wave_dt = 0.01, 0.005
        self.rng = np.random.default_rng(seed + 10000)
        self.y, self.x = np.mgrid[:grid_size, :grid_size]
        k = np.hypot(np.fft.fftfreq(grid_size)[None, :], np.fft.fftfreq(grid_size)[:, None])
        k[0, 0] = 1e-6
        landscape_rng = np.random.RandomState(42)
        spectrum = k**(-0.75/2) * np.exp(2j*np.pi*landscape_rng.rand(grid_size, grid_size))
        landscape = np.fft.ifft2(spectrum).real
        landscape = (landscape - landscape.min()) / np.ptp(landscape)
        # Preserve the literal archive's axis naming, even though np.gradient
        # returns the row derivative first.
        self.grad_x, self.grad_y = np.gradient(landscape)
        initial_rng = np.random.RandomState(seed)
        phase = initial_rng.uniform(0, 2*np.pi)
        offset = initial_rng.uniform(-5, 5, 2)
        center = np.array([8., 12.])
        wave = 1.2*np.exp(-((self.x-8)**2+(self.y-12)**2)/(2*8**2))*np.exp(1j*phase)
        self.phi = np.broadcast_to(wave, (count, grid_size, grid_size)).copy()
        self.prev = self.phi.copy()
        self.center = np.tile(center, (count, 1))
        self.center_vel = np.zeros((count, 2))
        self.scout_pos = np.tile(np.clip(center+offset, 0, grid_size-1), (count, 1))
        self.scout_vel = np.zeros((count, 2))
        self.binding = np.full(count, 0.1)
        self.equilibrium = np.full(count, np.linalg.norm(offset))
        self.coupling_phase = np.zeros(count)
        self.mass = np.full(count, 5.)
        self.amplitude = np.full(count, 1.2)
        self.phase = np.zeros(count)
        self.source_patch = np.exp(-((self.x-13)**2+(self.y-12)**2)/(2*2**2))
        self.clamp_count = 0
        self.energy_rescales = 0
        self.max_abs = float(np.max(np.abs(self.phi)))

    @property
    def real_state_size(self):
        # Two complex grids plus causal scalar/vector variables of one pair.
        return 4*self.n*self.n + 14

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        fresh = type(self)(self.seed, self.count, self.mode, self.n)
        self.__dict__.update(fresh.__dict__)

    def observe(self):
        values = self.phi[:, [10, 10, 14, 14], [11, 14, 11, 14]]
        return np.concatenate((values.real, values.imag), axis=1)

    def step(self, source=None, noise=None):
        if source is None:
            source = np.zeros(self.count)
        source = np.asarray(source, dtype=float)
        if source.shape != (self.count,) or not np.isfinite(source).all():
            raise ValueError('source must contain one finite scalar per episode')
        if noise is None:
            noise = 0.02*self.rng.standard_normal((self.count, 2))
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (self.count, 2) or not np.isfinite(noise).all():
            raise ValueError('noise must contain two finite components per episode')
        if not np.isfinite(self.phi).all():
            raise FloatingPointError('Nonfinite field; no silent reset is allowed')
        if self.mode != 'linear':
            energy = np.sum(np.abs(self.phi)**2, axis=(1, 2))
            scale = np.minimum(1., np.sqrt(1e6 / np.maximum(energy, 1e-30)))
            self.energy_rescales += int(np.sum(scale < 1))
            self.phi *= scale[:, None, None]
            self.prev *= scale[:, None, None]
        lap = (np.roll(self.phi, 1, axis=1) + np.roll(self.phi, -1, axis=1)
               + np.roll(self.phi, 1, axis=2) + np.roll(self.phi, -1, axis=2) - 4*self.phi)
        intensity = np.minimum(np.abs(self.phi)**2, 100.)
        if self.mode == 'linear':
            acceleration = 2.*lap + self.phi
        else:
            speed = 2. / (1. + 16.279*intensity + 1e-6)
            if self.mode == 'uncoupled':
                lap = np.zeros_like(lap)
            limited = np.minimum(np.abs(self.phi), 5.)
            collision = np.where(intensity > 0.328**2,
                0.25*limited**3*np.exp(3j*np.angle(self.phi)), 0.)
            acceleration = speed*lap + collision + self.phi - 0.2*limited**2*self.phi
        if self.mode == 'reciprocal':
            dx = self.x[None]-self.scout_pos[:, 0, None, None]
            dy = self.y[None]-self.scout_pos[:, 1, None, None]
            feedback = (0.1*np.clip(self.binding, 0, 1)*np.exp(1j*self.coupling_phase))[:, None, None]
            acceleration += feedback*np.exp(-(dx**2+dy**2)/(2*3**2))
        acceleration += source[:, None, None]*self.source_patch
        new = 2*self.phi-self.prev + self.wave_dt**2*acceleration
        magnitude = np.abs(new)
        if self.mode != 'linear':
            self.clamp_count += int(np.sum(magnitude > 10.))
            new *= np.minimum(1., 10./np.maximum(magnitude, 1e-30))
        self.prev, self.phi = self.phi, new
        self.max_abs = max(self.max_abs, float(np.max(np.abs(new))))
        if not np.isfinite(new).all():
            raise FloatingPointError('Nonfinite evolved field')
        self._track_centers()
        self._move_scouts(noise)
        self.equilibrium = 5.+3.*self.amplitude
        self.binding = 0.1*(1.+self.amplitude)
        relative = self.scout_vel-self.center_vel
        self.coupling_phase = (self.coupling_phase+0.01*np.linalg.norm(relative, axis=1)*self.dt) % (2*np.pi)

    def _track_centers(self):
        lower = np.maximum(0, np.trunc(self.center-15).astype(int))
        upper = np.minimum(self.n, np.trunc(self.center+15).astype(int))
        region = ((self.x[None] >= lower[:, 0, None, None])
            & (self.x[None] < upper[:, 0, None, None])
            & (self.y[None] >= lower[:, 1, None, None])
            & (self.y[None] < upper[:, 1, None, None]))
        weight = np.abs(self.phi)**2*region
        total = weight.sum(axis=(1, 2))
        new = np.column_stack(((weight*self.x).sum(axis=(1, 2)), (weight*self.y).sum(axis=(1, 2))))
        valid = total >= 1e-8
        new /= np.maximum(total[:, None], 1e-30)
        new[~valid] = self.center[~valid]
        displacement = new-self.center
        length = np.linalg.norm(displacement, axis=1)
        displacement *= np.minimum(1., 2./np.maximum(length, 1e-30))[:, None]
        self.center_vel[valid] = displacement[valid]/self.dt
        self.center += displacement
        idx = np.clip(self.center.astype(int), 0, self.n-1)
        values = self.phi[np.arange(self.count), idx[:, 1], idx[:, 0]]
        self.amplitude[valid] = np.minimum(10., np.abs(values[valid]))
        self.phase[valid] = np.angle(values[valid])

    def _move_scouts(self, noise):
        idx = np.clip(self.scout_pos.astype(int), 0, self.n-1)
        ix, iy = idx[:, 0], idx[:, 1]
        rows = np.arange(self.count)
        force = 0.01*np.column_stack((self.grad_x[iy, ix], self.grad_y[iy, ix]))
        displacement = self.center-self.scout_pos
        distance = np.linalg.norm(displacement, axis=1)+1e-6
        force += (np.clip(self.binding, 0, 1)*(distance-self.equilibrium)/distance)[:, None]*displacement
        field_value = self.phi[rows, iy, ix]
        gradient = np.column_stack((
            (self.phi[rows, iy, np.minimum(ix+1, self.n-1)]-self.phi[rows, iy, np.maximum(ix-1, 0)]).real/2,
            (self.phi[rows, np.minimum(iy+1, self.n-1), ix]-self.phi[rows, np.maximum(iy-1, 0), ix]).real/2))
        force += 0.3*np.minimum(100., np.abs(field_value)**2)[:, None]*gradient
        force += noise
        length = np.linalg.norm(force, axis=1)
        force *= np.minimum(1., 0.5/np.maximum(length, 1e-30))[:, None]
        self.scout_vel += force / np.clip(self.mass, 1, 20)[:, None]*self.dt
        velocity = np.linalg.norm(self.scout_vel, axis=1)
        self.scout_vel *= np.minimum(1., 2./np.maximum(velocity, 1e-30))[:, None]
        self.scout_vel *= 0.98
        self.scout_pos = np.clip(self.scout_pos+self.scout_vel*self.dt, 1, self.n-2)
        self.mass = np.clip(self.mass+0.001*length, 1, 20)
