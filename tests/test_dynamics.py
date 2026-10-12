"""Scientific checks: fidelity and causal separation, not success-gate tests."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class DynamicsTests(unittest.TestCase):
    def adapter(self, *args, **kwargs):
        from fieldcomputer import dynamics
        self.assertTrue(hasattr(dynamics, 'FieldBatch'), 'FieldBatch is not implemented')
        return dynamics.FieldBatch(*args, **kwargs)

    def test_agrees_with_literal_archive_over_unforced_steps(self):
        model = self.adapter(7, 1, 'reciprocal')
        import sys
        spec = importlib.util.spec_from_file_location('literal_archive', Path(__file__).parents[1] / 'archive/scout_com_instanton2.py')
        archive = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = archive
        spec.loader.exec_module(archive)
        original = archive.MultiScaleFieldComputer(grid_size=24)
        np.random.seed(7)
        original.create_instanton_scout_pair((8, 12), amplitude=1.2)
        for _ in range(20):
            state = np.random.get_state()
            noise = 0.02 * np.random.randn(1, 2)
            np.random.set_state(state)
            original.step()
            model.step(noise=noise)
            np.testing.assert_allclose(model.phi[0], original.phi, atol=1e-10, rtol=1e-12)
            np.testing.assert_allclose(model.prev[0], original.phi_prev, atol=1e-10, rtol=1e-12)
            for batch_attr, archive_attr in [('scout_pos', 'scout_pos'), ('scout_vel', 'scout_vel'), ('center', 'instanton_pos'), ('center_vel', 'instanton_vel'), ('binding', 'binding_strength'), ('equilibrium', 'equilibrium_distance'), ('coupling_phase', 'coupling_phase'), ('mass', 'information_mass')]:
                np.testing.assert_allclose(getattr(model, batch_attr)[0], getattr(original.pairs[0], archive_attr), atol=1e-10, rtol=1e-12)

    def test_reciprocal_feedback_changes_field(self):
        reciprocal = self.adapter(7, 2, 'reciprocal')
        driven = self.adapter(7, 2, 'driven')
        for _ in range(30):
            reciprocal.step(noise=np.zeros((2, 2)))
            driven.step(noise=np.zeros((2, 2)))
        self.assertGreater(np.linalg.norm(reciprocal.phi - driven.phi), 1e-6)

    def test_linear_probe_response_is_independent_of_history(self):
        model = self.adapter(19, 2, 'linear')
        for _ in range(40):
            model.step(source=np.array([4., -4.]), noise=np.zeros((2, 2)))
        probe, absent = model.clone(), model.clone()
        for _ in range(40):
            probe.step(source=np.ones(2), noise=np.zeros((2, 2)))
            absent.step(source=np.zeros(2), noise=np.zeros((2, 2)))
        delta = probe.phi - absent.phi
        np.testing.assert_allclose(delta[0], delta[1], atol=1e-10)
        self.assertEqual(model.clamp_count, 0)

    def test_erasing_causal_state_removes_history(self):
        model = self.adapter(31, 2, 'reciprocal')
        for _ in range(40):
            model.step(source=np.array([4., -4.]), noise=np.zeros((2, 2)))
        self.assertGreater(np.linalg.norm(model.phi[0] - model.phi[1]), 1e-3)
        model.reset()
        for _ in range(40):
            model.step(source=np.ones(2), noise=np.zeros((2, 2)))
        np.testing.assert_allclose(model.observe()[0], model.observe()[1], atol=1e-12)

    def test_linear_control_remains_linear_beyond_archive_amplitude_cap(self):
        model = self.adapter(7, 2, 'linear')
        for t in range(336):
            source = np.array([5.6, -5.6]) if 8 <= t < 80 or 112 <= t < 184 else np.zeros(2)
            model.step(source, np.zeros((2, 2)))
        probe, absent = model.clone(), model.clone()
        for t in range(108):
            probe.step(np.full(2, 4. if t < 48 else 0.), np.zeros((2, 2)))
            absent.step(np.zeros(2), np.zeros((2, 2)))
        self.assertEqual(probe.clamp_count, 0, 'A true linear control must never cap amplitudes')
        self.assertGreater(probe.max_abs, 10.)
        delta = probe.phi-absent.phi
        np.testing.assert_allclose(delta[0], delta[1], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
