import unittest
import numpy as np


class ExperimentTests(unittest.TestCase):
    def implementation(self, name):
        from fieldcomputer import experiment
        self.assertTrue(hasattr(experiment, name), f'{name} is not implemented')
        return getattr(experiment, name)

    def test_balanced_independent_splits_with_common_present_input(self):
        make = self.implementation('make_trials')
        trials = make(7, (64, 32, 64))
        for ids in trials['splits']:
            np.testing.assert_array_equal(np.sum(trials['targets'][ids], axis=0), 0)
        self.assertFalse(np.array_equal(trials['source'][:32], trials['source'][64:96]))
        self.assertEqual(trials['source'].shape, (160, 336))
        np.testing.assert_array_equal(trials['source'][:, 201:], 0.)

    def test_training_pca_does_not_fit_held_out_states(self):
        pca = self.implementation('training_pca')
        rng = np.random.default_rng(2)
        states = rng.normal(size=(80, 40))
        _, fit1 = pca(states, np.arange(40), 16)
        states[40:] *= 1000
        _, fit2 = pca(states, np.arange(40), 16)
        np.testing.assert_array_equal(fit1['mean'], fit2['mean'])
        np.testing.assert_array_equal(fit1['basis'], fit2['basis'])

    def test_quadratic_reader_recovers_known_parity_without_test_label_leakage(self):
        evaluate = self.implementation('evaluate')
        rng = np.random.default_rng(3)
        features = rng.normal(size=(640, 16))
        features[:, :2] = rng.choice([-1., 1.], size=(640, 2))
        targets = np.column_stack((features[:, 0], features[:, 0]*features[:, 1]))
        splits = (np.arange(256), np.arange(256, 384), np.arange(384, 640))
        result = evaluate(features, targets, splits, 'quadratic')
        self.assertGreater(result['tasks']['parity']['accuracy'], 0.99)
        changed = targets.copy()
        changed[splits[2]] *= -1
        result2 = evaluate(features, changed, splits, 'quadratic')
        np.testing.assert_array_equal(result['predictions'], result2['predictions'])
        self.assertEqual(result['alphas'], result2['alphas'])

    def test_no_advantage_gate_when_simple_control_is_better(self):
        gates = self.implementation('compute_gates')
        means = {}
        for model in ('reciprocal', 'driven', 'uncoupled', 'linear', 'leaky', 'delay_pca'):
            for observation in (('passive', 'incremental') if model in ('reciprocal', 'driven', 'uncoupled', 'linear') else ('passive',)):
                for family in ('linear', 'quadratic'):
                    means[(model, observation, family)] = {'first_sign': 0.9, 'parity': 0.9}
        means[('linear', 'passive', 'linear')]['parity'] = 0.5
        means[('linear', 'incremental', 'linear')]['parity'] = 0.5
        means[('delay_pca', 'passive', 'quadratic')] = {'first_sign': 1., 'parity': 1.}
        result = gates(means, [0.9]*5, [-0.1]*5, True)
        self.assertTrue(result['G1']['pass'])
        self.assertTrue(result['G2']['pass'])
        self.assertTrue(result['G3']['pass'])
        self.assertFalse(result['G4']['pass'])


if __name__ == '__main__':
    unittest.main()
