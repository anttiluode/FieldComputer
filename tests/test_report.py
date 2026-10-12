import unittest


class ReportTests(unittest.TestCase):
    def test_failed_advantage_is_reported_even_when_memory_passes(self):
        from fieldcomputer import report
        self.assertTrue(hasattr(report, 'render_report'), 'render_report is not implemented')
        gates = {name: {'pass': name != 'G4', 'meaning': name} for name in ('G0', 'G1', 'G2', 'G3', 'G4')}
        gates['G4'].update(difference=-.01, best_alternative='delay_pca')
        data = {'gates': gates, 'summary': [{'model': 'reciprocal', 'observation': 'passive', 'family': 'linear',
                'tasks': {'first_sign': {'mean': .91, 'min': .90, 'max': .92}, 'parity': {'mean': .85, 'min': .84, 'max': .86}}}],
            'fidelity': {'max_abs_error': 1e-14}, 'seeds': [7, 19, 31, 43, 59],
            'counts_per_seed': {'train': 256, 'validation': 128, 'test': 256}, 'diagnostics': [],
            'seconds': 60., 'source_sha256': {}, 'run_kind': 'frozen_full'}
        text = report.render_report(data)
        self.assertIn('| G4 | FAIL |', text)
        self.assertIn('91.0%', text)
        self.assertIn('No comparative advantage', text)
        self.assertIn('2318', text)
        self.assertIn('32 raw quadrature readings', text)
        self.assertIn('4636', text)


if __name__ == '__main__':
    unittest.main()
