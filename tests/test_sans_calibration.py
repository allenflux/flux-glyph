"""A separate calibration may change only two-choice verifier voting gates."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training import calibrate_sans_verifier as calibration
from training import calibrate_sans_blocking as blocking
from training import export_sans_verifier as export
from training.train_sans_verifier import POLICY, summarize


FAMILIES = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
            'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto']


def prediction(target, count=9, agreement=1.):
    probabilities = [.2/(count-1)]*count
    probabilities[target] = .8
    return {'predicted': target, 'score': .8, 'margin': .8-.2/(count-1), 'agreement': agreement,
            'probabilities': probabilities, 'passed': agreement >= .7}


class ParentFixture:
    """Mock completed training provenance; no actual neural weights are used."""
    def __init__(self, root, ambiguous_roboto=False):
        self.root = Path(root)
        self.args = SimpleNamespace(**{name: self.root/name for name in ('source_run', 'data', 'primary', 'output')})
        for name in ('source_run', 'data', 'primary'):
            getattr(self.args, name).mkdir()
        self.rows = [
            {'split': 'calibration', 'tile_count': 1, 'target': 5, 'family': 'SF Pro', 'domain': 'anchor', 'view': 'native', 'source_id': 'a', 'region_id': 'a'},
            {'split': 'calibration', 'tile_count': 3, 'target': 5, 'family': 'SF Pro', 'domain': 'anchor', 'view': 'half', 'source_id': 'a', 'region_id': 'a'},
            {'split': 'calibration', 'tile_count': 2 if ambiguous_roboto else 3, 'target': 8, 'family': 'Roboto', 'domain': 'new_native', 'view': 'native', 'source_id': 'b', 'region_id': 'b'}]
        self.records = [prediction(5), prediction(5, agreement=2/3), prediction(8, agreement=.5 if ambiguous_roboto else 2/3)]
        self.baseline = [{**prediction(5, count=8), 'known': True} for row in self.rows]
        calibration.dump(self.args.data/'MANIFEST.json', {'families': FAMILIES})
        for split in ('train', 'calibration'):
            folder = self.args.data/split
            folder.mkdir()
            (folder/'tiles.npy').write_bytes(b'mocked array bytes, never decoded')
            calibration.dump(folder/'metadata.json', {'split': split, 'rows': self.rows if split == 'calibration' else []})
            calibration.dump(folder/'MANIFEST.json', {'split': split, 'families': FAMILIES, 'test_pixels_opened': False,
                'root_manifest_sha256': calibration.sha(self.args.data/'MANIFEST.json'), 'rows': len(self.rows),
                'array': {'path': 'tiles.npy', 'sha256': calibration.sha(folder/'tiles.npy')},
                'metadata': {'path': 'metadata.json', 'sha256': calibration.sha(folder/'metadata.json')}})
        (self.args.primary/'model.onnx').write_bytes(b'primary')
        (self.args.primary/'rejection.onnx').write_bytes(b'rejection')
        calibration.dump(self.args.primary/'metadata.json', {'algorithm': 'region-cnn64x256-rejection-v2', 'families': FAMILIES[:-1],
            'model': {'path': 'model.onnx', 'sha256': calibration.sha(self.args.primary/'model.onnx')},
            'rejection': {'model': {'path': 'rejection.onnx', 'sha256': calibration.sha(self.args.primary/'rejection.onnx')}}})
        source = self.args.source_run
        calibration.dump(source/'BASELINE.json', {'families': FAMILIES[:-1], 'records': self.baseline})
        calibration.dump(source/'CALIBRATION_DECISIONS.json', {'families': FAMILIES, 'records': self.records, 'policy': POLICY})
        (source/'verifier.pth').write_bytes(b'original neural checkpoint mock')
        paths = [self.args.data/'MANIFEST.json', self.args.data/'train/MANIFEST.json', self.args.data/'calibration/MANIFEST.json',
                 self.args.primary/'metadata.json', self.args.primary/'model.onnx', self.args.primary/'rejection.onnx',
                 calibration.ROOT/'training/train_sans_verifier.py', calibration.ROOT/'training/prepare_sans_views.py',
                 calibration.ROOT/'training/region_network.py', calibration.ROOT/'training/network.py']
        bindings = {str(path.resolve()): calibration.sha(path) for path in paths}
        metrics, _ = summarize(self.rows, self.baseline, self.records, FAMILIES, FAMILIES[:-1])
        selected = {'step': 7500, 'calibration_nll': .2, 'metrics': metrics}
        self.selection = {'schema': calibration.TRAINING_SCHEMA, 'policy': POLICY, 'test_read': False,
            'optimizer_steps_executed': 8000, 'families': FAMILIES, 'selected': selected, 'history': [selected],
            'calibration_passed': metrics['passed'], 'passed': metrics['passed'], 'bindings': bindings,
            'state_before_sha256': '1'*64, 'state_after_sha256': '2'*64,
            'baseline_sha256': calibration.sha(source/'BASELINE.json'),
            'calibration_decisions_sha256': calibration.sha(source/'CALIBRATION_DECISIONS.json')}
        calibration.dump(source/'SELECTION.json', self.selection)
        calibration.dump(source/'POLICY.json', {'policy': POLICY, 'test_read': False, 'steps': 8000, 'bindings': bindings})
        self.checkpoint = {'selection_sha256': calibration.sha(source/'SELECTION.json'), 'families': FAMILIES, 'state_dict': {'mock': 1}}
        self.saved = {}
        self.torch = SimpleNamespace(load=self.load, save=self.save)

    def load(self, path, **kwargs):
        return self.checkpoint if Path(path) == self.args.source_run/'verifier.pth' else self.saved[str(path)]

    def save(self, value, path):
        self.saved[str(path)] = value
        Path(path).write_bytes(b'rewrapped checkpoint mock')

    def run(self):
        with patch.dict('sys.modules', {'torch': self.torch}), patch.object(calibration, 'state_sha', return_value='2'*64):
            calibration.main(self.args)
        return json.loads((self.args.output/'SELECTION.json').read_text())


class SansCalibrationTests(unittest.TestCase):
    def test_two_of_three_votes_change_only_passed_flag(self):
        row = {'split': 'calibration', 'tile_count': 3}
        record = prediction(4, agreement=2/3)
        changed = calibration.recompute_passed([record], [row], calibration.policy_for(2/3), 9)[0]
        self.assertFalse(record['passed'])
        self.assertTrue(changed['passed'])
        self.assertEqual({k: v for k, v in changed.items() if k != 'passed'}, {k: v for k, v in record.items() if k != 'passed'})

    def test_two_window_split_vote_and_zero_margin_never_pass(self):
        record = prediction(4, agreement=.5)
        self.assertFalse(calibration.recompute_passed([record], [{'split': 'calibration', 'tile_count': 2}], calibration.policy_for(2/3), 9)[0]['passed'])
        tied = {'predicted': 0, 'score': .5, 'margin': 0., 'agreement': 1., 'probabilities': [.5, .5]+[0.]*7, 'passed': False}
        self.assertFalse(calibration.recompute_passed([tied], [{'split': 'calibration', 'tile_count': 1}], calibration.policy_for(2/3), 9)[0]['passed'])

    def test_score_and_all_original_release_criteria_cannot_be_relaxed(self):
        base = {'schema': calibration.SCHEMA, 'policy': calibration.policy_for(2/3), 'grid_spec': calibration.GRID_SPEC,
                'optimizer_steps_executed': 0, 'weights_unchanged': True, 'test_read': False}
        for key in ('minimum_roboto_veto_recall', 'maximum_anchor_native_correct_loss_rate', 'minimum_wrong_name_reduction'):
            altered = copy.deepcopy(base)
            altered['policy'][key] = .01
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'only the predeclared'):
                calibration.effective_policy(altered)
        altered = copy.deepcopy(base)
        altered['policy']['gates']['min_score'] = .49
        with self.assertRaisesRegex(ValueError, 'only the predeclared'):
            calibration.effective_policy(altered)
        with self.assertRaisesRegex(ValueError, 'outside the predeclared'):
            calibration.policy_for(.5)

    def test_test_rows_are_not_accepted_by_the_calibration_stage(self):
        with self.assertRaisesRegex(ValueError, 'only CAL observations'):
            calibration.recompute_passed([prediction(4)], [{'split': 'test', 'tile_count': 1}], POLICY, 9)

    def test_parent_failed_selection_remains_failed_and_weights_are_only_rewrapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = ParentFixture(temporary)
            before = calibration.sha(case.args.source_run/'SELECTION.json')
            selected = case.run()
            self.assertTrue(selected['passed'])
            self.assertFalse(selected['parent']['calibration_passed'])
            self.assertEqual(before, calibration.sha(case.args.source_run/'SELECTION.json'))
            self.assertFalse(json.loads((case.args.source_run/'SELECTION.json').read_text())['passed'])
            self.assertEqual(selected['policy']['gates']['min_patch_agreement'], 2/3)
            self.assertEqual(selected['optimizer_steps_executed'], 0)
            self.assertTrue(selected['weights_unchanged'])
            self.assertEqual(case.saved[str(case.args.output/'verifier.pth')]['state_dict'], case.checkpoint['state_dict'])
            self.assertEqual(selected['state_after_sha256'], selected['parent']['state_after_sha256'])

    def test_both_grid_failures_remain_failed_and_no_new_checkpoint_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = ParentFixture(temporary, ambiguous_roboto=True)
            selected = case.run()
            self.assertFalse(selected['passed'])
            self.assertTrue((case.args.output/'FAILED.json').is_file())
            self.assertFalse((case.args.output/'verifier.pth').exists())
            self.assertFalse(case.saved)

    def test_exact_paths_and_complete_parent_bindings_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = ParentFixture(temporary)
            wrong = Path(temporary)/'different-data'
            with self.assertRaisesRegex(ValueError, 'exact paths differ'):
                calibration.validate_parent(case.args.source_run, wrong, case.args.primary)
            del case.selection['bindings'][str((case.args.primary/'model.onnx').resolve())]
            calibration.dump(case.args.source_run/'SELECTION.json', case.selection)
            with self.assertRaisesRegex(ValueError, 'exact paths differ'):
                calibration.validate_parent(case.args.source_run, case.args.data, case.args.primary)

    def test_calibration_cannot_claim_a_different_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = ParentFixture(temporary)
            selected = case.run()
            selected['state_after_sha256'] = '3'*64
            with self.assertRaisesRegex(ValueError, 'changed the trained parameters'):
                calibration.validate_calibrated_selection(selected, case.args.output, case.args.data, case.args.primary)


class SansBlockingCalibrationTests(unittest.TestCase):
    def prepare(self, temporary):
        case = ParentFixture(temporary, ambiguous_roboto=True)
        previous = case.run()
        self.assertFalse(previous['passed'])
        args = SimpleNamespace(source_run=case.args.source_run, previous_calibration=case.args.output,
            data=case.args.data, primary=case.args.primary, output=Path(temporary)/'blocking', objective=blocking.OBJECTIVE)
        return case, args

    def run_blocking(self, case, args):
        with patch.dict('sys.modules', {'torch': case.torch}), patch.object(blocking, 'state_sha', return_value='2'*64):
            blocking.main(args)
        return json.loads((args.output/'SELECTION.json').read_text())

    def test_product_objective_keeps_old_failures_and_independent_recall_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, args = self.prepare(temporary)
            source_sha = calibration.sha(args.source_run/'SELECTION.json')
            old_sha = calibration.sha(args.previous_calibration/'SELECTION.json')
            selected = self.run_blocking(case, args)
            self.assertTrue(selected['passed'])
            self.assertEqual(selected['policy']['gates']['min_patch_agreement'], 2/3)
            metrics = selected['selected']['metrics']
            self.assertEqual(metrics['roboto_wrong_name_blocking']['rate'], 1.)
            self.assertEqual(metrics['independent_roboto_diagnostic']['rate'], 0.)
            self.assertFalse(metrics['independent_roboto_diagnostic']['original_check_passed'])
            self.assertFalse(metrics['independent_roboto_diagnostic']['formal_name_coverage_claim'])
            self.assertFalse(metrics['original_release_criteria_passed'])
            self.assertEqual(selected['optimizer_steps_executed'], 0)
            self.assertEqual(selected['state_before_sha256'], selected['state_after_sha256'])
            self.assertEqual(case.saved[str(args.output/'verifier.pth')]['state_dict'], case.checkpoint['state_dict'])
            self.assertEqual(source_sha, calibration.sha(args.source_run/'SELECTION.json'))
            self.assertEqual(old_sha, calibration.sha(args.previous_calibration/'SELECTION.json'))
            self.assertFalse(json.loads((args.previous_calibration/'SELECTION.json').read_text())['passed'])
            self.assertFalse((args.data/'test').exists())
            self.assertEqual(export.validate(SimpleNamespace(run=args.output, data=args.data, primary=args.primary)), selected)

    def test_actual_blocking_excludes_already_rejected_and_zero_denominator_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = ParentFixture(temporary, ambiguous_roboto=True)
            rows = case.rows + [{**case.rows[-1], 'source_id': 'already-rejected'}]
            before = case.baseline + [{**case.baseline[-1], 'known': False}]
            metrics, _ = blocking.blocking_metrics(rows, before, case.records+[case.records[-1]], FAMILIES, FAMILIES[:-1])
            measured = metrics['roboto_wrong_name_blocking']
            self.assertEqual(measured['baseline_wrong_names'], 1)
            self.assertEqual(measured['prevented_wrong_names'], 1)
            self.assertEqual(measured['baseline_already_not_wrongly_named'], 1)
            before[-2] = {**before[-2], 'known': False}
            metrics, _ = blocking.blocking_metrics(rows, before, case.records+[case.records[-1]], FAMILIES, FAMILIES[:-1])
            self.assertIsNone(metrics['roboto_wrong_name_blocking']['rate'])
            self.assertFalse(metrics['checks']['roboto_wrong_name_blocking'])
            self.assertFalse(metrics['passed'])

    def test_other_release_failures_are_not_overridden_by_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = ParentFixture(temporary, ambiguous_roboto=True)
            records = [{**case.records[0], 'passed': False}, *case.records[1:]]
            metrics, _ = blocking.blocking_metrics(case.rows, case.baseline, records, FAMILIES, FAMILIES[:-1])
            self.assertTrue(metrics['checks']['roboto_wrong_name_blocking'])
            self.assertFalse(metrics['checks']['native_anchor_retention'])
            self.assertFalse(metrics['passed'])
            for key, value in metrics['original_release_checks'].items():
                if key != 'roboto_veto':
                    self.assertEqual(value, metrics['checks'][key])

    def test_objective_other_policy_changes_and_test_use_cannot_be_smuggled(self):
        base = {'schema': blocking.SCHEMA, 'policy': calibration.policy_for(2/3), 'objective': blocking.OBJECTIVE,
            'objective_protocol': blocking.PROTOCOL, 'grid_spec': blocking.GRID_SPEC,
            'optimizer_steps_executed': 0, 'weights_unchanged': True, 'test_read': False}
        mutations = [lambda row: row['policy']['gates'].update(min_score=.49),
                     lambda row: row['policy'].update(minimum_wrong_name_reduction=.1),
                     lambda row: row['objective_protocol'].update(minimum_roboto_wrong_name_blocking_rate=.7),
                     lambda row: row.update(objective='independent-recall'),
                     lambda row: row.update(test_read=True)]
        for mutate in mutations:
            altered = copy.deepcopy(base)
            mutate(altered)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                blocking.effective_policy(altered)

    def test_full_failed_lineage_and_exact_inputs_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, args = self.prepare(temporary)
            with self.assertRaisesRegex(ValueError, 'exact paths differ'):
                blocking.source_evidence(args.source_run, args.previous_calibration, Path(temporary)/'other-data', args.primary)
            failed = json.loads((args.previous_calibration/'FAILED.json').read_text())
            failed['export_allowed'] = True
            calibration.dump(args.previous_calibration/'FAILED.json', failed)
            with self.assertRaisesRegex(ValueError, 'changed or exported'):
                blocking.source_evidence(args.source_run, args.previous_calibration, args.data, args.primary)

    def test_old_failed_calibration_cannot_be_exported_and_explicit_objective_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, args = self.prepare(temporary)
            with self.assertRaisesRegex(ValueError, 'must pass'):
                export.validate(SimpleNamespace(run=args.previous_calibration, data=args.data, primary=args.primary))
            args.objective = None
            with self.assertRaisesRegex(ValueError, 'explicit --objective'):
                blocking.main(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
