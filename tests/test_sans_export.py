"""Export/test provenance guards run without Torch or opening held-out arrays."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from training import export_sans_verifier as export
from training import evaluate_sans_verifier as evaluate
from training.train_sans_verifier import POLICY, observations


FAMILIES = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
            'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto']


class SansExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(**{name: self.root/name for name in ('run', 'data', 'primary', 'region')},
                                    output=self.root/'TEST.json')
        for name in ('run', 'data', 'primary', 'region'):
            getattr(self.args, name).mkdir()
        for split in ('train', 'calibration'):
            folder = self.args.data/split
            folder.mkdir()
            export.dump(folder/'MANIFEST.json', {'split': split})
        export.dump(self.args.data/'MANIFEST.json', {'families': FAMILIES})
        for name, value in (('model.onnx', b'primary bytes'), ('rejection.onnx', b'rejector bytes')):
            (self.args.primary/name).write_bytes(value)
            (self.args.region/name).write_bytes(value)
        (self.args.run/'verifier.pth').write_bytes(b'frozen checkpoint bytes')
        (self.args.region/'verifier.onnx').write_bytes(b'verifier bytes')
        primary = {'algorithm': 'region-cnn64x256-rejection-v2', 'families': FAMILIES[:-1],
                   'model': {'path': 'model.onnx', 'sha256': export.sha(self.args.primary/'model.onnx')},
                   'rejection': {'model': {'path': 'rejection.onnx', 'sha256': export.sha(self.args.primary/'rejection.onnx')}}}
        export.dump(self.args.primary/'metadata.json', primary)
        export.dump(self.args.run/'CALIBRATION_DECISIONS.json', {'records': [], 'families': FAMILIES, 'policy': POLICY})
        export.dump(self.args.run/'BASELINE.json', {'records': [], 'families': FAMILIES[:-1]})
        paths = [self.args.data/'MANIFEST.json', self.args.data/'train/MANIFEST.json', self.args.data/'calibration/MANIFEST.json',
                 self.args.primary/'metadata.json', self.args.primary/'model.onnx', self.args.primary/'rejection.onnx',
                 export.ROOT/'training/train_sans_verifier.py', export.ROOT/'training/prepare_sans_views.py',
                 export.ROOT/'training/region_network.py', export.ROOT/'training/network.py']
        self.selection = {'schema': 'flux-glyph-sans-verifier-training-v1', 'calibration_passed': True, 'test_read': False,
                          'policy': copy.deepcopy(POLICY), 'families': list(FAMILIES),
                          'bindings': {str(path.resolve()): export.sha(path) for path in paths},
                          'calibration_decisions_sha256': export.sha(self.args.run/'CALIBRATION_DECISIONS.json'),
                          'baseline_sha256': export.sha(self.args.run/'BASELINE.json')}
        self.write_selection()
        metadata = copy.deepcopy(primary)
        metadata['algorithm'] = 'region-cnn64x256-consensus-v3'
        metadata['verifier'] = {'families': FAMILIES, 'temperature': POLICY['temperature'], 'gates': POLICY['gates'],
                                'model': {'path': 'verifier.onnx', 'sha256': export.sha(self.args.region/'verifier.onnx')}}
        export.dump(self.args.region/'metadata.json', metadata)
        sources = [export.ROOT/'training/export_sans_verifier.py', export.ROOT/'training/export_region_stable.py',
                   export.ROOT/'training/train_regions.py', export.ROOT/'training/calibrate_sans_verifier.py',
                   export.ROOT/'training/calibrate_sans_blocking.py']
        self.parity = {'schema': 'flux-glyph-sans-verifier-parity-v1', 'passed': True, 'test_read': False,
                       'selection_sha256': export.sha(self.args.run/'SELECTION.json'),
                       'checkpoint_sha256': export.sha(self.args.run/'verifier.pth'),
                       'source_bindings': {str(path.resolve()): export.sha(path) for path in sources},
                       'export_source_sha256': export.sha(export.ROOT/'training/export_sans_verifier.py'),
                       'metadata_sha256': export.sha(self.args.region/'metadata.json'),
                       'model_sha256': export.sha(self.args.region/'verifier.onnx'),
                       'primary_model_sha256': export.sha(self.args.primary/'model.onnx'),
                       'rejection_model_sha256': export.sha(self.args.primary/'rejection.onnx')}
        self.write_parity()

    def tearDown(self):
        self.temp.cleanup()

    def write_selection(self):
        export.dump(self.args.run/'SELECTION.json', self.selection)

    def write_parity(self):
        export.dump(self.args.run/'PARITY.json', self.parity)

    def test_unchanged_provenance_can_be_validated_without_test_data(self):
        self.assertEqual(export.validate(self.args), self.selection)
        self.assertEqual(evaluate.validate(self.args), (self.selection, self.parity))
        self.assertFalse((self.args.data/'test').exists())

    def test_failed_calibration_or_test_selected_run_is_rejected(self):
        for key, value in (('calibration_passed', False), ('test_read', True)):
            with self.subTest(key=key):
                previous = self.selection[key]
                self.selection[key] = value
                self.write_selection()
                with self.assertRaisesRegex(ValueError, 'pass unchanged CAL policy'):
                    export.validate(self.args)
                self.selection[key] = previous

    def test_modified_policy_is_rejected(self):
        self.selection['policy']['gates']['min_score'] = .25
        self.write_selection()
        with self.assertRaisesRegex(ValueError, 'unchanged CAL policy'):
            export.validate(self.args)

    def test_missing_source_binding_is_rejected(self):
        del self.selection['bindings'][str((self.args.primary/'model.onnx').resolve())]
        self.write_selection()
        with self.assertRaisesRegex(ValueError, 'inputs differ from training bindings'):
            export.validate(self.args)

    def test_other_view_directory_is_rejected_before_model_or_test_arrays(self):
        self.args.data = self.root/'other-views'
        with patch('prepare_sans_views.load_views') as loader, patch('flux_glyph.region_font.RegionFontClassifier') as model:
            with self.assertRaisesRegex(ValueError, 'inputs differ from training bindings'):
                evaluate.main(self.args)
            loader.assert_not_called()
            model.assert_not_called()

    def test_other_primary_directory_is_rejected_by_exact_path(self):
        self.args.primary = self.root/'other-primary'
        with self.assertRaisesRegex(ValueError, 'inputs differ from training bindings'):
            evaluate.validate(self.args)

    def test_source_bytes_change_invalidates_export(self):
        (self.args.primary/'rejection.onnx').write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'bound training input changed'):
            export.validate(self.args)

    def test_updated_data_hash_cannot_hide_wrong_family_order(self):
        export.dump(self.args.data/'MANIFEST.json', {'families': list(reversed(FAMILIES))})
        self.selection['bindings'][str((self.args.data/'MANIFEST.json').resolve())] = export.sha(self.args.data/'MANIFEST.json')
        self.write_selection()
        with self.assertRaisesRegex(ValueError, 'family order differs'):
            export.validate(self.args)

    def test_changed_calibration_records_are_rejected(self):
        (self.args.run/'CALIBRATION_DECISIONS.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'saved calibration changed'):
            export.validate(self.args)

    def test_changed_checkpoint_after_export_blocks_test(self):
        (self.args.run/'verifier.pth').write_bytes(b'other checkpoint')
        with self.assertRaisesRegex(ValueError, 'must be frozen before test'):
            evaluate.validate(self.args)

    def test_changed_lowering_provenance_blocks_test(self):
        self.parity['source_bindings'][str((export.ROOT/'training/export_region_stable.py').resolve())] = '0'*64
        self.write_parity()
        with self.assertRaisesRegex(ValueError, 'export/lowering source changed'):
            evaluate.validate(self.args)

    def test_changed_copied_primary_model_blocks_test(self):
        (self.args.region/'model.onnx').write_bytes(b'other font model')
        with self.assertRaisesRegex(ValueError, 'primary/rejection differs'):
            evaluate.validate(self.args)

    def test_new_metadata_hash_cannot_hide_changed_primary_thresholds(self):
        metadata = json.loads((self.args.region/'metadata.json').read_text())
        metadata['temperature'] = .25
        export.dump(self.args.region/'metadata.json', metadata)
        self.parity['metadata_sha256'] = export.sha(self.args.region/'metadata.json')
        self.write_parity()
        with self.assertRaisesRegex(ValueError, 'changed the primary font/rejection contract'):
            evaluate.validate(self.args)

    def test_incremental_roboto_prevention_excludes_already_rejected_regions(self):
        common = {'domain': 'new_native', 'view': 'native', 'family': 'Roboto', 'roboto_veto': True,
                  'before_correct': False, 'after_correct': False, 'after_wrong': False}
        details = [{**common, 'before_wrong': True}, {**common, 'before_wrong': False}]
        result = evaluate.diagnostics(details)['roboto']
        self.assertEqual(result['verifier_correct_and_above_gate'], 2)
        self.assertEqual(result['additional_wrong_names_prevented'], 1)
        self.assertEqual(result['baseline_already_not_wrongly_named'], 1)
        self.assertEqual(result['additional_prevention_rate_among_baseline_wrong_names'], 1)

    def test_nonfinite_predictions_and_out_of_bounds_tile_mapping_fail_closed(self):
        rows = [{'tile_start': 0, 'tile_count': 1}]
        with self.assertRaisesRegex(ValueError, 'nonfinite family logits'):
            observations(np.array([[np.nan, 1]]), rows, 1, POLICY['gates'])
        with self.assertRaisesRegex(ValueError, 'invalid observation tile mapping'):
            observations(np.array([[0., 1.]]), [{'tile_start': 1, 'tile_count': 1}], 1, POLICY['gates'])


if __name__ == '__main__':
    unittest.main()
