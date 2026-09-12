from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'training' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module('prepare_real_data_test', 'prepare_screenshots.py')
real = load_module('real_data_test', 'real_data.py')


class RealDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_temp.name) / 'valid'
        case_path = ROOT / 'tests/fixtures/font_accuracy/generated_manifest.json'
        case = next(row for row in json.loads(case_path.read_text())['cases']
                    if row['case_id'] == 'pingfang_21.0d1e1_regular--white_png_26')
        input_path = Path(cls.fixture_temp.name) / 'labels.jsonl'
        row = {'image': str(case_path.parent / case['input_file']), 'source_id': 'synthetic-source-1',
               'split': 'train', 'source_kind': 'source_font_synthetic_smoke',
               'regions': [{'bbox': case['expected_text_bbox'], 'text': case['text'],
                            'font_family': 'PingFang SC', 'script': 'han'}]}
        input_path.write_text(json.dumps(row, ensure_ascii=False) + '\n')
        prepare.prepare(input_path, cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / 'bundle'
        shutil.copytree(self.fixture, self.folder)
        self.families = prepare.family_names()
        self.scripts = prepare.script_families(self.families)

    def tearDown(self):
        self.temp.cleanup()

    def loader(self, **kwargs):
        return real.PreparedScreenshotData(self.folder, families=self.families, scripts=self.scripts,
                                           allow_empty_evaluation=kwargs.pop('allow_empty_evaluation', True), **kwargs)

    def change_manifest(self, edit):
        path = self.folder / 'manifest.json'
        manifest = json.loads(path.read_text())
        edit(manifest)
        path.write_text(json.dumps(manifest, ensure_ascii=False))

    def change_npz(self, edit):
        path = self.folder / 'train.npz'
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        edit(values)
        np.savez_compressed(path, **values)
        self.change_manifest(lambda m: m['splits']['train'].update(sha256=real.sha(path)))

    def test_valid_bundle_preserves_annotation_and_reports_empty_evaluation(self):
        loader = self.loader()
        self.assertEqual(loader.audit['empty_evaluation_splits'], ['calibration', 'test'])
        self.assertTrue(loader.audit['test_metadata_inspected_for_contract'])
        self.assertFalse(loader.audit['test_pixels_loaded'])
        dataset, = loader.datasets('train')
        self.assertEqual(dataset['chars'].tolist(), list('账单详情'))
        self.assertEqual(dataset['script'], 'han')
        self.assertEqual(dataset['x'].shape, (4, 64, 64))
        self.assertTrue(dataset['source']['contract_verified'])
        self.assertEqual(loader.datasets('calibration'), [])

    def test_empty_evaluation_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'empty evaluation partitions'):
            self.loader(allow_empty_evaluation=False)

    def test_empty_training_is_always_rejected(self):
        train = (self.folder / 'train.npz').read_bytes()
        calibration = (self.folder / 'calibration.npz').read_bytes()
        (self.folder / 'train.npz').write_bytes(calibration)
        (self.folder / 'calibration.npz').write_bytes(train)
        def swap(m):
            m['splits']['train'], m['splits']['calibration'] = m['splits']['calibration'], m['splits']['train']
            for split in ('train', 'calibration'):
                m['splits'][split]['file'] = split + '.npz'
            for row in m['sources'] + m['accepted_glyphs']:
                row['split'] = 'calibration'
        self.change_manifest(swap)
        with self.assertRaisesRegex(ValueError, 'train partition has no accepted'):
            self.loader()

    def test_tampered_partition_sha_is_rejected(self):
        path = self.folder / 'train.npz'
        path.write_bytes(path.read_bytes() + b'extra')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.loader()

    def test_family_order_cannot_silently_relabel_targets(self):
        self.change_manifest(lambda m: m['families'].reverse())
        with self.assertRaisesRegex(ValueError, 'family order'):
            self.loader()

    def test_script_and_target_must_match_annotation_even_with_updated_file_hash(self):
        for field, value, message in [('scripts', 'latin', 'character/script'),
                                       ('y', self.families.index('MiSans'), 'explicit glyph annotation'),
                                       ('source_ids', 'different-source', 'source_id differs')]:
            with self.subTest(field=field):
                shutil.copytree(self.fixture, self.folder, dirs_exist_ok=True)
                self.change_npz(lambda a: a[field].__setitem__(0, value))
                with self.assertRaisesRegex(ValueError, message):
                    self.loader()

    def test_source_id_and_source_hash_overlap_across_splits_is_rejected(self):
        def duplicate_source(m):
            duplicate = copy.deepcopy(m['sources'][0])
            duplicate['split'] = 'test'
            m['sources'].append(duplicate)
        self.change_manifest(duplicate_source)
        with self.assertRaisesRegex(ValueError, 'cross-split identity overlap'):
            self.loader()

    def test_mock_ocr_or_teacher_labels_cannot_be_passed_off_as_explicit_imports(self):
        for patch, message in [({'test_reader_injected': True}, 'mock/injected'),
                               ({'font_label_source': 'statusbar_teacher'}, 'annotation provenance')]:
            with self.subTest(patch=patch):
                shutil.copytree(self.fixture, self.folder, dirs_exist_ok=True)
                self.change_manifest(lambda m: m.update(patch))
                with self.assertRaisesRegex(ValueError, message):
                    self.loader()

    def test_changed_pixels_fail_per_glyph_binding_even_if_archive_hash_is_updated(self):
        self.change_npz(lambda a: a['x'].__setitem__((0, 0, 0), .5))
        loader = self.loader()
        with self.assertRaisesRegex(ValueError, 'glyph row hash'):
            loader.datasets('train')

    def test_bundle_changes_after_preflight_are_rejected(self):
        loader = self.loader()
        self.change_manifest(lambda m: m.update(extra='changed'))
        with self.assertRaisesRegex(ValueError, 'changed after preflight'):
            loader.datasets('train')


if __name__ == '__main__':
    unittest.main()
