"""Unit fixtures exercise contracts; they are not actual capture datasets."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    spec = importlib.util.spec_from_file_location('native_train', ROOT / 'training/train_screenshots.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@unittest.skipUnless(HAS_TORCH, 'Run with the standalone torch training environment')
class ScreenshotTrainingTests(unittest.TestCase):
    def fixture(self, root):
        """Invented data for structural unit testing, never passed to main()."""
        def write(path, value):
            path.write_text(json.dumps(value))
            return {'path': str(path), 'sha256': module.sha(path)}
        scenes = write(root / 'scenes.json', {'test_fixture_only': True})
        labels = write(root / 'labels.json', {'test_fixture_only': True})
        protocol = write(root / 'protocol.json', {'schema': 'ios-native-screen-capture-v1',
                         'source_kind': module.SOURCE_KIND, 'scenes_sha256': scenes['sha256']})
        manifest = {'schema': 'flux-glyph-native-captured-glyphs-v1', 'families': module.FAMILIES,
                    'scripts': module.SCRIPTS, 'splits': {}, 'sources': [], 'source_kind_counts': {module.SOURCE_KIND: 3},
                    'screenshot_count': 3, 'desktop_rendered_pixels_accepted': False, 'instrumented_boxes_used': True,
                    'accepted_native_verified_only': True, 'image_source': 'simctl_png', 'ui_content_is_generated': True,
                    'ocr_performed': False, 'input_labels': labels, 'scenes': scenes, 'capture_protocol': protocol,
                    'capture_protocol_sha256': protocol['sha256'], 'split_isolation': dict.fromkeys(module.IDENTITIES, 0),
                    'preprocessing': {'function': 'flux_glyph.neural_font.preprocess_glyph', 'padding_pixels': 4,
                                      'shape': [64, 64], 'dtype': 'float32', 'range': [0, 1],
                                      'runtime_source_sha256': module.sha(ROOT / 'src/flux_glyph/neural_font.py'),
                                      'glyph_extraction_source_sha256': module.sha(ROOT / 'src/flux_glyph/glyph_preprocess.py')}}
        for si, split in enumerate(module.SPLITS):
            folder = root / split
            folder.mkdir()
            raw = root / (split + '.png')
            raw.write_bytes(('not an image: unit fixture ' + split).encode())
            source = {'source_id': split, 'page_id': split, 'content_group_id': split, 'split': split,
                      'source_kind': module.SOURCE_KIND, 'image_path': str(raw), 'source_sha256': module.sha(raw),
                      'decoded_pixel_sha256': hashlib.sha256(split.encode()).hexdigest(),
                      'native_frame_sha256': 'e' * 64, 'scenes_sha256': scenes['sha256'], 'regions': []}
            rows = []
            x = np.random.default_rng(si).random((4, 64, 64), dtype=np.float32)
            for i, (family, script) in enumerate([('PingFang SC', 'han'), ('Noto Sans CJK SC', 'han'),
                                                 ('SF Pro', 'latin'), ('Helvetica', 'latin')]):
                text = chr(0x4e00 + si * 10 + i) if script == 'han' else chr(65 + si * 10 + i)
                rsha = hashlib.sha256((split + str(i)).encode()).hexdigest()
                rid, box = str(i), [0, 0, 10, 10]
                region = {'id': rid, 'text': text, 'script': script, 'font_family': family,
                          'font_match_verified': True, 'fallback_detected': False,
                          'region_rgb_sha256': rsha, 'bbox': box}
                source['regions'].append(region)
                native = {'font_match_verified': True, 'visible': True, 'font_family': family, 'character': text,
                          'font_postscript': family + '-TestOnly', 'text_index': 0, 'bbox': box, 'glyph_id': 1}
                rows.append({'array_row': i, 'target': module.FAMILIES.index(family), 'family': family, 'script': script,
                             'character': text, 'text': text, 'text_index': 0, 'source_kind': module.SOURCE_KIND,
                             'source_id': split, 'page_id': split, 'content_group_id': split,
                             'source_sha256': source['source_sha256'], 'decoded_pixel_sha256': source['decoded_pixel_sha256'],
                             'scenes_sha256': scenes['sha256'], 'region_id': rid, 'region_rgb_sha256': rsha,
                             'region_bbox': box, 'bbox': box, 'font_face': native['font_postscript'],
                             'native_glyph_provenance': native, 'glyph_sha256': hashlib.sha256(x[i].tobytes()).hexdigest()})
            metadata = write(folder / 'metadata.json', {'schema': 'flux-glyph-native-captured-metadata-v1', 'split': split, 'rows': rows})
            metadata['path'] = split + '/metadata.json'
            np.save(folder / 'glyphs.npy', x)
            manifest['splits'][split] = {'rows': 4, 'metadata': metadata,
                                         'glyphs': {'path': split + '/glyphs.npy', 'shape': [4, 64, 64], 'dtype': 'float32',
                                                    'sha256': module.sha(folder / 'glyphs.npy')},
                                         'family_counts': dict(Counter(row['family'] for row in rows)),
                                         'script_counts': dict(Counter(row['script'] for row in rows))}
            manifest['sources'].append(source)
        write(root / 'MANIFEST.json', manifest)
        return manifest

    def test_loader_registry_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            bundle = module.CapturedData(root)
            train = bundle.load('train')
            bundle.load('calibration')
            self.assertIsInstance(train['x'], np.memmap)
            self.assertEqual(bundle.families, ['Noto Sans CJK SC', 'PingFang SC', 'SF Pro', 'Helvetica'])
            self.assertFalse(bundle.audit['test_glyph_pixels_loaded'])
            path = root / 'test/glyphs.npy'
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 1
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                bundle.load('test')

    def test_duplicate_source_group_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            manifest['sources'][1]['content_group_id'] = manifest['sources'][0]['content_group_id']
            (root / 'MANIFEST.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'content_group_id crosses'):
                module.CapturedData(root)

    def test_region_uses_distinct_characters_and_source_groups(self):
        rows = [{'source_id': source, 'region_id': 'shared-name', 'script': 'latin', 'family': 'SF Pro',
                 'text': text, 'character': char, 'text_index': index}
                for source, text in [('page-a', 'AAB'), ('page-b', 'B')]
                for index, char in enumerate(text)]
        data = {'scripts': np.asarray(['latin'] * 4), 'y': np.zeros(4, dtype=np.int64), 'regions': module.group_regions(rows)}
        logits = np.log(np.asarray([[.9, .1], [.9, .1], [.1, .9], [.2, .8]]))
        glyph, regions = module.observations(logits, data, 'latin', ['SF Pro', 'Helvetica'], {'latin': ['SF Pro', 'Helvetica']})
        np.testing.assert_allclose(regions['probabilities'], [[.5, .5], [.2, .8]])
        self.assertEqual(len(regions['y']), 2)
        self.assertEqual(len(glyph['y']), 4)
        self.assertTrue(regions['complete'].all())

    def test_incomplete_regions_cannot_pass_gate(self):
        row = {'source_id': 'a', 'region_id': 'r', 'script': 'han', 'family': 'PingFang SC',
               'text': '中文', 'character': '中', 'text_index': 0}
        data = {'scripts': np.asarray(['han']), 'y': np.zeros(1, dtype=np.int64), 'regions': module.group_regions([row])}
        _, region = module.observations(np.asarray([[10., 0.]]), data, 'han', ['PingFang SC', 'Noto Sans CJK SC'],
                                        {'han': ['PingFang SC', 'Noto Sans CJK SC']})
        result = module.summary(region, ['PingFang SC', 'Noto Sans CJK SC'], {'min_score': .5, 'min_margin': .01})
        self.assertEqual(result['complete_rows'], 0)
        self.assertEqual(result['accepted']['rows'], 0)

    def test_balanced_sampler_visits_all_rows(self):
        data = {'scripts': np.asarray(['han'] * 20 + ['latin'] * 20),
                'y': np.asarray([0] * 17 + [1] * 3 + [2] * 15 + [3] * 5), 'x': np.empty(40)}
        sampler = module.BalancedSampler(data, 1)
        for _ in range(10):
            indices = sampler.batch(8)
            self.assertEqual(Counter(data['scripts'][indices]), {'han': 4, 'latin': 4})
            self.assertEqual(Counter(data['y'][indices]), {0: 2, 1: 2, 2: 2, 3: 2})
        self.assertTrue((sampler.visits > 0).all())

    def test_region_calibration_is_not_blocked_by_ambiguous_glyphs(self):
        rows, logits = [], []
        # One misleading glyph in each three-character region. All individual
        # glyphs have identical confidence/margin, so no glyph gate can retain
        # >=97% accuracy. Distinct-character region votes are all correct.
        for region in range(24):
            for index, character in enumerate('ABC'):
                rows.append({'source_id': 'page-' + str(region), 'region_id': 'r', 'script': 'latin',
                             'family': 'SF Pro', 'text': 'ABC', 'character': character, 'text_index': index})
                logits.append([3., 0.] if index < 2 else [0., 3.])
        data = {'scripts': np.asarray(['latin'] * len(rows)), 'y': np.zeros(len(rows), dtype=np.int64),
                'regions': module.group_regions(rows)}
        _, _, report = module.calibrate(np.asarray(logits), data, ['SF Pro', 'Helvetica'],
                                       {'latin': ['SF Pro', 'Helvetica']})
        result = report['latin']
        self.assertTrue(result['gate_found'])
        self.assertEqual(result['region']['accepted']['rows'], 24)
        self.assertEqual(result['region']['accepted']['precision'], 1.)
        self.assertLess(result['glyph']['accepted']['precision'], .97)
        self.assertEqual(result['gate_calibration_level'], 'complete_native_region')

    def test_warm_start_slices_head_by_name(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.pth'
            net = module.FontClassifier()
            torch.save({'families': module.FAMILIES, 'state_dict': net.state_dict()}, path)
            chosen = ['Noto Sans CJK SC', 'PingFang SC', 'SF Pro', 'Helvetica']
            sliced = module.active_model(chosen, path)
            self.assertEqual(sliced.family_head.out_features, 4)
            self.assertTrue(torch.equal(sliced.trunk[0].weight, net.trunk[0].weight))
            expected = net.family_head.weight[[module.FAMILIES.index(f) for f in chosen]]
            self.assertTrue(torch.equal(sliced.family_head.weight, expected))


if __name__ == '__main__':
    unittest.main()
