from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_screenshots', ROOT / 'training/prepare_screenshots.py')
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


class Reader:
    def __init__(self, text='中A文', confidence=.99, missing_token=False):
        self.calls = 0
        self.text = text
        self.confidence = confidence
        self.missing_token = missing_token

    def read(self, images):
        self.calls += 1
        tokens = [{'character': c, 'index': i, 'start_step': x, 'end_step': x + 1,
                   'peak_step': x, 'confidence': .99}
                  for i, (c, x) in enumerate(zip(self.text, (14, 39, 61)))]
        if self.missing_token:
            tokens = []
        return [{'text': self.text, 'confidence': self.confidence, 'tokens': tokens,
                 'metadata': {'timesteps': 80, 'input_width': 80, 'content_width': 80,
                              'orientation_degrees': 0}}]


class ScreenshotPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        image = Image.new('RGB', (80, 30), 'white')
        draw = ImageDraw.Draw(image)
        for box in [(5, 5, 23, 25), (34, 5, 45, 25), (51, 5, 71, 25)]:
            draw.rectangle(box, fill='black')
        self.image = self.folder / 'source.png'
        image.save(self.image)
        self.row = {'image': str(self.image), 'source_id': 'source-1', 'split': 'train',
                    'source_kind': 'synthetic_software_test',
                    'regions': [{'bbox': [0, 0, 80, 30], 'text': '中A文',
                                 'font_family': 'PingFang SC', 'script': 'han'}]}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        path = self.folder / 'labels.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        return path

    def test_mixed_text_font_label_applies_only_to_explicit_script(self):
        path = self.write([self.row])
        report = prepare_module.prepare(path, self.folder / 'out', reader=Reader())
        with np.load(self.folder / 'out/train.npz', allow_pickle=False) as data:
            self.assertEqual(data['chars'].tolist(), ['中', '文'])
            self.assertEqual(data['x'].shape, (2, 64, 64))
            self.assertEqual(data['x'].dtype, np.float32)
            self.assertEqual(data['y'].tolist(), [report['families'].index('PingFang SC')] * 2)
            self.assertEqual(data['scripts'].tolist(), ['han', 'han'])
        self.assertTrue(report['test_reader_injected'])
        self.assertFalse(report['independent_device_or_font_truth_verified_by_importer'])
        self.assertEqual(report['splits']['calibration']['rows'], 0)

    def test_missing_tokens_are_rejected_without_equal_width_fallback(self):
        report = prepare_module.prepare(self.write([self.row]), self.folder / 'out',
                                        reader=Reader(missing_token=True))
        self.assertEqual(report['accepted_count'], 0)
        self.assertEqual(report['rejection_counts'], {'segmentation_rejected': 2})

    def test_wrong_ocr_text_is_not_replaced_by_annotation(self):
        report = prepare_module.prepare(self.write([self.row]), self.folder / 'out', reader=Reader('中B文'))
        self.assertEqual(report['accepted_count'], 0)
        self.assertEqual(report['rejection_counts'], {'ocr_text_mismatch': 1})

    def test_whitespace_only_difference_does_not_relabel_mixed_latin(self):
        self.row['regions'][0]['text'] = '中 A 文'
        self.row['regions'][0]['script'] = 'latin'
        self.row['regions'][0]['font_family'] = 'Roboto'
        report = prepare_module.prepare(self.write([self.row]), self.folder / 'out', reader=Reader())
        self.assertEqual([row['character'] for row in report['accepted_glyphs']], ['A'])
        self.assertEqual(report['splits']['train']['family_counts'], {'Roboto': 1})

    def test_source_ids_and_hashes_cannot_cross_splits_before_ocr(self):
        for use_same_id in (False, True):
            with self.subTest(same_id=use_same_id):
                other = json.loads(json.dumps(self.row))
                other['split'] = 'test'
                other['source_id'] = self.row['source_id'] if use_same_id else 'renamed'
                if use_same_id:
                    changed = Image.open(self.image).convert('RGB')
                    changed.putpixel((0, 0), (0, 0, 0))
                    changed.save(self.folder / 'different.png')
                    other['image'] = str(self.folder / 'different.png')
                reader = Reader()
                with self.assertRaisesRegex(ValueError, 'Cross-split identity leakage'):
                    prepare_module.prepare(self.write([self.row, other]), self.folder / 'out', reader=reader)
                self.assertEqual(reader.calls, 0)
                self.assertFalse((self.folder / 'out').exists())

    def test_reencoded_identical_pixels_cannot_cross_splits(self):
        with Image.open(self.image) as image:
            image.save(self.folder / 'reencoded.png', compress_level=0)
        other = dict(self.row, image=str(self.folder / 'reencoded.png'), source_id='different-id', split='calibration')
        self.assertNotEqual(prepare_module.sha_bytes(self.image.read_bytes()),
                            prepare_module.sha_bytes((self.folder / 'reencoded.png').read_bytes()))
        with self.assertRaisesRegex(ValueError, 'source_rgb_sha256'):
            prepare_module.load_labels(self.write([self.row, other]), prepare_module.family_names())

    def test_missing_or_platform_label_and_invalid_bbox_are_rejected(self):
        for patch in ({'font_family': 'ios'}, {'font_family': 'android'}, {'font_family': None},
                      {'bbox': [-1, 0, 80, 30]}, {'bbox': [0, 0, 81, 30]},
                      {'bbox': [0, 0, 80.0, 30]}, {'script': 'latin', 'text': '中文'}):
            with self.subTest(patch=patch):
                row = json.loads(json.dumps(self.row))
                row['regions'][0].update(patch)
                with self.assertRaises(ValueError):
                    prepare_module.load_labels(self.write([row]), prepare_module.family_names())
        del self.row['regions'][0]['font_family']
        with self.assertRaisesRegex(ValueError, 'font_family'):
            prepare_module.load_labels(self.write([self.row]), prepare_module.family_names())

    def test_script_masks_and_supported_han_range_match_runtime(self):
        for family, script, text in [('PingFang SC', 'latin', 'ABC'),
                                     ('SF Pro', 'han', '中文')]:
            with self.subTest(family=family, script=script):
                row = json.loads(json.dumps(self.row))
                row['regions'][0].update(font_family=family, script=script, text=text)
                with self.assertRaisesRegex(ValueError, 'script mask'):
                    prepare_module.load_labels(self.write([row]), prepare_module.family_names())
        self.assertIsNone(prepare_module.scope('\u3400'))
        self.assertEqual(prepare_module.scope('\u4e00'), 'han')
        self.row['regions'][0]['text'] = '\u3400'
        with self.assertRaisesRegex(ValueError, 'no characters'):
            prepare_module.load_labels(self.write([self.row]), prepare_module.family_names())

    def test_real_pp_reader_smoke_uses_source_font_synthetic_fixture(self):
        """Actual OCR + gap segmentation, explicitly not a real-device sample."""
        case_path = ROOT / 'tests/fixtures/font_accuracy/generated_manifest.json'
        case = next(row for row in json.loads(case_path.read_text())['cases']
                    if row['case_id'] == 'pingfang_21.0d1e1_regular--white_png_26')
        row = {'image': str(case_path.parent / case['input_file']), 'source_id': case['case_id'],
               'split': 'train', 'source_kind': 'source_font_synthetic_smoke',
               'regions': [{'bbox': case['expected_text_bbox'], 'text': case['text'],
                            'font_family': case['expected_family'], 'script': 'han'}]}
        report = prepare_module.prepare(self.write([row]), self.folder / 'actual-ocr')
        self.assertFalse(report['test_reader_injected'])
        self.assertEqual([row['character'] for row in report['accepted_glyphs']], list(case['text']))
        self.assertEqual(report['splits']['train']['rows'], 4)
        self.assertEqual(report['rejected'], [])


if __name__ == '__main__':
    unittest.main()
