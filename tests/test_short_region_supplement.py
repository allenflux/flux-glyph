import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from training import prepare_short_region_supplement as short


class ShortRegionSupplementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / 'unified/train'; self.train.mkdir(parents=True)
        self.capture = self.root / 'capture'; self.capture.mkdir()
        self.output = self.root / 'output'
        self.families = ['Known', '__unknown__']
        self.patches = [
            patch.object(short, 'FAMILIES', self.families),
            patch.object(short, 'EXPECTED_NATIVE_IDENTITIES', 2),
            patch.object(short, 'ALLOWED_UNKNOWN_TRAIN', {'Old Unknown'}),
            patch.object(short, 'HELD_OUT_UNKNOWN', {'Held Out'}),
            patch.object(short, 'family_label', side_effect=lambda name: name if name == 'Known' else '__unknown__'),
            patch.object(short, 'inside_repo', side_effect=lambda value: Path(value).resolve()),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(patch.stopall)
        self.image = self.capture / 'train.png'
        canvas = Image.new('RGB', (80, 40), 'white'); draw = ImageDraw.Draw(canvas)
        self.boxes = [[5, 10, 17, 30], [22, 10, 34, 30], [39, 10, 51, 30], [56, 10, 68, 30]]
        for value in self.boxes:
            draw.rectangle((value[0], value[1], value[2] - 1, value[3] - 1), fill='black')
        canvas.save(self.image)
        self.image_sha = short.sha(self.image)
        self.labels = self.capture / 'labels.jsonl'
        records, rows = [], []
        for index, (family, training_family, font_sha) in enumerate((('Known', 'Known', 'a' * 64),
                                                                      ('Old Unknown', '__unknown__', 'b' * 64))):
            source_id, region_id = f'ios:train-{index}', f'train-{index}-r0'
            glyphs = [{'text_index': offset, 'character': character, 'glyph_id': offset + 1,
                       'bbox_screen_px': value, 'visible': True, 'font_match_verified': True,
                       'font_postscript': family + '-PS', 'font_source_sha256': font_sha}
                      for offset, (character, value) in enumerate(zip('ABCD', self.boxes))]
            region = {'id': region_id, 'status': 'ok', 'text': 'ABCD', 'script': 'latin',
                      'font_family': family, 'actual_font_postscript': family + '-PS',
                      'font_match_verified': True, 'fallback_detected': False, 'bbox_pixels': [0, 0, 80, 40],
                      'font_size_screen_px': 20, 'actual_text_color_hex': '#000000FF',
                      'font_source': {'kind': 'asset', 'sha256': font_sha}, 'glyphs': glyphs,
                      'font_runs': [{'postscript_name': family + '-PS', 'family': family, 'glyph_count': 4}]}
            records.append({'schema': 'fixture', 'split': 'train', 'source_id': source_id,
                            'source_sha256': self.image_sha, 'image': str(self.image),
                            'pixel_size': [80, 40], 'regions': [region]})
            rows.append({'family': training_family, 'target': index, 'source_font_family': family,
                         'font_face': family + '-PS', 'font_file_sha256': font_sha,
                         'source_sha256': self.image_sha, 'domain': 'ios', 'source_dataset': 'fixture',
                         'page_id': f'train-{index}', 'normalized_text_sha256': short.text_sha('ABCD'),
                         'source_id': source_id, 'region_id': region_id, 'native_font_verified': True,
                         'split': 'train', 'view': 'native'})
        # A malformed non-TRAIN line proves it is not decoded by the bounded scanner.
        self.labels.write_bytes(b'{"split":"calibration",invalid}\n' +
                                b''.join((json.dumps(item) + '\n').encode() for item in records))
        (self.train / 'rows.json').write_text(json.dumps(rows))
        manifest = {'schema': 'fixture', 'split': 'train', 'families': self.families, 'views': len(rows),
                    'metadata': {'path': 'rows.json', 'sha256': short.sha(self.train / 'rows.json')}}
        (self.train / 'MANIFEST.json').write_text(json.dumps(manifest))
        self.sources = {'ios_fixture': self.labels}

    def test_plan_only_never_opens_images_and_selects_every_available_length(self):
        with patch.object(short.Image, 'open') as opened:
            plan = short.prepare(self.train, self.output, sources=self.sources, plan_only=True)
        opened.assert_not_called()
        self.assertFalse((self.output / 'MANIFEST.json').exists())
        self.assertEqual(plan['coverage']['selected_subcrops'], 20)
        self.assertEqual(plan['coverage']['planned_views'], 80)
        self.assertEqual(plan['coverage']['selected_by_family_length']['Known'],
                         {'1': 4, '2': 3, '3': 2, '4': 1})
        self.assertTrue(all(item['split'] == 'train' for item in plan['candidates']))
        self.assertNotIn('text', plan['candidates'][0])

    def test_materialize_and_load_keep_four_one_tile_views_and_size_truth(self):
        short.prepare(self.train, self.output, sources=self.sources, plan_only=True)
        short.prepare(self.train, self.output, sources=self.sources)
        loaded = short.load_supplement(self.output)
        self.assertIsInstance(loaded['tiles'], np.memmap)
        self.assertEqual(len(loaded['rows']), 80)
        self.assertEqual(loaded['partition']['native_subcrops'], 20)
        self.assertLessEqual(loaded['partition']['bytes'], short.MAX_OUTPUT_BYTES)
        self.assertEqual(set(row['view'] for row in loaded['rows']), set(short.VIEWS))
        self.assertTrue(all(row['tile_count'] == 1 and row['native_font_verified'] is True
                            and row['index'] == index and row['glyph_count'] == row['selected_glyph_count']
                            and row['parent_source_id'] == row['source_id']
                            and row['parent_region_id'] != row['region_id']
                            for index, row in enumerate(loaded['rows'])))
        for row in loaded['rows']:
            self.assertAlmostEqual(np.exp(row['log_em_ratio']) * row['ink_height_px'], row['font_size_px'])

    def test_existing_plan_and_materialized_output_cannot_be_overwritten(self):
        short.prepare(self.train, self.output, sources=self.sources, plan_only=True)
        with self.assertRaisesRegex(ValueError, 'plan output must be new'):
            short.prepare(self.train, self.output, sources=self.sources, plan_only=True)
        short.prepare(self.train, self.output, sources=self.sources)
        before = short.sha(self.output / 'train/tiles.raw')
        with self.assertRaisesRegex(ValueError, 'must not be overwritten'):
            short.prepare(self.train, self.output, sources=self.sources)
        self.assertEqual(before, short.sha(self.output / 'train/tiles.raw'))

    def test_loader_rejects_coherently_rehashed_wrong_font_row(self):
        short.prepare(self.train, self.output, sources=self.sources, plan_only=True)
        short.prepare(self.train, self.output, sources=self.sources)
        path = self.output / 'train/rows.json'; rows = json.loads(path.read_text())
        rows[0]['family'] = 'Known' if rows[0]['family'] == '__unknown__' else '__unknown__'
        rows[0]['target'] = self.families.index(rows[0]['family'])
        path.write_text(json.dumps(rows))
        part_path = self.output / 'train/MANIFEST.json'; part = json.loads(part_path.read_text())
        part['metadata']['sha256'] = short.sha(path)
        part['counts'] = dict(short.Counter(row['family'] for row in rows))
        part['native_counts'] = dict(short.Counter(row['family'] for row in rows if row['view'] == 'native'))
        part_path.write_text(json.dumps(part))
        with self.assertRaisesRegex(ValueError, 'frozen candidate truth'):
            short.load_supplement(self.output)

    def test_source_change_during_materialization_cannot_publish_output(self):
        short.prepare(self.train, self.output, sources=self.sources, plan_only=True)
        original = short.apply_view; changed = False
        def mutate(image, view):
            nonlocal changed
            if not changed:
                changed = True
                self.labels.write_text(self.labels.read_text() + '\n')
            return original(image, view)
        with patch.object(short, 'apply_view', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during materialization'):
                short.prepare(self.train, self.output, sources=self.sources)
        self.assertFalse((self.output / 'MANIFEST.json').exists())
        self.assertFalse((self.output / 'train').exists())

    def test_crop_padding_shrinks_and_never_intersects_neighbor(self):
        selected = [[10, 10, 20, 30]]; neighbor = [[21, 10, 31, 30]]
        crop, padding = short.safe_crop(selected, neighbor, [0, 0, 50, 40], 50, 40)
        self.assertEqual((crop, padding), ([9, 9, 21, 31], 1))
        overlap = [[19, 10, 31, 30]]
        self.assertEqual(short.safe_crop(selected, overlap, [0, 0, 50, 40], 50, 40), (None, None))

    def test_selected_union_outside_parent_is_rejected_without_clamping_glyph(self):
        selected = [[8, 10, 20, 30]]
        self.assertEqual(short.safe_crop(selected, [], [10, 5, 40, 35], 50, 40), (None, None))
        selected = [[10, 3, 20, 30]]
        self.assertEqual(short.safe_crop(selected, [], [10, 5, 40, 35], 50, 40), (None, None))

    def test_punctuation_only_short_fragment_is_rejected(self):
        parent = {'family': 'Known', 'target': 0, 'domain': 'ios', 'source_dataset': 'fixture',
                  'source_font_family': 'Known', 'font_face': 'Known-PS', 'font_file_sha256': 'a' * 64,
                  'font_source_kind': 'asset', 'ttc_index': 0, 'source_id': 'ios:page', 'region_id': 'page-r0',
                  'page_id': 'page', 'image': str(self.image), 'source_sha256': self.image_sha,
                  'image_width': 80, 'image_height': 40, 'parent_bbox': [0, 0, 80, 40],
                  'native_font_size_px': 20, 'text_color_hex': '#000000FF', 'mapping': 'fixture',
                  'proof': str(self.labels), 'proof_sha256': short.sha(self.labels),
                  'glyphs': [{'proof_index': 0, 'text_index': 0, 'character': '-', 'glyph_id': 1,
                              'bbox': self.boxes[0]}], 'full_visible_bboxes': [self.boxes[0]]}
        rejected = short.Counter()
        self.assertEqual(list(short.candidate_rows(parent, rejected)), [])
        self.assertEqual(rejected, {'punctuation_only_fragment': 1})

    def test_android_rejects_non_bmp_or_non_one_to_one_glyph_mapping(self):
        parent = {'family': 'Known', 'target': 0, 'source_font_family': 'Known', 'font_face': 'Known-PS',
                  'font_file_sha256': 'a' * 64, 'source_sha256': self.image_sha, 'domain': 'android',
                  'source_dataset': 'fixture', 'page_id': 'page', 'normalized_text_sha256': short.text_sha('AB'),
                  'source_id': 'android:page', 'region_id': 'page-r0'}
        proof_path = self.capture / 'proof.json'
        label = {'split': 'train', 'source_id': 'android:page', 'page_id': 'page', 'region_id': 'page-r0',
                 'image': 'train.png', 'image_width': 80, 'image_height': 40, 'image_sha256': self.image_sha,
                 'proof': 'proof.json', 'bbox': [0, 0, 80, 40], 'text': 'AB', 'font_id': 'font',
                 'font_family': 'Known', 'training_family': 'Known', 'font_face': 'Known-PS',
                 'font_file_sha256': 'a' * 64, 'ttc_index': 0, 'font_size_screen_px': 20,
                 'color': '#000000', 'native_font_verified': True}
        glyph = {'glyph_id': 1, 'font_verified': True, 'position': [5, 20], 'bbox': self.boxes[0],
                 'font': {'postscript': 'Known-PS', 'sha256': 'a' * 64, 'ttc_index': 0}}
        proof_path.write_text(json.dumps({'split': 'train', 'page_id': 'page', 'regions': [
            {'id': 'page-r0', 'text': 'AB', 'font_id': 'font', 'font_verified': True, 'glyphs': [glyph]}]}))
        with self.assertRaisesRegex(short.CandidateRejected, 'glyph_count_mismatch'):
            short.android_parent(parent, label, self.labels, {})
        second = {**glyph, 'glyph_id': 2, 'position': [4, 20], 'bbox': self.boxes[1]}
        proof_path.write_text(json.dumps({'split': 'train', 'page_id': 'page', 'regions': [
            {'id': 'page-r0', 'text': 'AB', 'font_id': 'font', 'font_verified': True, 'glyphs': [glyph, second]}]}))
        with self.assertRaisesRegex(short.CandidateRejected, 'order'):
            short.android_parent(parent, label, self.labels, {})
        label['text'] = 'A😀'; parent['normalized_text_sha256'] = short.text_sha(label['text'])
        proof_path.write_text(json.dumps({'split': 'train', 'page_id': 'page', 'regions': [
            {'id': 'page-r0', 'text': label['text'], 'font_id': 'font', 'font_verified': True, 'glyphs': [glyph, glyph]}]}))
        with self.assertRaisesRegex(short.CandidateRejected, 'non_bmp'):
            short.android_parent(parent, label, self.labels, {})

    def test_held_out_unknown_family_is_rejected_from_train_plan(self):
        rows = json.loads((self.train / 'rows.json').read_text())
        rows[1]['source_font_family'] = 'Held Out'; rows[1]['font_face'] = 'Held Out-PS'
        (self.train / 'rows.json').write_text(json.dumps(rows))
        manifest = json.loads((self.train / 'MANIFEST.json').read_text())
        manifest['metadata']['sha256'] = short.sha(self.train / 'rows.json')
        (self.train / 'MANIFEST.json').write_text(json.dumps(manifest))
        lines = self.labels.read_bytes().splitlines(); record = json.loads(lines[2])
        region = record['regions'][0]; region['font_family'] = 'Held Out'; region['actual_font_postscript'] = 'Held Out-PS'
        for glyph in region['glyphs']:
            glyph['font_postscript'] = 'Held Out-PS'
        self.labels.write_bytes(lines[0] + b'\n' + lines[1] + b'\n' + (json.dumps(record) + '\n').encode())
        with self.assertRaisesRegex(ValueError, 'held-out unknown family'):
            short.build_plan(self.train, self.sources)


if __name__ == '__main__':
    unittest.main()
