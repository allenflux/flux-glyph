"""Audit split retention, native size targets and integrity of derived views."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from training import prepare_sans_views as views


FAMILIES = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans',
            'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number', 'Roboto']


class FakeDataset:
    """Small source fixtures; native collection validation is mocked explicitly."""
    def __init__(self, directory, domain):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.families = FAMILIES if domain == 'new_native' else FAMILIES[:-1]
        self.rows, self.sources, self.loaded = {}, {}, []
        partitions = {}
        for split_index, split in enumerate(views.SPLITS):
            source_id = domain + '-' + split
            image = Image.new('RGB', (131, 47), 'white')
            draw = ImageDraw.Draw(image)
            for number in range(6):
                x = 8 + number * 18
                draw.rectangle((x, 8 + split_index, x + 4 + (domain == 'anchor'), 37), fill='#333333')
                draw.rectangle((x, 10 + number, x + 10, 14 + number + split_index), fill='#333333')
            image_path = self.directory / (split + '.png')
            image.save(image_path)
            source = {'source_id': source_id, 'page_id': source_id + '-page', 'split': split,
                'source_kind': 'ios_simulator_screenshot', 'image': str(image_path),
                'source_sha256': views.sha(image_path), 'decoded_pixel_sha256': views.pixels_sha(image),
                'content_group_id': source_id + '-content'}
            self.sources[source_id] = source
            family = 'Roboto' if domain == 'new_native' else 'SF Pro'
            row = {'index': 0, 'source_id': source_id, 'region_id': source_id + '-region',
                'page_id': source['page_id'], 'content_group_id': source['content_group_id'],
                'source_sha256': source['source_sha256'], 'decoded_pixel_sha256': source['decoded_pixel_sha256'],
                'region_rgb_sha256': views.pixels_sha(image), 'bbox': [0, 0, 131, 47],
                'family': family, 'native_font_family': family, 'font_face': family + '-Regular',
                'font_size_screen_px': 36.0, 'split': split}
            self.rows[split] = [row]
            folder = self.directory / split
            folder.mkdir()
            views.dump(folder / 'metadata.json', {'rows': [row]})
            partitions[split] = {'metadata': {'path': split + '/metadata.json',
                                            'sha256': views.sha(folder / 'metadata.json')}}
        self.manifest = {'families': self.families, 'sources': list(self.sources.values()), 'splits': partitions}
        self.manifest_path = self.directory / 'MANIFEST.json'
        views.dump(self.manifest_path, self.manifest)
        self.manifest_sha = views.sha(self.manifest_path)

    def load(self, split):
        self.loaded.append(split)
        return {'rows': self.rows[split]}


class SansViewsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.anchor = FakeDataset(self.root / 'anchor', 'anchor')
        self.data = FakeDataset(self.root / 'new', 'new_native')
        self.snapshot = self.root / 'snapshot.py'
        self.snapshot.write_text('# test snapshot\n')
        self.output = self.root / 'views'
        self.open_patch = patch.object(views, '_open_dataset', side_effect=self.open_dataset)
        self.open_patch.start()

    def tearDown(self):
        self.open_patch.stop()
        self.temporary.cleanup()

    def open_dataset(self, directory, snapshot):
        return self.anchor if Path(directory) == self.anchor.directory else self.data

    def prepare(self, split='train', frozen=None):
        return views.prepare_views(self.data.directory, self.anchor.directory, self.snapshot,
                                   self.output, split, frozen)

    def rewrite_partition_metadata(self, callback):
        folder = self.output / 'train'
        metadata = json.loads((folder / 'metadata.json').read_text())
        callback(metadata['rows'])
        views.dump(folder / 'metadata.json', metadata)
        part = json.loads((folder / 'MANIFEST.json').read_text())
        part['metadata']['sha256'] = views.sha(folder / 'metadata.json')
        views.dump(folder / 'MANIFEST.json', part)

    def test_test_guard_precedes_any_dataset_open(self):
        with patch.object(views, '_open_dataset') as opened:
            with self.assertRaisesRegex(ValueError, 'requires --frozen-selection'):
                self.prepare('test')
            opened.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_or_previously_test_selected_checkpoint_cannot_open_test(self):
        path = self.root / 'selection.json'
        for value in ({'calibration_passed': False, 'test_read': False},
                      {'calibration_passed': True, 'test_read': True}, {'passed': True}):
            views.dump(path, value)
            with patch.object(views, '_open_dataset') as opened:
                with self.assertRaisesRegex(ValueError, 'pass calibration'):
                    self.prepare('test', path)
                opened.assert_not_called()

    def test_actual_resize_ratio_accounts_for_integer_pixel_rounding(self):
        image = Image.new('RGB', (131, 47), 'white')
        changed, vertical, horizontal = views.apply_view(image, 'half')
        self.assertEqual(changed.size, (66, 24))
        self.assertEqual(vertical, 24 / 47)
        self.assertEqual(horizontal, 66 / 131)
        self.assertNotEqual(vertical, .5)

    def test_anchor_sampling_is_seeded_without_replacement_and_eval_is_complete(self):
        rows = [{'family': family, 'source_id': f'{i:04}', 'region_id': family + str(i)}
                for family in ('PingFang', 'SF Pro') for i in range(500)]
        selected = views.select_rows(rows, 'train', 'anchor')
        self.assertEqual(Counter(r['family'] for r in selected), {'PingFang': 384, 'SF Pro': 384})
        self.assertEqual(selected, views.select_rows(list(reversed(rows)), 'train', 'anchor'))
        self.assertEqual(len({(r['source_id'], r['region_id']) for r in selected}), 768)
        self.assertEqual(len(views.select_rows(rows, 'calibration', 'anchor')), 1000)
        self.assertEqual(len(views.select_rows(rows, 'train', 'new_native')), 1000)

    def test_roundtrip_keeps_native_truth_and_four_views_and_appends_immutably(self):
        self.prepare()
        before = views.sha(self.output / 'MANIFEST.json')
        train_before = views.sha(self.output / 'train/MANIFEST.json')
        self.prepare('calibration')
        self.assertEqual(before, views.sha(self.output / 'MANIFEST.json'))
        self.assertEqual(train_before, views.sha(self.output / 'train/MANIFEST.json'))
        loaded = views.load_views(self.output, 'train')
        self.assertEqual(loaded['split'], 'train')
        self.assertIsInstance(loaded['tiles'], np.memmap)
        self.assertEqual(len(loaded['rows']), 8)
        self.assertEqual(set(loaded['targets']), {5, 8})
        self.assertEqual(set(r['view'] for r in loaded['rows']), set(views.RECIPES))
        half = next(r for r in loaded['rows'] if r['view'] == 'half')
        self.assertAlmostEqual(half['font_size_screen_px'], 36 * 24 / 47)
        self.assertAlmostEqual(np.exp(half['log_em_ratio']) * half['ink_height_px'], half['font_size_screen_px'])
        self.assertNotIn('test', self.anchor.loaded)
        self.assertNotIn('test', self.data.loaded)

    def test_duplicate_preparation_cannot_overwrite_existing_partition(self):
        self.prepare()
        before = views.sha(self.output / 'train/tiles.npy')
        with self.assertRaisesRegex(ValueError, 'overwriting is forbidden'):
            self.prepare()
        self.assertEqual(before, views.sha(self.output / 'train/tiles.npy'))

    def test_loader_rechecks_native_png_bytes(self):
        self.prepare()
        source = next(s for s in self.anchor.sources.values() if s['split'] == 'train')
        with Path(source['image']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'native source PNG changed'):
            views.load_views(self.output, 'train')

    def test_relabel_with_consistent_new_target_still_fails_native_truth_check(self):
        self.prepare()
        self.rewrite_partition_metadata(lambda rows: rows[0].update(family='PingFang', target=4))
        with self.assertRaisesRegex(ValueError, 'native truth/source differs'):
            views.load_views(self.output, 'train')

    def test_split_tamper_is_rejected_even_if_metadata_file_hash_is_updated(self):
        self.prepare()
        self.rewrite_partition_metadata(lambda rows: rows[0].update(split='test'))
        with self.assertRaisesRegex(ValueError, 'view tile indexing differs'):
            views.load_views(self.output, 'train')

    def test_tensor_tamper_fails_individual_view_hash_after_file_hash_update(self):
        self.prepare()
        folder = self.output / 'train'
        tiles = np.load(folder / 'tiles.npy')
        tiles[0, 0, 0, 0] = .123
        np.save(folder / 'tiles.npy', tiles)
        part = json.loads((folder / 'MANIFEST.json').read_text())
        part['array']['sha256'] = views.sha(folder / 'tiles.npy')
        views.dump(folder / 'MANIFEST.json', part)
        with self.assertRaisesRegex(ValueError, 'tensor pixels/hash differ'):
            views.load_views(self.output, 'train')

    def test_cross_dataset_split_collision_is_blocked_before_any_partition_is_loaded(self):
        source = next(s for s in self.data.sources.values() if s['split'] == 'test')
        original = next(s for s in self.anchor.sources.values() if s['split'] == 'train')
        source['source_sha256'] = original['source_sha256']
        with self.assertRaisesRegex(ValueError, 'crosses dataset splits'):
            self.prepare()
        self.assertEqual(self.anchor.loaded, [])
        self.assertEqual(self.data.loaded, [])

    def test_test_append_binds_the_passed_frozen_selection(self):
        self.prepare()
        before = views.sha(self.output / 'MANIFEST.json')
        selection = self.root / 'selection.json'
        views.dump(selection, {'calibration_passed': True, 'test_read': False})
        self.prepare('test', selection)
        loaded = views.load_views(self.output, 'test')
        self.assertEqual(loaded['partition_manifest']['frozen_selection']['sha256'], views.sha(selection))
        self.assertEqual(before, views.sha(self.output / 'MANIFEST.json'))
        views.dump(selection, {'calibration_passed': False, 'test_read': False})
        with self.assertRaisesRegex(ValueError, 'pass calibration'):
            views.load_views(self.output, 'test')


if __name__ == '__main__':
    unittest.main()
