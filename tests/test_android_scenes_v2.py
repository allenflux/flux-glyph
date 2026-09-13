import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import string
import tempfile
import unittest

from training.capture import generate_android_scenes_v2 as scenes


class AndroidRoundTwoScenesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.known = [f'Known{i}' for i in range(8)] + ['Roboto']
        self.fonts, self.maps = [], {}
        self.alnum = set(map(ord, string.ascii_letters + string.digits + '-.:/'))
        self.han = self.alnum | set(map(ord, scenes.HAN))
        for i, family in enumerate(self.known):
            for face in range(3 if i == 0 else 1):
                self.add_font(family, face, 'all', latin_only=family == 'Roboto')
        for split in scenes.SPLITS:
            for family, count in [('Sans', 4), ('Other', 1)]:
                for face in range(count):
                    self.add_font(split + family, face, split)
            self.add_font(split + 'Latin', 0, split, latin_only=True)
        self.registry = {'schema': 'flux-glyph-android-font-sources-v1', 'families': self.known,
                         'fonts': self.fonts}
        self.excluded = {scenes.normalize_text(text) for text in ['账单详情', 'ＢＡＬＡＮＣＥ', '12 : 34 : 56']}

    def add_font(self, family, face, split, *, latin_only=False):
        name = family + str(face)
        path = self.root / (name + '.ttf')
        path.write_bytes(name.encode())
        self.fonts.append({'id': name, 'family': family,
            'training_family': family if split == 'all' else '__unknown__',
            'path': str(path), 'sha256': scenes.sha(path), 'postscript': name, 'ttc_index': 0,
            'split': split, 'allowed_splits': list(scenes.SPLITS) if split == 'all' else [split],
            'scripts': ['latin'] if latin_only else ['han', 'latin']})
        self.maps[name] = self.alnum.copy() if latin_only else self.han.copy()

    def build(self, count=160):
        return scenes.build_scenes(self.registry, self.maps, self.excluded, {},
                                   {split: count for split in scenes.SPLITS})

    def test_old_exclusions_require_all_partitions_and_normalize_compatibility_case_space(self):
        path = self.root / 'labels.jsonl'
        rows = [{'split': split, 'text': text} for split, text in
                zip(scenes.SPLITS, ['Ｐａｙ Ｎｏｗ', '支付\u3000详情', 'Balance'])]
        path.write_text('\n'.join(json.dumps(row) for row in rows))
        excluded, counts = scenes.prior_texts(path)
        self.assertEqual(excluded, {'paynow', '支付详情', 'balance'})
        self.assertEqual(counts, {split: 1 for split in scenes.SPLITS})
        path.write_text(json.dumps(rows[0]))
        with self.assertRaisesRegex(ValueError, 'all three partitions'):
            scenes.prior_texts(path)

    def test_latin_only_sources_are_excluded_from_han_intersection_and_never_receive_han(self):
        generated = self.build()
        scenes.validate_round(generated, self.maps, self.excluded)
        fonts = {font['id']: font for font in self.fonts}
        for split in scenes.SPLITS:
            pages = [page for page in generated['pages'] if page['split'] == split]
            latin_unknown = []
            for page in pages:
                for row in page['regions']:
                    font = fonts[row['font_id']]
                    if 'han' not in font['scripts']:
                        self.assertNotEqual(row['script'], 'han')
                        self.assertTrue(set(map(ord, row['text'])) <= self.alnum)
                        if font['training_family'] == '__unknown__':
                            latin_unknown.append(row)
            self.assertEqual(len(latin_unknown), (len(pages) + 2) // 3)
        self.assertGreaterEqual(len(generated['design']['coverage']['test']['shared_han_alphabet']), 32)

    def test_known_size_color_and_text_are_paired_without_size_script_or_face_alias(self):
        generated = self.build(304)
        fonts = {font['id']: font for font in self.fonts}
        size_scripts, size_script_faces = defaultdict(set), defaultdict(set)
        for page in generated['pages']:
            known = [row for row in page['regions'] if fonts[row['font_id']]['training_family'] != '__unknown__']
            self.assertEqual(len(known), 9)
            self.assertEqual(len({(row['font_size_px'], row['color']) for row in known}), 1)
            cjk = [row for row in known if fonts[row['font_id']]['family'] != 'Roboto']
            self.assertEqual(len({(row['script'], row['text']) for row in cjk}), 1)
            target = next(row for row in cjk if fonts[row['font_id']]['family'] == 'Known0')
            latin = target['script'] != 'han'
            size_scripts[target['font_size_px']].add(latin)
            size_script_faces[target['font_size_px'], latin].add(target['font_id'])
        self.assertEqual(set(size_scripts), set(scenes.SIZES))
        self.assertTrue(all(value == {False, True} for value in size_scripts.values()))
        self.assertTrue(all(len(value) == 3 for value in size_script_faces.values()))

    def test_unknown_sampling_balances_families_before_face_counts(self):
        generated = self.build()
        fonts = {font['id']: font for font in self.fonts}
        for split in scenes.SPLITS:
            families, faces = Counter(), defaultdict(Counter)
            for page in generated['pages']:
                if page['split'] != split:
                    continue
                for row in page['regions']:
                    font = fonts[row['font_id']]
                    if font['training_family'] == '__unknown__' and 'han' in font['scripts']:
                        families[font['family']] += 1
                        faces[font['family']][font['id']] += 1
            self.assertLessEqual(max(families.values()) - min(families.values()), 1)
            for counts in faces.values():
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_fresh_text_split_contract_is_deterministic_and_fail_closed(self):
        generated = self.build(40)
        self.assertEqual(generated, self.build(40))
        scenes.validate_round(generated, self.maps, self.excluded)
        self.assertEqual(generated['seed'], 2026091303)
        self.assertTrue(all(page['id'].startswith('android-native-v2-') for page in generated['pages']))
        contaminated = copy.deepcopy(generated)
        contaminated['pages'][0]['regions'][0]['text'] = 'ＢＡＬＡＮＣＥ'
        with self.assertRaisesRegex(ValueError, 'prior partition text reused'):
            scenes.validate_round(contaminated, self.maps, self.excluded)
        cross_split = copy.deepcopy(generated)
        row = cross_split['pages'][0]['regions'][0]
        test_row = next(page for page in cross_split['pages'] if page['split'] == 'test')['regions'][0]
        test_row['text'] = row['text']
        with self.assertRaisesRegex(ValueError, 'text crosses partitions'):
            scenes.validate_round(cross_split, self.maps, self.excluded)

    def test_latin_sampler_contains_amount_date_time_and_english_shapes_with_cmap_filter(self):
        import random
        sampler = scenes.TextSampler(random.Random(scenes.SEED), self.excluded)
        texts = [sampler.sample('train', string.ascii_letters + string.digits + '-.:/', latin=True)
                 for _ in range(1000)]
        self.assertTrue(any(text.startswith('-') and '.' in text for text in texts))
        self.assertTrue(any(text.count('-') == 2 for text in texts))
        self.assertTrue(any(text.count(':') == 2 for text in texts))
        self.assertTrue(any(text in scenes.ENGLISH for text in texts))
        self.assertTrue(all(3 <= len(text) <= 10 for text in texts))
        restricted = [sampler.sample('train', string.ascii_letters + string.digits, latin=True)
                      for _ in range(100)]
        self.assertTrue(all(text.isalnum() for text in restricted))


if __name__ == '__main__':
    unittest.main()
