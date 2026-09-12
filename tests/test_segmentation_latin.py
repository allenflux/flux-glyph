from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flux_glyph.ppocr import PPReader
from flux_glyph.segmentation import segment_characters


def fixture(text, boxes, width=100):
    """Unequal, independently specified source-pixel ink and CTC anchors."""
    image = Image.new('RGB', (width, 30), 'white')
    draw = ImageDraw.Draw(image)
    tokens = []
    for index, (character, box) in enumerate(zip(text, boxes)):
        left, top, right, bottom = box
        if not character.isspace():
            draw.rectangle((left, top, right - 1, bottom - 1), fill='black')
        anchor = (left + right) // 2
        tokens.append(dict(character=character, index=index, start_step=anchor,
                           end_step=anchor + 1, peak_step=anchor, confidence=.99))
    metadata = dict(timesteps=width, input_width=width, content_width=width)
    return image, tokens, metadata


class LatinSegmentationTests(unittest.TestCase):
    def test_latin_requires_opt_in_and_keeps_complete_unequal_source_ink(self):
        boxes = [(5, 5, 9, 25), (10, 4, 27, 26), (28, 6, 40, 24)]
        image, tokens, metadata = fixture('1A2', boxes, width=46)
        default = segment_characters(image, '1A2', tokens, metadata=metadata)
        self.assertTrue(all(item['bbox'] is None for item in default['characters']))
        result = segment_characters(image, '1A2', tokens, metadata=metadata, segment_latin=True)
        self.assertEqual([item['status'] for item in result['characters']], ['ok'] * 3)
        widths = []
        for item, source in zip(result['characters'], boxes):
            left, top, right, bottom = item['bbox']
            self.assertLessEqual(left, source[0])
            self.assertGreaterEqual(right, source[2])
            self.assertEqual([top, bottom], [source[1], source[3]])
            pixels = np.asarray(image.crop(item['bbox']))[:, :, 0] == 0
            self.assertEqual(int(pixels.sum()), (source[2] - source[0]) * (source[3] - source[1]))
            widths.append(right - left)
        self.assertNotEqual(len(set(widths)), 1)
        self.assertTrue(all(boundary['minimum_gap_width'] == 1 and boundary['mass_max'] == 0
                            for boundary in result['diagnostics']['boundaries']))
        self.assertFalse(result['diagnostics']['equal_width_fallback_used'])

    def test_emitted_spaces_are_not_treated_as_glyph_centres(self):
        text = ' 1 2 '
        image, tokens, metadata = fixture(text, [(1, 4, 4, 20), (7, 4, 17, 24),
                                               (20, 4, 30, 20), (35, 4, 48, 24), (53, 4, 58, 20)], width=62)
        result = segment_characters(image, text, tokens, metadata=metadata, segment_latin=True)
        self.assertEqual([item['status'] for item in result['characters']],
                         ['uncertain', 'ok', 'uncertain', 'ok', 'uncertain'])
        self.assertEqual(result['diagnostics']['boundaries'][0]['between'], [1, 3])
        self.assertEqual(result['diagnostics']['boundaries'][0]['skipped_whitespace_indices'], [2])
        self.assertEqual(result['characters'][1]['bbox'][0], 7)
        self.assertEqual(result['characters'][3]['bbox'][2], 48)

    def test_only_exact_omitted_whitespace_can_reconcile_token_indices(self):
        image, indexed, metadata = fixture('1 2', [(5, 4, 16, 24), (20, 4, 24, 20), (28, 4, 40, 24)])
        omitted = [{**indexed[0], 'index': 0}, {**indexed[2], 'index': 1}]
        result = segment_characters(image, '1 2', omitted, metadata=metadata, segment_latin=True)
        self.assertEqual([item['status'] for item in result['characters']], ['ok', 'uncertain', 'ok'])
        self.assertEqual(result['diagnostics']['whitespace_reconciled_tokens'], 2)
        missing_glyph = segment_characters(image, '1A2', omitted, metadata=metadata, segment_latin=True)
        self.assertTrue(all(item['bbox'] is None for item in missing_glyph['characters']))
        self.assertEqual(missing_glyph['diagnostics']['whitespace_reconciled_tokens'], 0)

    def test_mixed_han_whitespace_still_uses_latin_as_ordering_neighbour(self):
        image, indexed, metadata = fixture('中 A文', [(5, 4, 23, 25), (25, 4, 29, 20),
                                                 (35, 4, 45, 25), (51, 4, 70, 25)])
        omitted = [{**token, 'index': index} for index, token in enumerate(indexed[i] for i in (0, 2, 3))]
        result = segment_characters(image, '中 A文', omitted, metadata=metadata)
        self.assertEqual([item['status'] for item in result['characters']], ['ok', 'uncertain', 'uncertain', 'ok'])
        self.assertEqual(result['characters'][0]['bbox'][0], 5)
        self.assertEqual(result['characters'][3]['bbox'][2], 70)

    def test_punctuation_is_a_neighbour_without_becoming_a_font_glyph(self):
        image, tokens, metadata = fixture('-1.2:3', [(5, 14, 12, 17), (16, 4, 24, 25),
                                                 (29, 22, 32, 25), (37, 4, 50, 25),
                                                 (55, 12, 58, 21), (63, 4, 77, 25)])
        result = segment_characters(image, '-1.2:3', tokens, metadata=metadata, segment_latin=True)
        self.assertEqual([item['status'] for item in result['characters']],
                         ['uncertain', 'ok', 'uncertain', 'ok', 'uncertain', 'ok'])

    def test_unrecognised_icon_is_not_absorbed_into_a_digit(self):
        image, tokens, metadata = fixture('234', [(22, 4, 35, 25), (42, 4, 56, 25), (63, 4, 76, 25)])
        ImageDraw.Draw(image).polygon([(4, 14), (13, 5), (13, 23)], fill='black')
        result = segment_characters(image, '234', tokens, metadata=metadata, segment_latin=True)
        self.assertEqual(result['characters'][0]['status'], 'uncertain')
        self.assertEqual(result['characters'][0]['reason'], 'multiple_ink_groups_for_latin_token')
        self.assertIsNone(result['characters'][0]['bbox'])
        self.assertEqual([item['status'] for item in result['characters'][1:]], ['ok', 'ok'])

    def test_touching_latin_glyphs_remain_uncertain(self):
        image, tokens, metadata = fixture('12', [(8, 4, 28, 25), (28, 4, 49, 25)])
        result = segment_characters(image, '12', tokens, metadata=metadata, segment_latin=True)
        self.assertTrue(all(item['bbox'] is None for item in result['characters']))
        self.assertFalse(result['diagnostics']['equal_width_fallback_used'])

    def test_real_pp_time_and_amount_keep_every_digit_source_pixel(self):
        root = Path(__file__).resolve().parents[1]
        font = ImageFont.truetype(str(root / 'assets' / 'annotation.otf'), 36)
        reader = PPReader(root / 'models' / 'pp')
        for text in ['22:43', '-100.00', '2026-08-01 22:43:11']:
            with self.subTest(text=text):
                image = Image.new('RGB', (round(font.getlength(text)) + 20, 54), 'white')
                ImageDraw.Draw(image).text((10, 0), text, fill='black', font=font)
                reading = reader.read([image])[0]
                self.assertEqual(reading['text'], text)
                result = segment_characters(image, text, reading['tokens'], metadata=reading['metadata'], segment_latin=True)
                for item in result['characters']:
                    if not item['character'].isdigit():
                        self.assertIsNone(item['bbox'])
                        continue
                    self.assertEqual(item['status'], 'ok')
                    index = item['index']
                    reference = Image.new('RGB', image.size, 'white')
                    ImageDraw.Draw(reference).text((10 + font.getlength(text[:index]), 0), item['character'],
                                                   fill='black', font=font)
                    source_ink = np.asarray(reference)[:, :, 0] < 128
                    left, top, right, bottom = item['bbox']
                    self.assertEqual(int(source_ink[top:bottom, left:right].sum()), int(source_ink.sum()))


if __name__ == '__main__':
    unittest.main()
