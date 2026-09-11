from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flux_glyph.segmentation import segment_characters
from flux_glyph.models import load_active
from flux_glyph.ppocr import PPReader, PPRegionDetector


def tokens(text, spans):
    return [{"character": char, "index": i, "start_step": a, "end_step": b, "peak_step": peak, "confidence": .99}
            for i, (char, (a, b, peak)) in enumerate(zip(text, spans))]


class SegmentationTests(unittest.TestCase):
    def test_ink_valleys_refine_han_without_equal_width(self):
        image = Image.new('RGB', (120, 36), 'white'); d = ImageDraw.Draw(image)
        d.rectangle((11, 8, 26, 27), fill='black'); d.rectangle((42, 6, 68, 29), fill='black'); d.rectangle((83, 10, 96, 25), fill='black')
        result = segment_characters(image, '中文甲', tokens('中文甲', [(3, 10, 6), (14, 25, 19), (27, 34, 30)]), metadata={'timesteps': 40, 'input_width': 40, 'content_width': 40})
        chars = result['characters']; self.assertEqual([c['status'] for c in chars], ['ok', 'ok', 'ok'])
        boxes = [c['bbox'] for c in chars]; self.assertTrue(all(a[2] <= b[0] for a, b in zip(boxes, boxes[1:])))
        self.assertNotEqual([b[2] - b[0] for b in boxes], [40, 40, 40]); self.assertTrue(result['diagnostics']['ctc_is_coarse_anchor_only'])
        self.assertFalse(result['diagnostics']['equal_width_fallback_used'])

    def test_no_valley_keeps_han_uncertain(self):
        image = Image.new('RGB', (80, 28), 'white'); ImageDraw.Draw(image).rectangle((8, 4, 67, 23), fill='black')
        result = segment_characters(image, '中文', tokens('中文', [(4, 15, 9), (22, 34, 28)]), metadata={'timesteps': 40, 'input_width': 40, 'content_width': 40})
        self.assertEqual([c['status'] for c in result['characters']], ['uncertain', 'uncertain'])
        self.assertTrue(all(c['bbox'] is None for c in result['characters']))

    def test_preserves_latin_and_missing_han_without_fake_box(self):
        image = Image.new('RGB', (60, 24), 'white'); ImageDraw.Draw(image).rectangle((5, 3, 16, 20), fill='black')
        result = segment_characters(image, '中A文', tokens('中A', [(2, 8, 4), (10, 15, 12)]), metadata={'timesteps': 20, 'input_width': 20, 'content_width': 20})
        self.assertEqual(result['characters'][1]['character'], 'A'); self.assertEqual(result['characters'][1]['status'], 'uncertain')
        self.assertIn('non_han', result['characters'][1]['reason']); self.assertEqual(result['characters'][2]['reason'], 'ctc_token_missing_or_mismatched')

    def test_single_han_uses_independent_left_and_right_edges(self):
        image = Image.new('RGB', (80, 32), 'white'); ImageDraw.Draw(image).rectangle((21, 5, 55, 26), fill='black')
        result = segment_characters(image, '中', tokens('中', [(8, 16, 12)]), metadata={'timesteps': 20, 'input_width': 20, 'content_width': 20})
        char = result['characters'][0]
        self.assertEqual(char['status'], 'ok')
        self.assertLess(char['bbox'][0], 30); self.assertGreater(char['bbox'][2], 48)

    def test_ctc_padding_token_is_not_promoted_to_source_box(self):
        image = Image.new('RGB', (60, 24), 'white'); ImageDraw.Draw(image).rectangle((8, 3, 25, 20), fill='black')
        result = segment_characters(image, '中', tokens('中', [(0, 3, 1)]),
                                    metadata={'timesteps': 20, 'input_width': 20, 'content_width': 14, 'padding_left': 4})
        self.assertEqual(result['characters'][0]['status'], 'uncertain')
        self.assertEqual(result['characters'][0]['reason'], 'ctc_anchor_outside_content_width')

    def test_low_confidence_and_roi_edge_contact_are_rejected(self):
        image = Image.new('RGB', (60, 24), 'white')
        ImageDraw.Draw(image).rectangle((0, 3, 25, 20), fill='black')
        result = segment_characters(image, '中', tokens('中', [(3, 12, 7)]),
                                    metadata={'timesteps': 20, 'input_width': 20, 'content_width': 20})
        self.assertEqual(result['characters'][0]['status'], 'uncertain')
        self.assertEqual(result['characters'][0]['reason'], 'foreground_touches_roi_edge')
        low = tokens('中', [(3, 12, 7)])
        low[0]['confidence'] = .79
        result = segment_characters(Image.new('RGB', (60, 24), 'white'), '中', low,
                                    metadata={'timesteps': 20, 'input_width': 20, 'content_width': 20})
        self.assertEqual(result['characters'][0]['reason'], 'low_ctc_token_confidence')

    def test_real_pp_expanded_detector_crop_keeps_complete_title_glyphs(self):
        fixtures = Path(__file__).resolve().parent / 'fixtures'
        with Image.open(fixtures / 'ui_title_billing_details.png') as source:
            source = source.convert('RGB')
            model_dir, _, _ = load_active(Path(__file__).resolve().parents[1] / 'models')
            detected = PPRegionDetector(model_dir / 'pp').detect(source)[0]
            left, top, right, bottom = detected['source_bbox']
            margin = max(2, min(12, round((bottom - top) * .08)))
            expanded = [max(0, left - margin), max(0, top - margin),
                        min(source.width, right + margin), min(source.height, bottom + margin)]
            crop = source.crop(expanded)
            reading = PPReader(model_dir / 'pp').read([crop])[0]
            result = segment_characters(crop, reading['text'], reading['tokens'], metadata=reading['metadata'])
        self.assertEqual(reading['text'], '账单详情')
        self.assertEqual([c['status'] for c in result['characters']], ['ok'] * 4)
        x0, y0, _, _ = expanded
        source_boxes = [[c['bbox'][0] + x0, c['bbox'][1] + y0, c['bbox'][2] + x0, c['bbox'][3] + y0]
                        for c in result['characters']]
        self.assertLessEqual(source_boxes[0][0], 10)
        self.assertGreaterEqual(source_boxes[0][2] - source_boxes[0][0], 48)
        self.assertGreaterEqual(source_boxes[-1][2], 219)
        widths = [box[2] - box[0] for box in source_boxes]
        self.assertLessEqual(max(widths) - min(widths), 2)
        self.assertTrue(all(boundary['minimum_gap_width'] >= 2 and boundary['low_ink_gap']
                            for boundary in result['diagnostics']['boundaries']))
        reference = json.loads((fixtures / 'ui_title_billing_details_vision.json').read_text())
        self.assertEqual(reference['text'], reading['text'])
        def iou(a, b):
            x, y, w, h = b; b = [x * source.width, source.height - (y + h) * source.height, (x + w) * source.width, source.height - y * source.height]
            left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            intersection = max(0, right-left) * max(0, bottom-top)
            return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)
        overlaps = [iou(box, ref['normalized_bbox_bottom_left']) for box, ref in zip(source_boxes, reference['vision_characters'])]
        self.assertGreater(sum(overlaps) / len(overlaps), .5)  # geometry smoke, not an accuracy claim

    def test_pp_detector_and_reader_segment_light_text_on_blue(self):
        fixtures = Path(__file__).resolve().parent / 'fixtures'
        source_path = fixtures / 'font_accuracy/inputs/pingfang_21.0d1e1_regular--blue_png_28.png'
        model_dir, _, _ = load_active(Path(__file__).resolve().parents[1] / 'models')
        with Image.open(source_path) as source:
            source = source.convert('RGB')
            detected = PPRegionDetector(model_dir / 'pp').detect(source)[0]
            crop = source.crop(detected['source_bbox'])
            reading = PPReader(model_dir / 'pp').read([crop])[0]
            result = segment_characters(crop, reading['text'], reading['tokens'], metadata=reading['metadata'])
        self.assertEqual(reading['text'], '转账到银行卡')
        self.assertEqual([c['status'] for c in result['characters']], ['ok'] * 6)
        boxes = [c['bbox'] for c in result['characters']]
        self.assertLessEqual(boxes[0][0], 2)
        self.assertGreaterEqual(boxes[-1][2], crop.width - 2)
        self.assertTrue(all(a[2] <= b[0] for a, b in zip(boxes, boxes[1:])))


if __name__ == '__main__':
    unittest.main()
