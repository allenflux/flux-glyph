"""OCR-free font/size inference on text-region pixels, with a standalone API."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SCHEMA = 'flux-glyph-region-font-v1'
ALGORITHM = 'region-cnn64x256-v1'
MAX_TILES = 8
MAX_PIXELS = 4_000_000


def preprocess_region(image):
    """Return region windows. No characters, script, OCR or glyph boxes enter."""
    result = {'status': 'unavailable', 'reason': 'low_quality_region', 'tiles': None}
    if (not isinstance(image, Image.Image) or min(image.size) < 6
            or image.width * image.height > MAX_PIXELS or max(image.size) > 16000):
        return result
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = np.median(border, axis=0)
    if np.quantile(np.max(np.abs(border - background), axis=1), .8) > 24:
        return {**result, 'reason': 'nonuniform_region_background'}
    distance = np.linalg.norm(pixels - background, axis=2)
    contrast = float(np.quantile(distance, .995))
    if contrast < 24:
        return result
    mask = distance >= max(12., contrast * .22)
    yy, xx = np.where(mask)
    if len(xx) < 12:
        return result
    l, t, r, b = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    height, width = b - t, r - l
    if height < 6 or width < 3:
        return result
    # Height/width ratio is retained. A region is not squashed into one square.
    scaled_width = max(1, round(width * 56 / height))
    if scaled_width > 8192:
        return {**result, 'reason': 'region_too_long'}
    ink = np.clip(distance[t:b, l:r] / contrast, 0, 1)
    scaled = np.asarray(Image.fromarray(ink).resize((scaled_width, 56), Image.Resampling.BICUBIC))
    canvas = np.zeros((64, max(256, scaled_width + 8)), dtype=np.float32)
    canvas[4:60, 4:4 + scaled_width] = np.clip(scaled, 0, 1)
    count = min(MAX_TILES, max(1, math.ceil((canvas.shape[1] - 32) / 224)))
    offsets = np.unique(np.rint(np.linspace(0, canvas.shape[1] - 256, count)).astype(int))
    tiles = np.stack([canvas[:, x:x + 256] for x in offsets])[:, None]
    return {'status': 'ok', 'reason': 'region_pixels', 'tiles': np.ascontiguousarray(tiles, dtype=np.float32),
            'ink_height_px': height, 'ink_bbox': [l, t, r, b], 'tile_count': len(tiles),
            'tile_valid_widths': [max(0, min(256, scaled_width + 4 - int(x))) for x in offsets],
            'background_rgb': np.rint(background).astype(int).tolist(),
            'source_size': list(image.size), 'normalized_width': scaled_width,
            'whole_width_covered': bool(len(offsets) == 1 or np.diff(offsets).max() <= 256)}


def aggregate_predictions(logits, log_em_ratio, *, temperature):
    values = np.asarray(logits, dtype=np.float64)
    values = (values - values.max(axis=1, keepdims=True)) / temperature
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    mean = probabilities.mean(axis=0)
    order = np.argsort(-mean, kind='stable')
    sizes = np.exp(np.asarray(log_em_ratio, dtype=np.float64).reshape(-1))
    ratio = float(np.exp(np.median(log_em_ratio)))
    spread = float((np.quantile(sizes, .9) - np.quantile(sizes, .1)) / ratio)
    return {'probabilities': mean, 'order': order, 'score': float(mean[order[0]]),
            'margin': float(mean[order[0]] - mean[order[1]]),
            'patch_agreement': float((probabilities.argmax(1) == order[0]).mean()),
            'em_ratio': ratio, 'size_relative_spread': spread}


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


class RegionFontClassifier:
    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / 'metadata.json'
        if path.stat().st_size > 256 * 1024:
            raise ValueError('Region metadata exceeds size limit')
        self.meta = json.loads(path.read_text())
        m = self.meta
        if m.get('schema') != SCHEMA or m.get('algorithm') != ALGORITHM:
            raise ValueError('Invalid region model contract')
        self.families = m.get('families')
        if (not isinstance(self.families, list) or not 2 <= len(self.families) <= 64
                or any(not isinstance(f, str) or not 1 <= len(f) <= 80 for f in self.families)
                or len(set(self.families)) != len(self.families)):
            raise ValueError('Invalid region font families')
        self.font_label_groups = m.get('font_label_groups', {})
        allowed_groups = {'PingFang': ['PingFang SC', 'PingFang TC', 'PingFang HK']}
        if (self.font_label_groups not in ({}, allowed_groups)
                or any(label not in self.families or any(name in self.families for name in native)
                       for label, native in self.font_label_groups.items())):
            raise ValueError('Invalid region font label groups')
        sources = m.get('font_sources', {})
        if (not isinstance(sources, dict) or (sources and set(sources) != set(self.families))
                or any(not isinstance(kinds, list) or not kinds or
                       any(kind not in ('system', 'asset') for kind in kinds) for kinds in sources.values())):
            raise ValueError('Invalid region font sources')
        gates = m.get('gates', {})
        if (not _number(m.get('temperature'), .01, 100)
                or any(not _number(gates.get(key), 0, 1) for key in ('min_score', 'min_margin', 'min_patch_agreement'))
                or not _number(m.get('max_size_relative_spread'), 0, 1)):
            raise ValueError('Invalid region gates')
        model = m.get('model', {})
        name = model.get('path')
        if (not isinstance(name, str) or Path(name).name != name or '\\' in name
                or not name.endswith('.onnx')):
            raise ValueError('Invalid region model path')
        path = self.directory / name
        if not path.resolve().is_relative_to(self.directory.resolve()) or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('Region model exceeds bounds')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != model.get('sha256'):
            raise ValueError('Region model SHA differs')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != 'tiles' or inputs[0].type != 'tensor(float)'
                or len(inputs[0].shape) != 4 or isinstance(inputs[0].shape[0], int)
                or inputs[0].shape[1:] != [1, 64, 256] or len(outputs) != 2
                or [o.name for o in outputs] != ['logits', 'log_em_ratio']
                or any(o.type != 'tensor(float)' for o in outputs)
                or len(outputs[0].shape) != 2 or outputs[0].shape[1] != len(self.families)
                or len(outputs[1].shape) != 1):
            raise ValueError('Region ONNX input/output contract differs')

    def predict(self, image):
        prepared = preprocess_region(image)
        result = {'method': 'region_neural_network', 'status': 'uncertain', 'family': None,
                  'candidates': [], 'score': None, 'margin': None, 'patch_agreement': None,
                  'reason_code': prepared['reason'], 'scope': 'Detected text region',
                  'font_size_px_estimate': None, 'size_relative_spread': None,
                  'ocr_performed': False, 'tile_count': prepared.get('tile_count', 0)}
        if prepared['status'] != 'ok':
            return result
        if not prepared['whole_width_covered']:
            return {**result,'reason_code':'region_too_long'}
        logits, ratios = self.session.run(['logits', 'log_em_ratio'], {'tiles': prepared['tiles']})
        if (logits.shape != (len(prepared['tiles']), len(self.families))
                or ratios.shape != (len(prepared['tiles']),) or not np.isfinite(logits).all()
                or not np.isfinite(ratios).all() or np.any(np.abs(ratios) > 3)):
            return {**result, 'reason_code': 'invalid_neural_output'}
        scores = aggregate_predictions(logits, ratios, temperature=self.meta['temperature'])
        result.update({k: scores[k] for k in ('score', 'margin', 'patch_agreement', 'size_relative_spread')})
        result['candidates'] = [{'family': self.families[int(i)], 'score': float(scores['probabilities'][i])}
                                for i in scores['order'][:3]]
        gate = self.meta['gates']
        if scores['patch_agreement'] < gate['min_patch_agreement']:
            result['reason_code'] = 'mixed_or_ambiguous_region'
        elif scores['score'] < gate['min_score']:
            result['reason_code'] = 'below_score_gate'
        elif scores['margin'] <= 1e-8 or scores['margin'] < gate['min_margin']:
            result['reason_code'] = 'ambiguous_neural_families'
        else:
            result.update(status='candidate', family=result['candidates'][0]['family'],
                          reason_code='region_neural_family_candidate')
            result['font_family_variants'] = self.font_label_groups.get(result['family'], [])
            if scores['size_relative_spread'] <= self.meta['max_size_relative_spread']:
                result['font_size_px_estimate'] = round(prepared['ink_height_px'] * scores['em_ratio'], 2)
        result['ink_height_px'] = prepared['ink_height_px']
        return result
