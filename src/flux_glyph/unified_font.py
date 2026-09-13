"""One unified native-mobile font CNN; no platform selection or device inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

if __package__:
    from .region_font import preprocess_region, aggregate_predictions
else:
    from region_font import preprocess_region, aggregate_predictions

SCHEMA = 'flux-glyph-unified-region-font-v1'
ALGORITHM = 'unified-region-cnn64x256-v1'
UNKNOWN = '__unknown__'


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def unified_metadata(metadata):
    """Validate and return the complete public inference contract, without paths to training data."""
    if (not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA
            or metadata.get('algorithm') != ALGORITHM or metadata.get('font_mode') != 'unified'
            or metadata.get('data_kind') != 'native_mobile_screenshots'
            or 'rejection' in metadata or 'verifier' in metadata):
        raise ValueError('Invalid unified model contract')
    families = metadata.get('families')
    if (not isinstance(families, list) or not 2 <= len(families) <= 128
            or any(not isinstance(f, str) or not 1 <= len(f) <= 80 or not f.strip() for f in families)
            or len(set(families)) != len(families) or families.count(UNKNOWN) != 1):
        raise ValueError('Unified model must contain 2..128 unique classes, including exactly one unknown')
    gates = metadata.get('gates')
    if (not _number(metadata.get('temperature'), .01, 100) or not isinstance(gates, dict)
            or any(not _number(gates.get(key), 0, 1) for key in ('min_score', 'min_margin', 'min_patch_agreement'))
            or not _number(metadata.get('max_size_relative_spread'), 0, 1)):
        raise ValueError('Invalid unified model gates')
    public = [family for family in families if family != UNKNOWN]
    groups = metadata.get('font_label_groups', {})
    if (not isinstance(groups, dict) or any(family not in public or not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not 1 <= len(name) <= 80 or name == UNKNOWN for name in names)
            or len(set(names)) != len(names) for family, names in groups.items())):
        raise ValueError('Invalid unified font label groups')
    sources = metadata.get('font_sources', {})
    if (not isinstance(sources, dict) or any(family not in public or not isinstance(kinds, list) or not kinds
            or any(kind not in ('system', 'asset') for kind in kinds) for family, kinds in sources.items())):
        raise ValueError('Invalid unified font sources')
    model = metadata.get('model')
    name = model.get('path') if isinstance(model, dict) else None
    digest = model.get('sha256') if isinstance(model, dict) else None
    if (not isinstance(name, str) or Path(name).name != name or '\\' in name or not name.endswith('.onnx')
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Invalid unified model path or SHA')
    result = {key: metadata[key] for key in ('schema', 'algorithm', 'font_mode', 'data_kind', 'families',
                                            'temperature', 'max_size_relative_spread')}
    result.update(model={'path': name, 'sha256': digest}, font_label_groups=groups, font_sources=sources,
                  gates={key: gates[key] for key in ('min_score', 'min_margin', 'min_patch_agreement')})
    return result


class UnifiedFontClassifier:
    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory/'metadata.json'
        if path.stat().st_size > 256*1024:
            raise ValueError('Unified metadata exceeds size limit')
        self.meta = json.loads(path.read_text())
        validated = unified_metadata(self.meta)
        self.output_families = validated['families']
        self.families = [family for family in self.output_families if family != UNKNOWN]
        self.font_label_groups = validated['font_label_groups']
        self.font_mode = 'unified'
        path = self.directory/validated['model']['path']
        if (not path.resolve().is_relative_to(self.directory.resolve()) or not path.is_file()
                or path.stat().st_size > 64*1024*1024):
            raise ValueError('Unified model missing or exceeds bounds')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != validated['model']['sha256']:
            raise ValueError('Unified model SHA differs')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != 'tiles' or inputs[0].type != 'tensor(float)'
                or len(inputs[0].shape) != 4 or isinstance(inputs[0].shape[0], int)
                or inputs[0].shape[1:] != [1, 64, 256] or len(outputs) != 2
                or [value.name for value in outputs] != ['logits', 'log_em_ratio']
                or any(value.type != 'tensor(float)' for value in outputs)
                or len(outputs[0].shape) != 2 or isinstance(outputs[0].shape[0], int) or outputs[0].shape[1] != len(self.output_families)
                or len(outputs[1].shape) != 1 or isinstance(outputs[1].shape[0], int)):
            raise ValueError('Unified ONNX input/output contract differs')

    def predict(self, image):
        prepared = preprocess_region(image)
        result = {'method': 'region_neural_network', 'font_mode': 'unified', 'status': 'uncertain', 'family': None,
                  'candidates': [], 'score': None, 'margin': None, 'patch_agreement': None,
                  'reason_code': prepared['reason'], 'scope': 'Detected text region',
                  'font_size_px_estimate': None, 'size_relative_spread': None, 'ocr_performed': False,
                  'tile_count': prepared.get('tile_count', 0), 'device_inference_performed': False}
        if prepared['status'] != 'ok':
            return result
        if not prepared['whole_width_covered']:
            return {**result, 'reason_code': 'region_too_long'}
        try:
            outputs = self.session.run(['logits', 'log_em_ratio'], {'tiles': prepared['tiles']})
        except Exception:
            return {**result, 'reason_code': 'invalid_neural_output'}
        if (not isinstance(outputs, (list, tuple)) or len(outputs) != 2
                or any(not isinstance(value, np.ndarray) or value.dtype != np.float32 for value in outputs)
                or outputs[0].shape != (len(prepared['tiles']), len(self.output_families)) or outputs[1].shape != (len(prepared['tiles']),)
                or any(not np.isfinite(value).all() for value in outputs) or np.any(np.abs(outputs[1]) > 3)):
            return {**result, 'reason_code': 'invalid_neural_output'}
        scores = aggregate_predictions(outputs[0], outputs[1], temperature=self.meta['temperature'])
        winner = self.output_families[int(scores['order'][0])]
        result['ink_height_px'] = prepared['ink_height_px']
        if winner == UNKNOWN:
            return {**result, 'status': 'out_of_scope', 'reason_code': 'unknown_font_rejected'}
        # Keep the original full-class softmax. The unknown class is never
        # removed and renormalized into a falsely confident named prediction.
        result.update({key: scores[key] for key in ('score', 'margin', 'patch_agreement', 'size_relative_spread')})
        result['candidates'] = [{'family': self.output_families[int(i)], 'score': float(scores['probabilities'][i])}
                                for i in scores['order'] if self.output_families[int(i)] != UNKNOWN][:3]
        gates = self.meta['gates']
        if scores['patch_agreement'] < gates['min_patch_agreement']:
            result['reason_code'] = 'mixed_or_ambiguous_region'
        elif scores['score'] < gates['min_score']:
            result['reason_code'] = 'below_score_gate'
        elif scores['margin'] <= 1e-8 or scores['margin'] < gates['min_margin']:
            result['reason_code'] = 'ambiguous_neural_families'
        else:
            result.update(status='candidate', family=winner, reason_code='region_neural_family_candidate',
                          font_family_variants=self.font_label_groups.get(winner, []))
            if scores['size_relative_spread'] <= self.meta['max_size_relative_spread']:
                result['font_size_px_estimate'] = round(prepared['ink_height_px']*scores['em_ratio'], 2)
        return result


# The independent download kit exposes the same small predict.py API.
RegionFontClassifier = UnifiedFontClassifier
