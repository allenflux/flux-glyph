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
REJECTION_ALGORITHM = 'region-cnn64x256-rejection-v2'
CONSENSUS_ALGORITHM = 'region-cnn64x256-consensus-v3'
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


def rejection_metadata(metadata):
    """Validate the optional gate; v1 never silently ignores a rejection model."""
    algorithm = metadata.get('algorithm')
    if algorithm != CONSENSUS_ALGORITHM and 'verifier' in metadata:
        raise ValueError('Verifier models require the v3 region algorithm')
    if algorithm == ALGORITHM:
        if 'rejection' in metadata:
            raise ValueError('Rejection models require the v2 region algorithm')
        return None
    if algorithm not in (REJECTION_ALGORITHM, CONSENSUS_ALGORITHM):
        raise ValueError('Invalid region model contract')
    value = metadata.get('rejection')
    if (not isinstance(value, dict) or value.get('schema') != 'flux-glyph-region-rejection-v1'
            or value.get('algorithm') != 'region-known-unknown-cnn64x256-v1'
            or value.get('labels') != ['unknown', 'known']
            or value.get('known_families') != metadata.get('families')
            or value.get('base_model_sha256') != metadata.get('model', {}).get('sha256')
            or value.get('aggregation') != 'mean_softmax_known_probability'
            or not _number(value.get('temperature'), .01, 100)
            or not _number(value.get('min_known_score'), 0, 1)):
        raise ValueError('Invalid region rejection metadata')
    model = value.get('model')
    name = model.get('path') if isinstance(model, dict) else None
    digest = model.get('sha256') if isinstance(model, dict) else None
    if (not isinstance(name, str) or Path(name).name != name or '\\' in name or not name.endswith('.onnx')
            or name == metadata.get('model', {}).get('path')
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Invalid region rejection model path or SHA')
    # Training provenance is retained in the full bundle, not the public kit.
    result = {key: value[key] for key in ('schema', 'algorithm', 'labels', 'known_families',
                                         'base_model_sha256', 'aggregation', 'temperature', 'min_known_score')}
    result['model'] = {'path': name, 'sha256': digest}
    return result


def verifier_metadata(metadata):
    """A v3 verifier is mandatory and bound to the unchanged primary model."""
    if metadata.get('algorithm') != CONSENSUS_ALGORITHM:
        if 'verifier' in metadata:
            raise ValueError('Verifier models require the v3 region algorithm')
        return None
    value = metadata.get('verifier')
    if not isinstance(value, dict):
        raise ValueError('Missing region verifier metadata')
    families, primary = value.get('families'), metadata.get('families')
    gates = value.get('gates')
    if (value.get('schema') != 'flux-glyph-region-verifier-v1'
            or value.get('algorithm') != 'region-font-verifier-cnn64x256-v1'
            or not isinstance(families, list) or not 2 <= len(families) <= 64
            or any(not isinstance(f, str) or not 1 <= len(f) <= 80 for f in families)
            or len(set(families)) != len(families) or not isinstance(primary, list)
            or any(f not in families for f in primary)
            or value.get('base_model_sha256') != metadata.get('model', {}).get('sha256')
            or not _number(value.get('temperature'), .01, 100) or not isinstance(gates, dict)
            or any(not _number(gates.get(k), 0, 1) for k in ('min_score', 'min_margin', 'min_patch_agreement'))):
        raise ValueError('Invalid region verifier metadata')
    model = value.get('model')
    name = model.get('path') if isinstance(model, dict) else None
    digest = model.get('sha256') if isinstance(model, dict) else None
    if (not isinstance(name, str) or Path(name).name != name or '\\' in name or not name.endswith('.onnx')
            or name in (metadata.get('model', {}).get('path'), metadata.get('rejection', {}).get('model', {}).get('path'))
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Invalid region verifier model path or SHA')
    result = {key: value[key] for key in ('schema', 'algorithm', 'families', 'base_model_sha256', 'temperature')}
    result['gates'] = {key: gates[key] for key in ('min_score', 'min_margin', 'min_patch_agreement')}
    result['model'] = {'path': name, 'sha256': digest}
    return result


def _gate_reason(scores, gates):
    if scores['patch_agreement'] < gates['min_patch_agreement']:
        return 'mixed_or_ambiguous_region'
    if scores['score'] < gates['min_score']:
        return 'below_score_gate'
    if scores['margin'] <= 1e-8 or scores['margin'] < gates['min_margin']:
        return 'ambiguous_neural_families'
    return None


class RegionFontClassifier:
    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / 'metadata.json'
        if path.stat().st_size > 256 * 1024:
            raise ValueError('Region metadata exceeds size limit')
        self.meta = json.loads(path.read_text())
        m = self.meta
        if m.get('schema') != SCHEMA or m.get('algorithm') not in (ALGORITHM, REJECTION_ALGORITHM, CONSENSUS_ALGORITHM):
            raise ValueError('Invalid region model contract')
        self.rejection_meta = rejection_metadata(m)
        self.verifier_meta = verifier_metadata(m)
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
        self.rejection_session = None
        if self.rejection_meta is not None:
            rejection_model = self.rejection_meta['model']
            rejection_path = self.directory / rejection_model['path']
            if (not rejection_path.resolve().is_relative_to(self.directory.resolve())
                    or not rejection_path.is_file() or rejection_path.stat().st_size > 64 * 1024 * 1024):
                raise ValueError('Region rejection model is missing or exceeds bounds')
            rejection_bytes = rejection_path.read_bytes()
            if hashlib.sha256(rejection_bytes).hexdigest() != rejection_model['sha256']:
                raise ValueError('Region rejection model SHA differs')
            self.rejection_session = ort.InferenceSession(rejection_bytes, sess_options=options, providers=['CPUExecutionProvider'])
            inputs, outputs = self.rejection_session.get_inputs(), self.rejection_session.get_outputs()
            if (len(inputs) != 1 or inputs[0].name != 'tiles' or inputs[0].type != 'tensor(float)'
                    or len(inputs[0].shape) != 4 or isinstance(inputs[0].shape[0], int)
                    or inputs[0].shape[1:] != [1, 64, 256] or len(outputs) != 1
                    or outputs[0].name != 'known_logits' or outputs[0].type != 'tensor(float)'
                    or len(outputs[0].shape) != 2 or isinstance(outputs[0].shape[0], int) or outputs[0].shape[1] != 2):
                raise ValueError('Region rejection ONNX input/output contract differs')
        self.verifier_session = None
        if self.verifier_meta is not None:
            model = self.verifier_meta['model']
            path = self.directory / model['path']
            if (not path.resolve().is_relative_to(self.directory.resolve()) or not path.is_file()
                    or path.stat().st_size > 64 * 1024 * 1024):
                raise ValueError('Region verifier model is missing or exceeds bounds')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != model['sha256']:
                raise ValueError('Region verifier model SHA differs')
            self.verifier_session = ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])
            inputs, outputs = self.verifier_session.get_inputs(), self.verifier_session.get_outputs()
            if (len(inputs) != 1 or inputs[0].name != 'tiles' or inputs[0].type != 'tensor(float)'
                    or len(inputs[0].shape) != 4 or isinstance(inputs[0].shape[0], int)
                    or inputs[0].shape[1:] != [1, 64, 256] or len(outputs) != 2
                    or [o.name for o in outputs] != ['logits', 'log_em_ratio']
                    or any(o.type != 'tensor(float)' for o in outputs)
                    or len(outputs[0].shape) != 2 or isinstance(outputs[0].shape[0], int)
                    or outputs[0].shape[1] != len(self.verifier_meta['families'])
                    or len(outputs[1].shape) != 1 or isinstance(outputs[1].shape[0], int)):
                raise ValueError('Region verifier ONNX input/output contract differs')

    def predict(self, image):
        prepared = preprocess_region(image)
        result = {'method': 'region_neural_network', 'status': 'uncertain', 'family': None,
                  'candidates': [], 'score': None, 'margin': None, 'patch_agreement': None,
                  'reason_code': prepared['reason'], 'scope': 'Detected text region',
                  'font_size_px_estimate': None, 'size_relative_spread': None,
                  'ocr_performed': False, 'tile_count': prepared.get('tile_count', 0)}
        rejection = getattr(self, 'rejection_meta', None)
        verifier = getattr(self, 'verifier_meta', None)
        if verifier is not None:
            result['verifier'] = {'method': 'region_neural_network', 'status': 'unavailable', 'family': None,
                                  'candidates': [], 'score': None, 'margin': None, 'patch_agreement': None}
        if rejection is not None:
            result['rejection'] = {'method': 'neural_network', 'status': 'unavailable', 'known_score': None,
                                   'min_known_score': rejection['min_known_score']}
        if prepared['status'] != 'ok':
            return result
        if not prepared['whole_width_covered']:
            return {**result,'reason_code':'region_too_long'}
        if rejection is not None:
            # This independent network runs before naming a font. Failure must
            # never fall through to a confident score among the eight classes.
            try:
                outputs = self.rejection_session.run(['known_logits'], {'tiles': prepared['tiles']})
            except Exception:
                return {**result, 'reason_code': 'invalid_rejection_output'}
            if (not isinstance(outputs, (list, tuple)) or len(outputs) != 1
                    or not isinstance(outputs[0], np.ndarray) or outputs[0].dtype != np.float32
                    or outputs[0].shape != (len(prepared['tiles']), 2) or not np.isfinite(outputs[0]).all()):
                return {**result, 'reason_code': 'invalid_rejection_output'}
            logits = outputs[0].astype(np.float64)
            probabilities = np.exp((logits - logits.max(axis=1, keepdims=True)) / rejection['temperature'])
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            known_score = float(probabilities[:, 1].mean())
            rejected = known_score < rejection['min_known_score']
            result['rejection'].update(status='rejected' if rejected else 'passed', known_score=known_score)
            if rejected:
                return {**result, 'status': 'out_of_scope', 'reason_code': 'unknown_font_rejected',
                        'ink_height_px': prepared['ink_height_px']}
        try:
            outputs = self.session.run(['logits', 'log_em_ratio'], {'tiles': prepared['tiles']})
        except Exception:
            return {**result, 'reason_code': 'invalid_neural_output'}
        if (not isinstance(outputs, (list, tuple)) or len(outputs) != 2
                or any(not isinstance(value, np.ndarray) or value.dtype != np.float32 for value in outputs)):
            return {**result, 'reason_code': 'invalid_neural_output'}
        logits, ratios = outputs
        if (logits.shape != (len(prepared['tiles']), len(self.families))
                or ratios.shape != (len(prepared['tiles']),) or not np.isfinite(logits).all()
                or not np.isfinite(ratios).all() or np.any(np.abs(ratios) > 3)):
            return {**result, 'reason_code': 'invalid_neural_output'}
        scores = aggregate_predictions(logits, ratios, temperature=self.meta['temperature'])
        if verifier is not None:
            try:
                outputs = self.verifier_session.run(['logits', 'log_em_ratio'], {'tiles': prepared['tiles']})
            except Exception:
                return {**result, 'reason_code': 'invalid_verifier_output'}
            if (not isinstance(outputs, (list, tuple)) or len(outputs) != 2
                    or any(not isinstance(value, np.ndarray) or value.dtype != np.float32 for value in outputs)
                    or outputs[0].shape != (len(prepared['tiles']), len(verifier['families']))
                    or outputs[1].shape != (len(prepared['tiles']),)
                    or any(not np.isfinite(value).all() for value in outputs)):
                return {**result, 'reason_code': 'invalid_verifier_output'}
            # The verifier's size output is unused: only the primary estimates size.
            verified = aggregate_predictions(outputs[0], np.zeros(len(outputs[0]), dtype=np.float32),
                                              temperature=verifier['temperature'])
            verifier_reason = _gate_reason(verified, verifier['gates'])
            family = verifier['families'][int(verified['order'][0])]
            if family not in self.families and verifier_reason is None:
                result['verifier']['status'] = 'out_of_scope'
                return {**result, 'status': 'out_of_scope', 'reason_code': 'verifier_font_out_of_scope',
                        'ink_height_px': prepared['ink_height_px']}
            result['verifier'].update(status='passed' if verifier_reason is None else 'below_gate', family=family,
                                      reason_code=verifier_reason,
                                      candidates=[{'family': verifier['families'][int(i)], 'score': float(verified['probabilities'][i])}
                                                  for i in verified['order'][:3]],
                                      **{k: verified[k] for k in ('score', 'margin', 'patch_agreement')})
        result.update({k: scores[k] for k in ('score', 'margin', 'patch_agreement', 'size_relative_spread')})
        result['candidates'] = [{'family': self.families[int(i)], 'score': float(scores['probabilities'][i])}
                                for i in scores['order'][:3]]
        gate = self.meta['gates']
        if verifier is not None and result['verifier']['family'] != result['candidates'][0]['family']:
            result['verifier']['status'] = 'disagreed'
            result['reason_code'] = 'neural_model_disagreement'
        elif verifier is not None and verifier_reason is not None:
            result['reason_code'] = 'verifier_' + verifier_reason
        elif scores['patch_agreement'] < gate['min_patch_agreement']:
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
