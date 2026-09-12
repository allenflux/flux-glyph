"""Import explicitly font-labelled screenshot regions; never infer font labels.

Run in the OCR environment, separately from the PyTorch training process:
  python training/prepare_screenshots.py --input labels.jsonl --output prepared

Each JSONL row contains image, source_id, split and regions. Each region contains
bbox [left, top, right, bottom], text, font_family and script (han or latin).
Coordinates refer to the EXIF-oriented source image. In mixed text, the supplied
family applies only to characters of the specified script. Punctuation is not
labelled. An optional source_kind records synthetic smoke fixtures explicitly.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flux_glyph.glyph_preprocess import extract_glyphs

SPLITS = ('train', 'calibration', 'test')


def family_names():
    """Read the trainer's literal class order without importing torch."""
    tree = ast.parse((Path(__file__).with_name('network.py')).read_text())
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'FAMILIES'
                      for target in node.targets)]
    if len(values) != 1 or not isinstance(values[0], list) or not values[0]:
        raise ValueError('training/network.py must declare a literal FAMILIES list')
    result = values[0]
    if any(not isinstance(name, str) or not name for name in result) or len(set(result)) != len(result):
        raise ValueError('Invalid trainer family names')
    return result


def script_families(families):
    """Read the trainer's script masks with a small data-expression evaluator."""
    def evaluate(node, names):
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(value, names) for value in node.elts]
        if isinstance(node, ast.Dict):
            return {evaluate(key, names): evaluate(value, names) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Slice):
            return slice(*(evaluate(value, names) if value else None for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            return evaluate(node.value, names)[evaluate(node.slice, names)]
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            generator = node.generators[0]
            if isinstance(generator.target, ast.Name) and not generator.ifs and not generator.is_async:
                return [evaluate(node.elt, dict(names, **{generator.target.id: value}))
                        for value in evaluate(generator.iter, names)]
        raise ValueError('Unsupported data expression in training/network.py SCRIPTS')

    tree = ast.parse(Path(__file__).with_name('network.py').read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'SCRIPTS' for target in node.targets)]
    if len(values) != 1:
        raise ValueError('training/network.py must declare SCRIPTS')
    masks = evaluate(values[0], {'FAMILIES': families})
    if (not isinstance(masks, dict) or set(masks) != {'han', 'latin'}
            or any(not names or len(set(names)) != len(names) or not set(names) <= set(families)
                   for names in masks.values())):
        raise ValueError('Invalid trainer script family masks')
    return masks


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def image_digest(image):
    return sha_bytes(str(image.size).encode() + b'\0RGB\0' + image.tobytes())


def scope(character):
    if '\u4e00' <= character <= '\u9fff':
        return 'han'
    if character.isascii() and character.isalnum():
        return 'latin'
    return None


def no_whitespace(text):
    return ''.join(character for character in text if not character.isspace())


def valid_bbox(box, size):
    return (isinstance(box, list) and len(box) == 4
            and all(type(value) is int for value in box)
            and 0 <= box[0] < box[2] <= size[0]
            and 0 <= box[1] < box[3] <= size[1])


def bind_identity(registry, kind, value, split):
    key = (kind, value)
    previous = registry.setdefault(key, split)
    if previous != split:
        raise ValueError(f'Cross-split identity leakage: {kind} belongs to {previous} and {split}')


def load_labels(input_file, families):
    """Validate all labels and source identities before running OCR or writing."""
    input_file = Path(input_file).resolve()
    masks = script_families(families)
    records, identities, labels = [], {}, {}
    for line_number, line in enumerate(input_file.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(key not in row for key in ('image', 'source_id', 'split', 'regions')):
            raise ValueError(f'Line {line_number}: image/source_id/split/regions are required')
        if not isinstance(row['source_id'], str) or not row['source_id'].strip():
            raise ValueError(f'Line {line_number}: source_id must be nonempty')
        if row['split'] not in SPLITS:
            raise ValueError(f'Line {line_number}: invalid split')
        if not isinstance(row['image'], str) or not row['image']:
            raise ValueError(f'Line {line_number}: image must be a local path')
        if not isinstance(row['regions'], list) or not row['regions']:
            raise ValueError(f'Line {line_number}: regions must be a nonempty list')
        path = Path(row['image']).expanduser()
        path = (input_file.parent / path).resolve() if not path.is_absolute() else path.resolve()
        raw = path.read_bytes()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        image.load()
        byte_sha = sha_bytes(raw)
        pixel_sha = image_digest(image)
        for kind, value in (('source_id', row['source_id']), ('source_file_sha256', byte_sha),
                            ('source_rgb_sha256', pixel_sha)):
            bind_identity(identities, kind, value, row['split'])
        validated = []
        for region_index, region in enumerate(row['regions']):
            prefix = f'Line {line_number}, region {region_index}'
            if not isinstance(region, dict) or any(key not in region for key in ('bbox', 'text', 'font_family', 'script')):
                raise ValueError(prefix + ': bbox/text/font_family/script are required')
            if not valid_bbox(region['bbox'], image.size):
                raise ValueError(prefix + ': bbox must be integer coordinates inside the oriented image')
            if not isinstance(region['text'], str) or not no_whitespace(region['text']):
                raise ValueError(prefix + ': text must be nonempty')
            if region['script'] not in ('han', 'latin'):
                raise ValueError(prefix + ': script must be han or latin')
            if not isinstance(region['font_family'], str) or region['font_family'] not in families:
                raise ValueError(prefix + ': an explicit known font_family is required; platform labels are invalid')
            if region['font_family'] not in masks[region['script']]:
                raise ValueError(prefix + ': font_family is outside the trainer/runtime script mask')
            if not any(scope(character) == region['script'] for character in region['text']):
                raise ValueError(prefix + ': text has no characters in the labelled script')
            if 'font_face' in region and (not isinstance(region['font_face'], str) or not region['font_face'].strip()):
                raise ValueError(prefix + ': font_face must be a nonempty string when supplied')
            crop_sha = image_digest(image.crop(region['bbox']))
            bind_identity(identities, 'region_rgb_sha256', crop_sha, row['split'])
            # Repeated mixed-script regions may have different scope labels,
            # but identical pixels cannot receive conflicting same-scope labels.
            label_key = (crop_sha, region['script'])
            previous = labels.setdefault(label_key, region['font_family'])
            if previous != region['font_family']:
                raise ValueError(prefix + ': conflicting font labels for identical same-script region pixels')
            validated.append(dict(region, region_index=region_index, crop_rgb_sha256=crop_sha))
        records.append({'image': str(path), 'source_id': row['source_id'], 'split': row['split'],
                        'source_kind': row.get('source_kind', 'user_supplied_screenshot'),
                        'line_number': line_number, 'source_sha256': byte_sha,
                        'source_rgb_sha256': pixel_sha, 'oriented_size': list(image.size), 'regions': validated})
    if not records:
        raise ValueError('No labelled screenshot records')
    return records


def prepare(input_file, output_dir, model_dir=ROOT / 'models', *, reader=None):
    """Prepare arrays and return manifest. reader injection is for tests only."""
    from flux_glyph.segmentation import segment_characters

    families = family_names()
    records = load_labels(input_file, families)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output directory must be new or empty: ' + str(output_dir))
    ocr_sources = None
    if reader is None:
        from flux_glyph.ppocr import PPReader
        from flux_glyph.models import load_active
        active, version, model_manifest = load_active(model_dir)
        reader = PPReader(active / 'pp')
        ocr_sources = {'model_version': version,
                       'files': [item for item in model_manifest['files'] if item['path'].startswith('pp/')]}
    buckets = {split: {key: [] for key in ('x', 'y', 'chars', 'faces', 'sizes', 'scripts', 'source_ids')}
               for split in SPLITS}
    accepted, rejected, seen = [], [], {}
    for row in records:
        # Detect edits during preparation instead of silently importing new pixels.
        if sha_bytes(Path(row['image']).read_bytes()) != row['source_sha256']:
            raise ValueError('Source changed after validation: ' + row['image'])
        with Image.open(row['image']) as opened:
            source = ImageOps.exif_transpose(opened).convert('RGB')
        for region in row['regions']:
            base = {key: row[key] for key in ('source_id', 'split', 'source_sha256', 'source_kind')}
            base.update({key: region[key] for key in ('region_index', 'bbox', 'text', 'font_family', 'script')})
            crop = source.crop(region['bbox'])
            if image_digest(crop) != region['crop_rgb_sha256']:
                raise ValueError('Region pixels changed after validation')
            results = reader.read([crop])
            if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
                raise ValueError('OCR reader returned an invalid result')
            reading = results[0]
            recognized = reading.get('text', '')
            if not isinstance(recognized, str) or no_whitespace(recognized) != no_whitespace(region['text']):
                rejected.append(dict(base, reason='ocr_text_mismatch', recognized_text=recognized))
                continue
            confidence = reading.get('confidence')
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < .8:
                rejected.append(dict(base, reason='low_ocr_confidence', recognized_text=recognized))
                continue
            metadata = reading.get('metadata', {})
            rotation = metadata.get('orientation_degrees', 0)
            if rotation not in (0, 180):
                rejected.append(dict(base, reason='unsupported_ocr_rotation'))
                continue
            oriented = crop.transpose(Image.Transpose.ROTATE_180) if rotation == 180 else crop
            segments = segment_characters(oriented, recognized, reading.get('tokens', []),
                                           metadata=metadata, segment_latin=True)
            pixels = np.asarray(oriented)
            border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
            background = tuple(np.median(border, axis=0).astype(np.uint8).tolist())
            for glyph in segments['characters']:
                character, index = glyph.get('character'), glyph.get('index')
                if type(index) is not int or not 0 <= index < len(recognized) or character != recognized[index]:
                    raise ValueError('Segmentation returned an invalid text index')
                if scope(character) != region['script']:
                    continue
                details = dict(base, character=character, text_index=index)
                if glyph.get('status') != 'ok' or not valid_bbox(glyph.get('bbox'), oriented.size):
                    rejected.append(dict(details, reason='segmentation_rejected',
                                         segmentation_reason=glyph.get('reason')))
                    continue
                raw = oriented.crop(glyph['bbox'])
                # Same four-pixel source-background padding as neural runtime.
                padded = Image.new('RGB', (raw.width + 8, raw.height + 8), background)
                padded.paste(raw, (4, 4))
                result = extract_glyphs(padded, 1)
                if result.glyphs is None or result.glyphs.shape != (1, 64, 64) or not np.isfinite(result.glyphs).all():
                    rejected.append(dict(details, reason='glyph_preprocessing_rejected',
                                         preprocessing_reason=result.diagnostics.get('reason')))
                    continue
                left, top, right, bottom = glyph['bbox']
                original_box = ([crop.width - right, crop.height - bottom, crop.width - left, crop.height - top]
                                if rotation else [left, top, right, bottom])
                source_box = [original_box[0] + region['bbox'][0], original_box[1] + region['bbox'][1],
                              original_box[2] + region['bbox'][0], original_box[3] + region['bbox'][1]]
                identity = (row['source_rgb_sha256'], tuple(source_box), region['script'])
                if identity in seen:
                    if seen[identity] != region['font_family']:
                        raise ValueError('Conflicting font labels for the same source glyph')
                    rejected.append(dict(details, reason='duplicate_source_glyph'))
                    continue
                seen[identity] = region['font_family']
                bucket = buckets[row['split']]
                array_row = len(bucket['x'])
                bucket['x'].append(result.glyphs[0].astype(np.float32))
                bucket['y'].append(families.index(region['font_family']))
                bucket['chars'].append(character)
                bucket['faces'].append(region.get('font_face', region['font_family'] + ':unspecified'))
                bucket['sizes'].append(raw.height)
                bucket['scripts'].append(region['script'])
                bucket['source_ids'].append(row['source_id'])
                accepted.append(dict(details, array_row=array_row, source_bbox=source_box,
                                     glyph_sha256=sha_bytes(result.glyphs[0].tobytes()), rotation_degrees=rotation))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = {}
    for split, bucket in buckets.items():
        array = np.stack(bucket['x']) if bucket['x'] else np.empty((0, 64, 64), dtype=np.float32)
        path = output_dir / (split + '.npz')
        np.savez_compressed(path, x=array, y=np.asarray(bucket['y'], dtype=np.int64),
                            chars=np.asarray(bucket['chars'], dtype=str), faces=np.asarray(bucket['faces'], dtype=str),
                            sizes=np.asarray(bucket['sizes'], dtype=np.int64),
                            scripts=np.asarray(bucket['scripts'], dtype=str),
                            source_ids=np.asarray(bucket['source_ids'], dtype=str))
        split_reports[split] = {'file': path.name, 'rows': len(array), 'sha256': sha_bytes(path.read_bytes()),
                                'family_counts': dict(Counter(families[y] for y in bucket['y'])),
                                'script_counts': dict(Counter(bucket['scripts']))}
    manifest = {'schema': 'explicit-font-screenshot-glyphs-v1',
                'input_file': str(Path(input_file).resolve()), 'input_sha256': sha_bytes(Path(input_file).read_bytes()),
                'families': families, 'network_source_sha256': sha_bytes(Path(__file__).with_name('network.py').read_bytes()),
                'font_label_source': 'explicit_region_annotation; not verified by OCR or inferred from device',
                'ocr_text_rule': 'exact Unicode equality after removing whitespace only; no fuzzy matching',
                'minimum_ocr_confidence': .8, 'ocr_sources': ocr_sources,
                'test_reader_injected': ocr_sources is None,
                'coordinate_space': 'EXIF-oriented original source pixels',
                'sizes_meaning': 'segmented source-glyph ink height in pixels, not font point size',
                'mixed_text_rule': 'font_family applies only to region.script characters; punctuation is excluded',
                'identity_checks': {'source_id_cross_split': 0, 'file_sha_cross_split': 0,
                                    'decoded_source_pixels_cross_split': 0, 'decoded_region_pixels_cross_split': 0},
                'source_kind_counts': dict(Counter(row['source_kind'] for row in records)),
                'sources': records, 'splits': split_reports, 'accepted_glyphs': accepted,
                'rejected': rejected, 'rejection_counts': dict(Counter(row['reason'] for row in rejected)),
                'accepted_count': len(accepted), 'labelled_screenshot_count': len({row['source_sha256'] for row in records}),
                'independent_device_or_font_truth_verified_by_importer': False}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    result = prepare(args.input, args.output, args.model_dir)
    print(json.dumps({'output': str(args.output.resolve()), 'accepted': result['accepted_count'],
                      'rejection_counts': result['rejection_counts']}, ensure_ascii=False))
    return 0 if result['accepted_count'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
