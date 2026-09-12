"""Validate prepared font-labelled screenshot bundles without torch or OCR.

Construct before training to validate all partition metadata and file hashes.
Only datasets(split) reads glyph pixels, so call datasets('test') after model
selection. Test labels/identities and the NPY header are inspected at preflight;
this is a structural audit, not a claim that no test metadata was accessed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

SPLITS = ('train', 'calibration', 'test')
ARRAY_KEYS = {'x', 'y', 'chars', 'faces', 'sizes', 'scripts', 'source_ids'}
LABEL_SOURCE = 'explicit_region_annotation; not verified by OCR or inferred from device'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError('Prepared screenshot data: ' + reason)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def character_script(character):
    if not isinstance(character, str) or len(character) != 1:
        return None
    if '\u4e00' <= character <= '\u9fff':
        return 'han'
    if character.isascii() and character.isalnum():
        return 'latin'
    return None


def bbox_valid(box, size):
    return (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
            and 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])


class PreparedScreenshotData:
    def __init__(self, folder, *, families, scripts, allow_empty_evaluation=False):
        self.folder = Path(folder).resolve()
        self.families = list(families)
        self.scripts = {script: list(names) for script, names in scripts.items()}
        require(set(self.scripts) == {'han', 'latin'} and len(set(self.families)) == len(self.families)
                and all(names and len(names) == len(set(names)) and set(names) <= set(self.families)
                        for names in self.scripts.values()),
                'invalid requested family/script configuration')
        self.manifest_path = self.folder / 'manifest.json'
        self.manifest_sha = sha(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        m = self.manifest
        require(m.get('schema') == 'explicit-font-screenshot-glyphs-v1', 'unsupported manifest schema')
        require(m.get('families') == self.families, 'manifest family order differs from trainer')
        require(m.get('font_label_source') == LABEL_SOURCE, 'explicit font annotation provenance is required')
        require(m.get('test_reader_injected') is False and isinstance(m.get('ocr_sources'), dict)
                and bool(m['ocr_sources'].get('files')), 'mock/injected OCR outputs are not training inputs')
        require(m.get('independent_device_or_font_truth_verified_by_importer') is False,
                'importer does not certify physical device or font truth')
        require(set(m.get('splits', {})) == set(SPLITS), 'all three partition files must be declared')
        require(isinstance(m.get('sources'), list) and m['sources'], 'source annotations are missing')
        require(isinstance(m.get('accepted_glyphs'), list), 'accepted glyph provenance is missing')
        self._identities = {}
        self._source_regions = {}
        for row in m['sources']:
            self._validate_source(row)
        self._accepted = {split: {} for split in SPLITS}
        for row in m['accepted_glyphs']:
            self._validate_accepted(row)
        self._metadata = {}
        self._hashes = {}
        for split in SPLITS:
            self._validate_partition(split)
        require(len(self._metadata['train']['y']) > 0, 'train partition has no accepted glyphs')
        empty = [split for split in ('calibration', 'test') if not len(self._metadata[split]['y'])]
        require(allow_empty_evaluation or not empty,
                'empty evaluation partitions require allow_empty_evaluation=True: ' + ', '.join(empty))
        require(m.get('accepted_count') == sum(len(x['y']) for x in self._metadata.values()),
                'manifest accepted count differs')
        require(m.get('labelled_screenshot_count') == len({x['source_sha256'] for x in m['sources']}),
                'manifest screenshot count differs')
        require(m.get('source_kind_counts') == dict(Counter(x['source_kind'] for x in m['sources'])),
                'source-kind counts differ')
        self.audit = {'schema': 'prepared-screenshot-contract-audit-v1', 'manifest_sha256': self.manifest_sha,
                      'partition_sha256': dict(self._hashes), 'source_identity_overlap': 0,
                      'empty_evaluation_splits': empty,
                      'empty_evaluation_meaning': 'No annotated-screenshot calibration/test claim for empty partitions; other datasets may supply evaluation.',
                      'test_metadata_inspected_for_contract': True, 'test_pixels_loaded': False,
                      'source_kind_counts': m['source_kind_counts'], 'font_label_source': LABEL_SOURCE,
                      'font_truth_certified_by_loader': False}

    def _bind(self, kind, value, split):
        previous = self._identities.setdefault((kind, value), split)
        require(previous == split, 'cross-split identity overlap: ' + kind)

    @staticmethod
    def _region_key(row, region):
        return (row['split'], row['source_id'], row['source_sha256'], region['region_index'],
                tuple(region['bbox']), region['text'], region['script'], region['font_family'])

    def _validate_source(self, row):
        require(isinstance(row, dict) and row.get('split') in SPLITS, 'invalid source partition')
        require(isinstance(row.get('source_id'), str) and bool(row['source_id'].strip()), 'missing source_id')
        require(isinstance(row.get('source_kind'), str) and bool(row['source_kind']), 'missing source_kind')
        size = row.get('oriented_size')
        require(isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size),
                'source dimensions are missing')
        for kind in ('source_sha256', 'source_rgb_sha256'):
            require(valid_sha(row.get(kind)), 'missing source hash: ' + kind)
            self._bind(kind, row[kind], row['split'])
        self._bind('source_id', row['source_id'], row['split'])
        require(isinstance(row.get('regions'), list) and row['regions'], 'source has no explicit regions')
        for region in row['regions']:
            require(isinstance(region, dict) and region.get('script') in self.scripts, 'invalid annotated script')
            require(type(region.get('region_index')) is int and region['region_index'] >= 0, 'invalid region index')
            require(bbox_valid(region.get('bbox'), size), 'invalid annotated source bbox')
            require(isinstance(region.get('text'), str) and bool(region['text'].strip()), 'missing annotation text')
            require(region.get('font_family') in self.scripts[region['script']], 'annotation family outside script mask')
            require(valid_sha(region.get('crop_rgb_sha256')), 'missing annotated region pixel hash')
            self._bind('region_rgb_sha256', region['crop_rgb_sha256'], row['split'])
            key = self._region_key(row, region)
            self._source_regions[key] = (row, region)

    def _validate_accepted(self, row):
        require(isinstance(row, dict) and row.get('split') in SPLITS, 'invalid accepted partition')
        require(type(row.get('array_row')) is int and row['array_row'] >= 0, 'invalid accepted array row')
        require(row['array_row'] not in self._accepted[row['split']], 'duplicate accepted array row')
        require(bbox_valid(row.get('bbox'), [2**31, 2**31]) and isinstance(row.get('text'), str),
                'missing accepted annotation binding')
        try:
            source, region = self._source_regions[self._region_key(row, row)]
        except (KeyError, TypeError):
            raise ValueError('Prepared screenshot data: glyph is not bound to an explicit source annotation') from None
        require(row.get('source_kind') == source['source_kind'], 'glyph source-kind differs from annotation')
        require(character_script(row.get('character')) == row.get('script'), 'accepted character/script mismatch')
        require(row['character'] in region['text'], 'accepted character absent from annotation')
        require(type(row.get('text_index')) is int and row['text_index'] >= 0, 'invalid accepted text index')
        require(valid_sha(row.get('glyph_sha256')), 'missing per-glyph pixel hash')
        require(bbox_valid(row.get('source_bbox'), source['oriented_size']), 'invalid glyph source bbox')
        box, area = row['source_bbox'], region['bbox']
        require(area[0] <= box[0] < box[2] <= area[2] and area[1] <= box[1] < box[3] <= area[3],
                'glyph bbox extends beyond its annotation')
        self._accepted[row['split']][row['array_row']] = (row, region)

    def _validate_partition(self, split):
        declared = self.manifest['splits'][split]
        require(declared.get('file') == split + '.npz' and valid_sha(declared.get('sha256')),
                'invalid declared partition file/hash')
        path = self.folder / declared['file']
        actual = sha(path)
        require(actual == declared['sha256'], 'partition SHA-256 mismatch: ' + split)
        self._hashes[split] = actual
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == ARRAY_KEYS and len(archive.files) == len(ARRAY_KEYS),
                    'partition array names differ or repeat: ' + split)
            metadata = {key: archive[key] for key in ARRAY_KEYS - {'x'}}
        require(metadata['y'].ndim == 1, 'targets must be a one-dimensional vector')
        n = len(metadata['y'])
        require(type(declared.get('rows')) is int and declared['rows'] == n, 'partition row count differs')
        require(set(self._accepted[split]) == set(range(n)), 'accepted row provenance is not complete')
        require(all(array.shape == (n,) for array in metadata.values()), 'metadata vector lengths differ')
        require(metadata['y'].dtype == np.dtype('int64') and metadata['sizes'].dtype == np.dtype('int64'),
                'targets and source sizes must be int64')
        require(all(metadata[key].dtype.kind == 'U' for key in ('chars', 'faces', 'scripts', 'source_ids')),
                'text metadata must be Unicode arrays')
        # Read only the NPY header. Test glyph values are not loaded pre-selection.
        with zipfile.ZipFile(path) as archive, archive.open('x.npy') as stream:
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), 'unsupported glyph NPY header version')
            read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = read_header(stream)
        require(shape == (n, 64, 64) and dtype == np.dtype('float32'), 'glyph array must be float32[N,64,64]')
        for i in range(n):
            char, script, target = str(metadata['chars'][i]), str(metadata['scripts'][i]), int(metadata['y'][i])
            require(character_script(char) == script, 'NPZ character/script mismatch')
            require(0 <= target < len(self.families) and self.families[target] in self.scripts[script],
                    'NPZ family is outside the script mask')
            row, region = self._accepted[split][i]
            require(char == row['character'] and script == row['script'] and self.families[target] == row['font_family'],
                    'NPZ target does not match explicit glyph annotation')
            require(str(metadata['source_ids'][i]) == row['source_id'], 'NPZ source_id differs from glyph provenance')
            self._bind('source_id', str(metadata['source_ids'][i]), split)
            require(str(metadata['faces'][i]) == region.get('font_face', row['font_family'] + ':unspecified'),
                    'NPZ face differs from annotation')
            require(int(metadata['sizes'][i]) == row['source_bbox'][3] - row['source_bbox'][1],
                    'NPZ source size differs from glyph box')
        require(declared.get('family_counts') == dict(Counter(self.families[int(y)] for y in metadata['y'])),
                'partition family counts differ')
        require(declared.get('script_counts') == dict(Counter(metadata['scripts'].tolist())),
                'partition script counts differ')
        self._metadata[split] = metadata

    def datasets(self, split):
        """Return only nonempty script datasets, with verified annotation labels."""
        require(split in SPLITS, 'invalid requested split')
        path = self.folder / (split + '.npz')
        require(sha(self.manifest_path) == self.manifest_sha and sha(path) == self._hashes[split],
                'bundle changed after preflight')
        with np.load(path, allow_pickle=False) as archive:
            pixels = archive['x']
        require(np.isfinite(pixels).all() and (not pixels.size or (pixels.min() >= 0 and pixels.max() <= 1)),
                'glyph pixels must be finite and within [0,1]')
        for i, glyph in enumerate(pixels):
            require(hashlib.sha256(glyph.tobytes()).hexdigest() == self._accepted[split][i][0]['glyph_sha256'],
                    'glyph row hash differs from accepted provenance')
        if split == 'test':
            self.audit['test_pixels_loaded'] = True
        metadata = self._metadata[split]
        result = []
        for script in self.scripts:
            rows = metadata['scripts'] == script
            if not rows.any():
                continue
            result.append({**{key: value[rows] for key, value in metadata.items()}, 'x': pixels[rows],
                           'valid': np.ones(int(rows.sum()), dtype=bool), 'script': script, 'real_data': True,
                           'name': str(self.folder) + '/' + split + '/' + script,
                           'source': {'path': str(path), 'sha256': self._hashes[split], 'rows': int(rows.sum()),
                                      'manifest_sha256': self.manifest_sha, 'font_label_source': LABEL_SOURCE,
                                      'source_kind_counts': self.manifest['source_kind_counts'],
                                      'contract_verified': True, 'font_truth_certified': False}})
        return result
