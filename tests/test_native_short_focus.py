"""Strict metadata/proof and deterministic sampling checks for native short focus."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from training import native_short_focus as focus
from training.prepare_unified_regions import FAMILIES


def _proof(path, page, region, text, face, font_sha, count):
    glyphs = [{'glyph_id': index + 1, 'font_verified': True,
               'font': {'postscript': face, 'sha256': font_sha, 'ttc_index': 0},
               'bbox': [index * 5, 0, index * 5 + 4, 8]} for index in range(count)]
    value = {'split': 'train', 'page_id': page, 'regions': [
        {'id': region, 'font_verified': True, 'text': text, 'glyphs': glyphs}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _identity(root, source, family, length, serial, proof_root, face=None):
    face = face or family.replace(' ', '') + '-Regular'
    font_sha = f'{serial + 1:064x}'[-64:]
    source_id = f'android:{source}-{serial}'
    region_id = f'{source}-{serial}-r0'
    page = f'{source}-{serial}'
    text = '字' * length
    proof = proof_root / f'{source}-{serial}.json'
    _proof(proof, page, region_id, text, face, font_sha, length)
    result = []
    for view in focus.VIEWS:
        result.append({'split': 'train', 'family': family, 'target': FAMILIES.index(family),
            'source_id': source_id, 'region_id': region_id, 'page_id': page,
            'source_font_family': family if family != '__unknown__' else source,
            'font_face': face, 'font_file_sha256': font_sha, 'ttc_index': 0,
            'domain': 'android', 'native_font_verified': True,
            'normalized_text_sha256': focus._text_sha(text), 'source_sha256': 'a' * 64,
            'tiles_sha256': f'{serial + 11:064x}'[-64:], 'tile_start': len(result),
            'tile_count': 1, 'view': view,
            **({} if root == 'original' else
               {'proof': str(proof), 'proof_sha256': focus.sha(proof)})})
    return result, {'split': 'train', 'source_id': source_id, 'region_id': region_id,
        'page_id': page, 'image_sha256': 'a' * 64, 'font_face': face,
        'font_file_sha256': font_sha, 'text': text,
        'proof': str(proof), 'proof_sha256': focus.sha(proof)}


def dataset_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(focus, 'ROOT', tmp_path)
    rows = {name: [] for name in focus.DATASETS}
    labels = []
    serial = 0
    proof_root = tmp_path / 'proofs'
    for family in focus.FOCUS_KNOWN_FAMILIES:
        for length in (3, 4):
            values, label = _identity('original', 'known', family, length, serial, proof_root)
            rows['original'] += values; labels.append(label); serial += 1
    # A second WenKai face lives in the merged-known partition and must be cycled.
    for length in (3, 4):
        values, _ = _identity('known', 'paired', 'LXGW WenKai', length, serial,
                              proof_root, face='LXGWWenKai-Light')
        rows['known'] += values; serial += 1
    for source in focus.FOCUS_UNKNOWN_SOURCES:
        partition = 'supplement' if source in focus.NEW_SOURCES else 'original'
        for length in (3, 4):
            values, label = _identity(partition, source, '__unknown__', length, serial, proof_root)
            rows[partition] += values
            if partition == 'original': labels.append(label)
            serial += 1
    labels_path = tmp_path / 'labels.jsonl'
    labels_path.write_text(''.join(json.dumps(row) + '\n' for row in labels))
    monkeypatch.setattr(focus, 'ANDROID_LABELS', labels_path)
    datasets = {}
    expected = {}
    for name, values in rows.items():
        root = tmp_path / name
        (root / 'train').mkdir(parents=True)
        rows_path = root / 'train/rows.json'; rows_path.write_text(json.dumps(values))
        part = {'split': 'train', 'families': FAMILIES, 'views': len(values),
                'tiles': len(values), 'metadata': {'path': 'rows.json', 'sha256': focus.sha(rows_path)}}
        part_path = root / 'train/MANIFEST.json'; part_path.write_text(json.dumps(part))
        manifest_path = root / 'MANIFEST.json'; manifest_path.write_text(json.dumps({'name': name}))
        expected[name] = {'root': root, 'manifest_sha256': focus.sha(manifest_path),
                          'partition_sha256': focus.sha(part_path), 'tiles': len(values)}
        datasets[name] = {'rows': deepcopy(values), 'tiles': np.zeros((len(values), 1)),
                          'families': FAMILIES, 'partition': deepcopy(part),
                          'manifest_sha256': focus.sha(manifest_path),
                          'partition_sha256': focus.sha(part_path)}
    monkeypatch.setattr(focus, 'DATASETS', expected)
    return datasets


def _strip_marker(row):
    return {key: value for key, value in row.items()
            if key not in (focus.SUPPLEMENT_MARKER, focus.KNOWN_SUPPLEMENT_MARKER)}


def test_actual_contract_has_canonical_fourteen_families_and_six_verified_unknown_sources():
    assert len(focus.FOCUS_KNOWN_FAMILIES) == 14
    assert list(focus.FOCUS_KNOWN_FAMILIES) == [
        family for family in FAMILIES if family in set(focus.FOCUS_KNOWN_FAMILIES)]
    assert focus.FOCUS_UNKNOWN_SOURCES == (
        'Lato', 'Liu Jian Mao Cao', 'Open Sans', 'Smiley Sans',
        'WenQuanYi Zen Hei', 'Zhuque Fangsong')
    assert set(focus.FOCUS_UNKNOWN_SOURCES) < set(focus.ALL_TRAIN_UNKNOWN_SOURCES)
    assert focus.SAMPLING['known_rows'] == 28 and focus.SAMPLING['unknown_rows'] == 4


def test_build_focus_binds_every_row_and_proof_and_sampler_is_deterministic(monkeypatch, tmp_path):
    datasets = dataset_fixture(monkeypatch, tmp_path)
    pools, proof = focus.build_focus(datasets)
    assert proof['identity_count'] == 14 * 2 + 2 + 6 * 2
    assert len(proof['known_pool_counts']) == 14 * 2 + 2
    assert len(proof['unknown_pool_counts']) == 6 * 2
    assert proof['row_binding_sha256'] == pools['row_binding_sha256']
    assert all(Path(path).is_file() and focus.sha(path) == digest
               for path, digest in proof['bindings'].items())
    left = focus.NativeShortFocusSampler(pools, 17)
    right = focus.NativeShortFocusSampler(deepcopy(pools), 17)
    left_batches = [left.batch() for _ in range(12)]
    right_batches = [right.batch() for _ in range(12)]
    key = lambda row: (row['source_id'], row['region_id'], row['view'])
    assert [[key(row) for row in batch] for batch in left_batches] == [
        [key(row) for row in batch] for batch in right_batches]
    for batch in left_batches:
        assert len(batch) == 32
        assert all('glyph_count' not in row for row in batch)
        assert sum(row['family'] == '__unknown__' for row in batch) == 4
        assert all(sum(row['family'] == family for row in batch) == 2
                   for family in focus.FOCUS_KNOWN_FAMILIES)
        for row in batch:
            source = ('known' if row.get(focus.KNOWN_SUPPLEMENT_MARKER) else
                      'supplement' if row.get(focus.SUPPLEMENT_MARKER) else 'original')
            assert _strip_marker(row) in datasets[source]['rows']
    report = left.report()
    assert report['steps'] == 12 and report['rows'] == 384
    assert report['by_glyph_count'] == {'3': 192, '4': 192}
    assert report['by_family']['__unknown__'] == 48
    assert set(report['by_unknown_source']) == set(focus.FOCUS_UNKNOWN_SOURCES)
    assert max(report['by_unknown_source'].values()) == min(report['by_unknown_source'].values())
    assert len(report['by_unknown_source_glyph_count']) == 12
    assert report['unique_native_identities_drawn'] > 0
    assert 1 <= report['per_identity_draws']['minimum'] <= report['per_identity_draws']['median']
    assert report['per_identity_draws']['median'] <= report['per_identity_draws']['maximum']
    wenkai_faces = {row['font_face']: row['rows'] for row in report['by_face']
                    if row['family'] == 'LXGW WenKai'}
    assert set(wenkai_faces) == {'LXGWWenKai-Regular', 'LXGWWenKai-Light'}
    assert max(wenkai_faces.values()) - min(wenkai_faces.values()) == 0


@pytest.mark.parametrize('fault', ['heldout', 'calibration', 'missing_view', 'two_tiles',
                                   'row_hash', 'font', 'glyph_count_fallback'])
def test_build_focus_rejects_split_source_view_row_and_glyph_proof_corruption(
        monkeypatch, tmp_path, fault):
    datasets = dataset_fixture(monkeypatch, tmp_path)
    target = next(row for row in datasets['original']['rows']
                  if row['family'] == focus.FOCUS_KNOWN_FAMILIES[0])
    identity = (target['source_id'], target['region_id'])
    matching = [row for row in datasets['original']['rows']
                if (row['source_id'], row['region_id']) == identity]
    if fault == 'heldout':
        values, label = _identity('original', 'Yusei Magic', '__unknown__', 3, 999,
                                  tmp_path/'proofs')
        datasets['original']['rows'] += values
        # Deliberately route a held-out source through an otherwise valid original label.
        with focus.ANDROID_LABELS.open('a') as stream: stream.write(json.dumps(label) + '\n')
    elif fault == 'calibration':
        for row in matching: row['split'] = 'calibration'
    elif fault == 'missing_view':
        datasets['original']['rows'].remove(matching[-1])
    elif fault == 'two_tiles':
        matching[-1]['tile_count'] = 2
    elif fault == 'row_hash':
        matching[-1]['normalized_text_sha256'] = '0' * 64
    elif fault == 'font':
        matching[-1]['font_face'] = 'WrongFace'
    # Keep the in-memory fixture equal to its newly corrupted on-disk row binding.
    root = focus.DATASETS['original']['root']; rows_path = root/'train/rows.json'
    rows_path.write_text(json.dumps(datasets['original']['rows']))
    datasets['original']['tiles'] = np.zeros((len(datasets['original']['rows']), 1))
    focus.DATASETS['original']['tiles'] = len(datasets['original']['rows'])
    datasets['original']['partition']['metadata']['sha256'] = focus.sha(rows_path)
    part_path = root/'train/MANIFEST.json'; part_path.write_text(json.dumps(datasets['original']['partition']))
    datasets['original']['partition_sha256'] = focus.sha(part_path)
    focus.DATASETS['original']['partition_sha256'] = focus.sha(part_path)
    if fault == 'glyph_count_fallback':
        label = json.loads(focus.ANDROID_LABELS.read_text().splitlines()[0])
        proof_path = Path(label['proof']); value = json.loads(proof_path.read_text())
        # Text remains three characters, but exact glyph proof has five entries.
        value['regions'][0]['glyphs'] += deepcopy(value['regions'][0]['glyphs'][:2])
        proof_path.write_text(json.dumps(value))
        label['proof_sha256'] = focus.sha(proof_path)
        lines = focus.ANDROID_LABELS.read_text().splitlines(); lines[0] = json.dumps(label)
        focus.ANDROID_LABELS.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError):
        focus.build_focus(datasets)


def test_sampler_rejects_post_build_raw_row_tampering(monkeypatch, tmp_path):
    pools, _ = focus.build_focus(dataset_fixture(monkeypatch, tmp_path))
    record = next(iter(next(iter(pools['known'].values())).values()))[0]
    for row in record['rows'].values():
        row['target'] = 24
    sampler = focus.NativeShortFocusSampler(pools, 1)
    with pytest.raises(ValueError, match='changed after proof'):
        for _ in range(20):
            sampler.batch()
