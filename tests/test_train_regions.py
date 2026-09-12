"""Whole-region training contract, independent of the optional torch runtime."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.train_regions import (AGREEMENT, DATA_SCHEMA, ROOT, SPLITS, RegionData,
                                    RegionSampler, calibrate, metrics, region_observations, sha)
from training.train_regions import training_test_history, system_focus_policy, sampling_report
from flux_glyph.region_font import aggregate_predictions


def dataset(counts=(2, 3)):
    rows, cursor = [], 0
    for target, count in enumerate(counts):
        rows.append({'tile_start': cursor, 'tile_count': count, 'ink_height_px': 40,
                     'font_size_screen_px': 50, 'whole_width_covered': True})
        cursor += count
    return {'tiles': np.zeros((cursor, 1, 64, 256), np.float32), 'rows': rows,
            'targets': np.arange(len(counts)), 'log_em_ratio': np.full(len(counts), np.log(1.25), np.float32)}


def test_exact_runtime_aggregation_including_even_tile_size():
    data = dataset()
    logits = np.asarray([[4., 1.], [2., .5], [0., 3.], [.2, 2.], [1., 2.]], np.float32)
    ratios = np.asarray([.1, .3, .15, .2, .25], np.float32)
    result = region_observations(logits, ratios, data, 1.25)
    for i, row in enumerate(data['rows']):
        block = slice(row['tile_start'], row['tile_start'] + row['tile_count'])
        expected = aggregate_predictions(logits[block], ratios[block], temperature=1.25)
        np.testing.assert_array_equal(result['probabilities'][i], expected['probabilities'])
        assert result['predicted'][i] == expected['order'][0]
        assert result['agreement'][i] == expected['patch_agreement']
        assert result['size_px'][i] == 40 * expected['em_ratio']
        assert result['size_spread'][i] == expected['size_relative_spread']
    assert result['size_px'][0] != 40 * np.exp(ratios[:2]).mean()


@pytest.mark.parametrize('failure', ['invalid_ratio', 'incomplete_width', 'patch_disagreement', 'size_spread'])
def test_runtime_rejections_are_honored(failure):
    data = dataset((2, 2))
    logits = np.asarray([[4., 0.], [4., 0.], [0., 4.], [0., 4.]], np.float32)
    ratio = np.zeros(4, np.float32)
    if failure == 'invalid_ratio':
        ratio[0] = 3.01
    if failure == 'incomplete_width':
        data['rows'][0]['whole_width_covered'] = False
    if failure == 'patch_disagreement':
        logits[1] = [0., 1.]
    if failure == 'size_spread':
        ratio[:2] = [0., .5]
    result = metrics(region_observations(logits, ratio, data), data, ['A', 'B'],
                     {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': AGREEMENT})
    assert result['accepted']['regions'] == (2 if failure == 'size_spread' else 1)
    assert result['accepted_size']['regions'] == 1


def test_balanced_sampler_exhausts_each_family_before_repeating():
    data = dataset((1, 2, 3, 1, 2, 1))
    data['targets'] = np.asarray([0, 0, 0, 0, 1, 1])
    sampler = RegionSampler(data, 42)
    for _ in range(4):
        tile_indices, labels, ratios = sampler.batch(2)
        assert sorted(labels.tolist()) == [0, 1]
        assert len(ratios) == 2
        assert ((tile_indices >= 0) & (tile_indices < len(data['tiles']))).all()
    assert sampler.visits[:4].tolist() == [1] * 4
    assert sampler.visits[4:].tolist() == [2, 2]


def test_default_weight_one_preserves_original_seeded_sampling_sequence():
    data = dataset((1, 2, 3, 1, 2, 1))
    data['targets'] = np.asarray([0, 0, 0, 0, 1, 1])
    data['log_em_ratio'] = np.arange(6, dtype=np.float32)
    default = RegionSampler(data, 42)
    explicit = RegionSampler(data, 42, families=['PingFang', 'MiSans'], system_family_weight=1)
    expected = [([9, 0, 7, 2, 7], [1, 0, 1, 0, 1], [5, 0, 4, 1, 4]),
                ([9, 4], [1, 0], [5, 2]),
                ([9, 6, 8, 6, 9, 0, 7, 2, 7], [1, 0, 1, 0, 1, 0, 1, 0, 1], [5, 3, 4, 3, 5, 0, 4, 1, 4])]
    for count, old in zip((5, 2, 9), expected):
        actual, weighted_one = default.batch(count), explicit.batch(count)
        for value, same, historical in zip(actual, weighted_one, old):
            np.testing.assert_array_equal(value, same)
            np.testing.assert_array_equal(value, historical)


def test_weight_two_retains_eight_competitors_and_exhausts_row_queues_across_batches():
    families = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans', 'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number']
    data = dataset((1,) * 32)
    data['targets'] = np.repeat(np.arange(8), 4)
    for row, target in zip(data['rows'], data['targets']):
        row.update(family=families[target], native_font_family=families[target], target=int(target))
    sampler = RegionSampler(data, 2026091294, families=families, system_family_weight=2)
    duplicate = RegionSampler(data, 2026091294, families=families, system_family_weight=2)
    labels = []
    for count in (53, 2, 55):  # Ten complete 11-slot cycles, crossing batch boundaries.
        batch, same = sampler.batch(count), duplicate.batch(count)
        labels.extend(batch[1].tolist())
        for left, right in zip(batch, same):
            np.testing.assert_array_equal(left, right)
    assert np.bincount(labels).tolist() == [10, 10, 10, 10, 20, 20, 20, 10]
    for target in range(8):
        visits = sampler.visits[data['targets'] == target]
        assert visits.max() - visits.min() <= 1
    report = sampling_report(sampler, data, families)
    assert report['family_weights'] == {name: 2 if name in ('PingFang', 'SF Pro', 'Helvetica') else 1 for name in families}
    assert report['family_visits']['PingFang'] == report['native_family_visits']['PingFang'] == 20


def test_reused_test_override_changes_report_without_mutating_prepared_manifest():
    manifest = {'test_history': 'fresh', 'test_scope': 'Original fresh capture scope'}
    before = copy.deepcopy(manifest)
    report = training_test_history(manifest, 'reused')
    assert report['test_history'] == 'reused' and 'regression' in report['test_scope']
    assert manifest == before
    assert training_test_history(manifest)['test_scope'] == before['test_scope']
    with pytest.raises(ValueError, match='cannot become fresh'):
        training_test_history({'test_history': 'reused'}, 'fresh')


def focus_args(tmp_path):
    families = ['HarmonyOS Sans SC', 'MiSans', 'Noto Sans CJK SC', 'OPPO Sans', 'PingFang', 'SF Pro', 'Helvetica', 'Alipay Number']
    path = tmp_path / 'parent.json'
    metadata = {'schema': 'flux-glyph-region-font-v1', 'algorithm': 'region-cnn64x256-v1', 'families': families,
                'temperature': .5, 'gates': {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .7},
                'max_size_relative_spread': .2, 'training': {'state_after_sha256': 'a' * 64}}
    path.write_text(json.dumps(metadata))
    return families, metadata, SimpleNamespace(parent_metadata=path, seed=2026091294, steps=4000, batch_size=64,
                                               learning_rate=3e-5, eval_every=250, system_family_weight=2,
                                               preserve_size_head=True, benchmark_steps=0)


def test_system_focus_policy_pins_parent_metadata_and_fixed_budget(tmp_path):
    families, metadata, args = focus_args(tmp_path)
    policy = system_focus_policy(args, families)
    assert policy['parent_metadata']['sha256'] == sha(args.parent_metadata)
    assert policy['fixed_policy'] == {key: metadata[key] for key in ('temperature', 'gates', 'max_size_relative_spread')}
    assert policy['selection_policy']['preserve_all_parent_accepted_correct_rows'] is True
    assert policy['selection_policy']['recalibrate'] is False
    args.steps = 3999
    with pytest.raises(ValueError, match='frozen'):
        system_focus_policy(args, families)


@pytest.mark.parametrize('key,value', [('temperature', .75), ('max_size_relative_spread', .3),
                                      ('gates', {'min_score': .5, 'min_margin': .01, 'min_patch_agreement': .5}),
                                      ('algorithm', 'region-cnn64x256-family-gates-v2'), ('training', {})])
def test_system_focus_rejects_parent_policy_drift(tmp_path, key, value):
    families, metadata, args = focus_args(tmp_path)
    metadata[key] = value
    args.parent_metadata.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='parent'):
        system_focus_policy(args, families)


def test_calibration_uses_complete_regions_and_patch_agreement():
    data = dataset((2,) * 24)
    data['targets'] = np.arange(24) % 2
    logits = np.full((48, 2), -3., np.float32)
    for i, target in enumerate(data['targets']):
        logits[2*i:2*i+2, target] = 3.
    _, gate, result = calibrate(logits, np.full(48, np.log(1.25), np.float32), data, ['A', 'B'])
    assert result['gate_found'] and result['metrics']['accepted']['regions'] == 24
    assert gate['min_patch_agreement'] == .7
    assert result['gate_selection_level'] == 'actual_whole_native_region'


def fixture_manifest(tmp_path):
    families = ['PingFang SC', 'SF Pro']
    partitions, sources = {}, []
    for index, split in enumerate(SPLITS):
        source = tmp_path / f'{split}.png'
        source.write_bytes(b'fixture source identity ' + split.encode())
        sid = f'source-{split}'
        source_row = {'source_id': sid, 'page_id': f'page-{split}', 'split': split,
                      'source_kind': 'ios_simulator_screenshot', 'image': str(source),
                      'source_sha256': sha(source), 'decoded_pixel_sha256': hashlib.sha256(split.encode()).hexdigest(),
                      'content_group_id': f'content-{split}'}
        sources.append(source_row)
        folder = tmp_path / split
        folder.mkdir()
        tiles = np.full((2, 1, 64, 256), .25, np.float32)
        np.save(folder / 'tiles.npy', tiles)
        rows = []
        for i, family in enumerate(families):
            rows.append({**{key: source_row[key] for key in ('source_id', 'page_id', 'split', 'source_sha256',
                                                              'decoded_pixel_sha256', 'content_group_id')},
                         'index': i, 'tile_start': i, 'tile_count': 1, 'family': family, 'target': i,
                         'native_font_verified': True, 'whole_width_covered': True, 'region_id': f'line-{i}',
                         'region_rgb_sha256': hashlib.sha256(f'{split}-{i}'.encode()).hexdigest(),
                         'ink_height_px': 40, 'font_size_screen_px': 50, 'log_em_ratio': np.log(1.25),
                         'tiles_sha256': hashlib.sha256(tiles[i:i+1].tobytes()).hexdigest()})
        (folder / 'metadata.json').write_text(json.dumps({'schema': DATA_SCHEMA, 'split': split, 'rows': rows}))
        partitions[split] = {'regions': 2, 'tiles': 2, 'family_counts': {f: 1 for f in families},
                             'array': {'path': f'{split}/tiles.npy', 'sha256': sha(folder / 'tiles.npy')},
                             'metadata': {'path': f'{split}/metadata.json', 'sha256': sha(folder / 'metadata.json')}}
    m = {'schema': DATA_SCHEMA, 'families': families, 'source_kind': 'ios_simulator_screenshot',
         'image_source': 'simctl_png', 'accepted_native_verified_only': True, 'network_inputs': ['image_tiles'],
         'ocr_performed': False, 'text_features_used': False, 'script_features_used': False,
         'character_segmentation_performed': False, 'splits': partitions, 'sources': sources,
         'preprocessing': {'source_sha256': sha(ROOT / 'src/flux_glyph/region_font.py'), 'shape': [1, 64, 256]},
         'split_isolation': {key: 0 for key in ('source_id', 'source_file_sha256', 'decoded_pixel_sha256',
                                                'page_id', 'content_group_id', 'region_rgb_sha256', 'normalized_region_text')}}
    for key in ('input_labels', 'scenes', 'capture_protocol'):
        path = tmp_path / f'{key}.json'
        path.write_text('{}')
        m[key] = {'path': str(path), 'sha256': sha(path)}
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(m))
    return m


def test_prepared_loader_validates_hashes_without_opening_test_array(tmp_path):
    m = fixture_manifest(tmp_path)
    # Test tensor is only validated when explicitly loaded after selection.
    (tmp_path / 'test/tiles.npy').write_bytes(b'corrupted held-out array')
    data = RegionData(tmp_path)
    assert len(data.load('train')['rows']) == 2
    assert len(data.load('calibration')['rows']) == 2
    assert set(data.loaded) == {'train', 'calibration'}
    with pytest.raises(ValueError, match='partition asset changed'):
        data.load('test')


@pytest.mark.parametrize('failure', ['source_overlap', 'font_truth', 'script_leak', 'ratio_truth', 'unknown_class'])
def test_prepared_contract_rejects_bad_provenance(tmp_path, failure):
    m = fixture_manifest(tmp_path)
    if failure == 'source_overlap':
        m['sources'][1]['content_group_id'] = m['sources'][0]['content_group_id']
    elif failure == 'unknown_class':
        m['families'][0] = 'invented font'
    else:
        path = tmp_path / 'train/metadata.json'
        metadata = json.loads(path.read_text())
        row = metadata['rows'][0]
        if failure == 'font_truth':
            row['native_font_verified'] = False
        elif failure == 'script_leak':
            row['script'] = 'han'
        else:
            row['font_size_screen_px'] = 60
        path.write_text(json.dumps(metadata))
        m['splits']['train']['metadata']['sha256'] = sha(path)
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        RegionData(tmp_path).load('train')
