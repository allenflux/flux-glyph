from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

import merge_unified_weight_pairs as merge


FAMILIES = [f'family-{index}' for index in range(24)] + ['__unknown__']
FAMILIES[10] = 'LXGW WenKai'
FAMILIES[11] = 'WenQuanYi Micro Hei'


def verified_loader_result(label, faces):
    """Return a compact post-loader structure; native proof loading is out of scope here."""
    rows = []
    blocks = []
    offset = 0
    for face_index, (family, font_face) in enumerate(faces):
        source_id = f'{label}:source-{face_index}'
        region_id = f'{label}:region-{face_index}'
        for view_index, view in enumerate(merge.VIEWS):
            count = 2 if view == 'native' else 1
            block = np.full((count, 1, 64, 256),
                            face_index * 100 + view_index * 10 + (1 if label == 'new' else 0),
                            dtype=np.float32)
            rows.append({
                'split': 'train',
                'original_split': 'train',
                'family': family,
                'target': merge.TARGETS[family],
                'source_id': source_id,
                'region_id': region_id,
                'source_font_family': family,
                'font_face': font_face,
                'domain': 'android',
                'source_dataset': 'android_paired_known_supplement',
                'native_font_verified': True,
                'view': view,
                'tile_start': offset,
                'tile_count': count,
                'fixture_marker': f'{label}-{face_index}-{view_index}',
            })
            blocks.append(block)
            offset += count
    tiles = np.concatenate(blocks, axis=0)
    return {
        'families': list(FAMILIES),
        'rows': rows,
        'tiles': tiles,
        'partition': {'split': 'train', 'families': list(FAMILIES), 'rejected': []},
    }


def sources():
    return (
        verified_loader_result('old', sorted(merge.OLD_FACES)),
        verified_loader_result('new', sorted(merge.NEW_FACES)),
    )


def test_assemble_keeps_old_regular_micro_and_only_new_light_medium_exactly():
    old, new = sources()
    old_before, new_before = deepcopy(old['rows']), deepcopy(new['rows'])
    result = merge._assemble_verified(old, new)

    assert old['rows'] == old_before
    assert new['rows'] == new_before
    assert {(row['family'], row['font_face']) for row in result['rows']} == merge.KEPT_FACES
    assert len(result['rows']) == 16
    assert len(result['excluded']) == 4
    assert {row['font_face'] for row in result['excluded']} == {'WenQuanYiMicroHei'}
    assert {row['reason'] for row in result['excluded']} == {
        'duplicate_new_microhei_face_excluded'
    }

    expected_blocks = []
    expected_rows = list(old['rows']) + [
        row for row in new['rows'] if row['font_face'] != 'WenQuanYiMicroHei'
    ]
    for row in expected_rows:
        source = old if row['source_id'].startswith('old:') else new
        expected_blocks.append(source['tiles'][row['tile_start']:row['tile_start'] + row['tile_count']])
    expected = np.concatenate(expected_blocks, axis=0)
    assert result['tiles'].dtype == np.float32
    assert result['tiles'].tobytes() == expected.tobytes()

    offset = 0
    for output_row, source_row, audit in zip(result['rows'], expected_rows, result['mapping']):
        assert output_row == {**source_row, 'tile_start': offset}
        assert audit['source_tile_start'] == source_row['tile_start']
        assert audit['output_tile_start'] == offset
        assert audit['tile_count'] == source_row['tile_count']
        offset += source_row['tile_count']


@pytest.mark.parametrize('mutation', ['relabel', 'pseudo_source', 'noncontiguous', 'missing_view'])
def test_assemble_rejects_invalid_already_verified_source_structure(mutation):
    old, new = sources()
    if mutation == 'relabel':
        new['rows'][0]['target'] = 24
    elif mutation == 'pseudo_source':
        new['rows'][0]['source_dataset'] = 'pseudo_labels'
    elif mutation == 'noncontiguous':
        old['rows'][1]['tile_start'] += 1
    else:
        removed = new['rows'].pop(0)
        count = removed['tile_count']
        new['tiles'] = new['tiles'][count:].copy()
        for row in new['rows']:
            row['tile_start'] -= count

    with pytest.raises(ValueError):
        merge._assemble_verified(old, new)


def test_assemble_rejects_duplicate_source_identity_even_when_new_row_is_excluded():
    old, new = sources()
    old_micro = [row for row in old['rows'] if row['font_face'] == 'WenQuanYiMicroHei']
    new_micro = [row for row in new['rows'] if row['font_face'] == 'WenQuanYiMicroHei']
    for old_row, new_row in zip(old_micro, new_micro):
        new_row['source_id'] = old_row['source_id']
        new_row['region_id'] = old_row['region_id']

    with pytest.raises(ValueError, match='conflicting duplicate source rows'):
        merge._assemble_verified(old, new)


def test_assemble_requires_source_ids_to_be_distinct_across_roots():
    old, new = sources()
    new_source = new['rows'][0]['source_id']
    old_source = old['rows'][0]['source_id']
    for row in new['rows']:
        if row['source_id'] == new_source:
            row['source_id'] = old_source

    with pytest.raises(ValueError, match='conflicting duplicate source rows'):
        merge._assemble_verified(old, new)


def test_rejection_selection_records_retained_and_excluded_source_decisions():
    old, new = sources()
    old_rejected = deepcopy(old['rows'][0])
    new_light_rejected = deepcopy(next(row for row in new['rows']
                                       if row['font_face'] == 'LXGWWenKai-Light'))
    new_micro_rejected = deepcopy(next(row for row in new['rows']
                                       if row['font_face'] == 'WenQuanYiMicroHei'))
    old['partition']['rejected'] = [old_rejected]
    new['partition']['rejected'] = [new_light_rejected, new_micro_rejected]

    kept, retained_audit, excluded_audit = merge._rejection_selection(old, new)

    assert kept == [old_rejected, new_light_rejected]
    assert [(row['source'], row['source_rejected_index']) for row in retained_audit] == [
        ('old', 0), ('new', 0)
    ]
    assert excluded_audit == [{
        'source': 'new',
        'source_rejected_index': 1,
        'source_id': new_micro_rejected['source_id'],
        'region_id': new_micro_rejected['region_id'],
        'view': new_micro_rejected['view'],
        'family': 'WenQuanYi Micro Hei',
        'font_face': 'WenQuanYiMicroHei',
        'reason': 'duplicate_new_microhei_face_excluded',
    }]


def test_coverage_accounts_for_accepted_and_rejected_views_per_native_region():
    old, new = sources()
    result = merge._assemble_verified(old, new)
    regular = [deepcopy(row) for row in result['rows'] if row['font_face'] == 'LXGWWenKai-Regular']
    for row in regular:
        row['source_id'] += '-second'
        row['region_id'] += '-second'
    result['rows'].extend(regular)
    moved = result['rows'].pop(0)
    rejected = [{**moved, 'rejection_reason': 'synthetic preprocessing rejection'}]

    counts = merge._coverage(result['rows'], rejected)

    assert counts == {
        'LXGW WenKai/LXGWWenKai-Light': 1,
        'LXGW WenKai/LXGWWenKai-Medium': 1,
        'LXGW WenKai/LXGWWenKai-Regular': 2,
        'WenQuanYi Micro Hei/WenQuanYiMicroHei': 1,
    }


def test_coverage_rejects_unaccounted_view():
    old, new = sources()
    result = merge._assemble_verified(old, new)
    result['rows'].pop(0)
    with pytest.raises(ValueError, match='all four requested views'):
        merge._coverage(result['rows'], [])


def test_parser_uses_distinct_fresh_merged_output():
    args = merge.parser().parse_args([])
    assert args.old.name == 'data'
    assert args.new.name == 'data'
    assert args.output == ROOT / 'artifacts/unified-font-v3/paired-wenkai-weight-capture-v1/merged-data'
    assert args.output != args.old and args.output != args.new
