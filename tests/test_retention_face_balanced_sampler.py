"""Face-balanced paired-known sampling uses synthetic TRAIN metadata only."""
from collections import Counter
from copy import deepcopy

import pytest

from training.retention_face_balanced_sampler import (
    FACE_NAMES, FACE_WEIGHTS, MICRO, SAMPLING, WENKAI, FaceBalancedSampler,
)
from training.retention_paired_known_sampler import KNOWN_SUPPLEMENT_MARKER
from training.retention_source_balanced_sampler import expected_unknown_source_counts
from training.retention_supplement_sampler import IOS_ANCHOR_FAMILIES, NEW_SOURCES, SUPPLEMENT_MARKER, VIEW_WEIGHTS

FAMILIES = [*IOS_ANCHOR_FAMILIES, WENKAI, MICRO,
            *[f'Joint {index}' for index in range(14)], '__unknown__']
OLD_UNKNOWN_SOURCES = tuple(f'Unknown {index}' for index in range(9))


def face_identity(family, weight):
    index = sum(map(ord, family + weight))
    face = FACE_NAMES.get(family, {}).get(weight, f'{family}-{weight}')
    return face, f'{index:064x}', 0


def fixtures():
    old = []
    for target, family in enumerate(FAMILIES):
        sources = OLD_UNKNOWN_SOURCES if family == '__unknown__' else (family,)
        weights = FACE_WEIGHTS[WENKAI] if family == WENKAI else FACE_WEIGHTS[MICRO] if family == MICRO else ('Regular', 'Bold')
        for source in sources:
            for domain in ('ios', 'android'):
                for weight in weights:
                    face, digest, ttc = face_identity(family if family != '__unknown__' else source, weight)
                    for view in ('native', 'half'):
                        old.append({'family': family, 'target': target, 'split': 'train', 'domain': domain,
                            'view': view, 'source_font_family': source, 'font_face': face,
                            'font_file_sha256': digest, 'ttc_index': ttc, 'native_font_verified': True,
                            'tile_start': len(old), 'tile_count': 1, 'source_id': f'old-{len(old)}',
                            'region_id': 'region', 'ink_height_px': 20, 'font_size_px': 20})

    def indices(families, one_face=False):
        result = [index for index, row in enumerate(old) if row['domain'] == 'ios'
                  and row['view'] == 'native' and row['family'] in families]
        return [index for index in result if not one_face or old[index]['font_face'].endswith('-Regular')]

    pools = [{'id': 'ios_native_pingfang', 'indices': indices(('PingFang',))},
        {'id': 'ios_native_latin', 'indices': indices(('SF Pro', 'Helvetica'))},
        {'id': 'ios_native_anchor_original8', 'indices': indices(IOS_ANCHOR_FAMILIES, True)}]
    new = [{'family': '__unknown__', 'target': 24, 'split': 'train', 'domain': 'android',
        'view': view, 'source_font_family': source, 'font_face': f'{source}-Regular',
        'font_file_sha256': f'{100 + index:064x}', 'ttc_index': 0, 'native_font_verified': True,
        'tile_start': index, 'tile_count': 1, 'source_id': f'new-{index}', 'region_id': 'region'}
        for index, (source, view) in enumerate((source, view) for source in NEW_SOURCES for view in VIEW_WEIGHTS)]
    known = []
    # The merger may place old Regular rows before new Light/Medium rows.
    for family, weights in ((WENKAI, ('Regular',)), (MICRO, ('Regular',)),
                            (WENKAI, ('Light', 'Medium'))):
        for weight in weights:
            face, digest, ttc = face_identity(family, weight)
            for view in VIEW_WEIGHTS:
                offset = len(known)
                known.append({'family': family, 'target': FAMILIES.index(family), 'split': 'train',
                    'domain': 'android', 'view': view, 'source_font_family': family,
                    'font_face': face, 'font_file_sha256': digest, 'ttc_index': ttc,
                    'native_font_verified': True, 'source_dataset': 'android_paired_known_supplement',
                    'tile_start': offset, 'tile_count': 1, 'source_id': f'known-{offset}',
                    'region_id': f'region-{offset}', 'pair_evidence': {
                        'intentional_train_text_pair': True, 'training_family': '__unknown__',
                        'source_id': f'unknown-{offset}', 'region_id': f'unknown-region-{offset}',
                        'font_family': f'paired-source-{offset}'}})
    return old, pools, new, known


def sampler(seed=2026091415):
    old, pools, new, known = fixtures()
    return FaceBalancedSampler(old, FAMILIES, seed, pools, new, known)


def test_wenkai_face_cycle_is_exact_and_reproducible():
    first, second = sampler(17), sampler(17)
    observed = []
    for _ in range(9):
        a, b = first.batch(), second.batch()
        assert a == b
        paired = [row for row in a if row.get(KNOWN_SUPPLEMENT_MARKER)]
        assert Counter(row['family'] for row in paired) == {WENKAI: 1, MICRO: 1}
        observed.append(next(weight for row in paired if row['family'] == WENKAI
                             for weight, face in FACE_NAMES[WENKAI].items() if row['font_face'] == face))
    assert observed == ['Light', 'Medium', 'Regular'] * 3


def test_exact_6000_face_source_slot_and_family_counts():
    replay = sampler()
    for _ in range(6000):
        replay.batch()
    report = replay.report()
    paired_faces = {next(weight for weight, face in FACE_NAMES[WENKAI].items()
                         if row['font_face'] == face): row['rows']
                    for row in report['known_supplement_face_rows'] if row['family'] == WENKAI}
    all_wenkai = Counter()
    for row in report['face_rows']:
        if row['family'] == WENKAI:
            all_wenkai[next(weight for weight, face in FACE_NAMES[WENKAI].items()
                            if row['font_face'] == face)] += row['rows']
    assert paired_faces == dict.fromkeys(FACE_WEIGHTS[WENKAI], 2000)
    assert dict(all_wenkai) == dict.fromkeys(FACE_WEIGHTS[WENKAI], 4000)
    assert {row['rows'] for row in report['known_supplement_face_rows'] if row['family'] == MICRO} == {6000}
    assert report['family_rows'][WENKAI] == report['family_rows'][MICRO] == 12000
    assert report['family_rows']['__unknown__'] == 96000
    assert report['unknown_source_rows'] == expected_unknown_source_counts(replay.unknown_source_order, 6000)
    supplement_rows = sum(report['supplement_source_rows'].values())
    assert report['slots'] == {'base_known': 276000, 'known_supplement': 12000,
        'base_unknown': 96000 - supplement_rows, 'supplement_unknown': supplement_rows,
        'ios_native_pingfang': 96000, 'ios_native_sfpro_helvetica': 48000,
        'ios_native_original_eight': 48000}
    assert sum(report['family_rows'].values()) == sum(report['slots'].values()) == 576000
    assert report['schema'] == 'flux-glyph-retention-face-balanced-sampling-v1'
    assert SAMPLING['known_supplement_wenkai_rows_per_face_at_6000_steps'] == 2000


@pytest.mark.parametrize('fault', ['missing_face', 'missing_view', 'relabel', 'offset', 'tile_count',
    'pair', 'face_identity', 'face_name', 'marker', 'duplicate'])
def test_invalid_merged_face_metadata_is_rejected(fault):
    old, pools, new, known = fixtures()
    if fault == 'missing_face':
        known = [row for row in known if row['font_face'] != FACE_NAMES[WENKAI]['Light']]
        for offset, row in enumerate(known): row['tile_start'] = offset
    elif fault == 'missing_view':
        known = [row for row in known if not (row['font_face'] == FACE_NAMES[WENKAI]['Medium'] and row['view'] == 'half')]
        for offset, row in enumerate(known): row['tile_start'] = offset
    elif fault == 'relabel': known[0]['target'] = 0
    elif fault == 'offset': known[1]['tile_start'] += 1
    elif fault == 'tile_count': known[0]['tile_count'] = 0
    elif fault == 'pair': known[0]['pair_evidence']['training_family'] = WENKAI
    elif fault == 'face_identity': known[1]['font_file_sha256'] = 'f'*64
    elif fault == 'face_name': known[0]['font_face'] = FACE_NAMES[MICRO]['Regular']
    elif fault == 'marker': known[0][SUPPLEMENT_MARKER] = True
    else:
        known[1]['source_id'], known[1]['region_id'], known[1]['view'] = (
            known[0]['source_id'], known[0]['region_id'], known[0]['view'])
    with pytest.raises(ValueError):
        FaceBalancedSampler(old, FAMILIES, 17, pools, new, known)


def test_inputs_are_not_mutated():
    old, pools, new, known = fixtures()
    before = deepcopy((old, pools, new, known))
    replay = FaceBalancedSampler(old, FAMILIES, 17, pools, new, known)
    replay.batch()
    assert (old, pools, new, known) == before
    assert all(KNOWN_SUPPLEMENT_MARKER not in row for row in known)
