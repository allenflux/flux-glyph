from collections import Counter
import copy
import json
import numpy as np
import pytest
import native_mobile_short_training as mobile


def fixture():
    datasets = {'ios': {'rows': []}, 'android': {'rows': []}}
    for source in (*mobile.KNOWN, *mobile.UNKNOWN_SOURCES):
        family = source if source in mobile.KNOWN else '__unknown__'
        dataset = 'ios' if source in ('PingFang', 'SF Pro', 'Helvetica') else 'android'
        for script in ('han', 'latin'):
            for length in (1, 2, 3, 4):
                for face in ('Regular', 'Bold'):
                    for identity_number in range(2):
                        identity = f'{source}/{script}/{length}/{face}/{identity_number}'
                        for view in mobile.VIEWS:
                            datasets[dataset]['rows'].append({'view': view, 'family': family,
                                'target': mobile.FAMILIES.index(family), 'font_face': source + '-' + face,
                                'glyph_count': length, 'source_id': source + '/page', 'region_id': identity,
                                'source_font_family': source, 'normalized_text_sha256': identity + '/text',
                                'native_font_size_px': 48, 'text_color_hex': '#333333',
                                'source_sha256': identity + '/source', 'frame_sha256': identity + '/proof',
                                'tiles_sha256': identity + '/' + view, 'region_rgb_sha256': identity + '/pixels',
                                'script': script})
    return datasets


def sampler(seed=123):
    datasets = fixture(); pools, proof = mobile.build_pools(datasets)
    return mobile.NativeMobileShortSampler({'pools': pools}, seed), datasets, proof


def test_balances_true_families_and_unknown_sources_independently_of_platform():
    value, _, _ = sampler()
    family_counts, lengths, sources = Counter(), Counter(), Counter()
    for _ in range(2400):
        rows = value.batch()
        assert len(rows) == 20
        assert Counter(r['family'] for r in rows) == {**dict.fromkeys(mobile.KNOWN, 1), '__unknown__': 3}
        family_counts.update(r['family'] for r in rows)
        lengths.update((r['source_font_family'], r['script'], r['glyph_count']) for r in rows)
        sources.update(r['source_font_family'] for r in rows)
    assert all(family_counts[f] == 2400 for f in mobile.KNOWN)
    assert family_counts['__unknown__'] == 7200
    assert all(sources[s] == 1800 for s in mobile.UNKNOWN_SOURCES)
    for source in (*mobile.KNOWN, *mobile.UNKNOWN_SOURCES):
        for script in ('han', 'latin'):
            for length in (1, 2, 3, 4):
                assert lengths[source, script, length] == (300 if source in mobile.KNOWN else 225)
    report = value.report()
    assert report['rows'] == sum(report['by_source'].values()) == 48000
    assert sum(report['by_identity'].values()) == 48000
    faces = Counter()
    for key, number in report['by_face'].items():
        source, script, length, face = json.loads(key)
        faces[source, script, length, face] += number
    for source in (*mobile.KNOWN, *mobile.UNKNOWN_SOURCES):
        for script in ('han', 'latin'):
            for length in (1, 2, 3, 4):
                assert abs(faces[source, script, length, source+'-Bold'] - faces[source, script, length, source+'-Regular']) <= 1


def test_reproducible_sampler_does_not_mutate_rows_or_global_rng():
    first, source, _ = sampler(); second, _, _ = sampler()
    snapshot = copy.deepcopy(source)
    np.random.seed(987); expected = np.random.rand(5)
    np.random.seed(987)
    for _ in range(12):
        a, b = first.batch(), second.batch()
        assert a == b
        a[0]['family'] = 'mutated caller copy'
    assert np.array_equal(np.random.rand(5), expected)
    assert first.report() == second.report() and source == snapshot


def test_cross_family_pixel_conflict_excludes_all_views_of_both_identities():
    datasets = fixture()
    a = next(r for r in datasets['ios']['rows'] if r['view'] == 'native')
    b = next(r for r in datasets['android']['rows'] if r['view'] == 'native')
    b['tiles_sha256'] = a['tiles_sha256']
    pools, proof = mobile.build_pools(datasets)
    excluded = {tuple(identity) for identity in proof['cross_dataset_conflicting_native_identities']}
    assert excluded == {('ios', a['source_id'], a['region_id']), ('android', b['source_id'], b['region_id'])}
    assert all(not (set(groups) & excluded) for faces in pools.values() for groups in faces.values())
    assert proof['native_regions'] == proof['native_regions_before_conflict_exclusion'] - 2


def test_missing_view_or_native_truth_disagreement_is_rejected():
    datasets = fixture(); datasets['ios']['rows'].pop()
    with pytest.raises(ValueError, match='four related views'):
        mobile.build_pools(datasets)
    datasets = fixture(); datasets['ios']['rows'][0]['glyph_count'] = 4
    with pytest.raises(ValueError, match='disagree on native truth'):
        mobile.build_pools(datasets)
