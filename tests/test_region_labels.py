import json

import pytest

from training.region_labels import font_label, label_groups, region_families, validate_groups
from training.train_regions import RegionData, sha
from test_train_regions import fixture_manifest


def test_region_grouping_preserves_native_identity_and_other_families():
    groups = label_groups(True)
    assert all(font_label(name, groups) == 'PingFang' for name in groups['PingFang'])
    assert font_label('SF Pro', groups) == 'SF Pro'
    assert font_label('PingFang TC', {}) == 'PingFang TC'
    assert 'PingFang' in region_families(groups)
    assert not set(groups['PingFang']) & set(region_families(groups))


@pytest.mark.parametrize('groups', [{'PingFang': ['PingFang TC']}, {'SF Pro': ['Alipay Number']}, {'PingFang SC': ['PingFang TC']}])
def test_incomplete_or_misleading_font_groups_are_rejected(groups):
    with pytest.raises(ValueError, match='grouping'):
        validate_groups(groups)


@pytest.mark.parametrize('native_family', ['PingFang TC', 'SF Pro', 'PingFang'])
def test_data_loader_requires_real_native_member_for_group_target(tmp_path, native_family):
    manifest = fixture_manifest(tmp_path)
    manifest['families'] = ['PingFang', 'SF Pro']
    manifest['font_label_groups'] = label_groups(True)
    for split, descriptor in manifest['splits'].items():
        path = tmp_path / descriptor['metadata']['path']
        metadata = json.loads(path.read_text())
        row = metadata['rows'][0]
        row.update(family='PingFang', native_font_family=native_family)
        path.write_text(json.dumps(metadata))
        descriptor['metadata']['sha256'] = sha(path)
        descriptor['family_counts'] = {'PingFang': 1, 'SF Pro': 1}
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(manifest))
    if native_family == 'PingFang TC':
        assert RegionData(tmp_path).load('train')['rows'][0]['native_font_family'] == native_family
    else:
        with pytest.raises(ValueError, match='native font identity'):
            RegionData(tmp_path).load('train')
