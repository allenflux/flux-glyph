"""Native font identities and explicit image-identifiable training labels."""
from training.capture.generate_scenes import capture_families

PINGFANG_GROUP = {'PingFang': ['PingFang SC', 'PingFang TC', 'PingFang HK']}


def label_groups(group_pingfang=False):
    return {key: list(value) for key, value in PINGFANG_GROUP.items()} if group_pingfang else {}


def validate_groups(groups):
    if groups not in ({}, PINGFANG_GROUP):
        raise ValueError('Unknown or incomplete native font label grouping')
    return groups


def font_label(native_family, groups):
    validate_groups(groups)
    for label, native_names in groups.items():
        if native_family in native_names:
            return label
    return native_family


def region_families(groups):
    return list(dict.fromkeys(font_label(name, groups) for name in capture_families()))
