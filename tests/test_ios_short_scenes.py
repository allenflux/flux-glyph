import hashlib
from collections import Counter

import pytest

from capture.generate_ios_short_scenes import corpus, generate, text_sha
from capture.prepare_captured import load_scenes


def test_train_only_native_request_contract(tmp_path):
    import json
    scenes = generate()
    path = tmp_path / 'Scenes.json'
    path.write_text(json.dumps(scenes))
    _, pages, _ = load_scenes(path)
    rows = [r for p in pages.values() for r in p['regions']]
    assert len(rows) == 704
    assert {p['split'] for p in pages.values()} == {'train'}
    assert Counter(len(r['text']) for r in rows) == dict.fromkeys(range(1, 5), 176)
    assert set(Counter(r['font_id'] for r in rows).values()) == {64}
    assert all(f['kind'] == 'system' and f['path'] is None for f in scenes['fonts'])
    assert scenes['test_read'] is False
    assert scenes == generate()


def test_cross_face_content_and_style_do_not_encode_family():
    rows = [r for p in generate()['pages'] for r in p['regions']]
    for kind in ('han', 'english', 'numeric'):
        by_face = {}
        for r in rows:
            if r['text_kind'] == kind:
                by_face.setdefault(r['font_id'], set()).add((r['text'], r['font_size'], r['color'], r['language']))
        pools = list(by_face.values())
        assert all(pool == pools[0] for pool in pools)


def test_full_text_hash_filter_preserves_train_only_contract():
    rows = [r for p in generate()['pages'] for r in p['regions']]
    forbidden = {text_sha(rows[0]['text'])}
    changed = generate(forbidden)
    assert not forbidden.intersection(text_sha(r['text']) for p in changed['pages'] for r in p['regions'])
    assert text_sha(' Ａ b ') == hashlib.sha256(b'ab').hexdigest()


def test_exhausted_numeric_space_fails_instead_of_reusing_heldout():
    import random
    with pytest.raises(ValueError, match='Insufficient'):
        corpus('numeric', 1, 8, random.Random(1), {text_sha(str(i)) for i in range(10)})
