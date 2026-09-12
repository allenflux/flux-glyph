"""Guards against admitting stale sources or leaking historical text."""
import json
from pathlib import Path

import pytest

from training.capture import prepare_sans_regions as subject
from training.capture.generate_scenes import sha


def fixture_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, 'COUNTS', {'train': 1})
    capture = tmp_path/'capture'
    capture.mkdir()
    plan = tmp_path/'plan'
    (plan/'assets').mkdir(parents=True)
    (plan/'assets/font.ttf').write_bytes(b'source font')
    previous = tmp_path/'old-scenes.json'
    previous.write_text(json.dumps({'pages': [{'regions': [{'text': 'Old text'}]}]}))
    scenes = {'pages': [{'id': 'p', 'split': 'train', 'regions': [{'text': 'New text'}]}],
              'fonts': [{'kind': 'asset', 'path': 'font.ttf', 'sha256': sha(plan/'assets/font.ttf')}]}
    (capture/'Scenes.json').write_text(json.dumps(scenes))
    sources = {'scenes_sha256': sha(capture/'Scenes.json'),
               'generator_sha256': sha(subject.ROOT/'training/capture/generate_sans_scenes.py'),
               'split_counts': {'train': 1}, 'android_native_rendering_validated': False,
               'old_test_pixels_opened': False, 'old_scene_sources': {str(previous): sha(previous)}}
    manifest = plan/'SOURCE_MANIFEST.json'
    manifest.write_text(json.dumps(sources))
    calls = []
    monkeypatch.setattr(subject, 'prepare', lambda *a, **k: calls.append(True))
    return capture, plan, previous, manifest, calls


def test_sans_prepare_rejects_changed_font_before_loading_captures(tmp_path, monkeypatch):
    capture, plan, _, manifest, calls = fixture_sources(tmp_path, monkeypatch)
    (plan/'assets/font.ttf').write_bytes(b'different face')
    with pytest.raises(AssertionError):
        subject.prepare_sans(capture/'labels.jsonl', tmp_path/'output', manifest)
    assert calls == []


def test_sans_prepare_rejects_changed_exclusion_source(tmp_path, monkeypatch):
    capture, _, previous, manifest, calls = fixture_sources(tmp_path, monkeypatch)
    previous.write_text(json.dumps({'pages': []}))
    with pytest.raises(AssertionError, match='Old text exclusion source changed'):
        subject.prepare_sans(capture/'labels.jsonl', tmp_path/'output', manifest)
    assert calls == []


def test_sans_prepare_rejects_old_text_even_if_exclusion_hash_is_current(tmp_path, monkeypatch):
    capture, _, previous, manifest, calls = fixture_sources(tmp_path, monkeypatch)
    previous.write_text(json.dumps({'pages': [{'regions': [{'text': 'NEW  TEXT'}]}]}))
    source = json.loads(manifest.read_text())
    source['old_scene_sources'][str(previous)] = sha(previous)
    manifest.write_text(json.dumps(source))
    with pytest.raises(AssertionError):
        subject.prepare_sans(capture/'labels.jsonl', tmp_path/'output', manifest)
    assert calls == []
