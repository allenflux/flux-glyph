"""Capture actual iOS Simulator frames and their native font instrumentation.

The app must already be installed. Images come from simctl's screenshot command;
this driver never renders text into an image or guesses font labels from iOS.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import time
import uuid

from PIL import Image

BUNDLE = 'tech.fluxglyph.capture'
SCHEMA = 'flux-glyph-capture-scenes-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def simctl(*args, timeout=60):
    result = subprocess.run(['xcrun', 'simctl', *map(str, args)], check=True,
                            capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def wait_frame(path, page_index, page_id, request_id, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            value = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            value = None
        if value and value.get('request_id') == request_id and value.get('error'):
            raise RuntimeError('Native capture error: ' + str(value['error']))
        if (value and value.get('page_index') == page_index and value.get('page_id') == page_id
                and value.get('request_id') == request_id):
            return value
        time.sleep(.025)
    raise TimeoutError(f'Native frame did not become ready: {page_id}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator', required=True)
    parser.add_argument('--scenes', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--frame-timeout', type=float, default=15)
    parser.add_argument('--settle-seconds', type=float, default=.12)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source = args.scenes.resolve()
    scenes = json.loads(source.read_text())
    if scenes.get('schema') != SCHEMA or scenes.get('platform') != 'ios':
        raise ValueError('Expected the explicit iOS scene protocol')
    pages = scenes['pages']
    ids = [page['id'] for page in pages]
    if len(set(ids)) != len(ids) or any(not re.fullmatch('[A-Za-z0-9_-]{1,100}', name) for name in ids):
        raise ValueError('Scene page IDs must be safe and unique')
    if any(page.get('split') not in ('train', 'calibration', 'test') for page in pages):
        raise ValueError('Every page must declare its split before capture')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = output / 'CAPTURE_PROTOCOL.json'
    scenes_sha = sha(source)
    if existing.exists():
        previous = json.loads(existing.read_text())
        if not args.resume or previous['scenes_sha256'] != scenes_sha or previous['simulator_id'] != args.simulator:
            raise ValueError('Existing capture requires --resume with the identical scenes and simulator')
    documents = Path(simctl('get_app_container', args.simulator, BUNDLE, 'data')) / 'Documents'
    app_bundle = Path(simctl('get_app_container', args.simulator, BUNDLE, 'app'))
    app_info = plistlib.loads((app_bundle / 'Info.plist').read_bytes())
    app_executable = app_bundle / app_info['CFBundleExecutable']
    documents.mkdir(exist_ok=True)
    shutil.copyfile(source, documents / 'Scenes.json')
    if args.assets:
        destination = documents / 'Fonts'
        destination.mkdir(exist_ok=True)
        for font in scenes.get('fonts', []):
            if font.get('kind') != 'asset':
                continue
            name = font['path']
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('Font asset paths must be basenames')
            path = args.assets / name
            if sha(path) != font['sha256']:
                raise ValueError('Source font SHA differs: ' + name)
            shutil.copyfile(path, destination / name)
    (output / 'images').mkdir(exist_ok=True)
    (output / 'frames').mkdir(exist_ok=True)
    atomic_json(existing, {'schema': 'ios-native-screen-capture-v1', 'source_kind': 'ios_simulator_screenshot',
                          'simulator_id': args.simulator, 'bundle_id': BUNDLE,
                          'scenes_sha256': scenes_sha, 'planned_pages': len(pages),
                          'capture_command': 'xcrun simctl io <simulator> screenshot --type=png <file>',
                          'capture_driver_sha256': sha(Path(__file__)),
                          'capture_app_executable_sha256': sha(app_executable),
                          'capture_app_source_sha256': sha(Path(__file__).parent / 'ios/FluxFontCapture/AppDelegate.swift'),
                          'device_list': json.loads(simctl('list', 'devices', 'available', '--json')),
                          'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    shutil.copyfile(source, output / 'Scenes.json')
    # Relaunch only this capture app to load the supplied fonts and scenes.
    try:
        simctl('terminate', args.simulator, BUNDLE)
    except subprocess.CalledProcessError:
        pass
    simctl('launch', args.simulator, BUNDLE)
    # The first app-opening animation happens outside our UIView's display
    # callbacks. Subsequent pages use the local command file, with no URL modal.
    time.sleep(1.2)
    ready_path = documents / 'frame-ready.json'
    stop = min(len(pages), args.start + args.limit) if args.limit is not None else len(pages)
    if not 0 <= args.start < stop:
        raise ValueError('Empty or invalid capture range')
    started = time.monotonic()
    for index in range(args.start, stop):
        page = pages[index]
        image_path = output / 'images' / (page['id'] + '.png')
        record_path = output / 'frames' / (page['id'] + '.json')
        if args.resume and record_path.exists():
            old = json.loads(record_path.read_text())
            if old['source_sha256'] != sha(image_path) or old['page_index'] != index:
                raise ValueError('Existing captured frame was changed')
            continue
        ready_path.unlink(missing_ok=True)
        request_id = uuid.uuid4().hex
        atomic_json(documents / 'command.json', {'page_index': index, 'request_id': request_id})
        frame = wait_frame(ready_path, index, page['id'], request_id, args.frame_timeout)
        if frame.get('error'):
            raise RuntimeError('Capture app rejected page: ' + str(frame['error']))
        time.sleep(args.settle_seconds)
        simctl('io', args.simulator, 'screenshot', '--type=png', image_path)
        with Image.open(image_path) as image:
            size = list(image.size)
        if size != frame.get('pixel_size'):
            raise ValueError(f'Screen dimensions disagree with native instrumentation: {size}, {frame.get("pixel_size")}')
        native = dict(frame)
        record = {'schema': 'ios-native-captured-frame-v1', 'page_index': index, 'page_id': page['id'],
                  'image': str(image_path), 'source_id': 'ios:' + page['id'], 'split': page['split'],
                  'source_kind': 'ios_simulator_screenshot', 'source_sha256': sha(image_path),
                  'scenes_sha256': scenes_sha, 'simulator_id': args.simulator, 'bundle_id': BUNDLE,
                  'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'native': native, 'regions': native.get('regions', []), 'pixel_size': size}
        atomic_json(record_path, record)
        if (index - args.start + 1) % 20 == 0 or index == args.start or index + 1 == stop:
            print(json.dumps({'captured': index - args.start + 1, 'page_index': index,
                              'total': stop - args.start, 'seconds': round(time.monotonic() - started, 2),
                              'regions': len(record['regions'])}), flush=True)
    inventory = documents / 'font-inventory.json'
    if inventory.exists():
        shutil.copyfile(inventory, output / 'font-inventory.json')
    rows = [json.loads(path.read_text()) for path in sorted((output / 'frames').glob('*.json'))]
    with (output / 'labels.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_json(output / 'CAPTURE_SUMMARY.json', {'captured_pages': len(rows),
                'source_kind': 'ios_simulator_screenshot', 'split_counts': {
                    split: sum(row['split'] == split for row in rows) for split in ('train', 'calibration', 'test')},
                'labels_sha256': sha(output / 'labels.jsonl')})


if __name__ == '__main__':
    main()
