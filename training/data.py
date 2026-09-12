"""Source-verified synthetic data; partitions stay isolated by character."""
from __future__ import annotations
import hashlib
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flux_glyph.glyph_preprocess import extract_glyphs
from network import FAMILIES


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def old_dataset(old_root, version, split):
    folder = old_root / 'runs' / version / 'data/arrays' / split
    expected = json.loads((folder.parent / 'PROTOCOL.json').read_text())['splits'][split]
    array_sha, metadata_sha = sha(folder / 'glyphs.npy'), sha(folder / 'metadata.json')
    if array_sha != expected['glyphs_sha256'] or metadata_sha != expected['metadata_sha256']:
        raise ValueError('Frozen legacy dataset checksum differs: ' + str(folder))
    rows = json.loads((folder / 'metadata.json').read_text())['rows']
    x = np.load(folder / 'glyphs.npy', mmap_mode='r')
    valid = np.array([row.get('valid', True) for row in rows], dtype=bool)
    y = np.array([FAMILIES.index(row['family_label']) for row in rows], dtype=np.int64)
    chars = np.array([row['character'] for row in rows])
    return {'x': x, 'y': y, 'chars': chars, 'valid': valid,
            'faces': np.array([row['font_id'] for row in rows]),
            'sizes': np.array([row['point_size'] for row in rows]),
            'name': version, 'script': 'han',
            'source': {'path': str(folder), 'array_sha256': array_sha,
                       'metadata_sha256': metadata_sha, 'frozen_checksums_verified': True, 'rows': len(x)}}


def npz_dataset(path, script):
    a = np.load(path, allow_pickle=False)
    return {**{key: a[key] for key in a.files}, 'valid': np.ones(len(a['x']), dtype=bool),
            'name': str(path), 'script': script,
            'source': {'path': str(path), 'sha256': sha(path), 'rows': len(a['x'])}}


def build_latin(old_root, output):
    # Reuse source validation/render helpers without importing production OCR
    # (OpenCV/ONNX Runtime are not requirements of the training environment).
    from fontTools.ttLib import TTFont
    import types
    source = ROOT / 'scripts/build_latin_references.py'
    parsed = ast.parse(source.read_text())
    nodes = [node for node in parsed.body if
             (isinstance(node, ast.FunctionDef) and node.name in {'sha', 'sources', 'font_for', 'render'}) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_HASHES' for t in node.targets))]
    namespace = {'hashlib': hashlib, 'Path': Path, 'json': json, 'TTFont': TTFont,
                 'ImageFont': ImageFont, 'Image': Image, 'ImageDraw': ImageDraw,
                 'ALPHABET': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.:,'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    builder = types.SimpleNamespace(**namespace)
    manifest = old_root / 'runs/alipay-font-generalization-v8-20260911/data/fonts.json'
    sources = builder.sources(manifest, Path('/System/Library/Fonts'))
    roboto = ROOT / 'artifacts/neural-font-v1/downloads/roboto-google-fonts/Roboto[wdth,wght].ttf'
    if sha(roboto) != 'd7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134':
        raise ValueError('Official Roboto source checksum differs')
    for weight in (400, 500, 700):
        sources.append({'id': f'roboto_{weight}', 'family': 'Roboto', 'path': roboto,
                        'postscript_name': 'Roboto-Regular', 'index': 0,
                        'variations': [weight, 100], 'characters': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'})
    # Six training digits; two calibration and two test digits. The whole
    # original render and all of its degradations stay in the same partition.
    partitions = {'train': '012457ABCDEFGHIJKLMNOPQabcdefghijklmnopq',
                  'calibration': '38RSTUVrstuv', 'test': '69WXYZwxyz'}
    sizes = {'train': [16, 22, 30, 42, 56], 'calibration': [20, 36], 'test': [24, 38]}
    assert not set(partitions['train']) & set(partitions['calibration'])
    assert not set(partitions['train']) & set(partitions['test'])
    assert not set(partitions['calibration']) & set(partitions['test'])
    audit = {'schema': 'neural-latin-source-data-v1', 'partitions': partitions, 'sizes': sizes,
             'split_unit': 'character; all sizes/views of a character are kept together',
             'source_manifest_sha256': sha(manifest), 'sources': [], 'splits': {}}
    for source in sources:
        audit['sources'].append({key: (str(value) if isinstance(value, Path) else value)
                                 for key, value in source.items()} | {'sha256': sha(source['path'])})
    for split, characters in partitions.items():
        path = output / (split + '.npz')
        if path.exists():
            raise FileExistsError(path)
        images, targets, cs, faces, ss, rejected = [], [], [], [], [], []
        for source in sources:
            for size in sizes[split]:
                font = builder.font_for(source, size)
                for char in characters:
                    if char not in source['characters']:
                        continue
                    image = builder.render(char, font)
                    for view in ('identity', 'jpeg'):
                        current = image
                        if view == 'jpeg':
                            buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=82 if split=='train' else 75)
                            buffer.seek(0); current = Image.open(buffer).convert('RGB')
                        result = extract_glyphs(current, 1)
                        if result.glyphs is None:
                            rejected.append({'face': source['id'], 'size': size, 'character': char,
                                             'view': view, 'reason': result.diagnostics['reason']})
                            continue
                        images.append(result.glyphs[0]); targets.append(FAMILIES.index(source['family']))
                        cs.append(char); faces.append(source['id']); ss.append(size)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=np.stack(images), y=np.array(targets, dtype=np.int64),
                            chars=np.array(cs), faces=np.array(faces), sizes=np.array(ss))
        audit['splits'][split] = {'path': str(path), 'rows': len(images), 'sha256': sha(path),
                                   'rejections': rejected, 'family_counts': {name: targets.count(i) for i, name in enumerate(FAMILIES) if i in targets}}
        print(json.dumps({'latin_built': split, 'rows': len(images), 'rejected': len(rejected)}), flush=True)
    dump(output / 'manifest.json', audit)
    return audit
