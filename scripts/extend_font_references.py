#!/usr/bin/env python3
"""Append hash-verified source-font glyphs without changing existing references."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import zipfile

import numpy as np
from PIL import Image
from fontTools.ttLib import TTCollection, TTFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flux_glyph.font_matcher import CompactFontBank, raster
from flux_glyph.models import file_sha


def resolve_source(row, characters):
    path = Path(row['path'])
    if file_sha(path) != row['source_sha256']:
        raise ValueError('Source font checksum mismatch: ' + row['font_id'])
    collection = TTCollection(path, lazy=True) if path.suffix.lower() in ('.ttc', '.otc') else None
    fonts = collection.fonts if collection else [TTFont(path, lazy=True)]
    try:
        matches = [(i, None) for i, font in enumerate(fonts)
                   if row['postscript_name'] in {n.toUnicode() for n in font['name'].names if n.nameID == 6}]
        if not matches:
            matches = [(i, [instance.coordinates[a.axisTag] for a in font['fvar'].axes])
                       for i, font in enumerate(fonts) if 'fvar' in font
                       for instance in font['fvar'].instances
                       if font['name'].getDebugName(instance.postscriptNameID) == row['postscript_name']]
        if len(matches) != 1:
            raise ValueError('Source PostScript face mismatch: ' + row['font_id'])
        index, axes = matches[0]
        cmap = fonts[index].getBestCmap()
        missing = [c for c in characters if not cmap.get(ord(c)) or cmap[ord(c)] == '.notdef']
        if missing:
            raise ValueError(f"Source font {row['font_id']} lacks {''.join(missing)}")
    finally:
        for font in fonts:
            font.close()
        if collection:
            collection.close()
    return index, axes


def extend(directory, source_manifest, characters_path):
    directory = Path(directory)
    bank = CompactFontBank(directory, 1)
    meta = bank.meta
    bank.archive.close()
    characters = sorted(set(Path(characters_path).read_text().strip()) - set(meta['characters']))
    if not characters:
        return {'added': 0, 'total': len(meta['characters'])}
    if any(not '\u4e00' <= c <= '\u9fff' for c in characters):
        raise ValueError('Extra characters must be Han characters')
    original_rows = json.loads(Path(source_manifest).read_text())
    rows = {row['font_id']: row for row in original_rows}
    if len(rows) != len(original_rows):
        raise ValueError('Duplicate source font IDs')
    sources = []
    for font_id in bank.font_ids:
        row = rows[font_id]
        if row['family_label'] != bank.face_family[font_id]:
            raise ValueError('Source family mismatch: ' + font_id)
        index, axes = resolve_source(row, characters)
        sources.append({**{key: row[key] for key in ('font_id', 'postscript_name', 'source_sha256')},
                        'face_index': index, 'variation_coordinates': axes})
    archive = directory / meta['archive']
    previous_sha = file_sha(archive)
    metadata_path = directory / 'metadata.json'
    previous_metadata = metadata_path.read_bytes()
    # Build and validate a complete bank before replacing either active file.
    with tempfile.TemporaryDirectory(prefix='.font-extension-', dir=directory.parent) as workspace:
        workspace = Path(workspace)
        rendered, staged = workspace / 'rendered', workspace / 'bank'
        rendered.mkdir()
        staged.mkdir()
        config = rendered / 'request.json'
        config.write_text(json.dumps({'fonts': [rows[id] for id in bank.font_ids], 'characters': characters}))
        subprocess.run(['swift', '-module-cache-path', str(workspace / 'module-cache'),
                        str(ROOT / 'scripts/render_font_extensions.swift'), str(config), str(rendered)], check=True)
        report = json.loads((rendered / 'render-report.json').read_text())
        if ([row['font_id'] for row in report['fonts']] != bank.font_ids or
                any(row['glyph_count'] != len(characters) or row['sizes'] != [20, 28, 44]
                    for row in report['fonts'])):
            raise ValueError('Renderer output provenance mismatch')
        # The renderer verifies full source URLs locally; the portable bundle
        # needs their identities and hashes, not workstation directory paths.
        for source in report['fonts']:
            source['source_basename'] = Path(source.pop('actual_url')).name
            source['source_url_verified'] = True
        preservation = append_rendered(archive, staged / archive.name, rendered, characters, bank.font_ids, meta)
        meta['archive_sha256'] = file_sha(staged / archive.name)
        meta.setdefault('extensions', []).append({
            'characters': ''.join(characters), 'source_manifest_sha256': file_sha(source_manifest),
            'character_list_sha256': file_sha(characters_path), 'sources': sources,
            'previous_archive_sha256': previous_sha, 'preservation': preservation,
            'renderer': 'CoreText exact source URL/PostScript; sizes 20/28/44; unchanged blur32 preprocessing',
            'render_report': report,
            'renderer_sha256': file_sha(ROOT / 'scripts/render_font_extensions.swift'),
            'builder_sha256': file_sha(Path(__file__)),
            'selection': 'Previously uncovered characters from the 100-image functional corpus; not font ground truth',
        })
        (staged / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
        shutil.copyfile(directory / 'GATES.json', staged / 'GATES.json')
        checked = CompactFontBank(staged, 1)
        checked.archive.close()
        # Keep a local rollback copy until both replacements have succeeded.
        backup = workspace / 'previous-archive.zip'
        shutil.copyfile(archive, backup)
        try:
            (staged / archive.name).replace(archive)
            (staged / 'metadata.json').replace(metadata_path)
        except BaseException:
            backup.replace(archive)
            metadata_path.write_bytes(previous_metadata)
            raise
    return {'added': len(characters), 'total': len(meta['characters'])}


def append_rendered(archive, temporary, rendered, characters, font_ids, meta):
    # Append preserves the compressed data for old members as well as their NPY
    # contents. Verify every old member again after adding the new references.
    shutil.copyfile(archive, temporary)
    with zipfile.ZipFile(archive) as source:
        previous_hashes = {name: hashlib.sha256(source.read(name)).hexdigest() for name in source.namelist()}
    with zipfile.ZipFile(temporary, 'a', zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for count, character in enumerate(characters, 1):
            packed = np.empty((len(font_ids), 3, 32, 32), dtype=np.uint8)
            for i, font_id in enumerate(font_ids):
                for j, size in enumerate((20, 28, 44)):
                    with Image.open(Path(rendered) / f'{font_id}_{size}_{ord(character)}.png') as image:
                        reference = raster(image)
                    if reference is None:
                        raise ValueError(f'Invalid rendered reference: {character} / {font_id}')
                    packed[i, j] = reference
            buffer = io.BytesIO()
            np.save(buffer, packed, allow_pickle=False)
            member = f'cjk_{ord(character):04X}.npy'
            if member in output.namelist():
                raise ValueError('Refusing to overwrite existing character: ' + character)
            output.writestr(member, buffer.getvalue())
            meta['characters'][character] = member
            if count % 25 == 0:
                print(f'Packed {count}/{len(characters)} additional characters', flush=True)
    with zipfile.ZipFile(temporary) as output:
        if any(hashlib.sha256(output.read(name)).hexdigest() != expected for name, expected in previous_hashes.items()):
            raise ValueError('Existing reference bytes changed')
    return {'existing_members_verified_unchanged': len(previous_hashes),
            'existing_member_hashes_sha256': hashlib.sha256(
                json.dumps(previous_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--characters', type=Path, default=ROOT / 'assets/mobile-extra-characters.txt')
    parser.add_argument('--directory', type=Path, default=ROOT / 'models/font')
    args = parser.parse_args()
    print(json.dumps(extend(args.directory, args.source_manifest, args.characters)))
