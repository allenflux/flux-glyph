"""Versioned offline model bundles; the serving process keeps one version until restart."""
from pathlib import Path,PurePosixPath
import hashlib
import json


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def verify_bundle(directory):
    root=Path(directory).resolve();manifest=json.loads((root/'MANIFEST.json').read_text())
    rows=manifest.get('files') if isinstance(manifest,dict) else None
    if not isinstance(rows,list) or not rows or len(rows)>10000:
        raise ValueError('Invalid model manifest')
    paths=set()
    for row in rows:
        relative=row.get('path') if isinstance(row,dict) else None
        canonical=PurePosixPath(relative) if isinstance(relative,str) else None
        valid_size=isinstance(row.get('bytes'),int) and not isinstance(row.get('bytes'),bool) and row['bytes']>=0 if isinstance(row,dict) else False
        valid_sha=isinstance(row.get('sha256'),str) and len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256']) if isinstance(row,dict) else False
        if (canonical is None or canonical.is_absolute() or not relative or '\\' in relative or '..' in canonical.parts or
                canonical.as_posix()!=relative or relative in paths or not valid_size or not valid_sha):
            raise ValueError('Invalid model manifest entry')
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Invalid model manifest path')
        paths.add(relative)
        if path.stat().st_size!=row['bytes'] or file_sha(path)!=row['sha256']:raise ValueError('Model file checksum mismatch: '+relative)
    required={'font/metadata.json','font/GATES.json','pp/onnx/paddle_ocr_det.onnx','pp/onnx/paddle_ocr_rec.onnx','pp/onnx/paddle_ocr_cls.onnx','pp/charset/ppocr_keys_v1.txt','pp/paddle_ocr_delivery.contract.json'}
    meta=json.loads((root/'font/metadata.json').read_text());archive=meta.get('archive') if isinstance(meta,dict) else None
    if not isinstance(archive,str) or PurePosixPath(archive).name!=archive:raise ValueError('Invalid font archive path')
    required.add('font/'+archive)
    if not required.issubset(paths):raise ValueError('Model bundle lacks required files')
    return manifest


def load_active(model_root):
    root=Path(model_root).resolve();active=root/'ACTIVE.json'
    relative=json.loads(active.read_text())['path'] if active.is_file() else '.'
    directory=(root/relative).resolve()
    if not directory.is_relative_to(root):raise ValueError('Active model escapes model root')
    manifest=verify_bundle(directory)
    version=manifest.get('version') or 'r13-pp-ink-v1-'+file_sha(directory/'font/metadata.json')[:8]
    return directory,version,manifest
