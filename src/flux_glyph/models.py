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
        actual_size=path.stat().st_size
        actual_sha=file_sha(path)
        if actual_size!=row['bytes'] or actual_sha!=row['sha256']:
            raise ValueError(
                f'Model file checksum mismatch: {relative}; bundle={root}; '
                f'expected bytes={row["bytes"]}, sha256={row["sha256"]}; '
                f'actual bytes={actual_size}, sha256={actual_sha}. '
                'Restore this file from the matching original model bundle. '
                'Preserve original bytes and line endings; do not regenerate MANIFEST.json to bypass validation.'
            )
    if any(path.startswith('region_neural/') for path in paths):
        required={'region_neural/metadata.json','pp/onnx/paddle_ocr_det.onnx','pp/paddle_ocr_delivery.contract.json'}
        if not required.issubset(paths):raise ValueError('Region bundle lacks required files')
        metadata=json.loads((root/'region_neural/metadata.json').read_text())
        model=metadata.get('model',{})
        name=model.get('path')
        if (not isinstance(name,str) or PurePosixPath(name).name!=name or '\\' in name
                or not name.endswith('.onnx')):raise ValueError('Invalid region model path')
        required.add('region_neural/'+name)
        if metadata.get('algorithm')=='region-cnn64x256-rejection-v2' or 'rejection' in metadata:
            from .region_font import rejection_metadata
            rejection=rejection_metadata(metadata)
            if rejection is not None:
                required.add('region_neural/'+rejection['model']['path'])
                entries={row['path']:row for row in rows}
                for declared in (model,rejection['model']):
                    entry=entries.get('region_neural/'+declared['path'])
                    if entry is None:raise ValueError('Region bundle lacks required rejection files')
                    if entry['sha256']!=declared.get('sha256'):
                        raise ValueError('Region bundle model metadata SHA differs')
        if not required.issubset(paths):raise ValueError('Region bundle lacks required files')
        return manifest
    required={'font/metadata.json','font/GATES.json','pp/onnx/paddle_ocr_det.onnx','pp/onnx/paddle_ocr_rec.onnx','pp/onnx/paddle_ocr_cls.onnx','pp/charset/ppocr_keys_v1.txt','pp/paddle_ocr_delivery.contract.json'}
    meta=json.loads((root/'font/metadata.json').read_text());archive=meta.get('archive') if isinstance(meta,dict) else None
    if not isinstance(archive,str) or PurePosixPath(archive).name!=archive:raise ValueError('Invalid font archive path')
    required.add('font/'+archive)
    if any(path.startswith('latin/') for path in paths):
        if 'latin/metadata.json' not in paths:raise ValueError('Model bundle lacks Latin metadata')
        latin=json.loads((root/'latin/metadata.json').read_text())
        latin_archive=latin.get('archive') if isinstance(latin,dict) else None
        gates=latin.get('gates',{}) if isinstance(latin,dict) else {}
        gate_path=gates.get('path') if isinstance(gates,dict) else None
        for name in (latin_archive,gate_path):
            if not isinstance(name,str) or not name or PurePosixPath(name).name!=name or '\\' in name:
                raise ValueError('Invalid Latin asset path')
            required.add('latin/'+name)
    if any(path.startswith('neural/') for path in paths):
        if 'neural/metadata.json' not in paths:raise ValueError('Model bundle lacks neural metadata')
        neural=json.loads((root/'neural/metadata.json').read_text())
        model=neural.get('model') if isinstance(neural,dict) else None
        name=model.get('path') if isinstance(model,dict) else None
        if (not isinstance(name,str) or not name or PurePosixPath(name).name!=name or
                name in ('.','..') or '\\' in name or not name.endswith('.onnx')):
            raise ValueError('Invalid neural model path')
        required.add('neural/'+name)
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
