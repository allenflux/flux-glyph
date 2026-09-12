#!/usr/bin/env python3
"""Build a separate, verified neural model bundle without changing ACTIVE.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flux_glyph.models import load_active, verify_bundle
from flux_glyph.neural_font import NeuralFontClassifier
from package_models import write_models_manifest
from model_release import safe_version, validate_runtime


def package(base: Path, neural: Path, output: Path, version: str, size_metrics: Path | None = None) -> dict:
    version = safe_version(version)
    selected, base_version, manifest = load_active(base)
    classifier = NeuralFontClassifier(neural)
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing bundles are not overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.neural-package-', dir=output.parent))
    try:
        for row in manifest['files']:
            if row['path'].startswith('neural/'):
                continue
            target = temporary / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(selected / row['path'], target)
        (temporary / 'neural').mkdir()
        for name in ('metadata.json', classifier.meta['model']['path']):
            shutil.copyfile(neural / name, temporary / 'neural' / name)
        if size_metrics is not None:
            from flux_glyph.text_style import SizeMetrics
            SizeMetrics.load(size_metrics)
            (temporary / 'style').mkdir(exist_ok=True)
            shutil.copyfile(size_metrics, temporary / 'style/size_metrics.json')
        packed = write_models_manifest(temporary)
        packed.update(version=version, base_version=base_version,
                      font_method='neural_network',
                      source_note='Trained 64px font CNN; PP OCR retained. Reference assets retained for bundle compatibility; neural inference does not call reference matching.')
        (temporary / 'MANIFEST.json').write_text(json.dumps(packed, ensure_ascii=False, indent=2) + '\n')
        verify_bundle(temporary)
        validate_runtime(temporary)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'directory': str(output), 'version': version, 'base_version': base_version,
            'font_method': 'neural_network', 'activated': False,
            'files': len(packed['files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'models')
    parser.add_argument('--neural', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default='r15-neural-font-v1')
    parser.add_argument('--size-metrics', type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.base, args.neural, args.output, args.version, args.size_metrics), indent=2))


if __name__ == '__main__':
    main()
