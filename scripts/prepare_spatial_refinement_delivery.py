#!/usr/bin/env python3
"""Finalize a validated spatial-refinement preview's development-state metadata."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import prepare_spatial_refinement_release as upstream
from flux_glyph.unified_font import unified_metadata

SCHEMA = 'flux-glyph-spatial-refinement-delivery-evidence-v1'
UPSTREAM_SOURCE = ROOT / 'scripts/prepare_spatial_refinement_release.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return upstream.sha(path)


def encoded(value):
    return upstream.encoded(value)


def _verify_bindings(bindings, label):
    require(isinstance(bindings, dict) and bindings, label + ' bindings are missing')
    require(all(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64
                and Path(path).is_file() and sha(path) == digest
                for path, digest in bindings.items()), label + ' binding changed')
    return bindings


def _staging_contract(staging, result, version):
    expected = {staging / name for name in ('model.onnx', 'metadata.json', 'RELEASE_EVIDENCE.json')}
    require(set(staging.iterdir()) == expected, 'Upstream staging contains unexpected files')
    evidence_path, model_path, metadata_path = (staging / 'RELEASE_EVIDENCE.json',
                                                 staging / 'model.onnx', staging / 'metadata.json')
    evidence, metadata = read(evidence_path), read(metadata_path)
    require(evidence == result and evidence.get('schema') == upstream.EVIDENCE_SCHEMA
            and evidence.get('version') == version
            and evidence.get('promotion_allowed') is True
            and evidence.get('release_tier') == 'preview'
            and evidence.get('stable_validation_passed') is False
            and evidence.get('test_passed') is False and evidence.get('test_read') is False
            and evidence.get('blind_test_performed') is False,
            'Upstream release evidence differs from the validated preview')
    expected_outputs = {str(model_path): sha(model_path), str(metadata_path): sha(metadata_path)}
    require(evidence.get('output_bindings') == expected_outputs,
            'Upstream staging model or metadata binding differs')
    _verify_bindings(evidence.get('input_bindings'), 'Upstream input')
    validation = metadata.get('validation', {})
    require(metadata.get('schema') == upstream.METADATA_SCHEMA
            and metadata.get('version') == version
            and metadata.get('release_tier') == 'experimental'
            and metadata.get('stable_validation_passed') is False
            and metadata.get('test_passed') is False
            and metadata.get('pre_development') is True
            and validation.get('promotion_allowed') is True
            and validation.get('development_holdout_evaluated') is True
            and validation.get('development_checks') == upstream.DEV_CHECKS
            and validation.get('stable_validation_passed') is False
            and validation.get('test_passed') is False and validation.get('test_read') is False
            and validation.get('blind_test_performed') is False,
            'Upstream metadata is not the exact validated post-DEV preview boundary')
    require(metadata.get('model') == {'path': 'model.onnx', 'sha256': sha(model_path)}
            and isinstance(metadata.get('families'), list) and len(metadata['families']) == 25
            and metadata['families'][-1] == '__unknown__'
            and model_path.stat().st_size <= 64 * 1024 * 1024
            and metadata_path.stat().st_size <= 256 * 1024,
            'Upstream staging is not one bounded 25-output unified CNN')
    unified_metadata(metadata)
    return evidence_path, model_path, metadata_path, evidence, metadata


def prepare_delivery(run, region, development_report, staging, output,
                     development_freeze=None, version=upstream.VERSION):
    """Run the frozen release validator, then correct its one stale lifecycle flag."""
    run, region, development_report, staging, output = map(
        lambda value: Path(value).resolve(),
        (run, region, development_report, staging, output))
    development_freeze = (None if development_freeze is None
                          else Path(development_freeze).resolve())
    require(staging != output and not staging.is_relative_to(output)
            and not output.is_relative_to(staging),
            'Staging and final region must be independent directories')
    require(not staging.exists(), 'Choose a new staging directory')
    require(not output.exists(), 'Choose a new final region directory')

    result = upstream.prepare_release(run, region, development_report, staging,
                                      development_freeze, version)
    evidence_path, model_path, metadata_path, old_evidence, old_metadata = \
        _staging_contract(staging, result, version)
    staging_bindings = {str(path): sha(path) for path in
                        (evidence_path, model_path, metadata_path)}

    final_metadata = copy.deepcopy(old_metadata)
    final_metadata['pre_development'] = False
    restored = copy.deepcopy(final_metadata); restored['pre_development'] = True
    require(restored == old_metadata
            and final_metadata['validation']['development_holdout_evaluated'] is True,
            'Delivery metadata may change only the post-DEV lifecycle flag')
    unified_metadata(final_metadata)
    metadata_bytes = encoded(final_metadata)

    source_bindings = {str(UPSTREAM_SOURCE.resolve()): sha(UPSTREAM_SOURCE),
                       str(Path(__file__).resolve()): sha(Path(__file__))}
    _verify_bindings(source_bindings, 'Delivery source')
    output.parent.mkdir(parents=True, exist_ok=True)
    final_model = output / 'model.onnx'; final_metadata_path = output / 'metadata.json'
    report = {'schema': SCHEMA, 'version': version, 'release_tier': 'preview',
        'promotion_allowed': True, 'calibration_promotion_allowed': True,
        'development_holdout_evaluated': True,
        'stable_validation_passed': False, 'test_passed': False, 'test_read': False,
        'blind_test_performed': False, 'model_count': 1, 'encoder_count': 1,
        'platform_routing': False, 'score_merging': False,
        'upstream_evidence': {'schema': upstream.EVIDENCE_SCHEMA,
                              'path': str(evidence_path), 'sha256': staging_bindings[str(evidence_path)]},
        'staging_bindings': staging_bindings,
        'upstream_input_bindings': copy.deepcopy(old_evidence['input_bindings']),
        'source_bindings': source_bindings,
        'metadata_change': {'field': 'pre_development', 'before': True, 'after': False,
                            'only_metadata_change': True,
                            'reason': 'The frozen dual-DEV release validation completed successfully.'},
        'model_bytes_unchanged': True,
        'output_bindings': {str(final_model): sha(model_path),
                            str(final_metadata_path): hashlib.sha256(metadata_bytes).hexdigest()}}

    temporary = Path(tempfile.mkdtemp(prefix='.spatial-refinement-delivery-', dir=output.parent))
    try:
        shutil.copyfile(model_path, temporary / 'model.onnx')
        (temporary / 'metadata.json').write_bytes(metadata_bytes)
        (temporary / 'RELEASE_EVIDENCE.json').write_bytes(encoded(report))
        require(sha(temporary / 'model.onnx') == sha(model_path)
                and sha(temporary / 'metadata.json') == report['output_bindings'][str(final_metadata_path)]
                and all(sha(path) == digest for path, digest in staging_bindings.items())
                and all(sha(path) == digest for path, digest in source_bindings.items())
                and all(sha(path) == digest for path, digest in old_evidence['input_bindings'].items()),
                'Model, metadata, staging, source, or upstream evidence changed during finalization')
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--region', type=Path, required=True)
    parser.add_argument('--development-report', type=Path, required=True)
    parser.add_argument('--development-freeze', type=Path)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default=upstream.VERSION)
    args = parser.parse_args()
    evidence = prepare_delivery(args.run, args.region, args.development_report,
        args.staging, args.output, args.development_freeze, args.version)
    print(json.dumps({'directory': str(args.output.resolve()), 'staging': str(args.staging.resolve()),
                      'version': evidence['version'],
                      'promotion_allowed': evidence['promotion_allowed']}, indent=2))
