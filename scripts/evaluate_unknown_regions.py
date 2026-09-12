#!/usr/bin/env python3
"""Frozen-model detector-to-font evaluation on held-out native unknown scenes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'training')]
from flux_glyph.models import verify_bundle
from flux_glyph.pipeline import FontPipeline
from training.capture.prepare_unknown_regions import load_partition
from training.train_regions import dump, require, sha
from scripts.evaluate_region_fonts import prohibit_ocr, pair_regions, validate_prediction


def evaluate(args):
    require(not args.output.exists(), 'use a new output; never replace a fixed test report')
    frozen = json.loads(args.frozen.read_text())
    require(frozen['test_read_for_selection'] is False and frozen['selected_region_metadata_sha256'] ==
            sha(args.models / 'region_neural/metadata.json'), 'bundle differs from the selected frozen model')
    data = load_partition(args.data, 'test', allow_test=True)
    require(sha(args.data / 'MANIFEST.json') == frozen['data_manifest_sha256'], 'test data differs from model selection')
    verify_bundle(args.models)
    args.output.mkdir(parents=True)
    lookup = {(row['source_id'], row['region_id']): row for row in data['rows']}
    records = [json.loads(line) for line in Path(data['manifest']['input_labels']['path']).read_text().splitlines()
               if json.loads(line)['split'] == 'test']
    rows, extra, pages = [], [], []
    with prohibit_ocr() as guard:
        pipeline = FontPipeline(args.models)
        for page_index, record in enumerate(records):
            image = Path(record['image'])
            require(sha(image) == record['source_sha256'], 'test source changed')
            output = args.output / record['page_id']
            result = pipeline.run(image, output, record['page_id'])
            validate_prediction(result)
            truths = record['regions']
            pairs = {i: (j, iou) for i, j, iou in pair_regions(truths, result['regions'])}
            used = {j for j, _ in pairs.values()}
            for index, truth in enumerate(truths):
                row = lookup[(record['source_id'], truth['id'])]
                matched = index in pairs
                prediction_index, iou = pairs.get(index, (None, None))
                font = result['regions'][prediction_index]['font'] if matched else {}
                accepted = font.get('status') in ('candidate', 'supported') and bool(font.get('family'))
                named = bool(font.get('candidates')) or bool(font.get('family'))
                rows.append({'source_id': row['source_id'], 'region_id': row['region_id'], 'family': row['family'],
                             'known_truth': row['label'] == 1, 'detected': matched, 'iou': iou,
                             'rejected': matched and font.get('reason_code') == 'unknown_font_rejected',
                             'family_candidate_displayed': bool(named), 'font_accepted': bool(accepted),
                             'accepted_correct': bool(accepted and font['family'] == row['family']),
                             'predicted_family': font.get('family'), 'font_status': font.get('status'),
                             'known_score': font.get('rejection', {}).get('known_score')})
            extra.extend({'source_id': record['source_id'], 'prediction_id': region['id']} for i, region in enumerate(result['regions']) if i not in used)
            pages.append({'source_id': record['source_id'], 'source_sha256': record['source_sha256'],
                          'result': str(output / 'result.json'), 'result_sha256': sha(output / 'result.json')})
            if (page_index + 1) % 10 == 0:
                print(json.dumps({'pages': page_index + 1, 'total': len(records)}), flush=True)
        guard_result = dict(guard)
    def summarize(items):
        known = [row for row in items if row['known_truth']]
        unknown = [row for row in items if not row['known_truth']]
        return {'regions': len(items), 'detected': sum(row['detected'] for row in items),
                'known': len(known), 'known_rejected': sum(row['rejected'] for row in known),
                'known_accepted_correct': sum(row['accepted_correct'] for row in known),
                'known_accepted_wrong': sum(row['font_accepted'] and not row['accepted_correct'] for row in known),
                'unknown': len(unknown), 'unknown_rejected': sum(row['rejected'] for row in unknown),
                'unknown_named_candidates': sum(row['family_candidate_displayed'] for row in unknown),
                'unknown_accepted_as_known': sum(row['font_accepted'] for row in unknown)}
    report = {'schema': 'flux-glyph-unknown-font-e2e-test-v1', 'metrics': summarize(rows), 'false_positives': len(extra),
              'by_family': {family: summarize([r for r in rows if r['family'] == family]) for family in sorted({r['family'] for r in rows})},
              'rows': rows, 'pages': pages, 'extras': extra, 'ocr_guard': guard_result,
              'frozen_selection_sha256': sha(args.frozen), 'data_manifest_sha256': sha(args.data / 'MANIFEST.json'),
              'model_manifest_sha256': sha(args.models / 'MANIFEST.json'), 'source_sha256': sha(__file__),
              'scope': 'Whole screenshots, geometry pairing only; held-out unknown font families in native controlled iOS scenes. Not Android-device validation.'}
    dump(args.output / 'report.json', report)
    print(json.dumps({'metrics': report['metrics'], 'false_positives': report['false_positives'], 'by_family': report['by_family']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models', 'data', 'frozen', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    evaluate(parser.parse_args())
