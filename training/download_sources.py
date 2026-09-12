"""Download only pinned OFL Roboto sources used by training/data.py.

No AdobeVFR or other research-only data is downloaded. Uses the standard library;
does not import OCR, PyTorch or execute downloaded content.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = '1c627bfa375fc51cf86fabeca4f6e08a95f0aa5c'
SOURCE_MANIFEST = ROOT / 'docs/sources/roboto/SOURCE_MANIFEST.json'
DEFAULT_OUTPUT = ROOT / 'artifacts/neural-font-v1/downloads/roboto-google-fonts'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(output=DEFAULT_OUTPUT, *, verify_only=False):
    output = Path(output).resolve()
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    report = {'schema': 'pinned-public-font-retrieval-v1', 'family': 'Roboto',
              'source_manifest_sha256': sha(SOURCE_MANIFEST), 'source_commit': PINNED_COMMIT,
              'license': 'SIL Open Font License 1.1', 'files': []}
    for item in manifest['files']:
        filename = item['file']
        if not isinstance(filename, str) or filename != Path(filename).name or filename in ('.', '..'):
            raise ValueError('Unsafe font source filename')
        expected_url = ('https://raw.githubusercontent.com/google/fonts/' + PINNED_COMMIT
                        + '/ofl/roboto/' + urllib.parse.quote(filename, safe=''))
        if item['url'] != expected_url or item['source_commit'] != PINNED_COMMIT:
            raise ValueError('Source URL must belong to the frozen Google Fonts commit')
        if type(item['bytes']) is not int or not 0 < item['bytes'] <= 2 * 1024 * 1024:
            raise ValueError('Source file exceeds bounded download size')
        path = output / filename
        reused = path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        if not reused:
            if verify_only:
                raise ValueError('Missing or modified source: ' + str(path))
            output.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with urllib.request.urlopen(expected_url, timeout=30) as response:
                    body = response.read(item['bytes'] + 1)
                if len(body) != item['bytes'] or hashlib.sha256(body).hexdigest() != item['sha256']:
                    raise ValueError('Downloaded source size/SHA-256 mismatch: ' + filename)
                with tempfile.NamedTemporaryFile(dir=output, prefix='.source-', delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(body)
                temporary.replace(path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        report['files'].append({'file': filename, 'path': str(path), 'url': expected_url,
                                'sha256': item['sha256'], 'bytes': item['bytes'],
                                'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'reused_verified_local_file': reused})
    if not verify_only:
        (output / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-only', action='store_true', help='Check existing files without network access or writes')
    args = parser.parse_args()
    report = download(args.output, verify_only=args.verify_only)
    print(json.dumps({'output': str(args.output.resolve()), 'verified_files': len(report['files']),
                      'source_commit': PINNED_COMMIT, 'verify_only': args.verify_only}))


if __name__ == '__main__':
    main()
