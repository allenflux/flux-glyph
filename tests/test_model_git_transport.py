"""Git checkout must preserve the exact bytes covered by the shipped manifest."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from flux_glyph.models import verify_bundle


@pytest.mark.skipif(shutil.which('git') is None,reason='Git is needed to verify deployment transport')
def test_model_bundle_survives_git_eol_conversion(tmp_path):
    project=Path(__file__).resolve().parents[1]
    staging=tmp_path/'source';staging.mkdir()
    shutil.copytree(project/'models',staging/'models')
    shutil.copy2(project/'.gitattributes',staging/'.gitattributes')
    env={**os.environ,'GIT_CONFIG_GLOBAL':os.devnull,'GIT_CONFIG_NOSYSTEM':'1'}
    def git(*args):
        return subprocess.run(['git','-C',str(staging),*args],env=env,check=True,capture_output=True,text=True)
    git('init','--quiet')
    git('config','core.autocrlf','input')
    git('add','.')
    for autocrlf in ('input','true'):
        checkout=tmp_path/f'checkout-{autocrlf}';checkout.mkdir()
        git('-c',f'core.autocrlf={autocrlf}','checkout-index','--all',f'--prefix={checkout}{os.sep}')
        expected=verify_bundle(project/'models')
        assert verify_bundle(checkout/'models')==expected
        for item in expected['files']:
            assert (checkout/'models'/item['path']).read_bytes()==(project/'models'/item['path']).read_bytes()
