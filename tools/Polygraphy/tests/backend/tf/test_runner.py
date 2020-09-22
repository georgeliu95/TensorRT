from polygraphy.backend.tf import TfRunner, SessionFromGraph

from tests.models.meta import TF_MODELS
from tests.common import check_file_non_empty

import tempfile
import pytest
import os


class TestTfRunner(object):
    def test_can_name_runner(self):
        NAME = "runner"
        runner = TfRunner(None, name=NAME)
        assert runner.name == NAME


    def test_basic(self):
        model = TF_MODELS["identity"]
        with TfRunner(SessionFromGraph(model.loader)) as runner:
            assert runner.is_active
            model.check_runner(runner)
        assert not runner.is_active


    @pytest.mark.skip(reason="Non-trivial to set up - requires CUPTI")
    def test_save_timeline(self):
        model = TF_MODELS["identity"]
        with tempfile.NamedTemporaryFile() as outpath:
            with TfRunner(SessionFromGraph(model.loader), allow_growth=True, save_timeline=outpath.name) as runner:
                model.check_runner(runner)
                check_file_non_empty(outpath.name)
