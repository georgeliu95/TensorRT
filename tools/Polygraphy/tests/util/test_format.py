from polygraphy.logger import G_LOGGER

from polygraphy.util.format import FormatManager, DataFormat

import pytest


class FormatTestCase:
    def __init__(self, shape, format):
        self.shape = shape
        self.format = format

EXPECTED_FORMATS = [
    FormatTestCase((1, 3, 480, 960), DataFormat.NCHW),
    FormatTestCase((1, 3, 224, 224), DataFormat.NCHW),
    FormatTestCase((1, 224, 224, 3), DataFormat.NHWC),
    FormatTestCase((1, 9, 9, 3), DataFormat.NHWC),
]

@pytest.mark.parametrize("test_case", EXPECTED_FORMATS)
def test_format_deduction(test_case):
    assert test_case.format == FormatManager.determine_format(test_case.shape)
