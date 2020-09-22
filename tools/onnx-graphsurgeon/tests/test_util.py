from onnx_graphsurgeon.util import misc


def test_combine_dicts_second_overwrites_first():
    x = {"a": 1}
    y = {"a": 2}
    z = misc.combine_dicts(x, y)
    assert z["a"] == 2
