from onnx_graphsurgeon.logger.logger import G_LOGGER
import onnx_graphsurgeon as gs

import subprocess as sp
import numpy as np
import onnxruntime
import tempfile
import pytest
import onnx
import sys
import os


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))
EXAMPLES_ROOT = os.path.join(ROOT_DIR, "examples")

EXAMPLES = [
    ("01_creating_a_model", ["test_globallppool.onnx"]),
    ("02_creating_a_model_with_initializer", ["test_conv.onnx"]),
    ("03_isolating_a_subgraph", ["model.onnx", "subgraph.onnx"]),
    ("04_modifying_a_model", ["model.onnx", "modified.onnx"]),
    ("05_folding_constants", ["model.onnx", "folded.onnx"]),
]

# Extract any ``` blocks from the README
def load_commands_from_readme(readme):
    def ignore_command(cmd):
        return "pip" in cmd

    commands = []
    with open(readme, 'r') as f:
        in_command_block = False
        for line in f.readlines():
            if not in_command_block and "```" in line:
                in_command_block = True
            elif in_command_block:
                if "```" in line:
                    in_command_block = False
                elif not ignore_command(line):
                    commands.append(line.strip())
    return commands


def infer_model(path):
    model = onnx.load(path)
    graph = gs.import_onnx(model)

    feed_dict = {}
    for tensor in graph.inputs:
        feed_dict[tensor.name] = np.random.random_sample(size=tensor.shape).astype(tensor.dtype)

    output_names = [out.name for out in graph.outputs]

    sess = onnxruntime.InferenceSession(model.SerializeToString())
    outputs = sess.run(output_names, feed_dict)
    G_LOGGER.info("Inference outputs: {:}".format(outputs))
    return outputs


@pytest.mark.parametrize("example_dir,artifacts", EXAMPLES)
def test_examples(example_dir, artifacts):
    example_dir = os.path.join(EXAMPLES_ROOT, example_dir)
    readme = os.path.join(example_dir, "README.md")
    commands = load_commands_from_readme(readme)
    for command in commands:
        G_LOGGER.info(command)
        assert sp.run(["bash", "-c", command], cwd=example_dir, env={"PYTHONPATH": ROOT_DIR}).returncode == 0

    for artifact in artifacts:
        artifact_path = os.path.join(example_dir, artifact)
        assert os.path.exists(artifact_path)
        assert infer_model(artifact_path)
        os.remove(artifact_path)
