"""Guard the workflow inheritance and dependency direction after migration."""

import ast
from pathlib import Path

import pytest

from nipoppy.workflows.base import BaseDatasetWorkflow, BaseWorkflow
from nipoppy.workflows.bids_conversion import BIDSificationRunner
from nipoppy.workflows.execution import PipelineExecutionWorkflow
from nipoppy.workflows.extractor import ExtractionRunner
from nipoppy.workflows.processing_runner import ProcessingRunner
from nipoppy.workflows.tracker import PipelineTracker


@pytest.mark.parametrize(
    "runner", [ProcessingRunner, BIDSificationRunner, ExtractionRunner]
)
def test_execution_inheritance_is_flat(runner):
    assert runner.__bases__ == (PipelineExecutionWorkflow,)
    assert PipelineExecutionWorkflow.__bases__ == (BaseWorkflow,)
    assert BaseDatasetWorkflow not in runner.__mro__
    assert [cls for cls in runner.__mro__ if issubclass(cls, BaseWorkflow)] == [
        runner,
        PipelineExecutionWorkflow,
        BaseWorkflow,
    ]


def test_old_runner_layer_is_removed():
    workflows = Path(__file__).resolve().parents[3] / "nipoppy" / "workflows"
    assert not (workflows / "runner.py").exists()


def test_tracker_inherits_dataset_base_directly():
    assert PipelineTracker.__bases__ == (BaseDatasetWorkflow,)
    assert [
        cls for cls in PipelineTracker.__mro__ if issubclass(cls, BaseWorkflow)
    ] == [
        PipelineTracker,
        BaseDatasetWorkflow,
        BaseWorkflow,
    ]


def test_old_pipeline_layer_is_removed():
    workflows = Path(__file__).resolve().parents[3] / "nipoppy" / "workflows"
    assert not (workflows / "pipeline.py").exists()


def test_shared_modules_do_not_import_concrete_workflows():
    workflows = Path(__file__).resolve().parents[3] / "nipoppy" / "workflows"
    forbidden = {
        "nipoppy.workflows.processing_runner",
        "nipoppy.workflows.bids_conversion",
        "nipoppy.workflows.extractor",
        "nipoppy.workflows.runner",
        "nipoppy.workflows.pipeline",
        "nipoppy.workflows.execution",
    }
    for path in [
        workflows / "execution.py",
        *sorted((workflows / "services").glob("*.py")),
    ]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden, f"{path}: {node.module}"
            if isinstance(node, ast.FunctionDef):
                assert node.name != "__getattr__", f"Generic forwarding in {path}"
