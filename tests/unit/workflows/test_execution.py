"""Tests for pipeline assembly and the new shallow execution base."""

import pytest

from nipoppy.env import PipelineTypeEnum
from nipoppy.exceptions import WorkflowError
from nipoppy.workflows.execution import PipelineExecutionWorkflow
from nipoppy.workflows.factories import create_pipeline_dependencies
from nipoppy.workflows.services.cohort import SelectionOptions
from nipoppy.workflows.services.models import ExecutionOptions, PipelineRequest


def test_dependencies_are_lazy(tmp_path):
    dependencies = create_pipeline_dependencies(
        tmp_path / "missing-study",
        "process",
        PipelineRequest(PipelineTypeEnum.PROCESSING, "missing-pipeline"),
        SelectionOptions("sub-01", "ses-A"),
        ExecutionOptions(),
    )
    assert dependencies["selection"].participant_id == "01"
    assert dependencies["selection"].session_id == "A"
    assert dependencies["pipeline"].version is None
    assert not (tmp_path / "missing-study").exists()


def test_execution_modes_are_mutually_exclusive():
    with pytest.raises(WorkflowError, match="mutually exclusive"):
        ExecutionOptions(hpc="slurm", write_subcohort="cohort.tsv")


def test_pipeline_execution_base_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        PipelineExecutionWorkflow(None, None, None, None, None, None, None)
