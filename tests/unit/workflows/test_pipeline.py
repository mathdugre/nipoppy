"""Tests for PipelineExecutionWorkflow."""

import builtins
import importlib
import json
import re
from collections.abc import Generator
from contextlib import nullcontext
from dataclasses import replace
from functools import cached_property
from pathlib import Path

import pandas as pd
import pytest
import pytest_mock
from fids import fids

from nipoppy.config.boutiques import BoutiquesConfig
from nipoppy.config.pipeline import (
    BIDSificationPipelineConfig,
    ExtractionPipelineConfig,
    ProcessingPipelineConfig,
)
from nipoppy.config.pipeline_step import AnalysisLevelType, ProcPipelineStepConfig
from nipoppy.config.schema import get_current_schema_version
from nipoppy.config.tracker import TrackerConfig
from nipoppy.container import ApptainerHandler
from nipoppy.env import (
    BIDS_SESSION_PREFIX,
    DEFAULT_PIPELINE_STEP_NAME,
    FAKE_SESSION_ID,
    ConfigType,
    ContainerCommandEnum,
    PipelineTypeEnum,
)
from nipoppy.exceptions import (
    ConfigError,
    FileOperationError,
    InvalidArgumentError,
    ReturnCode,
    WorkflowError,
)
from nipoppy.logger import get_logger
from nipoppy.workflows.execution import PipelineExecutionWorkflow
from nipoppy.workflows.factories import create_pipeline_dependencies
from nipoppy.workflows.services.artifacts import get_pipeline_version
from nipoppy.workflows.services.bids import build_bids_db
from nipoppy.workflows.services.cohort import (
    SelectionOptions,
    WorkItem,
    apply_analysis_level,
    check_filter_args,
)
from nipoppy.workflows.services.execution import iter_results, log_summary, run_item
from nipoppy.workflows.services.models import ExecutionOptions, PipelineRequest
from tests.conftest import (
    create_empty_dataset,
    create_pipeline_config_files,
    datetime_fixture,  # noqa F401
    get_config,
    prepare_dataset,
)

logger = get_logger()


class PipelineWorkflow(PipelineExecutionWorkflow):
    """Minimal domain hooks for exercising the composed modules."""

    def __init__(
        self,
        dpath_root,
        pipeline_name,
        pipeline_version=None,
        pipeline_step=None,
        participant_id=None,
        session_id=None,
        use_subcohort=None,
        hpc=None,
        write_subcohort=None,
        n_jobs=None,
        fpath_layout=None,
        verbose=False,
        dry_run=False,
        _skip_logfile=False,
        _show_progress=False,
    ):
        request = PipelineRequest(
            PipelineTypeEnum.PROCESSING, pipeline_name, pipeline_version, pipeline_step
        )
        selection = SelectionOptions(participant_id, session_id, use_subcohort)
        execution = ExecutionOptions(
            hpc=hpc,
            write_subcohort=write_subcohort,
            n_jobs=n_jobs or 1,
            fpath_layout=fpath_layout,
            verbose=verbose,
            dry_run=dry_run,
            skip_logfile=_skip_logfile,
            show_progress=_show_progress,
        )
        super().__init__(
            **create_pipeline_dependencies(
                dpath_root, "test", request, selection, execution
            )
        )
        self.directories = []

    @cached_property
    def derivatives(self):
        """Derivatives path for the dummy domain."""
        return self.dataset.study.layout.get_dpath_pipeline(
            self.pipeline.name, self.pipeline.version
        )

    @cached_property
    def output(self):
        """Output path for the dummy domain."""
        return self.dataset.study.layout.get_dpath_pipeline_output(
            self.pipeline.name, self.pipeline.version
        )

    def template_values(self):
        """Expose the two domain paths in the explicit namespace."""
        return {
            "dpath_pipeline": lambda: self.derivatives,
            "dpath_pipeline_output": lambda: self.output,
        }

    def setup_directories(self):
        """Return paths selected by setup tests."""
        return self.directories

    def select_candidates(self):
        """Select BIDSified pairs for the dummy execution hook."""
        return self.dataset.curation_status_table.get_bidsified_participants_sessions(
            participant_id=self.selection.participant_id,
            session_id=self.selection.session_id,
        )

    def execute_item(self, item):
        """Return success or fail explicitly for the error-isolation tests."""
        logger.info(f"Running on {item.participant_id}, {item.session_id}")
        if item.participant_id == "FAIL":
            raise RuntimeError("FAIL")
        return "SUCCESS"


@pytest.fixture(scope="function")
def workflow(tmp_path: Path):
    workflow = PipelineWorkflow(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="my_pipeline",
        pipeline_version="1.0",
        pipeline_step=DEFAULT_PIPELINE_STEP_NAME,
    )

    # write config
    config = get_config()
    config.save(workflow.dataset.study.layout.fpath_config)

    create_empty_dataset(workflow.dataset.study.layout.dpath_root)

    create_pipeline_config_files(
        workflow.dataset.study.layout.dpath_pipelines,
        bidsification_pipelines=[
            {
                "NAME": "bids_converter",
                "VERSION": "1.0",
                "CONTAINER_INFO": {"FILE": "path"},
                "STEPS": [{"NAME": "step1"}, {"NAME": "step2"}],
            },
            {
                "NAME": "bids_converter",
                "VERSION": "0.1",
            },
        ],
        processing_pipelines=[
            {
                "NAME": "fmriprep",
                "VERSION": "23.1.3",
                "STEPS": [{}],
            },
            {
                "NAME": "my_pipeline",
                "VERSION": "1.0",
                "CONTAINER_INFO": {
                    "FILE": "[[NIPOPPY_DPATH_CONTAINERS]]/my_container.sif",
                    "URI": "docker://fake/uri",
                },
                "STEPS": [{}],
            },
            {
                "NAME": "my_pipeline",
                "VERSION": "2.0",
                "STEPS": [{}],
            },
        ],
        extraction_pipelines=[
            {
                "NAME": "extractor1",
                "VERSION": "0.1.0",
                "PROC_DEPENDENCIES": [{"NAME": "pipeline1", "VERSION": "v1"}],
            },
        ],
    )
    return workflow


@pytest.fixture(scope="function")
def reimport_joblib(mocker: pytest_mock.MockerFixture):
    """Fixture for tests where joblib import was manipulated."""
    yield

    # fmt: off
    mocker.stopall()
    import nipoppy.workflows.services.execution
    importlib.reload(nipoppy.workflows.services.execution)
    from nipoppy.workflows.execution import PipelineExecutionWorkflow  # noqa F401
    from nipoppy.workflows.services.execution import (
        JOBLIB_INSTALLED,
    )
    assert JOBLIB_INSTALLED
    # fmt: on


def _mock_joblib_import_error(mocker: pytest_mock.MockerFixture, error_message: str):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "joblib":
            raise ImportError(error_message)
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=fake_import)


def test_joblib_import_fails(
    mocker: pytest_mock.MockFixture,
):
    """Test that ImportError is propagated if joblib exists but import fails."""
    # pretend that joblib is not installed
    error_message = "Unexpected exception during import"
    _mock_joblib_import_error(mocker, error_message)

    # reload the module
    import nipoppy.workflows.services.execution

    with pytest.raises(ImportError, match=error_message):
        importlib.reload(nipoppy.workflows.services.execution)


@pytest.mark.parametrize(
    "analysis_level,expected",
    [
        (
            AnalysisLevelType.participant_session,
            [("S01", "BL"), ("S01", "FU"), ("S02", "BL"), ("S02", "FU")],
        ),
        (AnalysisLevelType.participant, [("S01", None), ("S02", None)]),
        (AnalysisLevelType.session, [(None, "BL"), (None, "FU")]),
        (AnalysisLevelType.group, [(None, None)]),
    ],
)
def test_apply_analysis_level(analysis_level, expected):
    participants_sessions = [("S01", "BL"), ("S01", "FU"), ("S02", "BL"), ("S02", "FU")]
    assert apply_analysis_level(participants_sessions, analysis_level) == expected


@pytest.mark.parametrize(
    "dname_pipelines,pipeline_name,expected_version",
    [
        ("processing", "fmriprep", "23.1.3"),
        ("processing", "my_pipeline", "2.0"),
        ("bidsification", "bids_converter", "1.0"),
        ("extraction", "extractor1", "0.1.0"),
    ],
)
def test_get_pipeline_version(
    dname_pipelines: str,
    pipeline_name: str,
    expected_version: str,
    workflow: PipelineExecutionWorkflow,
):
    assert (
        get_pipeline_version(
            pipeline_name,
            workflow.dataset.study.layout.dpath_pipelines / dname_pipelines,
        )
        == expected_version
    )


def test_get_pipeline_version_invalid_name(tmp_path: Path):
    with pytest.raises(WorkflowError, match="No config found for pipeline"):
        get_pipeline_version("pipeline1", tmp_path)


@pytest.mark.parametrize(
    "args",
    [
        {
            "dpath_root": "my_dataset",
            "pipeline_name": "my_pipeline",
            "pipeline_version": "1.0",
        },
        {
            "dpath_root": "my_dataset",
            "pipeline_name": "my_other_pipeline",
            "pipeline_version": "2.0",
        },
    ],
)
def test_init(args):
    workflow = PipelineWorkflow(**args)
    assert isinstance(workflow, PipelineExecutionWorkflow)
    assert workflow.pipeline.name == args["pipeline_name"]
    assert workflow.pipeline.version == args["pipeline_version"]
    assert isinstance(workflow.selection, SelectionOptions)
    assert isinstance(workflow.derivatives, Path)
    assert isinstance(workflow.output, Path)
    assert isinstance(workflow.workspace.work, Path)
    assert isinstance(workflow.workspace.bids_db, Path)


@pytest.mark.no_xdist
def test_init_n_jobs_but_no_joblib(
    tmp_path: Path,
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    reimport_joblib,
):
    """Test that an error is raised when joblib is not installed and n_jobs > 1."""
    dpath_root = tmp_path / "my_dataset"
    mocker.patch("nipoppy.workflows.services.execution.JOBLIB_INSTALLED", False)
    with pytest.raises(SystemExit):
        PipelineWorkflow(
            dpath_root=dpath_root,
            pipeline_name="my_pipeline",
            n_jobs=2,
            _skip_logfile=True,
        )

    assert any(
        [
            record.levelname == "ERROR"
            and (
                "An additional dependency is required to enable local parallelization "
                "with --n-jobs. Install it with: pip install nipoppy[parallel]"
            )
            in record.message
            for record in caplog.records
        ]
    )


def test_init_hpc_write_subcohort():
    args = {
        "dpath_root": "my_dataset",
        "pipeline_name": "my_pipeline",
        "hpc": "slurm",
        "write_subcohort": "list.tsv",
    }
    with pytest.raises(
        WorkflowError,
        match="HPC job submission and writing a list of participants and sessions are mutually exclusive.",  # noqa: E501
    ):
        PipelineWorkflow(**args)


@pytest.mark.parametrize(
    "participant_id,session_id,participant_expected,session_expected",
    [
        ("01", "BL", "01", "BL"),
        ("sub-01", "ses-BL", "01", "BL"),
    ],
)
def test_init_participant_session(
    participant_id, session_id, participant_expected, session_expected
):
    workflow = PipelineWorkflow(
        dpath_root="my_dataset",
        pipeline_name="my_pipeline",
        pipeline_version="1.0",
        participant_id=participant_id,
        session_id=session_id,
    )
    assert workflow.selection.participant_id == participant_expected
    assert workflow.selection.session_id == session_expected


def test_init_n_jobs_logfile():
    with pytest.raises(
        WorkflowError, match="n_jobs is not supported when _skip_logfile is False."
    ):
        PipelineWorkflow(
            dpath_root="my_dataset",
            pipeline_name="my_pipeline",
            n_jobs=2,
            _skip_logfile=False,
        )


def test_pipeline_version_optional():
    workflow = PipelineWorkflow(
        dpath_root="my_dataset",
        pipeline_name="my_pipeline",
    )
    assert workflow.pipeline.version is None


def test_pipeline_config(workflow: PipelineWorkflow, mocker: pytest_mock.MockFixture):
    mocked_process_template_json = mocker.patch.object(
        workflow.pipeline, "render", wraps=workflow.pipeline.render
    )

    assert isinstance(workflow.pipeline.config, ProcessingPipelineConfig)

    # make sure substitutions are processed
    mocked_process_template_json.assert_called_once()


def test_fpath_container(workflow: PipelineWorkflow, mocker: pytest_mock.MockFixture):
    fpath_container = (
        workflow.dataset.study.layout.dpath_containers / "my_container.sif"
    )
    fpath_container.parent.mkdir(parents=True, exist_ok=True)
    fpath_container.touch()

    mocked = mocker.patch(
        "nipoppy.workflows.services.boutiques.get_container_handler",
        return_value=ApptainerHandler(),
    )

    assert workflow.pipeline.container_file == fpath_container
    mocked.assert_called_once_with(workflow.pipeline.config.CONTAINER_CONFIG)


def test_fpath_container_custom(workflow: PipelineWorkflow):
    fpath_custom = workflow.dataset.study.layout.dpath_root / "my_container.sif"
    workflow.pipeline.config.CONTAINER_INFO.FILE = fpath_custom
    fpath_custom.touch()
    assert workflow.pipeline.container_file == fpath_custom


def test_fpath_container_not_specified_apptainer(workflow: PipelineWorkflow):
    workflow.pipeline.config.CONTAINER_INFO.FILE = None
    workflow.pipeline.step.CONTAINER_CONFIG.COMMAND = ContainerCommandEnum.APPTAINER
    with pytest.raises(
        WorkflowError,
        match=(
            "Error in container config for pipeline.*"
            "Path to container image must be specified"
        ),
    ):
        workflow.pipeline.container_file


def test_fpath_container_not_specified_docker(
    workflow: PipelineWorkflow, mocker: pytest_mock.MockFixture
):
    workflow.pipeline.config.CONTAINER_INFO.FILE = None
    workflow.pipeline.step.CONTAINER_CONFIG.COMMAND = ContainerCommandEnum.DOCKER

    mocker.patch(
        "nipoppy.container.DockerHandler.is_image_downloaded", return_value=True
    )

    # no error expected
    workflow.pipeline.container_file == workflow.pipeline.config.CONTAINER_INFO.FILE


@pytest.mark.parametrize("container_uri", [None, "docker://some/uri:tag"])
def test_fpath_container_not_found(workflow: PipelineWorkflow, container_uri):
    workflow.pipeline.config.CONTAINER_INFO.URI = container_uri
    error_message = (
        "No container image file found for pipeline "
        f"{workflow.pipeline.name} {workflow.pipeline.version}"
    )
    if container_uri is not None:
        error_message += (
            ". This file can be downloaded to the appropriate path by running the "
            "following command:\n\n"
            f"apptainer pull {workflow.pipeline.config.CONTAINER_INFO.FILE}"
            f" {workflow.pipeline.config.CONTAINER_INFO.URI}"
        )
    with pytest.raises(FileOperationError, match=error_message):
        workflow.pipeline.container_file


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,pipeline_step,descriptor",
    [
        ("fmriprep", "23.1.3", None, {}),
        ("my_pipeline", "1.0", None, {"key2": "value2"}),
    ],
)
def test_descriptor(
    workflow: PipelineWorkflow,
    pipeline_name,
    pipeline_version,
    pipeline_step,
    descriptor,
    tmp_path: Path,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version
    workflow.pipeline.step_name = pipeline_step

    fpath_descriptor = tmp_path / "descriptor.json"
    fpath_descriptor.write_text(json.dumps(descriptor))

    workflow.pipeline.step.DESCRIPTOR_FILE = fpath_descriptor.name
    workflow.pipeline.bundle = fpath_descriptor.parent
    assert workflow.pipeline.descriptor == descriptor


def test_descriptor_none(workflow: PipelineWorkflow):
    with pytest.raises(ConfigError, match="No descriptor file specified for pipeline"):
        workflow.pipeline.descriptor


@pytest.mark.parametrize(
    "variables,expected_descriptor",
    [
        ({"TO_REPLACE1": "value1"}, {"key1": "value1"}),
        ({"[[TO_REPLACE1]]": "value1"}, {"key1": "[[TO_REPLACE1]]"}),
    ],
)
def test_descriptor_pipeline_variables(
    tmp_path: Path, workflow: PipelineWorkflow, variables, expected_descriptor
):
    # set variables for substitution
    workflow.dataset.study.config.PIPELINE_VARIABLES.PROCESSING[workflow.pipeline.name][
        workflow.pipeline.version
    ] = variables

    # set descriptor file and write descriptor content
    fpath_descriptor = tmp_path / "descriptor.json"
    fpath_descriptor.write_text(json.dumps({"key1": "[[TO_REPLACE1]]"}))

    workflow.pipeline.step.DESCRIPTOR_FILE = fpath_descriptor.name
    workflow.pipeline.bundle = fpath_descriptor.parent

    assert workflow.pipeline.descriptor == expected_descriptor


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,invocation",
    [
        ("my_pipeline", "1.0", {"key1": "val1", "key2": "val2"}),
        ("fmriprep", "23.1.3", {}),
    ],
)
def test_invocation(
    pipeline_name,
    pipeline_version,
    invocation,
    tmp_path: Path,
    workflow: PipelineWorkflow,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version

    fpath_invocation = tmp_path / "invocation.json"
    fpath_invocation.write_text(json.dumps(invocation))

    workflow.pipeline.step.INVOCATION_FILE = fpath_invocation.name
    workflow.pipeline.bundle = fpath_invocation.parent

    assert workflow.pipeline.invocation == invocation


def test_invocation_none(workflow: PipelineWorkflow):
    with pytest.raises(ConfigError, match="No invocation file specified for pipeline"):
        workflow.pipeline.invocation


@pytest.mark.parametrize(
    "variables,expected_invocation",
    [
        ({"TO_REPLACE1": "value1"}, {"key1": "value1"}),
        ({"[[TO_REPLACE1]]": "value1"}, {"key1": "[[TO_REPLACE1]]"}),
    ],
)
def test_invocation_pipeline_variables(
    tmp_path: Path, workflow: PipelineWorkflow, variables, expected_invocation
):
    # set variables for substitution
    workflow.dataset.study.config.PIPELINE_VARIABLES.PROCESSING[workflow.pipeline.name][
        workflow.pipeline.version
    ] = variables

    # set invocation file and write invocation content
    fpath_invocation = tmp_path / "invocation.json"
    fpath_invocation.write_text(json.dumps({"key1": "[[TO_REPLACE1]]"}))

    workflow.pipeline.step.INVOCATION_FILE = fpath_invocation.name
    workflow.pipeline.bundle = fpath_invocation.parent

    assert workflow.pipeline.invocation == expected_invocation


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,pipeline_step,tracker_config_data",
    [
        ("fmriprep", "23.1.3", None, {"PATHS": ["path1"]}),
        ("my_pipeline", "1.0", None, {"PATHS": ["path1", "path2"]}),
    ],
)
def test_tracker_config(
    workflow: PipelineWorkflow,
    pipeline_name,
    pipeline_version,
    pipeline_step,
    tracker_config_data,
    tmp_path: Path,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version
    workflow.pipeline.step_name = pipeline_step

    fpath_tracker_config = tmp_path / "tracker_config.json"
    fpath_tracker_config.write_text(json.dumps(tracker_config_data))

    workflow.pipeline.step.TRACKER_CONFIG_FILE = fpath_tracker_config.name
    workflow.pipeline.bundle = fpath_tracker_config.parent
    assert workflow.pipeline.tracker == TrackerConfig(**tracker_config_data)


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,patterns",
    [
        ("my_pipeline", "1.0", ["PATTERN1", "PATTERN2"]),
        ("fmriprep", "23.1.3", ["PATTERN3", "PATTERN4"]),
    ],
)
def test_pybids_ignore_patterns(
    pipeline_name,
    pipeline_version,
    patterns,
    tmp_path: Path,
    workflow: PipelineWorkflow,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version

    fpath_patterns = tmp_path / "pybids_ignore_patterns.json"
    fpath_patterns.write_text(json.dumps(patterns))

    workflow.pipeline.step.PYBIDS_IGNORE_FILE = fpath_patterns.name
    workflow.pipeline.bundle = fpath_patterns.parent

    assert workflow.pipeline.ignore_patterns == [
        re.compile(pattern) for pattern in patterns
    ]


def test_pybids_ignore_patterns_no_file(workflow: PipelineWorkflow):
    workflow.pipeline.step.PYBIDS_IGNORE_FILE = None
    assert workflow.pipeline.ignore_patterns == []


def test_pybids_ignore_patterns_invalid_format(
    workflow: PipelineWorkflow, tmp_path: Path
):
    fpath_patterns = tmp_path / "pybids_ignore_patterns.json"
    fpath_patterns.write_text(json.dumps({"key": "value"}))
    workflow.pipeline.step.PYBIDS_IGNORE_FILE = fpath_patterns

    with pytest.raises(ConfigError, match="Expected a list of strings"):
        workflow.pipeline.ignore_patterns


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,dname_pipelines,pipeline_class",
    [
        ("fmriprep", "23.1.3", "processing", ProcessingPipelineConfig),
        ("my_pipeline", "2.0", "processing", ProcessingPipelineConfig),
        ("bids_converter", "1.0", "bidsification", BIDSificationPipelineConfig),
        ("extractor1", "0.1.0", "extraction", ExtractionPipelineConfig),
    ],
)
def test_get_pipeline_config(
    pipeline_name,
    pipeline_version,
    dname_pipelines,
    pipeline_class,
    workflow: PipelineWorkflow,
):
    dpath_pipeline_bundle = (
        workflow.dataset.study.layout.dpath_pipelines
        / dname_pipelines
        / f"{pipeline_name}-{pipeline_version}"
    )
    assert isinstance(
        workflow.pipeline.load_config(
            dpath_pipeline_bundle,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_class=pipeline_class,
        ),
        pipeline_class,
    )


def test_get_pipeline_config_json5(workflow: PipelineWorkflow, tmp_path: Path):
    (tmp_path / "config.json").write_text(
        f"""
{{
  // Comments and trailing commas should be supported
  "NAME": "{workflow.pipeline.name}",
  "VERSION": "{workflow.pipeline.version}",
  "PIPELINE_TYPE": "processing",
  "SCHEMA_VERSION": "{get_current_schema_version(ConfigType.PIPELINE)}",
  "STEPS": [
    {{}},
  ],
}}
""".strip()
    )

    config = workflow.pipeline.load_config(
        tmp_path,
        pipeline_name=workflow.pipeline.name,
        pipeline_version=workflow.pipeline.version,
        pipeline_class=ProcessingPipelineConfig,
    )
    assert isinstance(config, ProcessingPipelineConfig)


def test_get_pipeline_config_invalid(workflow: PipelineWorkflow):
    pipeline_name = "new_pipeline"
    pipeline_version = "1.0.0"
    dpath_pipeline_bundle = (
        workflow.dataset.study.layout.dpath_pipelines
        / "processing"
        / f"{pipeline_name}-{pipeline_version}"
    )
    config_dict = {
        "NAME": pipeline_name,
        "VERSION": "2.0.0",  # different version
        "PIPELINE_TYPE": "processing",
        "SCHEMA_VERSION": get_current_schema_version(ConfigType.PIPELINE),
    }
    dpath_pipeline_bundle.mkdir(parents=True)
    (dpath_pipeline_bundle / "config.json").write_text(json.dumps(config_dict))
    with pytest.raises(
        RuntimeError,
        match=(
            "Expected pipeline config to have "
            f'NAME="{pipeline_name}" and VERSION="{pipeline_version}"'
        ),
    ):
        workflow.pipeline.load_config(
            dpath_pipeline_bundle,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_class=ProcessingPipelineConfig,
        )


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,dname_pipelines,pipeline_class",
    [
        ("not_a_pipeline", "23.1.3", "processing", ProcessingPipelineConfig),
        ("my_pipeline", "not_a_version", "processing", ProcessingPipelineConfig),
        ("bids_converter", "2.0", "bidsification", BIDSificationPipelineConfig),
        ("bids_converter", "1.0", "extraction", ExtractionPipelineConfig),
    ],
)
def test_get_pipeline_config_missing(
    pipeline_name,
    pipeline_version,
    dname_pipelines,
    pipeline_class,
    workflow: PipelineWorkflow,
):
    dpath_pipeline_bundle = (
        workflow.dataset.study.layout.dpath_pipelines
        / dname_pipelines
        / f"{pipeline_name}-{pipeline_version}"
    )
    with pytest.raises(FileOperationError, match="Pipeline config file not found at"):
        workflow.pipeline.load_config(
            dpath_pipeline_bundle,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_class=pipeline_class,
        )


@pytest.mark.parametrize("return_str", [True, False])
def test_process_template_json(workflow: PipelineWorkflow, return_str):
    # add user-defined substitution variables
    workflow.dataset.study.config.SUBSTITUTIONS = {
        "USER_SUBSTITUTION": "val1",
        "OTHER_USER_SUBSTITUTION": "val2",
    }

    class Test:
        extra2 = "extra_obj_attribute"

    processed = workflow.templates.render(
        {
            "[[NIPOPPY_BIDS_PARTICIPANT_ID]]": "[[NIPOPPY_PARTICIPANT_ID]]",
            "[[NIPOPPY_BIDS_SESSION_ID]]": "[[NIPOPPY_SESSION_ID]]",
            "[[NIPOPPY_DPATH_PIPELINE]]": "[[NIPOPPY_DPATH_BIDS]]",
            "[[NIPOPPY_EXTRA1]]": "[[NIPOPPY_EXTRA2]]",
            "USER_SUBSTITUTION": "OTHER_USER_SUBSTITUTION",
        },
        participant_id="01",
        session_id="1",
        extra1="extra_kwarg",
        objs=[Test()],
        return_str=return_str,
    )

    if return_str:
        assert isinstance(processed, str)
    else:
        assert isinstance(processed, dict)
        processed = json.dumps(processed)

    # check that everything was replaced
    for pattern in [
        "[[NIPOPPY_BIDS_PARTICIPANT_ID]]",
        "[[NIPOPPY_PARTICIPANT_ID]]",
        "[[NIPOPPY_BIDS_SESSION_ID]]",
        "[[NIPOPPY_SESSION_ID]]",
        "[[NIPOPPY_DPATH_PIPELINE]]",
        "[[NIPOPPY_DPATH_BIDS]]",
        "[[NIPOPPY_EXTRA1]]",
        "[[NIPOPPY_EXTRA2]]",
        "USER_SUBSTITUTION",
        "OTHER_USER_SUBSTITUTION",
    ]:
        assert pattern not in processed


def test_boutiques_config(tmp_path: Path):
    workflow = PipelineWorkflow(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="my_pipeline",
        pipeline_version="1.0",
    )
    workflow.pipeline.descriptor = {
        "custom": {
            "nipoppy": {"CONTAINER_CONFIG": {"ARGS": ["--pipeline-specific-arg"]}}
        }
    }

    boutiques_config = workflow.pipeline.boutiques_config
    assert isinstance(boutiques_config, BoutiquesConfig)
    # should not be the default
    assert boutiques_config != BoutiquesConfig()


def test_boutiques_config_default(tmp_path: Path):
    workflow = PipelineWorkflow(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="no_boutiques_config",
        pipeline_version="1.0",
    )
    workflow.pipeline.descriptor = {}
    boutiques_config = workflow.pipeline.boutiques_config
    assert isinstance(boutiques_config, BoutiquesConfig)
    # expect the default
    assert boutiques_config == BoutiquesConfig()


def test_boutiques_config_invalid(tmp_path: Path):
    workflow = PipelineWorkflow(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="bad_boutiques_config",
        pipeline_version="1.0",
    )
    workflow.pipeline.descriptor = {"custom": {"nipoppy": {"INVALID_ARG": "value"}}}

    with pytest.raises(WorkflowError, match="Error when loading the Boutiques config"):
        workflow.pipeline.boutiques_config


@pytest.mark.parametrize(
    "participant_id,session_id,expected_count",
    [
        (None, None, 12),
        ("01", None, 6),
        ("02", None, 6),
        (None, "1", 5),
        (None, "2", 5),
        (None, "3", 2),
        ("01", "3", 2),
        ("02", "3", 0),
    ],
)
def test_set_up_bids_db(
    workflow: PipelineWorkflow,
    participant_id,
    session_id,
    expected_count,
    tmp_path: Path,
):
    dpath_pybids_db = tmp_path / "bids_db"
    fids.create_fake_bids_dataset(
        output_dir=workflow.dataset.study.layout.dpath_bids,
        subjects="01",
        sessions=["1", "2", "3"],
        datatypes=["anat"],
    )
    fids.create_fake_bids_dataset(
        output_dir=workflow.dataset.study.layout.dpath_bids,
        subjects="02",
        sessions=["1", "2"],
        datatypes=["anat", "func"],
    )
    bids_layout = build_bids_db(
        workflow.dataset.study.layout.dpath_bids,
        ignore_patterns=workflow.pipeline.ignore_patterns,
        dpath_pybids_db=dpath_pybids_db,
        participant_id=participant_id,
        session_id=session_id,
    )
    assert dpath_pybids_db.exists()
    assert len(bids_layout.get(extension=".nii.gz")) == expected_count


def test_set_up_bids_db_ignore_patterns(workflow: PipelineWorkflow, tmp_path: Path):
    dpath_pybids_db = tmp_path / "bids_db"
    participant_id = "01"
    session_id = "1"

    fids.create_fake_bids_dataset(
        output_dir=workflow.dataset.study.layout.dpath_bids,
    )

    pybids_ignore_patterns = workflow.pipeline.ignore_patterns[:]

    build_bids_db(
        workflow.dataset.study.layout.dpath_bids,
        ignore_patterns=workflow.pipeline.ignore_patterns,
        dpath_pybids_db=dpath_pybids_db,
        participant_id=participant_id,
        session_id=session_id,
    )

    assert pybids_ignore_patterns == workflow.pipeline.ignore_patterns


@pytest.mark.no_xdist
def test_set_up_bids_db_no_session(
    workflow: PipelineWorkflow,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Create a fake BIDS dataset with no session-level folders.

    Make sure:
    - Check that the ignore pattern is not added to the BIDS layout.
    - Check if files are found in the BIDS layout not ignored.
    """
    dpath_pybids_db = tmp_path / "bids_db"
    participant_id = "01"
    session_id = FAKE_SESSION_ID

    fids.create_fake_bids_dataset(
        output_dir=workflow.dataset.study.layout.dpath_bids,
        subjects=participant_id,
        sessions=None,
    )

    bids_layout = build_bids_db(
        workflow.dataset.study.layout.dpath_bids,
        ignore_patterns=workflow.pipeline.ignore_patterns,
        dpath_pybids_db=dpath_pybids_db,
        participant_id=participant_id,
        session_id=session_id,
    )

    assert f".*?/{BIDS_SESSION_PREFIX}(?!{session_id})" not in caplog.text
    assert len(bids_layout.get(extension=".nii.gz")) > 0


@pytest.mark.parametrize(
    "pipeline_name,expected_version",
    [("fmriprep", "23.1.3"), ("my_pipeline", "2.0")],
)
@pytest.mark.no_xdist
def test_check_pipeline_version(
    pipeline_name,
    expected_version,
    workflow: PipelineWorkflow,
    caplog: pytest.LogCaptureFixture,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = None  # should be inferred
    workflow.pipeline.resolve_version()
    assert workflow.pipeline.version == expected_version
    assert f"using version {expected_version}" in caplog.text


@pytest.mark.parametrize(
    "variables,valid", [({"var1": "val1"}, True), ({"var2": None}, False)]
)
def test_check_pipeline_variables(workflow: PipelineWorkflow, variables, valid):
    workflow.dataset.study.config.PIPELINE_VARIABLES.PROCESSING[workflow.pipeline.name][
        workflow.pipeline.version
    ] = variables
    with (
        nullcontext()
        if valid
        else pytest.raises(ConfigError, match="Variable .* is not set in the config")
    ):
        assert workflow.pipeline.check_variables() is None


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,expected_step",
    [
        ("fmriprep", "23.1.3", DEFAULT_PIPELINE_STEP_NAME),
        ("my_pipeline", "1.0", DEFAULT_PIPELINE_STEP_NAME),
    ],
)
@pytest.mark.no_xdist
def test_check_pipeline_step(
    pipeline_name,
    pipeline_version,
    expected_step,
    workflow: PipelineWorkflow,
    caplog: pytest.LogCaptureFixture,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version
    workflow.pipeline.step_name = None
    workflow.pipeline.resolve_step()
    assert workflow.pipeline.step_name == expected_step
    assert f"using step {expected_step}" in caplog.text


def test_run_setup_pipeline_version_step(workflow: PipelineWorkflow):
    workflow.pipeline.version = None
    workflow.pipeline.step_name = None
    create_empty_dataset(workflow.dataset.study.layout.dpath_root)
    workflow.run_setup()
    assert workflow.pipeline.version == "2.0"
    assert workflow.pipeline.step_name == DEFAULT_PIPELINE_STEP_NAME


@pytest.mark.parametrize("dry_run", [True, False])
def test_run_setup_create_directories(workflow: PipelineWorkflow, dry_run: bool):
    workflow.dry_run = dry_run

    assert len(workflow.directories) == 0
    workflow.directories = [workflow.derivatives]

    workflow.run_setup()
    assert workflow.derivatives.exists() == (not dry_run)

    # run again, should not fail even if directories already exist
    workflow.run_setup()


def test_run_setup_check_filter_args_compatibility_called(
    workflow: PipelineWorkflow,
    mocker: pytest_mock.MockFixture,
):
    create_empty_dataset(workflow.dataset.study.layout.dpath_root)
    mocked_check = mocker.patch(
        "nipoppy.workflows.execution.check_filter_args", wraps=check_filter_args
    )

    workflow.run_setup()

    mocked_check.assert_called_once_with(
        workflow.pipeline.step.ANALYSIS_LEVEL,
        workflow.selection.participant_id,
        workflow.selection.session_id,
    )


@pytest.mark.parametrize("show_progress", [True, False])
def test_get_results_generator_no_joblib(
    workflow: PipelineWorkflow,
    show_progress: bool,
    mocker: pytest_mock.MockFixture,
    reimport_joblib,
):
    workflow.execution = replace(workflow.execution, n_jobs=1)
    workflow.execution = replace(workflow.execution, show_progress=show_progress)
    participants_sessions = [("01", "1"), ("01", "2"), ("01", "3"), ("02", "1")]

    # pretend that joblib is not installed
    _mock_joblib_import_error(mocker, "No module named 'joblib'")

    # also mock joblib.delayed which is not supposed to be called
    # when joblib is not installed
    mocked_delayed = mocker.patch("nipoppy.workflows.services.execution.delayed")

    # reload the module
    # fmt: off
    import nipoppy.workflows.services.execution
    importlib.reload(nipoppy.workflows.services.execution)
    # check that mocking/reloading worked
    from nipoppy.workflows.services.execution import (  # noqa: F401
        JOBLIB_INSTALLED,
    )
    assert not JOBLIB_INSTALLED
    # fmt: on

    results = get_results_generator(workflow, participants_sessions)
    mocked_delayed.assert_not_called()
    assert isinstance(results, Generator)
    assert len(list(results)) == len(participants_sessions)


def test_get_results_generator_with_progress_bar(
    workflow: PipelineWorkflow, mocker: pytest_mock.MockFixture
):
    workflow.execution = replace(workflow.execution, show_progress=True)
    participants_sessions = [("001", "1")]

    mocked_track = mocker.patch(
        "nipoppy.workflows.services.execution.rich.progress.track"
    )

    get_results_generator(workflow, participants_sessions)
    mocked_track.assert_called_once()


@pytest.mark.parametrize(
    "show_progress,participants_sessions", [(False, [("001", "1")]), (True, [])]
)
def test_get_results_generator_no_progress_bar(
    workflow: PipelineWorkflow,
    show_progress,
    participants_sessions,
    mocker: pytest_mock.MockFixture,
):
    workflow.execution = replace(workflow.execution, show_progress=show_progress)

    mocked_track = mocker.patch(
        "nipoppy.workflows.services.execution.rich.progress.track"
    )

    get_results_generator(workflow, participants_sessions)
    mocked_track.assert_not_called()


@pytest.mark.parametrize(
    "participant_id,session_id,expected_count",
    [(None, None, 4), ("01", None, 3), ("01", "2", 1)],
)
def test_run_main(
    workflow: PipelineWorkflow,
    participant_id,
    session_id,
    expected_count,
    mocker: pytest_mock.MockFixture,
):
    workflow.selection = replace(workflow.selection, participant_id=participant_id)
    workflow.selection = replace(workflow.selection, session_id=session_id)

    participants_and_sessions = {"01": ["1", "2", "3"], "02": ["1"]}
    manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
        dpath_bidsified=workflow.dataset.study.layout.dpath_bids,
    )
    manifest.save_with_backup(workflow.dataset.study.layout.fpath_manifest)

    mocked_log_summary_message = mocker.patch(
        "nipoppy.workflows.execution.log_summary", return_value=None
    )

    workflow.run_main()
    assert workflow.result.n_total == expected_count
    assert workflow.result.n_success == expected_count
    assert workflow.result.payloads == tuple(["SUCCESS"] * expected_count)
    mocked_log_summary_message.assert_called_once()


def test_run_main_analysis_level(
    workflow: PipelineWorkflow,
    mocker: pytest_mock.MockFixture,
):
    mocked = mocker.patch(
        "nipoppy.workflows.services.cohort.apply_analysis_level",
        wraps=apply_analysis_level,
    )
    participants_and_sessions = {"01": ["1", "2", "3"], "02": ["1"]}
    manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions
    )
    manifest.save_with_backup(workflow.dataset.study.layout.fpath_manifest)
    workflow.run_main()
    assert mocked.call_count == 1


@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_run_main_n_jobs(workflow: PipelineWorkflow, n_jobs: int):
    """Smoke test for parallel execution (with joblib installed)."""
    # fmt: off
    # make sure joblib is installed
    from nipoppy.workflows.services.execution import JOBLIB_INSTALLED
    assert JOBLIB_INSTALLED, "joblib must be installed"
    # fmt: on

    workflow.execution = replace(workflow.execution, n_jobs=n_jobs, skip_logfile=True)
    participants_and_sessions = {"01": ["1", "2", "3"], "02": ["1"]}
    manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
        dpath_bidsified=workflow.dataset.study.layout.dpath_bids,
    )
    manifest.save_with_backup(workflow.dataset.study.layout.fpath_manifest)
    workflow.run_main()


def test_run_main_catch_errors(workflow: PipelineWorkflow):
    workflow.selection = replace(workflow.selection, participant_id="FAIL")
    workflow.selection = replace(workflow.selection, session_id="1")

    participants_and_sessions = {
        workflow.selection.participant_id: [workflow.selection.session_id]
    }
    manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
        dpath_bidsified=workflow.dataset.study.layout.dpath_bids,
    )
    manifest.save_with_backup(workflow.dataset.study.layout.fpath_manifest)
    workflow.run_main()
    assert workflow.result.n_total == 1
    assert workflow.result.n_success == 0
    assert workflow.return_code == ReturnCode.PARTIAL_SUCCESS


@pytest.mark.parametrize("write_subcohort", ["list.tsv", "to_run.tsv"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_run_main_write_subcohort(
    workflow: PipelineWorkflow,
    write_subcohort: str,
    dry_run: bool,
    tmp_path: Path,
):
    write_subcohort = tmp_path / write_subcohort

    workflow.selection = replace(workflow.selection, participant_id="01")
    workflow.selection = replace(workflow.selection, session_id="1")
    workflow.execution = replace(workflow.execution, write_subcohort=write_subcohort)
    workflow.dry_run = dry_run

    participants_and_sessions = {
        workflow.selection.participant_id: [workflow.selection.session_id]
    }
    manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
        dpath_bidsified=workflow.dataset.study.layout.dpath_bids,
    )
    workflow.dataset.study.manifest = manifest
    workflow.run_main()

    if not dry_run:
        assert write_subcohort.exists()
        assert write_subcohort.read_text().strip() == "01\t1"
    else:
        assert not write_subcohort.exists()


@pytest.mark.parametrize(
    "original_participants_sessions_to_run,list_content,final_participants_sessions_to_run",
    [
        (
            [("01", "A"), ("02", "B"), ("03", "C")],
            [("01", "A"), ("02", "B")],
            [("01", "A"), ("02", "B")],
        ),
        (
            [("01", "A"), ("02", "B"), ("03", "C")],
            [("01", "A"), ("02", "B"), ("04", "D")],
            [("01", "A"), ("02", "B")],
        ),
        (
            [],
            [("01", "A")],
            [],
        ),
    ],
)
def test_run_main_use_subcohort(
    original_participants_sessions_to_run: list[tuple[str, str]],
    list_content: list[tuple[str, str]],
    final_participants_sessions_to_run: list[tuple[str, str]],
    workflow: PipelineWorkflow,
    tmp_path: Path,
    mocker: pytest_mock.MockFixture,
):
    fpath_list = tmp_path / "to_run.tsv"
    pd.DataFrame(list_content).to_csv(fpath_list, sep="\t", header=False, index=False)

    workflow.selection = replace(workflow.selection, use_subcohort=fpath_list)

    mocker.patch.object(
        workflow,
        "select_candidates",
        return_value=original_participants_sessions_to_run,
    )
    mocked_run_single = mocker.patch.object(workflow, "execute_item")

    workflow.run_main()

    for participant_id, session_id in final_participants_sessions_to_run:
        mocked_run_single.assert_any_call(WorkItem(participant_id, session_id))


def test_run_main_use_subcohort_not_found(
    workflow: PipelineWorkflow, tmp_path: Path, mocker: pytest_mock.MockFixture
):
    fpath_list = tmp_path / "to_run.tsv"
    workflow.selection = replace(workflow.selection, use_subcohort=fpath_list)

    assert not fpath_list.exists(), "File cannot exist"

    mocker.patch.object(
        workflow,
        "select_candidates",
        return_value=[("01", "A")],
    )

    with pytest.raises(
        FileOperationError,
        match=f"Subcohort file {fpath_list} not found",
    ):
        workflow.run_main()


def test_run_main_use_subcohort_empty_file(
    workflow: PipelineWorkflow,
    tmp_path: Path,
    mocker: pytest_mock.MockFixture,
):
    fpath_list: Path = tmp_path / "to_run.tsv"
    workflow.selection = replace(workflow.selection, use_subcohort=fpath_list)

    fpath_list.touch()

    mocker.patch.object(
        workflow,
        "select_candidates",
        return_value=[("01", "A")],
    )

    with pytest.raises(RuntimeError, match=f"Subcohort file {fpath_list} is empty"):
        workflow.run_main()


@pytest.mark.parametrize(
    "n_success,n_total,analysis_level,expected_message",
    [
        (0, 0, "participant_session", "No participants or sessions to run"),
        (
            1,
            2,
            "participant_session",
            "Ran for {0} out of {1} participants or sessions",
        ),
        (
            0,
            1,
            "participant_session",
            "Ran for {0} out of {1} participants or sessions",
        ),
        (
            2,
            2,
            "participant_session",
            "Ran for {0} out of {1} participants or sessions",
        ),
        (
            1,
            1,
            "group",
            "Ran on the entire study",
        ),
    ],
)
@pytest.mark.no_xdist
def test_log_summary_message(
    n_success,
    n_total,
    analysis_level,
    expected_message: str,
    workflow: PipelineWorkflow,
    caplog: pytest.LogCaptureFixture,
):
    workflow.result = replace(workflow.result, n_success=n_success)
    workflow.result = replace(workflow.result, n_total=n_total)

    workflow.pipeline.config.STEPS = [
        ProcPipelineStepConfig(ANALYSIS_LEVEL=analysis_level)
    ]

    report_summary(workflow)

    assert expected_message.format(n_success, n_total) in caplog.text


@pytest.mark.no_xdist
def test_log_summary_message_no_participants_warning(
    workflow: PipelineWorkflow, caplog: pytest.LogCaptureFixture
):
    workflow.result = replace(workflow.result, n_success=0)
    workflow.result = replace(workflow.result, n_total=0)
    report_summary(workflow)
    assert any(
        record.levelname == "WARNING"
        and "No participants or sessions to run" in record.message
        for record in caplog.records
    ), "Wrong log message or level"
    assert workflow.return_code == ReturnCode.NO_PARTICIPANTS_OR_SESSIONS_TO_RUN, (
        "Wrong return code"
    )


@pytest.mark.parametrize(
    "n_success,n_total,expected_message",
    [
        (0, 0, "No participants or sessions to run"),
        (0, 1, "Failed to submit HPC jobs"),
        (2, 2, "Successfully submitted 2 HPC job(s)"),
    ],
)
@pytest.mark.no_xdist
def test_log_summary_message_hpc(
    n_success,
    n_total,
    expected_message,
    workflow: PipelineWorkflow,
    caplog: pytest.LogCaptureFixture,
):
    workflow.execution = replace(workflow.execution, hpc="slurm")

    workflow.result = replace(workflow.result, n_success=n_success)
    workflow.result = replace(workflow.result, n_total=n_total)
    report_summary(workflow)

    assert expected_message in caplog.text


@pytest.mark.parametrize(
    "analysis_level,participant_id,session_id",
    [
        (AnalysisLevelType.participant_session, "01", "1"),
        (AnalysisLevelType.participant_session, None, None),
        (AnalysisLevelType.participant, "01", None),
        (AnalysisLevelType.session, None, "1"),
        (AnalysisLevelType.group, None, None),
    ],
)
def test_check_filter_args_compatibility(
    analysis_level, participant_id, session_id, workflow: PipelineWorkflow
):
    workflow.pipeline.step.ANALYSIS_LEVEL = analysis_level
    workflow.selection = replace(workflow.selection, participant_id=participant_id)
    workflow.selection = replace(workflow.selection, session_id=session_id)

    check_filter_args(
        workflow.pipeline.step.ANALYSIS_LEVEL,
        workflow.selection.participant_id,
        workflow.selection.session_id,
    )


@pytest.mark.parametrize(
    "analysis_level,participant_id,session_id,expected_message",
    [
        (
            AnalysisLevelType.group,
            "01",
            None,
            "The --participant-id flag(s) are given, "
            "but the pipeline is run at the group level",
        ),
        (
            AnalysisLevelType.group,
            None,
            "1",
            "The --session-id flag(s) are given, "
            "but the pipeline is run at the group level",
        ),
        (
            AnalysisLevelType.group,
            "01",
            "1",
            "The --participant-id and --session-id flag(s) are given, "
            "but the pipeline is run at the group level",
        ),
        (
            AnalysisLevelType.session,
            "01",
            None,
            "The --participant-id flag(s) are given, "
            "but the pipeline is run at the session level",
        ),
        (
            AnalysisLevelType.participant,
            None,
            "1",
            "The --session-id flag(s) are given, "
            "but the pipeline is run at the participant level",
        ),
    ],
)
def test_check_filter_args_compatibility_error(
    analysis_level,
    participant_id,
    session_id,
    expected_message,
    workflow: PipelineWorkflow,
):
    workflow.pipeline.step.ANALYSIS_LEVEL = analysis_level
    workflow.selection = replace(workflow.selection, participant_id=participant_id)
    workflow.selection = replace(workflow.selection, session_id=session_id)

    with pytest.raises(InvalidArgumentError, match=re.escape(expected_message)):
        check_filter_args(
            workflow.pipeline.step.ANALYSIS_LEVEL,
            workflow.selection.participant_id,
            workflow.selection.session_id,
        )


@pytest.mark.no_xdist
def test_log_summary_message_write_subcohort(
    workflow: PipelineWorkflow, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    workflow.execution = replace(
        workflow.execution, write_subcohort=tmp_path / "subcohort.tsv"
    )

    workflow.result = replace(workflow.result, n_success=0)
    workflow.result = replace(workflow.result, n_total=0)
    report_summary(workflow)

    assert "No participants or sessions to run" not in caplog.text
    assert "Wrote subcohort to" in caplog.text


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version,pipeline_step,participant_id,session_id,expected_stem",
    [
        (
            "my_pipeline",
            "1.0",
            "step1",
            "sub1",
            None,
            "nipoppy/test/my_pipeline-1.0-step1/my_pipeline-1.0-step1-sub1",
        ),
        (
            "my_pipeline",
            None,
            None,
            "sub1",
            None,
            "nipoppy/test/my_pipeline-2.0-default/my_pipeline-2.0-default-sub1",
        ),
        (
            "fmriprep",
            None,
            None,
            None,
            "1",
            "nipoppy/test/fmriprep-23.1.3-default/fmriprep-23.1.3-default-1",
        ),
    ],
)
def test_generate_fpath_log(
    pipeline_name,
    pipeline_version,
    pipeline_step,
    participant_id,
    session_id,
    expected_stem,
    workflow: PipelineWorkflow,
    datetime_fixture,  # noqa F811
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version
    workflow.pipeline.step_name = pipeline_step
    workflow.selection = replace(workflow.selection, participant_id=participant_id)
    workflow.selection = replace(workflow.selection, session_id=session_id)
    fpath_log = workflow.generate_fpath_log()
    assert (
        fpath_log
        == workflow.dataset.study.layout.dpath_logs
        / f"{expected_stem}-20240404_1234.log"
    )


def get_results_generator(workflow, pairs):
    """Exercise ordered local execution with the workflow's domain hook."""

    def execute(participant, session):
        return run_item(
            lambda p, s: workflow.execute_item(WorkItem(p, s)),
            workflow.pipeline.name,
            workflow.pipeline.version,
            participant,
            session,
        )

    return iter_results(
        pairs,
        execute,
        n_jobs=workflow.execution.n_jobs,
        show_progress=workflow.execution.show_progress,
    )


def report_summary(workflow):
    """Apply the summary policy to an explicitly provided result."""
    code = log_summary(
        workflow.result,
        workflow.execution.write_subcohort,
        workflow.execution.hpc,
        workflow.pipeline.step.ANALYSIS_LEVEL,
        workflow.dataset.study.layout.fpath_curation_status,
    )
    if code is not None:
        workflow.return_code = code
