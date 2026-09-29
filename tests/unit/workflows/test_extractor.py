"""Tests for ExtractionWorkflow."""

from pathlib import Path

import pytest
import pytest_mock

from nipoppy.config.pipeline import (
    ExtractionPipelineConfig,
    ExtractionPipelineStepConfig,
    PipelineInfo,
)
from nipoppy.env import DEFAULT_PIPELINE_STEP_NAME
from nipoppy.exceptions import FileOperationError
from nipoppy.tabular.processing_status import ProcessingStatusTable
from nipoppy.utils.bids import (
    participant_id_to_bids_participant_id,
    session_id_to_bids_session_id,
)
from nipoppy.workflows.extractor import ExtractionRunner
from nipoppy.workflows.factories import create_extraction_workflow
from nipoppy.workflows.services.cohort import SelectionOptions, WorkItem
from tests.conftest import (
    create_empty_dataset,
    create_pipeline_config_files,
    get_config,
)


@pytest.fixture(scope="function")
def extractor(tmp_path: Path) -> ExtractionRunner:
    extractor = create_extraction_workflow(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="fs_extractor",
        pipeline_version="2.0.0",
        pipeline_step=DEFAULT_PIPELINE_STEP_NAME,
    )
    extractor.dataset.study.config = get_config()
    create_empty_dataset(extractor.dataset.study.layout.dpath_root)
    create_pipeline_config_files(
        extractor.dataset.study.layout.dpath_pipelines,
        processing_pipelines=[
            {
                "NAME": "freesurfer",
                "VERSION": "7.3.2",
                "STEPS": [
                    # field unique to ProcPipelineStepConfig
                    {"TRACKER_CONFIG_FILE": "tracker_config.json"}
                ],
            },
        ],
        extraction_pipelines=[
            {
                "NAME": "fs_extractor",
                "VERSION": "2.0.0",
                "PROC_DEPENDENCIES": [
                    {
                        "NAME": "freesurfer",
                        "VERSION": "7.3.2",
                    },
                ],
                "STEPS": [{}],
            },
            {
                "NAME": "fs_fmriprep_extractor",
                "VERSION": "1.0.0",
                "PROC_DEPENDENCIES": [
                    {
                        "NAME": "freesurfer",
                        "VERSION": "6.0.1",
                    },
                    {
                        "NAME": "fmriprep",
                        "VERSION": "20.0.7",
                    },
                ],
                "STEPS": [{}],
            },
        ],
    )
    return extractor


@pytest.mark.parametrize(
    "attribute,expected",
    [
        ("derivatives", "derivatives/freesurfer/7.3.2"),
        ("output", "derivatives/freesurfer/7.3.2/output"),
        ("idp", "derivatives/freesurfer/7.3.2/idp"),
    ],
)
def test_paths(extractor: ExtractionRunner, attribute, expected):
    assert (
        getattr(extractor, attribute)
        == extractor.dataset.study.layout.dpath_root / expected
    )


def test_setup(extractor: ExtractionRunner):
    assert not extractor.idp.exists()
    extractor.run_setup()
    assert extractor.idp.exists()


def test_dpath_pipeline(extractor: ExtractionRunner):
    assert (
        extractor.derivatives
        == extractor.dataset.study.layout.dpath_derivatives / "freesurfer" / "7.3.2"
    )


def test_proc_pipeline_info(extractor: ExtractionRunner):
    # check that proc_pipeline_info calls _get_pipeline_config with the
    # correct pipeline_class
    assert isinstance(extractor.source, PipelineInfo)


def test_proc_pipeline_info_error(extractor: ExtractionRunner):
    bad_version = "invalid_version"
    extractor.pipeline.config.PROC_DEPENDENCIES[0].VERSION = bad_version
    with pytest.raises(
        FileOperationError, match=f"Pipeline config file not found at .* {bad_version}"
    ):
        extractor.source


@pytest.mark.parametrize(
    "processing_status_data,pipeline_name,pipeline_version,participant_id,session_id,expected",  # noqa: E501
    [
        (
            [
                [
                    "S01",
                    "1",
                    "freesurfer",
                    "7.3.2",
                    ProcessingStatusTable.status_success,
                ],
                [
                    "S01",
                    "2",
                    "freesurfer",
                    "7.3.2",
                    ProcessingStatusTable.status_incomplete,
                ],
                ["S01", "3", "freesurfer", "7.3.2", ProcessingStatusTable.status_fail],
                [
                    "S02",
                    "1",
                    "freesurfer",
                    "7.3.2",
                    ProcessingStatusTable.status_unavailable,
                ],
                [
                    "S02",
                    "2",
                    "freesurfer",
                    "7.3.2",
                    ProcessingStatusTable.status_success,
                ],
            ],
            "fs_extractor",
            "2.0.0",
            None,
            None,
            [("S01", "1"), ("S02", "2")],
        ),
        (
            [
                [
                    "S01",
                    "1",
                    "freesurfer",
                    "7.3.2",
                    ProcessingStatusTable.status_success,
                ],
            ],
            "fs_extractor",
            "2.0.0",
            "S02",  # S02 is not in processing status table
            "1",
            [],
        ),
        (
            [
                [
                    "P01",
                    "A",
                    "freesurfer",
                    "6.0.1",
                    ProcessingStatusTable.status_success,
                ],
                ["P01", "B", "freesurfer", "6.0.1", ProcessingStatusTable.status_fail],
                [
                    "P02",
                    "B",
                    "freesurfer",
                    "6.0.1",
                    ProcessingStatusTable.status_success,
                ],
                ["P01", "A", "fmriprep", "20.0.7", ProcessingStatusTable.status_fail],
                [
                    "P01",
                    "B",
                    "fmriprep",
                    "20.0.7",
                    ProcessingStatusTable.status_success,
                ],
                [
                    "P02",
                    "B",
                    "fmriprep",
                    "20.0.7",
                    ProcessingStatusTable.status_success,
                ],
            ],
            "fs_fmriprep_extractor",
            "1.0.0",
            "P02",
            "B",
            [("P02", "B")],
        ),
    ],
)
def test_get_participants_sessions_to_run(
    processing_status_data,
    pipeline_name,
    pipeline_version,
    participant_id,
    session_id,
    expected,
    extractor: ExtractionRunner,
):
    extractor.pipeline.name = pipeline_name
    extractor.pipeline.version = pipeline_version
    extractor.pipeline.step_name = DEFAULT_PIPELINE_STEP_NAME
    extractor.dataset.processing_status_table = ProcessingStatusTable().add_or_update_records(
        records=[
            {
                ProcessingStatusTable.col_participant_id: data[0],
                ProcessingStatusTable.col_session_id: data[1],
                ProcessingStatusTable.col_bids_participant_id: participant_id_to_bids_participant_id(
                    data[0]
                ),
                ProcessingStatusTable.col_bids_session_id: session_id_to_bids_session_id(
                    data[1]
                ),
                ProcessingStatusTable.col_pipeline_name: data[2],
                ProcessingStatusTable.col_pipeline_version: data[3],
                ProcessingStatusTable.col_pipeline_step: DEFAULT_PIPELINE_STEP_NAME,
                ProcessingStatusTable.col_status: data[4],
            }
            for data in processing_status_data
        ]
    )
    extractor.selection = SelectionOptions(participant_id, session_id)
    assert list(extractor.select_candidates()) == expected


def test_run_single(
    extractor: ExtractionRunner,
    mocker: pytest_mock.MockerFixture,
):
    mocked_execute = mocker.patch.object(extractor, "execute_pipeline")
    extractor.execute_item(WorkItem("S01", "BL"))
    mocked_execute.assert_called_once_with(
        WorkItem("S01", "BL"), [extractor.idp, extractor.output]
    )


def test_check_pipeline_version(extractor: ExtractionRunner):
    extractor.pipeline.name = "fs_extractor"
    extractor.pipeline.version = None

    extractor.pipeline.resolve_version()
    assert extractor.pipeline.version == "2.0.0"


def test_pipeline_config(extractor: ExtractionRunner):
    assert isinstance(extractor.pipeline.config, ExtractionPipelineConfig)


def test_pipeline_step_config(extractor: ExtractionRunner):
    assert isinstance(extractor.pipeline.step, ExtractionPipelineStepConfig)


def test_generate_cli_command_for_hpc(
    extractor: ExtractionRunner,
    mocker: pytest_mock.MockFixture,
):
    assert extractor.hpc_extra_flags() == []
