"""Tests for BidsConversionWorkflow."""

from dataclasses import replace
from pathlib import Path

import pytest
import pytest_mock

from nipoppy.config.pipeline import BIDSificationPipelineConfig
from nipoppy.tabular.curation_status import CurationStatusTable
from nipoppy.workflows.bids_conversion import BIDSificationRunner
from nipoppy.workflows.factories import create_bidsification_workflow
from nipoppy.workflows.services.cohort import SelectionOptions, WorkItem
from tests.conftest import (
    create_empty_dataset,
    create_pipeline_config_files,
    get_config,
)


@pytest.fixture
def workflow(tmp_path: Path) -> BIDSificationRunner:
    workflow = create_bidsification_workflow(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="heudiconv",
        pipeline_version="0.12.2",
        pipeline_step="prepare",
    )
    workflow.dataset.study.config = get_config()
    create_empty_dataset(workflow.dataset.study.layout.dpath_root)
    create_pipeline_config_files(
        workflow.dataset.study.layout.dpath_pipelines,
        bidsification_pipelines=[
            {
                "NAME": "heudiconv",
                "VERSION": "0.12.2",
                "STEPS": [
                    {"NAME": "prepare"},
                    {"NAME": "convert", "UPDATE_STATUS": True},
                ],
            },
            {
                "NAME": "dcm2bids",
                "VERSION": "3.1.0",
                "STEPS": [
                    {"NAME": "prepare"},
                    {"NAME": "convert", "UPDATE_STATUS": True},
                ],
            },
        ],
    )
    return workflow


@pytest.mark.parametrize(
    "pipeline_name,expected_version",
    [
        ("heudiconv", "0.12.2"),
        ("dcm2bids", "3.1.0"),
    ],
)
def test_check_pipeline_version(
    pipeline_name,
    expected_version,
    workflow: BIDSificationRunner,
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = None
    workflow.pipeline.resolve_version()
    assert workflow.pipeline.version == expected_version


@pytest.mark.parametrize(
    "pipeline_name,pipeline_version",
    [
        ("heudiconv", "0.12.2"),
        ("dcm2bids", "3.1.0"),
    ],
)
def test_pipeline_config(
    pipeline_name, pipeline_version, workflow: BIDSificationRunner
):
    workflow.pipeline.name = pipeline_name
    workflow.pipeline.version = pipeline_version
    assert isinstance(workflow.pipeline.config, BIDSificationPipelineConfig)


def test_dpath_pipeline_error(workflow: BIDSificationRunner):
    with pytest.raises(
        RuntimeError, match='"dpath_pipeline" attribute is not available for '
    ):
        workflow._unavailable_pipeline_path()


def test_setup(workflow: BIDSificationRunner):
    # check that the working directory is created
    assert not workflow.workspace.work.exists()
    workflow.run_setup()
    assert workflow.workspace.work.exists()


@pytest.mark.parametrize("update_status", [True, False])
def test_run_single(
    update_status, workflow: BIDSificationRunner, mocker: pytest_mock.MockerFixture
):
    workflow.dataset.curation_status_table = CurationStatusTable()
    workflow.pipeline.step.UPDATE_STATUS = update_status

    mocked_execute = mocker.patch.object(workflow, "execute_pipeline")

    mocked_set_status = mocker.patch.object(
        workflow.dataset.curation_status_table, "set_status"
    )

    workflow.execute_item(WorkItem("01", "1"))

    mocked_execute.assert_called_once_with(
        WorkItem("01", "1"),
        [
            workflow.dataset.study.layout.dpath_post_reorg,
            workflow.dataset.study.layout.dpath_bids,
        ],
    )

    if update_status:
        mocked_set_status.assert_called_once()
    else:
        mocked_set_status.assert_not_called()


@pytest.mark.parametrize(
    "table",
    [
        CurationStatusTable(),
        CurationStatusTable(
            data={
                CurationStatusTable.col_participant_id: ["01"],
                CurationStatusTable.col_visit_id: ["1"],
                CurationStatusTable.col_session_id: ["1"],
                CurationStatusTable.col_datatype: "['anat']",
                CurationStatusTable.col_participant_dicom_dir: ["01"],
                CurationStatusTable.col_in_pre_reorg: [True],
                CurationStatusTable.col_in_post_reorg: [True],
                CurationStatusTable.col_in_bids: [True],
            }
        ).validate(),
    ],
)
def test_write_status_file(table: CurationStatusTable, workflow: BIDSificationRunner):
    workflow.pipeline.step_name = "convert"
    workflow.dataset.curation_status_table = table

    workflow._write_status_file()

    assert workflow.dataset.study.layout.fpath_curation_status.exists()
    assert CurationStatusTable.load(
        workflow.dataset.study.layout.fpath_curation_status
    ).equals(table)


def test_write_status_file_simulate(workflow: BIDSificationRunner):
    workflow.pipeline.step_name = "convert"
    workflow.execution = replace(workflow.execution, simulate=True)
    workflow.dataset.curation_status_table = CurationStatusTable()

    workflow._write_status_file()

    assert not workflow.dataset.study.layout.fpath_curation_status.exists()


def test_write_status_file_no_update(workflow: BIDSificationRunner):
    workflow.pipeline.step_name = "prepare"
    workflow.dataset.curation_status_table = CurationStatusTable()

    workflow._write_status_file()

    assert not workflow.dataset.study.layout.fpath_curation_status.exists()


def test_run_main(workflow: BIDSificationRunner, mocker: pytest_mock.MockerFixture):
    mocked_write_status_file = mocker.patch.object(workflow, "_write_status_file")
    mocker.patch.object(workflow, "select_candidates", return_value=[("01", "1")])
    workflow.run_main()
    mocked_write_status_file.assert_called_once()


@pytest.mark.parametrize(
    "status_data,participant_id,session_id,expected",
    [
        (
            [
                ["S01", "1", True, False],
                ["S01", "2", True, True],
                ["S02", "3", False, False],
            ],
            None,
            None,
            [("S01", "1")],
        ),
        (
            [
                ["P01", "A", True, False],
                ["P01", "B", True, False],
                ["P02", "B", True, False],
            ],
            "P01",
            "B",
            [("P01", "B")],
        ),
    ],
)
def test_get_participants_sessions_to_run(
    status_data, participant_id, session_id, expected, workflow: BIDSificationRunner
):
    workflow.dataset.curation_status_table = (
        CurationStatusTable().add_or_update_records(
            records=[
                {
                    CurationStatusTable.col_participant_id: data[0],
                    CurationStatusTable.col_session_id: data[1],
                    CurationStatusTable.col_in_post_reorg: data[2],
                    CurationStatusTable.col_in_bids: data[3],
                    CurationStatusTable.col_visit_id: data[1],
                    CurationStatusTable.col_datatype: None,
                    CurationStatusTable.col_participant_dicom_dir: "",
                    CurationStatusTable.col_in_pre_reorg: False,
                }
                for data in status_data
            ]
        )
    )
    workflow.selection = SelectionOptions(participant_id, session_id)
    assert list(workflow.select_candidates()) == expected


def test_generate_cli_command_for_hpc(
    workflow: BIDSificationRunner,
    mocker: pytest_mock.MockFixture,
):
    assert workflow.hpc_extra_flags() == []
