"""Tests for the PipelineTracker class."""

import json
import logging
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
import pytest_mock

from nipoppy.config.pipeline_step import AnalysisLevelType
from nipoppy.env import DEFAULT_PIPELINE_STEP_NAME
from nipoppy.exceptions import InvalidArgumentError, ReturnCode
from nipoppy.tabular.curation_status import CurationStatusTable
from nipoppy.tabular.manifest import Manifest
from nipoppy.tabular.processing_status import ProcessingStatusTable
from nipoppy.workflows import base
from nipoppy.workflows.services.archive import tar_directory
from nipoppy.workflows.services.execution import BatchResult
from nipoppy.workflows.tracker import PipelineTracker
from tests.conftest import (
    MOCKED_DATETIME,
    create_empty_dataset,
    create_pipeline_config_files,
    get_config,
    prepare_dataset,
)


@pytest.fixture(scope="function")
def tracker(tmp_path: Path):
    participants_and_sessions = {
        "01": ["1", "2"],
        "02": ["1", "2"],
    }

    tracker = PipelineTracker(
        dpath_root=tmp_path / "my_dataset",
        pipeline_name="test_pipeline",
        pipeline_version="0.1.0",
        pipeline_step=DEFAULT_PIPELINE_STEP_NAME,
    )

    create_empty_dataset(tracker.dpath_root)

    manifest: Manifest = prepare_dataset(
        participants_and_sessions_manifest=participants_and_sessions,
        participants_and_sessions_bidsified=participants_and_sessions,
    )
    manifest.save_with_backup(tracker.study.layout.fpath_manifest)

    tracker.study.config = get_config()

    fname_tracker_config = "tracker_config.json"
    create_pipeline_config_files(
        tracker.study.layout.dpath_pipelines,
        processing_pipelines=[
            {
                "NAME": tracker.pipeline.name,
                "VERSION": tracker.pipeline.version,
                "STEPS": [
                    {
                        "NAME": tracker.pipeline.step_name,
                        "TRACKER_CONFIG_FILE": fname_tracker_config,
                    }
                ],
            },
        ],
    )
    tracker_config = {
        "PATHS": [
            "[[NIPOPPY_PARTICIPANT_ID]]/[[NIPOPPY_BIDS_SESSION_ID]]/results.txt",
            "file.txt",
        ],
        "PARTICIPANT_SESSION_DIR": "[[NIPOPPY_PARTICIPANT_ID]]/[[NIPOPPY_BIDS_SESSION_ID]]",
    }

    (tracker.pipeline.bundle / fname_tracker_config).write_text(
        json.dumps(tracker_config)
    )

    return tracker


@pytest.mark.parametrize(
    "analysis_level",
    tuple(AnalysisLevelType),
)
def test_apply_analysis_level_success(analysis_level):
    participants_sessions = [("S01", "BL"), ("S01", "FU"), ("S02", "BL"), ("S02", "FU")]
    assert (
        PipelineTracker.apply_analysis_level(participants_sessions, analysis_level)
        == participants_sessions
    )


def test_run_setup(tracker: PipelineTracker):
    tracker.run_setup()
    assert tracker.processing_status_table.empty
    assert not tracker.derivatives.exists(), "Tracker should not create directory"


def test_run_setup_existing_processing_status_file(tracker: PipelineTracker):
    processing_status_table = ProcessingStatusTable(
        data={
            ProcessingStatusTable.col_participant_id: ["01"],
            ProcessingStatusTable.col_session_id: ["1"],
            ProcessingStatusTable.col_pipeline_name: ["some_pipeline"],
            ProcessingStatusTable.col_pipeline_version: ["some_version"],
            ProcessingStatusTable.col_pipeline_step: ["some_step"],
            ProcessingStatusTable.col_status: [ProcessingStatusTable.status_success],
        }
    ).validate()
    processing_status_table.save_with_backup(
        tracker.study.layout.fpath_processing_status
    )

    tracker.run_setup()

    assert tracker.processing_status_table.equals(processing_status_table)


@pytest.mark.no_xdist
def test_run_setup_existing_bad_processing_status_file(
    tracker: PipelineTracker, caplog: pytest.LogCaptureFixture
):
    # processing status file with wrong columns
    bad_processing_status_table = pd.DataFrame([{"col1": "val1"}])
    bad_processing_status_table.to_csv(
        tracker.study.layout.fpath_processing_status, index=False
    )

    tracker.run_setup()

    assert any(
        [
            record.levelno == logging.WARNING
            and "Failed to load existing processing status file at " in record.message
            for record in caplog.records
        ]
    )
    assert tracker.processing_status_table.empty


@pytest.mark.parametrize(
    "relative_paths,expected_status",
    [
        (
            ["dirA/01_ses-1.txt", "dirA/dirB/file.txt"],
            ProcessingStatusTable.status_success,
        ),
        (["**/*.txt"], ProcessingStatusTable.status_success),
        (["*file.txt"], ProcessingStatusTable.status_fail),
        (
            ["dirA/01_ses-1.txt", "dirA/dirB/file.txt", "missing.txt"],
            ProcessingStatusTable.status_fail,
        ),
    ],
)
def test_check_status(tracker: PipelineTracker, relative_paths, expected_status):
    for relative_path_to_write in ["dirA/01_ses-1.txt", "dirA/dirB/file.txt"]:
        fpath = tracker.output / relative_path_to_write
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.touch()

    assert tracker.check_status(relative_paths) == expected_status


@pytest.mark.parametrize(
    "relative_paths,relative_dpath_to_tar,expected_status",
    [
        (
            ["dirA/01_ses-1.txt", "dirA/dirB/file.txt"],
            "dirA",
            ProcessingStatusTable.status_success,
        ),
        (
            ["dirA/*.txt", "dirA/dirB/file.txt"],
            "dirA",
            ProcessingStatusTable.status_success,
        ),
        (["**/*.txt"], "dirA", ProcessingStatusTable.status_success),
        # # FAILING, see note in check_status
        # (["*file.txt"], "dirA", ProcessingStatus.status_fail),
        (
            ["dirA/01_ses-1.txt", "dirA/dirB/file.txt", "missing.txt"],
            "dirA",
            ProcessingStatusTable.status_fail,
        ),
        (
            ["dirA/01_ses-1.txt", "dirA/dirB/file.txt"],
            "dirA/dirB",
            ProcessingStatusTable.status_success,
        ),
    ],
)
def test_check_status_with_tarball(
    tracker: PipelineTracker,
    relative_paths: list[str],
    relative_dpath_to_tar: str,
    expected_status,
):
    for relative_path_to_write in ["dirA/01_ses-1.txt", "dirA/dirB/file.txt"]:
        fpath = tracker.output / relative_path_to_write
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.touch()

    # use PipelineRunner to tar the directory
    dpath_to_tar = tracker.output / relative_dpath_to_tar
    tar_directory(dpath_to_tar)

    assert not dpath_to_tar.exists()
    assert (
        tracker.check_status(relative_paths, relative_dpath_to_tar) == expected_status
    )


@pytest.mark.parametrize(
    "curation_status_data,participant_id,session_id,expected",
    [
        (
            [
                ["S01", "1", False],
                ["S01", "2", True],
                ["S02", "3", False],
            ],
            None,
            None,
            [("S01", "2")],
        ),
        (
            [
                ["P01", "A", False],
                ["P01", "B", True],
                ["P02", "B", True],
            ],
            "P01",
            "B",
            [("P01", "B")],
        ),
    ],
)
def test_get_participants_sessions_to_run(
    curation_status_data, participant_id, session_id, expected, tmp_path: Path
):
    tracker = PipelineTracker(
        dpath_root=tmp_path,
        pipeline_name="",
        pipeline_version="",
    )
    tracker.curation_status_table = CurationStatusTable().add_or_update_records(
        records=[
            {
                CurationStatusTable.col_participant_id: data[0],
                CurationStatusTable.col_session_id: data[1],
                CurationStatusTable.col_visit_id: data[1],
                CurationStatusTable.col_in_bids: data[2],
                CurationStatusTable.col_datatype: None,
                CurationStatusTable.col_participant_dicom_dir: "",
                CurationStatusTable.col_in_pre_reorg: False,
                CurationStatusTable.col_in_post_reorg: False,
            }
            for data in curation_status_data
        ]
    )

    assert [
        tuple(x)
        for x in tracker.get_participants_sessions_to_run(
            participant_id=participant_id, session_id=session_id
        )
    ] == expected


@pytest.mark.parametrize(
    "participant_id,session_id,expected_status",
    [
        ("01", "1", ProcessingStatusTable.status_success),
        ("02", "2", ProcessingStatusTable.status_fail),
    ],
)
def test_run_single(
    participant_id, session_id, expected_status, tracker: PipelineTracker
):
    for relative_path_to_write in [
        "01/ses-1/results.txt",
        "file.txt",
        "02/ses-1/results.txt",
    ]:
        fpath = tracker.output / relative_path_to_write
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.touch()

    for relative_dpath_to_tar in ["01/ses-1", "02/ses-1"]:
        dpath_to_tar = tracker.output / relative_dpath_to_tar
        tar_directory(dpath_to_tar)

    assert not dpath_to_tar.exists()
    assert (
        tracker.run_single(participant_id, session_id)[ProcessingStatusTable.col_status]
        == expected_status
    )


def test_run_single_no_config(tracker: PipelineTracker):
    tracker.pipeline.config.STEPS[0].TRACKER_CONFIG_FILE = None
    with pytest.raises(ValueError, match="No tracker config file specified for"):
        tracker.run_single("01", "1")


@pytest.mark.parametrize(
    "records,expected_processing_status_table",
    [
        ([], ProcessingStatusTable()),
        (
            [
                {
                    ProcessingStatusTable.col_participant_id: "01",
                    ProcessingStatusTable.col_session_id: "1",
                    ProcessingStatusTable.col_pipeline_name: "some_pipeline",
                    ProcessingStatusTable.col_pipeline_version: "some_version",
                    ProcessingStatusTable.col_pipeline_step: "some_step",
                    ProcessingStatusTable.col_status: ProcessingStatusTable.status_success,  # noqa: E501
                }
            ],
            ProcessingStatusTable(
                data={
                    ProcessingStatusTable.col_participant_id: ["01"],
                    ProcessingStatusTable.col_session_id: ["1"],
                    ProcessingStatusTable.col_pipeline_name: ["some_pipeline"],
                    ProcessingStatusTable.col_pipeline_version: ["some_version"],
                    ProcessingStatusTable.col_pipeline_step: ["some_step"],
                    ProcessingStatusTable.col_status: [
                        ProcessingStatusTable.status_success
                    ],
                }
            ).validate(),
        ),
    ],
)
def test_update_status_file(
    tracker: PipelineTracker,
    records,
    expected_processing_status_table: ProcessingStatusTable,
):
    tracker.result = BatchResult.from_results([(True, record) for record in records])
    tracker._update_status_file()

    assert tracker.study.layout.fpath_processing_status.exists()
    assert ProcessingStatusTable.load(
        tracker.study.layout.fpath_processing_status
    ).equals(expected_processing_status_table)


def test_run_main(tracker: PipelineTracker, mocker: pytest_mock.MockFixture):
    mocked_update_status_file = mocker.patch.object(tracker, "_update_status_file")
    tracker.run_main()
    mocked_update_status_file.assert_called_once()


def test_run_no_create_work_directory(tracker: PipelineTracker):
    tracker.run()
    assert not tracker.workspace.work.exists()


def test_run_no_rm_work_directory(tracker: PipelineTracker):
    tracker.workspace.work.mkdir(parents=True)
    tracker.run()
    assert tracker.workspace.work.exists()


def test_construction_is_lazy_and_reuses_study(tmp_path, mocker):
    opened = mocker.spy(base, "open_dataset")
    for loader in [
        "load_pipeline_config",
        "load_tracker_config",
        "load_descriptor",
        "load_invocation",
    ]:
        mocker.patch(
            f"nipoppy.workflows.services.artifacts.{loader}",
            side_effect=AssertionError("Construction must not load artifacts"),
        )
    tracker = PipelineTracker(tmp_path / "missing", "uninstalled")

    assert "uninstalled" in str(tracker)
    opened.assert_called_once()
    assert tracker.dataset.study is tracker.study
    assert tracker.pipeline.study is tracker.study
    assert tracker.workspace.layout is tracker.study.layout
    assert "config" not in tracker.study.__dict__
    assert "manifest" not in tracker.study.__dict__
    assert "processing_status_table" not in tracker.__dict__
    assert "derivatives" not in tracker.__dict__
    assert "output" not in tracker.__dict__
    assert tracker.result.n_total == 0
    assert tracker.execution.skip_logfile and tracker.execution.show_progress
    assert not tracker.dpath_root.exists()


@pytest.mark.parametrize("version", [None, "0.1.0"])
@pytest.mark.parametrize("step", [None, DEFAULT_PIPELINE_STEP_NAME])
def test_setup_resolves_omitted_identity(tracker, version, step, mocker):
    create_pipeline_config_files(
        tracker.study.layout.dpath_pipelines,
        processing_pipelines=[
            {
                "NAME": tracker.pipeline.name,
                "VERSION": "0.2.0",
                "STEPS": [
                    {"NAME": "track"},
                    {"NAME": DEFAULT_PIPELINE_STEP_NAME},
                ],
            }
        ],
    )
    workflow = PipelineTracker(tracker.dpath_root, tracker.pipeline.name, version, step)
    workflow.study.config = tracker.study.config
    file_handler = mocker.patch("nipoppy.workflows.base.logger.add_file_handler")
    workflow.run_setup()

    assert workflow.pipeline.version == (version or "0.2.0")
    assert workflow.pipeline.step_name == (
        "track" if version is None and step is None else DEFAULT_PIPELINE_STEP_NAME
    )
    assert workflow.processing_status_table.empty
    assert workflow.output == workflow.study.layout.get_dpath_pipeline_output(
        workflow.pipeline.name, workflow.pipeline.version
    )
    file_handler.assert_not_called()


def _set_bidsified_pairs(tracker):
    tracker.curation_status_table = CurationStatusTable().add_or_update_records(
        [
            {
                "participant_id": participant,
                "session_id": session,
                "visit_id": session,
                "datatype": None,
                "participant_dicom_dir": "",
                "in_pre_reorg": False,
                "in_post_reorg": False,
                "in_bids": True,
            }
            for participant, session in [
                ("01", "1"),
                ("01", "2"),
                ("02", "1"),
                ("02", "2"),
            ]
        ]
    )


@pytest.mark.parametrize("analysis_level", tuple(AnalysisLevelType))
@pytest.mark.parametrize("dry_run", [False, True])
def test_run_persists_complete_ordered_status_serial_and_parallel(
    tracker, analysis_level, dry_run
):
    _set_bidsified_pairs(tracker)
    tracker.pipeline.step.ANALYSIS_LEVEL = analysis_level
    tracker.dry_run = dry_run
    for path in ["01/ses-1/results.txt", "file.txt"]:
        output = tracker.output / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    results = []
    for n_jobs in [1, 2]:
        tracker.execution = replace(tracker.execution, n_jobs=n_jobs)
        tracker.run()
        results.append(tracker.result)
        assert tracker.return_code == ReturnCode.SUCCESS
        assert tracker.result.n_success == tracker.result.n_total == 4
        assert [
            (record["participant_id"], record["session_id"])
            for record in tracker.result.payloads
        ] == [("01", "1"), ("01", "2"), ("02", "1"), ("02", "2")]
        assert [record["status"] for record in tracker.result.payloads] == [
            ProcessingStatusTable.status_success,
            *[ProcessingStatusTable.status_fail] * 3,
        ]
        expected = ProcessingStatusTable().add_or_update_records(
            tracker.result.payloads
        )
        assert tracker.processing_status_table.equals(expected)
        assert tracker.study.layout.fpath_processing_status.exists() == (not dry_run)
        if not dry_run:
            saved = ProcessingStatusTable.load(
                tracker.study.layout.fpath_processing_status
            )
            pd.testing.assert_frame_equal(saved, expected[saved.columns])
    assert results[0] == results[1]


@pytest.mark.parametrize("analysis_level", tuple(AnalysisLevelType))
def test_run_empty_selection(tracker, analysis_level):
    tracker.pipeline.step.ANALYSIS_LEVEL = analysis_level
    tracker.curation_status_table = CurationStatusTable()
    tracker.run()

    assert tracker.result.n_total == tracker.result.n_success == 0
    assert tracker.result.payloads == ()
    assert tracker.return_code == ReturnCode.NO_PARTICIPANTS_OR_SESSIONS_TO_RUN
    assert ProcessingStatusTable.load(
        tracker.study.layout.fpath_processing_status
    ).empty


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_run_main_isolates_item_exception(tracker, n_jobs, mocker):
    _set_bidsified_pairs(tracker)
    tracker.execution = replace(tracker.execution, n_jobs=n_jobs)
    run_single = tracker.run_single

    def execute(participant, session):
        if (participant, session) == ("01", "2"):
            raise RuntimeError("item error")
        return run_single(participant, session)

    mocker.patch.object(tracker, "run_single", side_effect=execute)
    # Test execution independently of the existing persistence policy for None payloads.
    update = mocker.patch.object(tracker, "_update_status_file")
    tracker.run()

    assert tracker.result.n_total == 4
    assert tracker.result.n_success == 3
    assert tracker.result.payloads[1] is None
    assert tracker.result.payloads[2]["participant_id"] == "02"
    assert tracker.return_code == ReturnCode.PARTIAL_SUCCESS
    update.assert_called_once()


def test_run_retains_status_for_other_pipelines_and_backup(tracker, datetime_fixture):
    _set_bidsified_pairs(tracker)
    original = ProcessingStatusTable().add_or_update_records(
        {
            "participant_id": "01",
            "session_id": "1",
            "pipeline_name": "other_pipeline",
            "pipeline_version": "1.0",
            "pipeline_step": "other_step",
            "status": ProcessingStatusTable.status_success,
        }
    )
    original.save_with_backup(tracker.study.layout.fpath_processing_status)
    previous_backup = tracker.study.layout.fpath_processing_status.resolve()
    previous_bytes = previous_backup.read_bytes()
    datetime_fixture.datetime.now.return_value = MOCKED_DATETIME + timedelta(minutes=1)
    tracker.run()

    saved = ProcessingStatusTable.load(tracker.study.layout.fpath_processing_status)
    assert len(saved) == 5
    assert (
        saved[saved["pipeline_name"] == "other_pipeline"]
        .reset_index(drop=True)
        .equals(original)
    )
    assert tracker.study.layout.fpath_processing_status.resolve() != previous_backup
    assert previous_backup.read_bytes() == previous_bytes


@pytest.mark.parametrize(
    "analysis_level,participant,session",
    [
        (AnalysisLevelType.group, "01", None),
        (AnalysisLevelType.participant, None, "1"),
        (AnalysisLevelType.session, "01", None),
    ],
)
def test_setup_rejects_incompatible_filters(
    tracker, analysis_level, participant, session
):
    tracker.pipeline.step.ANALYSIS_LEVEL = analysis_level
    tracker.selection = replace(
        tracker.selection, participant_id=participant, session_id=session
    )
    with pytest.raises(InvalidArgumentError):
        tracker.run_setup()


@pytest.mark.parametrize("stage", ["run_setup", "run_main"])
def test_run_cleans_up_after_failure(tracker, stage, mocker):
    mocker.patch.object(tracker, stage, side_effect=RuntimeError("workflow error"))
    cleanup = mocker.patch.object(tracker, "run_cleanup")
    with pytest.raises(RuntimeError, match="workflow error"):
        tracker.run()
    cleanup.assert_called_once()
