"""Characterize pipeline behavior through the CLI and resulting files.

The external Boutiques command and scheduler adapter are replaced, and dotenv
loading is isolated. Selection, configuration, validation, rendering, status
persistence, and cleanup remain real.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tarfile
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from nipoppy.cli.cli import cli
from nipoppy.cli.groups import _load_dotenv_files
from nipoppy.config.main import Config
from nipoppy.env import PipelineTypeEnum
from nipoppy.exceptions import ReturnCode
from nipoppy.layout import DatasetLayout
from nipoppy.tabular.curation_status import CurationStatusTable
from nipoppy.tabular.processing_status import ProcessingStatusTable
from tests.conftest import (
    MOCKED_DATETIME,
    create_empty_dataset,
    create_pipeline_config_files,
    get_config,
    prepare_dataset,
)

PIPELINES = {"process": "processor", "bidsify": "converter", "extract": "extractor"}
PIPELINE_TYPES = {
    "process": PipelineTypeEnum.PROCESSING,
    "bidsify": PipelineTypeEnum.BIDSIFICATION,
    "extract": PipelineTypeEnum.EXTRACTION,
}
ELIGIBLE = {
    "process": [("01", "B"), ("02", "A")],
    "bidsify": [("03", "A"), ("04", "A")],
    "extract": [("01", "B"), ("02", "A")],
}


@pytest.fixture(params=tuple(PIPELINES))
def command(request):
    return request.param


@pytest.fixture
def workflow_dataset(tmp_path, monkeypatch):
    """Use the existing dataset helpers to install all three pipeline types."""
    # Do not load the user's dotenv files or leak variables between CLI invocations.
    monkeypatch.setattr("nipoppy.cli.groups._load_dotenv_files", lambda root: None)
    monkeypatch.delenv("NIPOPPY_LAYOUT", raising=False)
    layout = DatasetLayout(tmp_path / "study")
    create_empty_dataset(layout.dpath_root)
    get_config(container_config={"COMMAND": None}).save(layout.fpath_config)
    participants = {"01": ["A", "B"], **{p: ["A"] for p in ("02", "03", "04", "05")}}
    manifest = prepare_dataset(participants_and_sessions_manifest=participants)
    manifest.save_with_backup(layout.fpath_manifest)
    CurationStatusTable(
        [
            {
                "participant_id": participant,
                "visit_id": session,
                "session_id": session,
                "datatype": ["anat"],
                "participant_dicom_dir": f"{participant}/{session}",
                "in_pre_reorg": True,
                "in_post_reorg": participant != "05",
                "in_bids": participant in ("01", "02"),
            }
            for participant, sessions in participants.items()
            for session in sessions
        ]
    ).validate().save_with_backup(layout.fpath_curation_status)
    completed = {
        "processor": [("01", "A")],
        "dependency-one": [("01", "A"), ("01", "B"), ("02", "A")],
        "dependency-two": [("01", "B"), ("02", "A"), ("03", "A")],
    }
    ProcessingStatusTable(
        [
            {
                "participant_id": participant,
                "session_id": session,
                "pipeline_name": name,
                "pipeline_version": "2.0.0",
                "pipeline_step": "run",
                "status": "SUCCESS",
            }
            for name, pairs in completed.items()
            for participant, session in pairs
        ]
    ).validate().save_with_backup(layout.fpath_processing_status)
    common_step = {
        "DESCRIPTOR_FILE": "descriptor.json",
        "INVOCATION_FILE": "invocation.json",
    }
    configs = {}
    for command_name, pipeline_type in (
        ("process", PipelineTypeEnum.PROCESSING),
        ("bidsify", PipelineTypeEnum.BIDSIFICATION),
        ("extract", PipelineTypeEnum.EXTRACTION),
    ):
        pipelines = []
        for version in ("1.0.0", "2.0.0"):
            step = dict(common_step, NAME="run")
            if command_name == "process":
                step.update(
                    GENERATE_PYBIDS_DATABASE=False, TRACKER_CONFIG_FILE="tracker.json"
                )
            if command_name == "bidsify":
                step["UPDATE_STATUS"] = True
            config = {
                "NAME": PIPELINES[command_name],
                "VERSION": version,
                "STEPS": [step, dict(step, NAME="followup")],
            }
            if command_name == "extract":
                config["PROC_DEPENDENCIES"] = [
                    {"NAME": name, "VERSION": "2.0.0", "STEP": "run"}
                    for name in ("dependency-one", "dependency-two")
                ]
            pipelines.append(config)
        configs[pipeline_type] = pipelines
    configs[PipelineTypeEnum.PROCESSING].extend(
        {
            "NAME": name,
            "VERSION": "2.0.0",
            "STEPS": [{"NAME": "run", "GENERATE_PYBIDS_DATABASE": False}],
        }
        for name in ("dependency-one", "dependency-two")
    )
    create_pipeline_config_files(
        layout.dpath_pipelines,
        processing_pipelines=configs[PipelineTypeEnum.PROCESSING],
        bidsification_pipelines=configs[PipelineTypeEnum.BIDSIFICATION],
        extraction_pipelines=configs[PipelineTypeEnum.EXTRACTION],
    )
    descriptor = {
        "name": "contract-test",
        "tool-version": "1.0",
        "description": "Pipeline workflow compatibility fixture",
        "schema-version": "0.5",
        "command-line": "echo [PARTICIPANT] [SESSION] [OUTPUT]",
        "inputs": [
            {"id": key, "name": key, "type": "String", "value-key": f"[{key.upper()}]"}
            for key in ("participant", "session", "output")
        ],
    }
    for command_name, pipeline_type in (
        ("process", PipelineTypeEnum.PROCESSING),
        ("bidsify", PipelineTypeEnum.BIDSIFICATION),
        ("extract", PipelineTypeEnum.EXTRACTION),
    ):
        output_key = {
            "process": "DPATH_PIPELINE_OUTPUT",
            "bidsify": "DPATH_BIDS",
            "extract": "DPATH_PIPELINE_IDP",
        }[command_name]
        for version in ("1.0.0", "2.0.0"):
            bundle = layout.get_dpath_pipeline_bundle(
                pipeline_type, PIPELINES[command_name], version
            )
            (bundle / "descriptor.json").write_text(json.dumps(descriptor))
            (bundle / "invocation.json").write_text(
                json.dumps(
                    {
                        "participant": "[[NIPOPPY_PARTICIPANT_ID]]",
                        "session": "[[NIPOPPY_SESSION_ID]]",
                        "output": f"[[NIPOPPY_{output_key}]]",
                    }
                )
            )
            if command_name == "process":
                (bundle / "tracker.json").write_text(
                    json.dumps(
                        {
                            "SCHEMA_VERSION": "1.0",
                            "PATHS": [
                                "[[NIPOPPY_PARTICIPANT_ID]]_[[NIPOPPY_SESSION_ID]]/result.txt"
                            ],
                            "PARTICIPANT_SESSION_DIR": (
                                "[[NIPOPPY_PARTICIPANT_ID]]_[[NIPOPPY_SESSION_ID]]"
                            ),
                        }
                    )
                )
    return layout


@pytest.fixture
def bosh_commands(monkeypatch):
    commands = []

    def record(args, *, quiet, dry_run):
        commands.append((args, dry_run))
        return 0

    monkeypatch.setattr("nipoppy.workflows.runner._run_command", record)
    return commands


def _invoke(layout, command, *options):
    return CliRunner().invoke(
        cli,
        [
            command,
            "--study",
            str(layout.dpath_root),
            "--pipeline",
            PIPELINES[command],
            *options,
        ],
    )


def _invocations(commands):
    return [
        json.loads(args[4] if args[2] == "simulate" else args[5])
        for args, _ in commands
    ]


def _set_step(layout, command, **changes):
    pipeline_type = PIPELINE_TYPES[command]
    path = (
        layout.get_dpath_pipeline_bundle(pipeline_type, PIPELINES[command], "2.0.0")
        / "config.json"
    )
    config = json.loads(path.read_text())
    config["STEPS"][0].update(changes)
    path.write_text(json.dumps(config))


def _work_paths(layout):
    return [
        layout.get_dpath_pipeline_work("processor", "2.0.0", "run"),
        layout.get_dpath_pybids_db("processor", "2.0.0", "run"),
    ]


def test_cli_eligibility_defaults_and_rendered_commands(
    workflow_dataset, command, bosh_commands
):
    layout = workflow_dataset
    result = _invoke(layout, command)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    invocations = _invocations(bosh_commands)
    assert [(i["participant"], i["session"]) for i in invocations] == ELIGIBLE[command]
    expected_output = {
        "process": layout.dpath_derivatives / "processor/2.0.0/output",
        "bidsify": layout.dpath_bids,
        "extract": layout.dpath_derivatives / "dependency-one/2.0.0/idp",
    }[command]
    assert {i["output"] for i in invocations} == {str(expected_output)}
    for args, dry_run in bosh_commands:
        assert args[:4] == ["bosh", "exec", "launch", "--stream"]
        assert args[6:] == ["--no-pull", "--no-container"]
        assert (
            json.loads(args[4])["command-line"]
            == "echo [PARTICIPANT] [SESSION] [OUTPUT]"
        )
        assert not dry_run
    assert list(
        (
            layout.dpath_logs / "nipoppy" / command / f"{PIPELINES[command]}-2.0.0-run"
        ).glob("*.log")
    )


@pytest.mark.parametrize("mode", ["normal", "dry-run", "simulate", "both"])
def test_cli_mode_status_and_cleanup(workflow_dataset, command, bosh_commands, mode):
    layout = workflow_dataset
    original_status = layout.fpath_curation_status.read_bytes()
    options = []
    if mode in ("dry-run", "both"):
        options.append("--dry-run")
    if mode in ("simulate", "both"):
        options.append("--simulate")
    result = _invoke(layout, command, *options)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert len(bosh_commands) == 2
    assert {args[2] for args, _ in bosh_commands} == {
        "simulate" if "--simulate" in options else "launch"
    }
    assert {dry_run for _, dry_run in bosh_commands} == {"--dry-run" in options}
    status = CurationStatusTable.load(layout.fpath_curation_status)
    for participant in ("03", "04"):
        assert bool(status.get_status(participant, "A", "in_bids")) == (
            command == "bidsify" and mode == "normal"
        )
    if command != "bidsify" or mode != "normal":
        assert layout.fpath_curation_status.read_bytes() == original_status
    assert layout.fpath_curation_status.is_symlink()
    if command == "process":
        assert all(not p.exists() for p in _work_paths(layout))
    else:
        work = layout.get_dpath_pipeline_work(PIPELINES[command], "2.0.0", "run")
        assert work.exists() == ("--dry-run" not in options)


@pytest.mark.parametrize("failures", [set(), {0}, {0, 1}])
def test_cli_item_failures_continue_and_preserve_status(
    workflow_dataset, command, monkeypatch, failures
):
    layout = workflow_dataset
    invocations = []

    def execute(args, *, quiet, dry_run):
        index = len(invocations)
        invocations.append(json.loads(args[5]))
        if index in failures:
            raise subprocess.CalledProcessError(7, args)
        return 0

    monkeypatch.setattr("nipoppy.workflows.runner._run_command", execute)
    result = _invoke(layout, command)
    assert result.exit_code == (
        ReturnCode.PARTIAL_SUCCESS if failures else ReturnCode.SUCCESS
    ), result.output
    assert [(i["participant"], i["session"]) for i in invocations] == ELIGIBLE[command]
    if command == "process":
        assert _work_paths(layout)[0].exists() == bool(failures)
    if command == "bidsify":
        status = CurationStatusTable.load(layout.fpath_curation_status)
        for index, participant in enumerate(("03", "04")):
            assert bool(status.get_status(participant, "A", "in_bids")) == (
                index not in failures
            )


@pytest.mark.parametrize("export_flag", ["--write-subcohort", "--write-list"])
def test_cli_export_filters_eligibility_before_writing(
    workflow_dataset, command, bosh_commands, export_flag, tmp_path
):
    layout = workflow_dataset
    source = tmp_path / "selection.tsv"
    source.write_text("01\tA\n01\tB\n03\tA\n99\tA\n")
    output = tmp_path / "selected.tsv"
    original_status = layout.fpath_curation_status.read_bytes()
    result = _invoke(
        layout, command, "--use-subcohort", str(source), export_flag, str(output)
    )
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert output.read_text() == ("03\tA\n" if command == "bidsify" else "01\tB\n")
    assert not bosh_commands
    assert layout.fpath_curation_status.read_bytes() == original_status


@pytest.mark.parametrize(
    "level,expected",
    [
        ("participant_session", [("01", "B"), ("02", "A")]),
        ("participant", [("01", "None"), ("02", "None")]),
        ("session", [("None", "B"), ("None", "A")]),
        ("group", [("None", "None")]),
    ],
)
def test_cli_analysis_levels(
    workflow_dataset, command, bosh_commands, level, expected, recwarn
):
    _set_step(
        workflow_dataset,
        command,
        ANALYSIS_LEVEL=level,
        **({"UPDATE_STATUS": False} if command == "bidsify" else {}),
    )
    result = _invoke(workflow_dataset, command)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    if command == "bidsify":
        expected = {
            "participant_session": [("03", "A"), ("04", "A")],
            "participant": [("03", "None"), ("04", "None")],
            "session": [("None", "A")],
            "group": [("None", "None")],
        }[level]
    assert [
        (i["participant"], i["session"]) for i in _invocations(bosh_commands)
    ] == expected
    expected_warnings = {
        "participant_session": set(),
        "participant": {"Replacing [[NIPOPPY_SESSION_ID]] with None"},
        "session": {"Replacing [[NIPOPPY_PARTICIPANT_ID]] with None"},
        "group": {
            "Replacing [[NIPOPPY_PARTICIPANT_ID]] with None",
            "Replacing [[NIPOPPY_SESSION_ID]] with None",
        },
    }
    assert {str(w.message) for w in recwarn} == expected_warnings[level]


@pytest.mark.parametrize("group", [False, True])
def test_cli_empty_selection_and_group_execution(
    workflow_dataset, command, bosh_commands, tmp_path, group, recwarn
):
    selection = tmp_path / "unmatched.tsv"
    selection.write_text("99\tZ\n")
    if group:
        _set_step(
            workflow_dataset,
            command,
            ANALYSIS_LEVEL="group",
            **({"UPDATE_STATUS": False} if command == "bidsify" else {}),
        )
    result = _invoke(workflow_dataset, command, "--use-subcohort", str(selection))
    assert result.exit_code == (
        ReturnCode.SUCCESS if group else ReturnCode.NO_PARTICIPANTS_OR_SESSIONS_TO_RUN
    ), result.output
    assert len(bosh_commands) == int(group)
    assert len(recwarn) == (2 if group else 0)


@pytest.mark.parametrize(
    "flag,value", [("--participant-id", "sub-01"), ("--session-id", "ses-A")]
)
def test_cli_group_rejects_incompatible_filters(
    workflow_dataset, command, bosh_commands, flag, value
):
    _set_step(
        workflow_dataset,
        command,
        ANALYSIS_LEVEL="group",
        **({"UPDATE_STATUS": False} if command == "bidsify" else {}),
    )
    result = _invoke(workflow_dataset, command, flag, value)
    assert result.exit_code == ReturnCode.INVALID_ARGUMENT, result.output
    assert not bosh_commands


@pytest.mark.parametrize("keep", [False, True])
def test_processing_setup_failure_cleanup(workflow_dataset, bosh_commands, keep):
    layout = workflow_dataset
    paths = _work_paths(layout)
    for path in paths:
        path.mkdir(parents=True)
        (path / "existing.txt").write_text("pre-existing work")
    bundle = layout.get_dpath_pipeline_bundle(
        PipelineTypeEnum.PROCESSING, "processor", "2.0.0"
    )
    (bundle / "descriptor.json").write_text("{}")
    result = _invoke(layout, "process", *(["--keep-workdir"] if keep else []))
    assert result.exit_code == ReturnCode.INVALID_CONFIG, result.output
    assert not bosh_commands
    # Existing behavior: zero successful / zero attempted items allows cleanup.
    assert all(path.exists() == keep for path in paths)


@pytest.mark.parametrize("mode", ["run", "export", "dry-run"])
@pytest.mark.parametrize("keep", [False, True])
def test_processing_existing_work_cleanup(
    workflow_dataset, bosh_commands, tmp_path, mode, keep
):
    layout = workflow_dataset
    paths = _work_paths(layout)
    for path in paths:
        path.mkdir(parents=True)
        (path / "existing.txt").write_text("pre-existing work")
    options = ["--keep-workdir"] if keep else []
    if mode == "export":
        options.extend(["--write-subcohort", str(tmp_path / "export.tsv")])
    if mode == "dry-run":
        options.append("--dry-run")
    result = _invoke(layout, "process", *options)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert all(path.exists() == (keep or mode == "dry-run") for path in paths)


@pytest.mark.parametrize("simulate", [False, True])
def test_processing_archives_only_after_real_launch(
    workflow_dataset, monkeypatch, simulate
):
    layout = workflow_dataset
    output = layout.dpath_derivatives / "processor/2.0.0/output"
    monkeypatch.setenv("COPYFILE_DISABLE", "1")

    def execute(args, *, quiet, dry_run):
        invocation = json.loads(args[4] if simulate else args[5])
        path = (
            output
            / f"{invocation['participant']}_{invocation['session']}"
            / "result.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pipeline result")
        return 0

    monkeypatch.setattr("nipoppy.workflows.runner._run_command", execute)
    result = _invoke(layout, "process", "--tar", *(["--simulate"] if simulate else []))
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    for participant, session in ELIGIBLE["process"]:
        directory = output / f"{participant}_{session}"
        archive = directory.with_suffix(".tar")
        assert archive.exists() == (not simulate)
        assert directory.exists() == simulate
        if not simulate:
            with tarfile.open(archive) as tar:
                assert {m.name for m in tar.getmembers() if m.isfile()} == {
                    f"{participant}_{session}/result.txt"
                }


@pytest.fixture
def queue_adapter(workflow_dataset, mocker):
    """Replace the scheduler only; run command generation and submission policy."""
    workflow_dataset.dpath_hpc.mkdir(parents=True)
    adapter = mocker.Mock()
    adapter.get_queue_status.return_value = []
    adapter.submit_job.return_value = "1234"
    mocker.patch("nipoppy.workflows.services.hpc.QueueAdapter", return_value=adapter)
    return adapter


def test_cli_hpc_child_commands_and_flags(
    workflow_dataset, command, bosh_commands, queue_adapter, tmp_path
):
    layout = workflow_dataset
    root_with_spaces = tmp_path / "study with spaces"
    shutil.move(layout.dpath_root, root_with_spaces)
    layout = DatasetLayout(root_with_spaces)
    options = ["--hpc", "slurm", "--simulate", "--keep-workdir", "--verbose"]
    if command == "process":
        options.append("--tar")
    result = _invoke(layout, command, *options)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert not bosh_commands
    queue_adapter.submit_job.assert_called_once()
    submission = queue_adapter.submit_job.call_args.kwargs
    assert (
        list(
            zip(
                submission["NIPOPPY_PARTICIPANT_IDS"], submission["NIPOPPY_SESSION_IDS"]
            )
        )
        == ELIGIBLE[command]
    )
    assert submission["NIPOPPY_PIPELINE_NAME"] == PIPELINES[command]
    assert submission["NIPOPPY_PIPELINE_VERSION"] == "2.0.0"
    assert submission["NIPOPPY_PIPELINE_STEP"] == "run"
    assert submission["NIPOPPY_DPATH_ROOT"] == layout.dpath_root
    assert Path(submission["working_directory"]) == layout.get_dpath_pipeline_work(
        PIPELINES[command], "2.0.0", "run"
    )
    assert submission["NIPOPPY_DPATH_LOGS"] == layout.dpath_logs / "hpc"
    for child, (participant, session) in zip(
        submission["NIPOPPY_COMMANDS"], ELIGIBLE[command]
    ):
        expected = [
            "nipoppy",
            command,
            "--study",
            str(layout.dpath_root),
            "--pipeline",
            PIPELINES[command],
            "--pipeline-version",
            "2.0.0",
            "--pipeline-step",
            "run",
            "--participant-id",
            participant,
            "--session-id",
            session,
            "--keep-workdir",
            "--verbose",
        ]
        if command == "process":
            expected.append("--tar")
        assert shlex.split(child) == expected


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_cli_hpc_capacity_counts_and_processing_cleanup(
    workflow_dataset, command, bosh_commands, queue_adapter, limit
):
    layout = workflow_dataset
    config = Config.load(layout.fpath_config)
    config.HPC_QUEUE_LIMIT = 2
    queue_adapter.get_queue_status.return_value = ["queued job"] * (2 - limit)
    config.save(layout.fpath_config)
    result = _invoke(layout, command, "--hpc", "slurm")
    # Existing behavior: a truncated/empty submission does not set partial success.
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert not bosh_commands
    if limit:
        queue_adapter.submit_job.assert_called_once()
        submission = queue_adapter.submit_job.call_args.kwargs
        assert len(submission["NIPOPPY_COMMANDS"]) == limit
        assert (
            list(
                zip(
                    submission["NIPOPPY_PARTICIPANT_IDS"],
                    submission["NIPOPPY_SESSION_IDS"],
                )
            )
            == ELIGIBLE[command][:limit]
        )
    else:
        queue_adapter.submit_job.assert_not_called()
    if command == "process":
        assert _work_paths(layout)[0].exists() == (limit != 2)


def test_cli_hpc_dry_run_side_effects(
    workflow_dataset, command, bosh_commands, queue_adapter
):
    layout = workflow_dataset
    work = layout.get_dpath_pipeline_work(PIPELINES[command], "2.0.0", "run")
    work.mkdir(parents=True)
    error = work / "pysqa.err"
    error.write_text("previous submission error")
    result = _invoke(layout, command, "--hpc", "slurm", "--dry-run")
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert not bosh_commands
    queue_adapter.submit_job.assert_not_called()
    queue_adapter.get_queue_status.assert_called_once()
    assert not error.exists()
    assert (layout.dpath_logs / "hpc").is_dir()
    assert work.exists()


def test_cli_hpc_submission_failure(
    workflow_dataset, command, bosh_commands, queue_adapter
):
    layout = workflow_dataset

    def fail(**kwargs):
        (Path(kwargs["working_directory"]) / "pysqa.err").write_text(
            "scheduler rejected job"
        )
        return None

    queue_adapter.submit_job.side_effect = fail
    result = _invoke(layout, command, "--hpc", "slurm")
    assert result.exit_code == ReturnCode.WORKFLOW_FAILURE, result.output
    assert not bosh_commands
    work = layout.get_dpath_pipeline_work(PIPELINES[command], "2.0.0", "run")
    assert work.exists() == (command != "process")


@pytest.mark.parametrize(
    "source", ["cli", "environment", "dotenv", "cli-over-environment"]
)
def test_cli_custom_layout_sources(
    workflow_dataset, command, bosh_commands, monkeypatch, tmp_path, source
):
    layout = workflow_dataset
    layout_file = tmp_path / "custom layout.json"
    specification = layout.config.model_dump(mode="json")
    specification["dpath_work"]["path"] = "custom/work"
    layout_file.write_text(json.dumps(specification))
    custom = DatasetLayout(layout.dpath_root, fpath_config=layout_file)
    custom.dpath_work.mkdir(parents=True)
    bundle = layout.get_dpath_pipeline_bundle(
        PIPELINE_TYPES[command], PIPELINES[command], "2.0.0"
    )
    invocation_file = bundle / "invocation.json"
    invocation = json.loads(invocation_file.read_text())
    invocation["output"] = "[[NIPOPPY_DPATH_PIPELINE_WORK]]"
    invocation_file.write_text(json.dumps(invocation))
    options = []
    if source in ("cli", "cli-over-environment"):
        options.extend(["--layout", str(layout_file)])
    if source == "environment":
        monkeypatch.setenv("NIPOPPY_LAYOUT", str(layout_file))
    if source == "cli-over-environment":
        monkeypatch.setenv("NIPOPPY_LAYOUT", str(tmp_path / "missing.json"))
    if source == "dotenv":
        dotenv = layout.dpath_root / ".env"
        dotenv.write_text(f'NIPOPPY_LAYOUT="{layout_file}"\n')
        monkeypatch.setattr(
            "nipoppy.cli.groups._load_dotenv_files",
            lambda root: _load_dotenv_files(root, [str(dotenv)]),
        )
    result = _invoke(layout, command, *options)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    expected_work = custom.get_dpath_pipeline_work(PIPELINES[command], "2.0.0", "run")
    assert {i["output"] for i in _invocations(bosh_commands)} == {str(expected_work)}


def test_cli_prefixed_id_filtering(workflow_dataset, command, bosh_commands):
    participant, session = ELIGIBLE[command][0]
    result = _invoke(
        workflow_dataset,
        command,
        "--participant-id",
        f"sub-{participant}",
        "--session-id",
        f"ses-{session}",
    )
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert [(i["participant"], i["session"]) for i in _invocations(bosh_commands)] == [
        (participant, session)
    ]


@pytest.mark.parametrize("alias", ["--dataset", "run"])
def test_cli_processing_aliases(workflow_dataset, bosh_commands, alias):
    args = [
        "run" if alias == "run" else "process",
        "--dataset" if alias == "--dataset" else "--study",
        str(workflow_dataset.dpath_root),
        "--pipeline",
        "processor",
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert [
        (i["participant"], i["session"]) for i in _invocations(bosh_commands)
    ] == ELIGIBLE["process"]


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0"])
@pytest.mark.parametrize("step", ["run", "followup"])
def test_cli_explicit_pipeline_version_and_step(
    workflow_dataset, command, bosh_commands, version, step
):
    result = _invoke(
        workflow_dataset,
        command,
        "--pipeline-version",
        version,
        "--pipeline-step",
        step,
    )
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    expected = ELIGIBLE[command]
    if command == "process" and (version, step) != ("2.0.0", "run"):
        expected = [("01", "A"), *expected]
    assert [
        (i["participant"], i["session"]) for i in _invocations(bosh_commands)
    ] == expected
    if command == "process":
        assert {i["output"] for i in _invocations(bosh_commands)} == {
            str(workflow_dataset.dpath_derivatives / f"processor/{version}/output")
        }
    assert list(
        (
            workflow_dataset.dpath_logs
            / "nipoppy"
            / command
            / f"{PIPELINES[command]}-{version}-{step}"
        ).glob("*.log")
    )


def test_bidsification_disabled_status_updates(workflow_dataset, bosh_commands):
    layout = workflow_dataset
    _set_step(layout, "bidsify", UPDATE_STATUS=False)
    original = layout.fpath_curation_status.read_bytes()
    result = _invoke(layout, "bidsify")
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert len(bosh_commands) == 2
    assert layout.fpath_curation_status.read_bytes() == original


def test_bidsification_retains_previous_status_backup(
    datetime_fixture, workflow_dataset, bosh_commands
):
    layout = workflow_dataset
    previous_backup = layout.fpath_curation_status.resolve()
    original = previous_backup.read_bytes()
    datetime_fixture.datetime.now.return_value = MOCKED_DATETIME + timedelta(minutes=1)
    result = _invoke(layout, "bidsify")
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert layout.fpath_curation_status.is_symlink()
    assert layout.fpath_curation_status.resolve() != previous_backup
    assert previous_backup.read_bytes() == original
    assert layout.fpath_curation_status.read_bytes() != original


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_empty_export_is_successful(
    workflow_dataset, command, bosh_commands, tmp_path, dry_run
):
    selection = tmp_path / "unmatched.tsv"
    selection.write_text("99\tZ\n")
    output = tmp_path / "export.tsv"
    options = ["--use-subcohort", str(selection), "--write-subcohort", str(output)]
    if dry_run:
        options.append("--dry-run")
    result = _invoke(workflow_dataset, command, *options)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert not bosh_commands
    assert output.exists() == (not dry_run)
    if not dry_run:
        assert output.read_bytes() == b""


def test_cli_empty_subcohort_is_a_workflow_error(
    workflow_dataset, command, bosh_commands, tmp_path
):
    selection = tmp_path / "empty.tsv"
    selection.touch()
    result = _invoke(workflow_dataset, command, "--use-subcohort", str(selection))
    assert result.exit_code == ReturnCode.WORKFLOW_FAILURE, result.output
    assert not bosh_commands


def test_cli_invocation_substitution_order(
    workflow_dataset, command, bosh_commands, recwarn
):
    layout = workflow_dataset
    config = Config.load(layout.fpath_config)
    config.SUBSTITUTIONS = {
        "USER_VALUE": "user",
        "[[CHOICE]]": "user-choice",
        "[[NIPOPPY_PARTICIPANT_ID]]": "user-id",
    }
    config.PIPELINE_VARIABLES.set_variables(
        PIPELINE_TYPES[command],
        PIPELINES[command],
        "2.0.0",
        {"CHOICE": "pipeline-choice"},
    )
    config.save(layout.fpath_config)
    bundle = layout.get_dpath_pipeline_bundle(
        PIPELINE_TYPES[command], PIPELINES[command], "2.0.0"
    )
    path = bundle / "invocation.json"
    invocation = json.loads(path.read_text())
    invocation["participant"] = "USER_VALUE/[[CHOICE]]/[[NIPOPPY_PARTICIPANT_ID]]"
    path.write_text(json.dumps(invocation))
    result = _invoke(layout, command)
    assert result.exit_code == ReturnCode.SUCCESS, result.output
    assert {i["participant"] for i in _invocations(bosh_commands)} == {
        "user/pipeline-choice/user-id"
    }
    assert len(recwarn) == 1
    assert "Unable to replace [[NIPOPPY_PARTICIPANT_ID]]" in str(recwarn[0].message)


def test_cli_invalid_descriptor_stops_execution(
    workflow_dataset, command, bosh_commands
):
    layout = workflow_dataset
    bundle = layout.get_dpath_pipeline_bundle(
        PIPELINE_TYPES[command], PIPELINES[command], "2.0.0"
    )
    (bundle / "descriptor.json").write_text("{}")
    result = _invoke(layout, command)
    assert result.exit_code == ReturnCode.INVALID_CONFIG, result.output
    assert not bosh_commands
    work = layout.get_dpath_pipeline_work(PIPELINES[command], "2.0.0", "run")
    # Setup creates directories before bundle validation; processing finally cleans.
    assert work.exists() == (command != "process")


def test_cli_runner_option_contract(command, workflow_dataset):
    params = {p.name: p for p in cli.commands[command].params}
    expected_defaults = {
        "dataset": None,
        "pipeline_name": PIPELINES[command],
        "pipeline_version": None,
        "pipeline_step": None,
        "participant_id": None,
        "session_id": None,
        "use_subcohort": None,
        "write_subcohort": None,
        "write_list": None,
        "hpc": None,
        "fpath_layout": None,
        "simulate": False,
        "keep_workdir": False,
        "verbose": False,
        "dry_run": False,
    }
    if command == "process":
        expected_defaults["tar"] = False
    assert set(params) == {"dpath_root", *expected_defaults}
    context = cli.commands[command].make_context(
        command,
        ["--study", str(workflow_dataset.dpath_root), "--pipeline", PIPELINES[command]],
    )
    for name, default in expected_defaults.items():
        assert context.params[name] == default
        assert params[name].required == (name == "pipeline_name")
    assert params["dataset"].hidden
    assert params["write_list"].hidden
    assert params["fpath_layout"].envvar == "NIPOPPY_LAYOUT"
    assert set(cli.context_settings["help_option_names"]) == {"-h", "--help"}
