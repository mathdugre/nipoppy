"""Tests for the BaseWorkflow class."""

import logging
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from nipoppy.env import BUG_REPORT_URL, DISCORD_URL
from nipoppy.exceptions import (
    MissingDependencyError,
    NipoppyError,
    ReturnCode,
)
from nipoppy.workflows.base import (
    BaseWorkflow,
    LogPrefix,
    _log_command,
    _run_command,
    handle_exception,
)
from nipoppy.zenodo_api import ZenodoAPIError


@pytest.fixture()
def workflow():
    class DummyWorkflow(BaseWorkflow):
        def run_main(self):
            pass

    workflow = DummyWorkflow(name="my_workflow")

    return workflow


def test_abstract_class():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseWorkflow(None, None)


def test_init(workflow: BaseWorkflow):
    assert workflow.name == "my_workflow"
    assert workflow.return_code is None


@pytest.mark.parametrize("command", ["echo x", "echo y"])
@pytest.mark.no_xdist
def test_log_command(command, caplog: pytest.LogCaptureFixture):
    _log_command(command)
    assert caplog.records
    record = caplog.records[-1]
    assert record.levelno == logging.INFO
    assert record.message.startswith(LogPrefix.RUN)
    assert command in record.message


class MyCustomException(NipoppyError):
    code = 999


class TestHandleException:
    @pytest.mark.parametrize(
        "exception,return_code",
        [
            (NipoppyError("error"), NipoppyError.code),
            (MyCustomException("error"), 999),
            (MissingDependencyError("error"), ReturnCode.MISSING_DEPENDENCY),
        ],
    )
    def test_handle_nipoppy_exception(
        self,
        exception: NipoppyError,
        return_code: int,
        caplog: pytest.LogCaptureFixture,
    ):
        """Return the exception code and log the error for NipoppyError subclasses."""
        assert handle_exception(exception) == return_code
        assert any(
            r.levelno == logging.ERROR and "error" in r.message for r in caplog.records
        )

    @pytest.mark.parametrize("hint", ["", "This is a hint."])
    def test_log_exception_custom_hint(
        self, hint: str, caplog: pytest.LogCaptureFixture
    ):
        """Log the custom troubleshooting hint when one is given."""
        handle_exception(NipoppyError("Invalid project config", hint=hint))
        assert any(
            f"Troubleshooting: {hint}" in record.message for record in caplog.records
        )

    def test_rendered_output_is_not_parsed_as_markup(self, rendered_console):
        """Show hints and messages verbatim on the console, brackets included."""
        hint = "pip install nipoppy[parallel]"
        handle_exception(MissingDependencyError("x", hint=hint))
        handle_exception(NipoppyError("bad [/] tag"))

        rendered = rendered_console.getvalue()
        assert hint in rendered
        assert "bad [/] tag" in rendered

    @pytest.mark.parametrize("exception", [Exception("boom"), RuntimeError("boom")])
    def test_handle_unknown_exception(
        self, exception: Exception, caplog: pytest.LogCaptureFixture
    ):
        """Map unexpected errors to UNKNOWN_FAILURE with traceback and bug report info."""
        assert handle_exception(exception) == ReturnCode.UNKNOWN_FAILURE
        assert any(r.levelno == logging.ERROR and r.exc_info for r in caplog.records)
        assert any(BUG_REPORT_URL in record.message for record in caplog.records)
        assert any(DISCORD_URL in record.message for record in caplog.records)

    def test_handle_pydantic_validation_error(self, caplog: pytest.LogCaptureFixture):
        """Map pydantic validation errors to INVALID_CONFIG with a hint."""

        class MockedModel(BaseModel):
            field: int

        with pytest.raises(ValidationError) as exc_info:
            MockedModel(field="invalid")

        assert handle_exception(exc_info.value) == ReturnCode.INVALID_CONFIG
        hint = "Review your configuration fields and value types"
        assert any(
            f"Troubleshooting: {hint}" in record.message for record in caplog.records
        )

    def test_handle_zenodo_api_error(self, caplog: pytest.LogCaptureFixture):
        """Map Zenodo API errors to KNOWN_FAILURE."""
        assert handle_exception(ZenodoAPIError("zenodo failed")) == (
            ReturnCode.KNOWN_FAILURE
        )
        assert any("zenodo failed" in record.message for record in caplog.records)

    def test_handle_keyboard_interrupt(self, caplog: pytest.LogCaptureFixture):
        """Map KeyboardInterrupt to TERMINATED."""
        assert handle_exception(KeyboardInterrupt()) == ReturnCode.TERMINATED
        assert any(
            r.levelno == logging.ERROR and "Interrupted by user" in r.message
            for r in caplog.records
        )


class TestRun:
    def test_run(self, workflow: BaseWorkflow):
        """A successful run sets the return code to SUCCESS."""
        assert workflow.run() is None
        assert workflow.return_code == ReturnCode.SUCCESS

    @pytest.mark.parametrize("method", ["run_setup", "run_main", "run_cleanup"])
    def test_run_exception(self, workflow: BaseWorkflow, mocker, method: str):
        """Set the return code and re-raise the same error from any run phase."""
        exception = MyCustomException("failed")
        mocker.patch.object(workflow, method, side_effect=exception)

        with pytest.raises(MyCustomException) as exc_info:
            workflow.run()

        assert exc_info.value is exception
        assert workflow.return_code == MyCustomException.code


# TODO we might want to move these tests to a separate test file or reorganize them
# Previously, were using the BaseWorkflow.run_command method, which has been extracted
# to a standalone function. The tests have been adapted accordingly.
@pytest.mark.no_xdist
def test_log_command_no_markup(caplog: pytest.LogCaptureFixture):
    # message with closing tag
    message = "[/]"

    # this should not raise a rich markup error
    _run_command(["echo", message])
    assert message in caplog.text


def test_run_command(tmp_path: Path):
    fpath = tmp_path / "test.txt"
    process = _run_command(["touch", fpath])
    assert process.returncode == 0
    assert fpath.exists()


def test_run_command_single_string(tmp_path: Path):
    fpath = tmp_path / "test.txt"
    process = _run_command(f"touch {fpath}", shell=True)
    assert process.returncode == 0
    assert fpath.exists()


def test_run_command_dry_run(tmp_path: Path):
    fpath = tmp_path / "test.txt"
    command = _run_command(["touch", fpath], dry_run=True)
    assert command == f"touch {fpath}"
    assert not fpath.exists()


def test_run_command_check():
    with pytest.raises(subprocess.CalledProcessError):
        _run_command(["which", "probably_fake_command"], check=True)


@pytest.mark.no_xdist
def test_run_command_no_markup(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    # text with closing tag
    text = "[/]"

    # this should not raise a rich markup error
    fpath_txt = tmp_path / "test.txt"
    fpath_txt.write_text(text)
    _run_command(["cat", fpath_txt])
    assert text in caplog.text


@pytest.mark.no_xdist
def test_run_command_quiet(caplog: pytest.LogCaptureFixture):
    message = "This should be printed"
    _run_command(["echo", message], quiet=True)
    assert LogPrefix.RUN not in caplog.text
    assert message in caplog.text
