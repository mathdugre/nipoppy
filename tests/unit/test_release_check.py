"""Tests for the non-blocking release check."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from nipoppy.cli import release_check
from nipoppy.cli.cli import cli

RELEASE_URL = "https://github.com/nipoppy/nipoppy/releases/tag/{version}"
LATEST_VERSION = "1.1.0"


@pytest.fixture(autouse=True)
def configure_release_check(monkeypatch: pytest.MonkeyPatch):
    """Enable the check and use a predictable installed version by default."""
    monkeypatch.delenv(release_check.DISABLE_RELEASE_CHECK_ENVVAR, raising=False)
    monkeypatch.setattr(release_check, "PROGRAM_VERSION", "1.0.0")


@pytest.fixture(autouse=True)
def isolate_release_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate the release cache in a temporary directory for each test."""
    monkeypatch.setattr(
        release_check,
        "FPATH_RELEASE_CACHE",
        tmp_path / ".release_cache.json",
    )


def _release_url(version: str) -> str:
    return RELEASE_URL.format(version=version)


def _add_release_response(httpx_mock, version: str = LATEST_VERSION, *, tag_prefix="v"):
    httpx_mock.add_response(
        url=release_check.RELEASES_API_URL,
        json={
            "tag_name": f"{tag_prefix}{version}",
            "html_url": _release_url(version),
        },
    )


def _write_cached_release(version: str, timestamp: float | None = None) -> None:
    release_check.FPATH_RELEASE_CACHE.write_text(
        json.dumps(
            {
                "version": version,
                "url": _release_url(version),
                "timestamp": time.time() if timestamp is None else timestamp,
            }
        )
    )


def _wait_for_check(
    check: release_check.ReleaseCheck | None, timeout: float = 2.0
) -> None:
    """Wait for a background check to complete."""
    assert check is not None
    check.finish(timeout=timeout)


def test_newer_release_is_reported(httpx_mock, caplog: pytest.LogCaptureFixture):
    _add_release_response(httpx_mock)

    with caplog.at_level(logging.WARNING, logger="nipoppy"):
        _wait_for_check(release_check.ReleaseCheck.start())

    assert (
        "A newer version of Nipoppy is available: 1.0.0 -> "
        "[link=https://github.com/nipoppy/nipoppy/releases/tag/1.1.0]1.1.0[/link]. "
        "Please upgrade manually based on your installation method "
        "(e.g., 'pip install --upgrade nipoppy')."
    ) in caplog.text


def test_no_warning_for_current_release(
    monkeypatch: pytest.MonkeyPatch,
    httpx_mock,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setattr(release_check, "PROGRAM_VERSION", LATEST_VERSION)
    _add_release_response(httpx_mock, tag_prefix="")

    with caplog.at_level(logging.WARNING, logger="nipoppy"):
        _wait_for_check(release_check.ReleaseCheck.start())

    assert "A newer version of Nipoppy is available" not in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        {"json": {"missing": "tag_name"}},
        {"text": "not json"},
    ],
    ids=["missing-tag", "malformed-json"],
)
def test_malformed_release_responses_are_ignored(httpx_mock, response: dict):
    httpx_mock.add_response(url=release_check.RELEASES_API_URL, **response)

    check = release_check.ReleaseCheck.start()
    _wait_for_check(check)

    assert check.release is None


def test_disabled_release_check_makes_no_request(
    monkeypatch: pytest.MonkeyPatch, mocker
):
    monkeypatch.setenv(release_check.DISABLE_RELEASE_CHECK_ENVVAR, "1")
    request = mocker.patch("nipoppy.cli.release_check.httpx.get")

    assert release_check.ReleaseCheck.start() is None
    request.assert_not_called()


@pytest.mark.parametrize(
    ("args", "succeeds"),
    [
        (["--help"], True),
        (["--version"], True),
        (["pipeline", "--help"], True),
        (["not-a-command"], False),
    ],
    ids=["help", "version", "subcommand-help", "command-failure"],
)
def test_release_check_runs_and_finishes_around_cli(
    args: list[str], succeeds: bool, mocker
):
    handle = mocker.MagicMock(spec=release_check.ReleaseCheck)
    start = mocker.patch("nipoppy.cli.cli.ReleaseCheck.start", return_value=handle)

    result = CliRunner().invoke(cli, args)

    assert (result.exit_code == 0) is succeeds
    start.assert_called_once_with()
    handle.finish.assert_called_once_with()


def test_release_check_none_on_cli_invocation(mocker):
    start = mocker.patch("nipoppy.cli.cli.ReleaseCheck.start", return_value=None)

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    start.assert_called_once_with()


def test_release_check_failure_does_not_prevent_cli(mocker):
    mocker.patch(
        "nipoppy.cli.release_check.ReleaseCheck._start_thread",
        side_effect=RuntimeError("thread start failed"),
    )
    result = CliRunner().invoke(cli, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_release_check_is_bounded_and_daemonized(mocker):
    request = mocker.patch(
        "nipoppy.cli.release_check.httpx.get",
        side_effect=httpx.ConnectError("offline"),
    )
    thread_spy = mocker.spy(release_check.threading, "Thread")

    _wait_for_check(release_check.ReleaseCheck.start())

    thread_spy.assert_called_once()
    assert thread_spy.call_args.kwargs["daemon"] is True
    request.assert_called_once()
    assert (
        request.call_args.kwargs["timeout"]
        == release_check.RELEASE_CHECK_REQUEST_TIMEOUT
    )


def test_unknown_current_version_is_ignored(
    monkeypatch: pytest.MonkeyPatch, httpx_mock
):
    monkeypatch.setattr(release_check, "PROGRAM_VERSION", "unknown")
    _add_release_response(httpx_mock)

    check = release_check.ReleaseCheck.start()
    _wait_for_check(check)

    assert check.release is None
    assert str(httpx_mock.get_request().url) == release_check.RELEASES_API_URL


def test_release_check_finish_timeout(mocker):
    check = release_check.ReleaseCheck()
    check.release = release_check.ReleaseInfo("1.1.0", _release_url("1.1.0"))
    wait = mocker.patch.object(check._completed, "wait", return_value=False)
    report = mocker.patch.object(check, "report")

    check.finish()

    wait.assert_called_once_with(timeout=release_check.CLI_WAIT_TIMEOUT)
    report.assert_not_called()


def test_successful_check_updates_cache(httpx_mock):
    _add_release_response(httpx_mock)

    _wait_for_check(release_check.ReleaseCheck.start())

    cached = json.loads(release_check.FPATH_RELEASE_CACHE.read_text())
    assert cached["version"] == LATEST_VERSION
    assert cached["url"] == _release_url(LATEST_VERSION)
    assert isinstance(cached["timestamp"], float)


def test_valid_cache_with_newer_version_reports_immediately(
    caplog: pytest.LogCaptureFixture, mocker
):
    _write_cached_release("2.0.0")
    request = mocker.patch("nipoppy.cli.release_check.httpx.get")
    thread_spy = mocker.spy(release_check.threading, "Thread")

    with caplog.at_level(logging.WARNING, logger="nipoppy"):
        check = release_check.ReleaseCheck.start()

    assert check is not None
    assert (
        "A newer version of Nipoppy is available: 1.0.0 -> "
        f"[link={_release_url('2.0.0')}]2.0.0[/link]."
    ) in caplog.text
    thread_spy.assert_not_called()
    request.assert_not_called()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nipoppy"):
        check.finish()
    assert "A newer version of Nipoppy is available" not in caplog.text


def test_valid_cache_with_current_version_makes_no_request(
    caplog: pytest.LogCaptureFixture, mocker
):
    _write_cached_release("1.0.0")
    request = mocker.patch("nipoppy.cli.release_check.httpx.get")
    thread_spy = mocker.spy(release_check.threading, "Thread")

    with caplog.at_level(logging.WARNING, logger="nipoppy"):
        check = release_check.ReleaseCheck.start()

    assert check is not None
    assert check.release is None
    assert "A newer version of Nipoppy is available" not in caplog.text
    thread_spy.assert_not_called()
    request.assert_not_called()


def test_expired_cache_triggers_background_check(
    httpx_mock, caplog: pytest.LogCaptureFixture
):
    _write_cached_release(
        "1.0.5",
        timestamp=time.time() - release_check.RELEASE_CACHE_TTL - 3600,
    )
    _add_release_response(httpx_mock)

    with caplog.at_level(logging.WARNING, logger="nipoppy"):
        check = release_check.ReleaseCheck.start()
        assert "A newer version of Nipoppy is available" not in caplog.text
        _wait_for_check(check)

    assert "1.1.0" in caplog.text
    cached = json.loads(release_check.FPATH_RELEASE_CACHE.read_text())
    assert cached["version"] == LATEST_VERSION


@pytest.mark.parametrize(
    "corrupted_content",
    [
        "not json",
        "[]",
        json.dumps({"version": "1.1.0"}),
        json.dumps({"version": 123, "url": "url", "timestamp": "now"}),
    ],
    ids=["malformed-json", "wrong-shape", "missing-fields", "invalid-types"],
)
def test_corrupted_cache_triggers_background_check(httpx_mock, corrupted_content: str):
    release_check.FPATH_RELEASE_CACHE.write_text(corrupted_content)
    _add_release_response(httpx_mock)

    check = release_check.ReleaseCheck.start()
    _wait_for_check(check)

    assert check.release is not None
    assert check.release.version == LATEST_VERSION


def test_cache_write_failure_raises(mocker):
    mocker.patch.object(
        release_check.fileops,
        "mkdir",
        side_effect=PermissionError("Read only"),
    )

    with pytest.raises(PermissionError):
        release_check._write_cache(
            release_check.ReleaseInfo("1.1.0", "https://example.com")
        )


def test_offline_network_failure_is_cached_and_prevents_another_check(mocker):
    request = mocker.patch(
        "nipoppy.cli.release_check.httpx.get",
        side_effect=httpx.ConnectError("offline"),
    )
    thread_spy = mocker.spy(release_check.threading, "Thread")

    first_check = release_check.ReleaseCheck.start()
    assert first_check is not None
    _wait_for_check(first_check)

    assert first_check.release is None
    assert request.call_count == 1
    assert thread_spy.call_count == 1
    assert release_check.FPATH_RELEASE_CACHE.is_file()

    request.reset_mock()
    thread_spy.reset_mock()
    check = release_check.ReleaseCheck.start()

    assert check is not None
    assert check.release is None
    thread_spy.assert_not_called()
    request.assert_not_called()


@pytest.mark.parametrize(
    "release",
    [
        release_check.ReleaseInfo("2.0.0", _release_url("2.0.0")),
        None,
    ],
    ids=["release-result", "failed-check"],
)
def test_cache_read_write_round_trip(release: release_check.ReleaseInfo | None):
    release_check._write_cache(release)

    cached = release_check._read_cache()

    assert cached is not None
    cached_release, timestamp = cached
    assert cached_release == release
    assert isinstance(timestamp, float)


@pytest.mark.parametrize(
    "latest,current,expected",
    [
        ("1.1.0", "1.0.0", True),
        ("1.0.0", "1.0.0", False),
        ("0.9.0", "1.0.0", False),
        ("invalid", "1.0.0", False),
        ("1.1.0", "invalid", False),
    ],
)
def test_is_newer(
    monkeypatch: pytest.MonkeyPatch, latest: str, current: str, expected: bool
):
    monkeypatch.setattr(release_check, "PROGRAM_VERSION", current)
    assert release_check._is_newer(latest) is expected
