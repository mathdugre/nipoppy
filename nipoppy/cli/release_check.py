"""Check for a newer Nipoppy release without delaying CLI execution."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Final

import httpx
from packaging.version import InvalidVersion, Version

from nipoppy.env import FPATH_USER_CONFIG, PROGRAM_VERSION
from nipoppy.logger import get_logger
from nipoppy.utils import fileops

RELEASES_API_URL: Final = "https://api.github.com/repos/nipoppy/nipoppy/releases/latest"
RELEASE_CHECK_REQUEST_TIMEOUT: Final = 2.0
CLI_WAIT_TIMEOUT: Final = 0.2
DISABLE_RELEASE_CHECK_ENVVAR: Final = "NIPOPPY_DISABLE_RELEASE_CHECK"
RELEASE_CACHE_TTL: Final = 86400.0  # 24 hours in seconds
FPATH_RELEASE_CACHE: Final = FPATH_USER_CONFIG.parent / ".release_cache.json"


@dataclass(frozen=True)
class ReleaseInfo:
    """Information about a newer Nipoppy release.

    Parameters
    ----------
    version : str
        The version string of the newer release.
    url : str
        The URL of the release on GitHub.
    """

    version: str
    url: str


def _read_cache() -> tuple[ReleaseInfo | None, float] | None:
    """Read cached release data if its timestamp is valid."""
    try:
        data = json.loads(FPATH_RELEASE_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    timestamp = data.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return None
    now = time.time()
    # Allow up to 60 seconds of clock skew and reject expired entries.
    if not (-60.0 <= now - timestamp < RELEASE_CACHE_TTL):
        return None
    version = data.get("version")
    url = data.get("url")
    if version is None and url is None:
        return None, float(timestamp)
    if not isinstance(version, str) or not isinstance(url, str):
        return None
    return ReleaseInfo(version=version, url=url), float(timestamp)


def _write_cache(
    release: ReleaseInfo | None = None,
) -> None:
    """Cache release information or the time of a failed check."""
    fileops.mkdir(FPATH_RELEASE_CACHE.parent)
    data: dict[str, str | float | None] = {
        "version": release.version if release is not None else None,
        "url": release.url if release is not None else None,
        "timestamp": time.time(),
    }
    FPATH_RELEASE_CACHE.write_text(json.dumps(data), encoding="utf-8")


def _is_newer(
    latest_version_str: str,
) -> bool:
    """Check whether the latest version is newer than the installed version."""
    try:
        return Version(latest_version_str) > Version(PROGRAM_VERSION)
    except InvalidVersion:
        return False


def _get_newer_release() -> ReleaseInfo | None:
    """Cache the latest release and return it if newer than the installed version."""
    response = httpx.get(
        RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "nipoppy",
        },
        timeout=RELEASE_CHECK_REQUEST_TIMEOUT,
    )
    release = response.json()

    latest_release = ReleaseInfo(
        version=str(Version(release["tag_name"].removeprefix("v"))),
        url=release["html_url"],
    )
    _write_cache(latest_release)

    if not _is_newer(latest_release.version):
        return None

    return latest_release


class ReleaseCheck:
    """Handle for a background release check."""

    def __init__(self) -> None:
        self._completed = threading.Event()
        self._reported = False
        self.release: ReleaseInfo | None = None

    @classmethod
    def start(cls) -> ReleaseCheck | None:
        """Start a release check or report immediately from cached release data.

        Returns
        -------
        ReleaseCheck | None
            Handle for the release check, or None if disabled or
            starting the check failed.
        """
        if os.environ.get(DISABLE_RELEASE_CHECK_ENVVAR):
            return None

        try:
            check = cls()
            cached = _read_cache()
            if cached is not None:
                cached_release, _ = cached
                if cached_release is not None and _is_newer(cached_release.version):
                    check.release = cached_release
                    check.report()
                check._completed.set()
                return check

            check._start_thread()
            return check
        except Exception:
            # Release check must not prevent Nipoppy from starting.
            return None

    def _start_thread(self) -> None:
        """Start the release check in a background daemon thread."""
        thread = threading.Thread(
            target=self._run,
            name="nipoppy-release-check",
            daemon=True,
        )
        thread.start()

    def _run(self) -> None:
        """Execute the release check and record the result."""
        try:
            self.release = _get_newer_release()
        except Exception:
            try:
                _write_cache(None)
            except Exception:
                pass
        finally:
            self._completed.set()

    def finish(self, timeout: float = CLI_WAIT_TIMEOUT) -> None:
        """Wait for the check to complete and report a newer release if found.

        Parameters
        ----------
        timeout : float, optional
            Maximum number of seconds to wait, by default CLI_WAIT_TIMEOUT.
        """
        try:
            if not self._reported:
                if self._completed.wait(timeout=timeout):
                    self.report()
        except Exception:
            # Reporting is best effort and must not replace the CLI result.
            pass

    def report(self) -> None:
        """Report a newer release to the user via logger if available."""
        if self.release is not None and not self._reported:
            self._reported = True
            logger = get_logger()
            logger.warning(
                f"A newer version of Nipoppy is available: {PROGRAM_VERSION} -> "
                f"[link={self.release.url}]{self.release.version}[/link]. "
                "Please upgrade manually based on your installation method "
                "(e.g., 'pip install --upgrade nipoppy')."
            )
