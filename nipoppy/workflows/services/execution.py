"""Local work-item execution and outcome reporting."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, NamedTuple

import rich

from nipoppy.config.pipeline_step import AnalysisLevelType
from nipoppy.console import _INDENT, CONSOLE_STDOUT
from nipoppy.exceptions import ReturnCode
from nipoppy.logger import get_logger

logger = get_logger()
try:
    from joblib import Parallel, delayed

    JOBLIB_INSTALLED = True
except ImportError as error:
    if str(error).startswith("No module named 'joblib'"):
        JOBLIB_INSTALLED = False
    else:
        raise


class ItemResult(NamedTuple):
    """Success and optional domain payload for one work item."""

    success: bool
    payload: Any = None


@dataclass(frozen=True)
class BatchResult:
    """Counts and ordered payloads from local execution or scheduler submission."""

    n_success: int = 0
    n_total: int = 0
    payloads: tuple = ()

    @classmethod
    def from_results(cls, results):
        """Collect counts and domain payloads without dropping failures."""
        results = list(results)
        return cls(
            sum(result[0] for result in results),
            len(results),
            tuple(result[1] for result in results),
        )

    @property
    def return_code(self):
        """Return the local execution status, including partial failures."""
        return (
            ReturnCode.PARTIAL_SUCCESS
            if self.n_total and self.n_success != self.n_total
            else ReturnCode.SUCCESS
        )


def require_parallel_support(n_jobs):
    """Report the optional dependency in the same way as the CLI."""
    if not JOBLIB_INSTALLED and n_jobs not in (None, 1):
        logger.error(
            "An additional dependency is required to enable local parallelization "
            "with --n-jobs. Install it with: pip install nipoppy[parallel]",
            extra={"markup": False},
        )
        sys.exit(ReturnCode.MISSING_DEPENDENCY)


def run_item(execute, pipeline_name, pipeline_version, participant_id, session_id):
    """
    Run a single participant/session and handle exceptions.

    This is a helper function for parallelization with joblib.
    Returns (True, <result>) if the run was successful, (False, None) otherwise.
    """
    try:
        # success
        return ItemResult(True, execute(participant_id, session_id))
    except Exception as exception:
        logger.error(
            f"Error running {pipeline_name} {pipeline_version}"
            f" on participant {participant_id}, session {session_id}"
            f": {exception}"
        )

    # failure
    return ItemResult(False, None)


def iter_results(
    participants_sessions,
    execute,
    n_jobs=1,
    show_progress=False,
    description="Working...",
):
    """Run an ordered batch, optionally with threads and progress reporting."""
    participants_sessions = list(participants_sessions)
    n_total = len(participants_sessions)
    if JOBLIB_INSTALLED:
        wrapper_func = delayed(execute)
    else:
        wrapper_func = execute

    results_generator = (
        wrapper_func(participant_id, session_id)
        for participant_id, session_id in participants_sessions
    )

    if JOBLIB_INSTALLED:
        results_generator = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            return_as="generator",
        )(results_generator)

    if show_progress and n_total != 0:
        results_generator = rich.progress.track(
            results_generator,
            description=f"{' ' * _INDENT}{description}",
            total=n_total,
            console=CONSOLE_STDOUT,
        )

    return results_generator


def log_summary(result, write_subcohort, hpc, analysis_level, fpath_curation_status):
    """Log a summary message."""
    if write_subcohort is not None:
        logger.success(f"Wrote subcohort to {write_subcohort}")
    elif result.n_total == 0:
        logger.warning(
            "No participants or sessions to run. Make sure there are no mistakes "
            "in the input arguments, the dataset's manifest or config file, and/or "
            "check the curation status file at "
            f"{fpath_curation_status}"
        )
        return ReturnCode.NO_PARTICIPANTS_OR_SESSIONS_TO_RUN
    elif hpc is not None:
        if result.n_success == 0:
            logger.failure("Failed to submit HPC jobs")
        else:
            logger.success(f"Successfully submitted {result.n_success} HPC job(s)")
    else:
        if analysis_level == AnalysisLevelType.group:
            log_msg = "Ran on the entire study"
        else:
            log_msg = (
                f"Ran for {result.n_success} out of "
                f"{result.n_total} participants or sessions"
            )

        if result.n_success == 0:
            logger.error(log_msg)
        elif result.n_success == result.n_total:
            logger.success(log_msg)
        else:
            logger.warning(log_msg)


def execute_locally(
    items, execute, pipeline_name, pipeline_version, **options
) -> BatchResult:
    """Execute domain work with shared failure isolation and ordered collection."""
    require_parallel_support(options.get("n_jobs", 1))

    def wrapped(participant_id, session_id):
        return run_item(
            execute, pipeline_name, pipeline_version, participant_id, session_id
        )

    return BatchResult.from_results(iter_results(items, wrapped, **options))
