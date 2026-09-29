"""Validate archive configuration and remove source data only after valid tar output."""

from pathlib import Path
from tarfile import is_tarfile

from nipoppy.env import EXT_TAR
from nipoppy.exceptions import ConfigError, FileOperationError
from nipoppy.logger import get_logger
from nipoppy.utils import fileops
from nipoppy.utils.command import _run_command

logger = get_logger()


def check_archive_config(enabled, step, load_tracker):
    """
    Make sure that conditions for tarring are met if tarring is requested.

    Specifically, check that dpath to tar is specified in the tracker config
    """
    if not enabled:
        return

    if step.TRACKER_CONFIG_FILE is None:
        raise ConfigError(
            "Tarring requested but there is no tracker config file. "
            "Specify the TRACKER_CONFIG_FILE field for the pipeline step in "
            "the global config file, then make sure the PARTICIPANT_SESSION_DIR "
            "field is specified in the TRACKER_CONFIG_FILE file."
        )
    if load_tracker().PARTICIPANT_SESSION_DIR is None:
        raise ConfigError(
            "Tarring requested but no participant-session directory specified. "
            "The PARTICIPANT_SESSION_DIR field in the tracker config must set "
            "in the tracker config file at "
            f"{step.TRACKER_CONFIG_FILE}"
        )


def tar_directory(dpath, dry_run=False, run_command=None):
    """Tar a directory and delete it."""
    dpath = Path(dpath)
    if not dpath.exists():
        raise FileOperationError(f"Not tarring {dpath} since it does not exist")
    if not dpath.is_dir():
        raise FileOperationError(f"Not tarring {dpath} since it is not a directory")

    tar_flags = "-cvf"
    fpath_tarred = dpath.with_suffix(EXT_TAR)

    (run_command or _run_command)(
        f"tar {tar_flags} {fpath_tarred} -C {dpath.parent} {dpath.name}",
        dry_run=dry_run,
    )

    # make sure that the tarfile was created successfully before removing
    # original directory
    if fpath_tarred.exists() and is_tarfile(fpath_tarred):
        fileops.rm(dpath, dry_run=dry_run)
    else:
        logger.error(f"Failed to tar {dpath} to {fpath_tarred}")

    return fpath_tarred
