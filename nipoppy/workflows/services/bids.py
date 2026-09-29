"""Build filtered PyBIDS databases and resolve exact-match injected paths."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nipoppy.env import (
    BIDS_PATH_INJECTION_PREFIX,
    BIDS_SESSION_PREFIX,
    BIDS_SUBJECT_PREFIX,
    FAKE_SESSION_ID,
)
from nipoppy.exceptions import WorkflowError
from nipoppy.logger import get_logger
from nipoppy.utils.bids import add_pybids_ignore_patterns, create_bids_db

if TYPE_CHECKING:
    import bids
logger = get_logger()


def get_bids_paths_to_inject(
    bids_layout: bids.BIDSLayout, injection_map: dict[str, dict]
) -> dict[str, str]:
    """Require one filename per injection query and normalize placeholder keys."""
    bids_paths = {}
    for key, kwargs in injection_map.items():
        bids_path = bids_layout.get(**kwargs, return_type="filename")
        if len(bids_path) != 1:
            raise WorkflowError(
                f"Expected exactly one file to match criteria {kwargs}, got: {bids_path}"  # noqa: E501
            )

        bids_paths[f"{BIDS_PATH_INJECTION_PREFIX}{key}".lower()] = bids_path[0]

    return bids_paths


def build_bids_db(
    dpath_bids, dpath_pybids_db, ignore_patterns, participant_id=None, session_id=None
):
    """Set up the BIDS database."""
    dpath_pybids_db: Path = Path(dpath_pybids_db)

    pybids_ignore_patterns = ignore_patterns.copy()

    if participant_id is not None:
        add_pybids_ignore_patterns(
            current=pybids_ignore_patterns,
            new=f"^(?!/{BIDS_SUBJECT_PREFIX}({participant_id}))",
        )
    if (session_id is not None) and (session_id != FAKE_SESSION_ID):
        add_pybids_ignore_patterns(
            current=pybids_ignore_patterns,
            new=f".*?/{BIDS_SESSION_PREFIX}(?!{session_id})",
        )

    logger.info(
        f"Building BIDSLayout with {len(pybids_ignore_patterns)} ignore "
        f"patterns: {pybids_ignore_patterns}"
    )

    if dpath_pybids_db.exists() and list(dpath_pybids_db.iterdir()):
        logger.warning(
            f"Overwriting existing BIDS database directory: {dpath_pybids_db}"
        )

    logger.debug(f"Path to BIDS data: {dpath_bids}")
    bids_layout: bids.BIDSLayout = create_bids_db(
        dpath_bids=dpath_bids,
        dpath_pybids_db=dpath_pybids_db,
        ignore_patterns=pybids_ignore_patterns,
        reset_database=True,
    )

    # list all the files in BIDSLayout
    # since we are selecting for specific a specific subject and
    # session, there should not be too many files
    filenames = bids_layout.get(return_type="filename")
    logger.debug(f"Found {len(filenames)} files in BIDS database:")
    for filename in filenames:
        logger.debug(filename)

    if len(filenames) == 0:
        logger.warning("BIDS database is empty")

    return bids_layout
