"""Cohort selection after domain-specific eligibility has been determined."""

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from nipoppy.config.pipeline_step import AnalysisLevelType
from nipoppy.env import StrOrPathLike
from nipoppy.exceptions import FileOperationError, InvalidArgumentError, WorkflowError
from nipoppy.utils.bids import check_participant_id, check_session_id


class WorkItem(NamedTuple):
    """One unit of work at any supported analysis level."""

    participant_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class SelectionOptions:
    """Validated CLI filters and an optional participant-session file."""

    participant_id: str | None = None
    session_id: str | None = None
    use_subcohort: StrOrPathLike | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "participant_id", check_participant_id(self.participant_id)
        )
        object.__setattr__(self, "session_id", check_session_id(self.session_id))


def apply_analysis_level(participants_sessions, analysis_level):
    """Filter participant-session pairs to run based on the analysis level."""
    if analysis_level == AnalysisLevelType.group:
        return [(None, None)]

    if analysis_level == AnalysisLevelType.participant:
        participants = []
        for participant, _ in participants_sessions:
            if participant not in participants:
                participants.append(participant)
        return [(participant, None) for participant in participants]

    if analysis_level == AnalysisLevelType.session:
        sessions = []
        for _, session in participants_sessions:
            if session not in sessions:
                sessions.append(session)
        return [(None, session) for session in sessions]

    return list(participants_sessions)


def filter_by_subcohort(participants_sessions, fpath):
    """Filter participants/sessions by subcohort file."""
    try:
        df_participants_sessions = pd.read_csv(fpath, header=None, sep="\t", dtype=str)
    except FileNotFoundError as e:
        raise FileOperationError(f"Subcohort file {fpath} not found") from e
    except pd.errors.EmptyDataError as e:
        raise WorkflowError(f"Subcohort file {fpath} is empty") from e

    return set(participants_sessions) & set(
        df_participants_sessions.itertuples(index=False, name=None)
    )


def write_subcohort(participants_sessions, fpath, dry_run=False):
    """Write participants/sessions to file."""
    if not dry_run:
        pd.DataFrame(participants_sessions).to_csv(
            fpath, header=False, index=False, sep="\t"
        )


def check_filter_args(analysis_level, participant_id, session_id):
    """Warn if participant/session ID filters cannot be applied."""
    ignored_flags = []
    if participant_id is not None and analysis_level in (
        AnalysisLevelType.group,
        AnalysisLevelType.session,
    ):
        ignored_flags.append("--participant-id")
    if session_id is not None and analysis_level in (
        AnalysisLevelType.group,
        AnalysisLevelType.participant,
    ):
        ignored_flags.append("--session-id")

    if len(ignored_flags) > 0:
        raise InvalidArgumentError(
            (
                f"The {' and '.join(ignored_flags)} flag(s) are given, "
                f"but the pipeline is run at the {analysis_level.value} level"
            ),
            hint="Remove the incompatible flag(s)",
        )


def plan_work_items(
    candidates, selection: SelectionOptions, analysis_level: AnalysisLevelType
) -> list[WorkItem]:
    """Apply subcohort membership before collapsing to the requested analysis level."""
    if selection.use_subcohort is not None:
        candidates = filter_by_subcohort(candidates, selection.use_subcohort)
    return [
        WorkItem(*pair) for pair in apply_analysis_level(candidates, analysis_level)
    ]
