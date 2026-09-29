"""Inputs to pipeline execution, separated from resolved runtime state."""

from dataclasses import dataclass

from nipoppy.env import PipelineTypeEnum, StrOrPathLike
from nipoppy.exceptions import WorkflowError


@dataclass(frozen=True)
class PipelineRequest:
    """Pipeline identity as requested, before default version and step resolution."""

    pipeline_type: PipelineTypeEnum
    name: str
    version: str | None = None
    step: str | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    """Execution and filesystem modes shared by pipeline workflows."""

    simulate: bool = False
    keep_workdir: bool = False
    dry_run: bool = False
    verbose: bool = False
    hpc: str | None = None
    write_subcohort: StrOrPathLike | None = None
    fpath_layout: StrOrPathLike | None = None
    n_jobs: int = 1
    show_progress: bool = False
    skip_logfile: bool = False

    def __post_init__(self):
        if self.hpc is not None and self.write_subcohort is not None:
            raise WorkflowError(
                "HPC job submission and writing a list of participants and sessions "
                "are mutually exclusive."
            )
        if self.n_jobs != 1 and not self.skip_logfile:
            raise WorkflowError("n_jobs is not supported when _skip_logfile is False.")
