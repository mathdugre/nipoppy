"""Small dataset-aware pipeline base retained for processing status tracking."""

from abc import abstractmethod
from functools import cached_property
from pathlib import Path

from nipoppy.env import PipelineTypeEnum, StrOrPathLike
from nipoppy.workflows.base import BaseDatasetWorkflow
from nipoppy.workflows.factories import create_pipeline_dependencies
from nipoppy.workflows.services.cohort import SelectionOptions, check_filter_args
from nipoppy.workflows.services.execution import (
    BatchResult,
    execute_locally,
    log_summary,
    require_parallel_support,
)
from nipoppy.workflows.services.models import ExecutionOptions, PipelineRequest


class BasePipelineWorkflow(BaseDatasetWorkflow):
    """Sequence tracking setup and local execution through the extracted modules.

    Executable pipelines use PipelineExecutionWorkflow directly. This base only
    retains the dataset-aware lifecycle and the two hooks used by PipelineTracker.
    """

    progress_bar_description = "Working..."

    def __init__(
        self,
        dpath_root: StrOrPathLike,
        name: str,
        pipeline_name: str,
        pipeline_version: str | None = None,
        pipeline_step: str | None = None,
        participant_id: str | None = None,
        session_id: str | None = None,
        n_jobs: int = 1,
        fpath_layout: StrOrPathLike | None = None,
        verbose: bool = False,
        dry_run: bool = False,
        _skip_logfile: bool = True,
        _show_progress: bool = False,
    ):
        super().__init__(
            dpath_root, name, fpath_layout, verbose, dry_run, _skip_logfile
        )
        dependencies = create_pipeline_dependencies(
            dpath_root,
            name,
            PipelineRequest(
                PipelineTypeEnum.PROCESSING,
                pipeline_name,
                pipeline_version,
                pipeline_step,
            ),
            SelectionOptions(participant_id, session_id),
            ExecutionOptions(
                n_jobs=n_jobs,
                skip_logfile=_skip_logfile,
                show_progress=_show_progress,
                fpath_layout=fpath_layout,
                verbose=verbose,
                dry_run=dry_run,
            ),
            study=self.study,
        )
        self.pipeline = dependencies["pipeline"]
        self.selection = dependencies["selection"]
        self.execution = dependencies["execution"]
        self.templates = dependencies["templates"]
        self.workspace = dependencies["workspace"]
        self.dataset = dependencies["dataset"]
        self.templates.values.update(
            {
                "dpath_pipeline": lambda: self.derivatives,
                "dpath_pipeline_output": lambda: self.output,
            }
        )
        require_parallel_support(n_jobs)
        self.result = BatchResult()
        self.templates.values.update(
            {
                "n_success": lambda: self.result.n_success,
                "n_total": lambda: self.result.n_total,
            }
        )

    def __str__(self):
        """Describe requested identity without forcing lazy artifact loading."""
        return self._str_helper(
            components=[
                self.pipeline.name,
                self.pipeline.version,
                self.pipeline.step_name,
            ]
        )

    @cached_property
    def derivatives(self) -> Path:
        """Derivatives root for the pipeline being tracked."""
        return self.study.layout.get_dpath_pipeline(
            self.pipeline.name, self.pipeline.version
        )

    @cached_property
    def output(self) -> Path:
        """Output files checked by the concrete tracker."""
        return self.study.layout.get_dpath_pipeline_output(
            self.pipeline.name, self.pipeline.version
        )

    def run_setup(self):
        """Validate the study and resolve the processing pipeline being tracked."""
        super().run_setup()
        self.pipeline.resolve_version()
        self.pipeline.check_variables()
        self.pipeline.resolve_step()
        check_filter_args(
            self.pipeline.step.ANALYSIS_LEVEL,
            self.selection.participant_id,
            self.selection.session_id,
        )

    def run_main(self):
        """Execute tracker items with ordered result collection and progress."""
        items = self.get_participants_sessions_to_run(
            self.selection.participant_id, self.selection.session_id
        )
        items = self.apply_analysis_level(items, self.pipeline.step.ANALYSIS_LEVEL)
        self.result = execute_locally(
            items,
            self.run_single,
            self.pipeline.name,
            self.pipeline.version,
            n_jobs=self.execution.n_jobs,
            show_progress=self.execution.show_progress,
            description=self.progress_bar_description,
        )
        self.return_code = self.result.return_code
        code = log_summary(
            self.result,
            None,
            None,
            self.pipeline.step.ANALYSIS_LEVEL,
            self.study.layout.fpath_curation_status,
        )
        if code is not None:
            self.return_code = code

    @abstractmethod
    def get_participants_sessions_to_run(self, participant_id, session_id):
        """Return participant-session pairs to track."""

    @abstractmethod
    def apply_analysis_level(self, participants_sessions, analysis_level):
        """Choose tracker work items from the selected pairs."""

    @abstractmethod
    def run_single(self, participant_id, session_id):
        """Return one processing-status record."""
