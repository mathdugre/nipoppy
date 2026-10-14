"""PipelineTracker workflow."""

import tarfile
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

from nipoppy.config.pipeline_step import AnalysisLevelType
from nipoppy.config.tracker import TrackerConfig
from nipoppy.env import EXT_TAR, PipelineTypeEnum, StrOrPathLike
from nipoppy.exceptions import NipoppyError
from nipoppy.logger import get_logger
from nipoppy.tabular.processing_status import ProcessingStatusTable
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

logger = get_logger()


class PipelineTracker(BaseDatasetWorkflow):
    """Pipeline tracker."""

    progress_bar_description = "Tracking..."

    def __init__(
        self,
        dpath_root: StrOrPathLike,
        pipeline_name: str,
        pipeline_version: str | None = None,
        pipeline_step: str | None = None,
        participant_id: str = None,
        session_id: str = None,
        n_jobs: int = 1,
        fpath_layout: StrOrPathLike | None = None,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        super().__init__(
            dpath_root=dpath_root,
            name="track_processing",
            fpath_layout=fpath_layout,
            verbose=verbose,
            dry_run=dry_run,
            _skip_logfile=True,
        )
        dependencies = create_pipeline_dependencies(
            dpath_root,
            self.name,
            PipelineRequest(
                PipelineTypeEnum.PROCESSING,
                pipeline_name,
                pipeline_version,
                pipeline_step,
            ),
            SelectionOptions(participant_id, session_id),
            ExecutionOptions(
                n_jobs=n_jobs,
                skip_logfile=True,
                show_progress=True,
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
        """Validate the dataset and pipeline, then load or initialize status."""
        super().run_setup()
        self.pipeline.resolve_version()
        self.pipeline.check_variables()
        self.pipeline.resolve_step()
        check_filter_args(
            self.pipeline.step.ANALYSIS_LEVEL,
            self.selection.participant_id,
            self.selection.session_id,
        )
        if self.study.layout.fpath_processing_status.exists():
            try:
                self.processing_status_table = ProcessingStatusTable.load(
                    self.study.layout.fpath_processing_status
                )
                logger.info(
                    f"Found existing processing status file with shape"
                    f" {self.processing_status_table.shape}"
                    f" at {self.study.layout.fpath_processing_status}"
                )
            except NipoppyError as e:
                if "Error when validating the " in str(e):
                    logger.warning(
                        "Failed to load existing processing status file at "
                        f"{self.study.layout.fpath_processing_status}. Generating a new"
                        f" processing status table.\nOriginal error:\n{e}"
                    )
                    self.processing_status_table = ProcessingStatusTable()
        else:
            self.processing_status_table = ProcessingStatusTable()
            logger.info("Initialized empty processing status table")

    def check_status(
        self,
        relative_paths: StrOrPathLike,
        relative_dpath_tarred: StrOrPathLike | None = None,
    ):
        """Check the processing status based on a list of expected paths."""
        # collect list of paths in tarball if it exists
        paths_tarred = []
        if relative_dpath_tarred is not None:
            fpath_tarball = self.output / f"{relative_dpath_tarred}{EXT_TAR}"
            if fpath_tarball.exists():
                with tarfile.open(fpath_tarball) as tarball:
                    paths_tarred = tarball.getnames()

        for relative_path in relative_paths:
            relative_path = Path(relative_path)
            logger.debug(f"Checking path {self.output / relative_path}")

            matches_glob = list(self.output.glob(str(relative_path)))
            logger.debug(f"Matches: {matches_glob}")

            # also check tarball paths if applicable/needed
            if (not matches_glob) and (relative_dpath_tarred is not None):
                # NOTE
                # The behaviour of Path.match() is not exactly the same as glob
                # but it should be good enough for now. What we really need is
                # Path.full_match(), but that was only introduced in Python 3.13
                matches_tarred = [
                    path_tarred
                    for path_tarred in paths_tarred
                    if Path(path_tarred).match(
                        str(
                            Path(relative_path).relative_to(
                                Path(relative_dpath_tarred).parent
                            )
                        ),
                    )
                ]
                logger.debug(f"Matches in tarball: {matches_tarred}")
            else:
                matches_tarred = []

            if not (matches_glob or matches_tarred):
                return ProcessingStatusTable.status_fail

        return ProcessingStatusTable.status_success

    def get_participants_sessions_to_run(
        self, participant_id: str | None, session_id: str | None
    ):
        """Get participant-session pairs with BIDS data to run the tracker on."""
        return self.curation_status_table.get_bidsified_participants_sessions(
            participant_id=participant_id, session_id=session_id
        )

    @staticmethod
    def apply_analysis_level(
        participants_sessions: Iterable[str],
        analysis_level: AnalysisLevelType,
    ) -> list[tuple[str]]:
        """Tracker: level is always participant-session."""
        return list(participants_sessions)

    def run_single(self, participant_id: str, session_id: str):
        """Run tracker on a single participant/session."""
        # replace template strings in the tracker config
        tracker_config = TrackerConfig(
            **self.templates.render(
                self.pipeline.tracker.model_dump(mode="json"),
                participant_id=participant_id,
                session_id=session_id,
            )
        )

        # check status and update processing status file
        status = self.check_status(
            tracker_config.PATHS, tracker_config.PARTICIPANT_SESSION_DIR
        )
        logger.debug(f"Status: {status}")
        processing_status_record = {
            ProcessingStatusTable.col_participant_id: participant_id,
            ProcessingStatusTable.col_session_id: session_id,
            ProcessingStatusTable.col_pipeline_name: self.pipeline.name,
            ProcessingStatusTable.col_pipeline_version: self.pipeline.version,
            ProcessingStatusTable.col_pipeline_step: self.pipeline.step_name,
            ProcessingStatusTable.col_status: status,
        }
        return processing_status_record

    def _update_status_file(self):
        """Update the processing status file."""
        self.processing_status_table = (
            self.processing_status_table.add_or_update_records(self.result.payloads)
        )
        logger.info(
            "New/updated processing status table shape: "
            f"{self.processing_status_table.shape}"
        )
        self.processing_status_table.save_with_backup(
            self.study.layout.fpath_processing_status,
            dry_run=self.dry_run,
        )

    def run_main(self):
        """Track selected pairs, summarize results and persist status."""
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
        self._update_status_file()
