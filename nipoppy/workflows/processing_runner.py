"""Processing eligibility, PyBIDS inputs, archive policy and cleanup."""

from functools import cached_property
from pathlib import Path

from nipoppy.config.tracker import TrackerConfig
from nipoppy.exceptions import FileOperationError
from nipoppy.logger import get_logger
from nipoppy.utils import fileops
from nipoppy.workflows.execution import PipelineExecutionWorkflow
from nipoppy.workflows.services.archive import check_archive_config, tar_directory
from nipoppy.workflows.services.bids import build_bids_db, get_bids_paths_to_inject
from nipoppy.workflows.services.cohort import WorkItem

logger = get_logger()


class ProcessingRunner(PipelineExecutionWorkflow):
    """Run processing pipelines and retain their domain-specific policies."""

    def __init__(self, *, tar: bool = False, **dependencies):
        self.tar = tar
        super().__init__(**dependencies)

    @cached_property
    def derivatives(self) -> Path:
        """Derivatives root of the requested processing pipeline."""
        return self.dataset.study.layout.get_dpath_pipeline(
            self.pipeline.name, self.pipeline.version
        )

    @cached_property
    def output(self) -> Path:
        """Output directory of the requested processing pipeline."""
        return self.dataset.study.layout.get_dpath_pipeline_output(
            self.pipeline.name, self.pipeline.version
        )

    def template_values(self):
        """Register the processing paths and archive flag lazily."""
        return {
            "dpath_pipeline": lambda: self.derivatives,
            "dpath_pipeline_output": lambda: self.output,
            "tar": lambda: self.tar,
        }

    def setup_directories(self):
        """Create processing output and working directories during setup."""
        return [self.output, self.workspace.work]

    def hpc_extra_flags(self):
        """Propagate the processing archive option to scheduler child commands."""
        return ["--tar"] if self.tar else []

    def run_setup(self):
        """Validate archive requirements and fail early for a missing container."""
        super().run_setup()
        check_archive_config(
            self.tar, self.pipeline.step, lambda: self.pipeline.tracker
        )
        try:
            self.pipeline.container_file
        except FileOperationError:
            raise
        except Exception:
            pass

    def select_candidates(self):
        """Run BIDSified participant/sessions without a completed processing record."""
        filters = dict(
            participant_id=self.selection.participant_id,
            session_id=self.selection.session_id,
        )
        completed = set(
            self.dataset.processing_status_table.get_completed_participants_sessions(
                pipeline_name=self.pipeline.name,
                pipeline_version=self.pipeline.version,
                pipeline_step=self.pipeline.step_name,
                **filters,
            )
        )
        candidates = (
            self.dataset.curation_status_table.get_bidsified_participants_sessions(
                **filters
            )
        )
        return (pair for pair in candidates if pair not in completed)

    def execute_item(self, item: WorkItem):
        """Build optional PyBIDS inputs, execute the pipeline, then archive output."""
        logger.info(
            f"Running for participant {item.participant_id}, session {item.session_id}"
        )
        layout = self.dataset.study.layout
        injected = {}
        if self.pipeline.step.GENERATE_PYBIDS_DATABASE:
            database = build_bids_db(
                layout.dpath_bids,
                self.workspace.bids_db,
                self.pipeline.ignore_patterns,
                *item,
            )
            injected = get_bids_paths_to_inject(
                bids_layout=database,
                injection_map=self.pipeline.config.BIDS_PATH_INJECTION_MAP,
            )
        payload = self.execute_pipeline(
            item,
            [
                layout.dpath_bids,
                self.output,
                self.workspace.work,
                self.workspace.bids_db,
            ],
            **injected,
        )
        if self.tar and not self.execution.simulate:
            tracker = TrackerConfig(
                **self.templates.render(
                    self.pipeline.tracker.model_dump(mode="json"),
                    participant_id=item.participant_id,
                    session_id=item.session_id,
                )
            )
            tar_directory(
                self.output / tracker.PARTICIPANT_SESSION_DIR, dry_run=self.dry_run
            )
        return payload

    def run_cleanup(self):
        """Retain the established count-based cleanup policy, including zero/zero."""
        if self.result.n_success == self.result.n_total:
            if not self.execution.keep_workdir:
                for directory in [self.workspace.bids_db, self.workspace.work]:
                    if directory.exists():
                        fileops.rm(directory, dry_run=self.dry_run)
            else:
                logger.info("Keeping working / intermediary files.")
        else:
            logger.info(
                "Some pipeline segments failed. Keeping working / intermediary files."
            )
        return super().run_cleanup()
