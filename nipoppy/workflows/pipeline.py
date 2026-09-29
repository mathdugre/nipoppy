"""Base class for pipeline workflows."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from nipoppy.config.pipeline import (
    BasePipelineConfig,
    BIDSificationPipelineConfig,
    ExtractionPipelineConfig,
    ProcessingPipelineConfig,
)
from nipoppy.config.pipeline_step import AnalysisLevelType, ProcPipelineStepConfig
from nipoppy.config.tracker import TrackerConfig
from nipoppy.env import (
    PipelineTypeEnum,
    StrOrPathLike,
)
from nipoppy.exceptions import (
    ReturnCode,
    WorkflowError,
)
from nipoppy.logger import get_logger
from nipoppy.utils import fileops
from nipoppy.utils.bids import (
    check_participant_id,
    check_session_id,
)
from nipoppy.utils.utils import (
    get_pipeline_tag,
)
from nipoppy.workflows.base import BaseDatasetWorkflow
from nipoppy.workflows.services.artifacts import (
    check_pipeline_variables,
    get_pipeline_version,
    load_descriptor,
    load_invocation,
    load_pipeline_config,
    load_pybids_ignore_patterns,
    load_tracker_config,
)
from nipoppy.workflows.services.bids import build_bids_db
from nipoppy.workflows.services.boutiques import (
    load_boutiques_config,
    resolve_container_file,
)
from nipoppy.workflows.services.cohort import (
    apply_analysis_level,
    check_filter_args,
    filter_by_subcohort,
    write_subcohort,
)
from nipoppy.workflows.services.execution import (
    BatchResult,
    iter_results,
    log_summary,
    run_item,
)
from nipoppy.workflows.services.templates import TemplateRenderer

if TYPE_CHECKING:
    pass
try:
    import joblib  # noqa: F401 - retain availability checks until Tracker migration

    JOBLIB_INSTALLED = True

except ImportError as error:
    if str(error).startswith("No module named 'joblib'"):
        JOBLIB_INSTALLED = False
    else:
        raise

logger = get_logger()


class BasePipelineWorkflow(BaseDatasetWorkflow, ABC):
    """A workflow for a pipeline that has a Boutiques descriptor."""

    dname_hpc_logs = "hpc"
    fname_hpc_error = "pysqa.err"
    fname_job_script = "run_queue.sh"

    _pipeline_type = PipelineTypeEnum.PROCESSING

    _pipeline_type_to_pipeline_class_map = {
        PipelineTypeEnum.PROCESSING: ProcessingPipelineConfig,
        PipelineTypeEnum.BIDSIFICATION: BIDSificationPipelineConfig,
        PipelineTypeEnum.EXTRACTION: ExtractionPipelineConfig,
    }

    progress_bar_description = "Working..."  # default description used by rich

    def __init__(
        self,
        dpath_root: StrOrPathLike,
        name: str,
        pipeline_name: str,
        pipeline_version: str | None = None,
        pipeline_step: str | None = None,
        participant_id: str = None,
        session_id: str = None,
        use_subcohort: StrOrPathLike | None = None,
        hpc: str | None = None,
        write_subcohort: StrOrPathLike | None = None,
        n_jobs: int | None = None,
        fpath_layout: StrOrPathLike | None = None,
        verbose: bool = False,
        dry_run=False,
        _skip_logfile: bool = False,
        _show_progress: bool = False,
    ):
        if hpc is not None and write_subcohort is not None:
            raise WorkflowError(
                "HPC job submission and writing a list of participants and sessions "
                "are mutually exclusive."
            )

        if n_jobs is not None and not _skip_logfile:
            raise WorkflowError("n_jobs is not supported when _skip_logfile is False.")
        if n_jobs is None:
            n_jobs = 1

        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.pipeline_step = pipeline_step
        self.participant_id = check_participant_id(participant_id)
        self.session_id = check_session_id(session_id)
        self.use_subcohort = use_subcohort
        self.hpc = hpc
        self.write_subcohort = write_subcohort
        self.n_jobs = n_jobs
        self._show_progress = _show_progress

        super().__init__(
            dpath_root=dpath_root,
            name=name,
            fpath_layout=fpath_layout,
            verbose=verbose,
            dry_run=dry_run,
            _skip_logfile=_skip_logfile,
        )

        # the message logged in run_cleanup will depend on
        # the final values for these attributes (updated in run_main)
        self.n_success = 0
        self.n_total = 0

        self.run_single_results = None

        if not JOBLIB_INSTALLED and self.n_jobs not in (None, 1):
            logger.error(
                "An additional dependency is required to enable local parallelization "
                "with --n-jobs. Install it with: pip install nipoppy[parallel]",
                extra={"markup": False},
            )
            sys.exit(ReturnCode.MISSING_DEPENDENCY)

    @cached_property
    def dpaths_to_check(self) -> list[Path]:
        """Directory paths to create if needed during the setup phase."""
        return []

    @cached_property
    def dpath_pipeline(self) -> Path:
        """Return the path to the pipeline's derivatives directory."""
        return self.study.layout.get_dpath_pipeline(
            pipeline_name=self.pipeline_name, pipeline_version=self.pipeline_version
        )

    @cached_property
    def dpath_pipeline_output(self) -> Path:
        """Return the path to the pipeline's output directory."""
        return self.study.layout.get_dpath_pipeline_output(
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
        )

    @cached_property
    def dpath_pipeline_work(self) -> Path:
        """Return the path to the pipeline's working directory."""
        return self.study.layout.get_dpath_pipeline_work(
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_step=self.pipeline_step,
            participant_id=self.participant_id,
            session_id=self.session_id,
        )

    @cached_property
    def dpath_pipeline_bids_db(self) -> Path:
        """Return the path to the pipeline's BIDS database directory."""
        return self.study.layout.get_dpath_pybids_db(
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_step=self.pipeline_step,
            participant_id=self.participant_id,
            session_id=self.session_id,
        )

    @cached_property
    def dpath_pipeline_bundle(self) -> Path:
        """Path to the pipeline bundle directory."""
        return self.study.layout.get_dpath_pipeline_bundle(
            self._pipeline_type,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
        )

    @cached_property
    def pipeline_config(self) -> ProcessingPipelineConfig:
        """Get the user config object for the processing pipeline."""
        return self._get_pipeline_config(
            self.dpath_pipeline_bundle,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_class=self._pipeline_type_to_pipeline_class_map[
                self._pipeline_type
            ],
        )

    @cached_property
    def pipeline_step_config(self) -> ProcPipelineStepConfig:
        """Get the user config for the pipeline step."""
        return self.pipeline_config.get_step_config(step_name=self.pipeline_step)

    @cached_property
    def fpath_container(self):
        """Load container execution inputs using the Boutiques module."""
        return resolve_container_file(self.pipeline_config, self.pipeline_step_config)

    @cached_property
    def descriptor(self) -> dict:
        """Delegate pipeline loading to the artifact module."""
        return load_descriptor(
            self.study,
            self.pipeline_config,
            self.pipeline_step_config,
            self.dpath_pipeline_bundle,
        )

    @cached_property
    def invocation(self) -> dict:
        """Delegate pipeline loading to the artifact module."""
        return load_invocation(
            self.study,
            self.pipeline_config,
            self.pipeline_step_config,
            self.dpath_pipeline_bundle,
        )

    @cached_property
    def tracker_config(self) -> TrackerConfig:
        """Delegate pipeline loading to the artifact module."""
        return load_tracker_config(
            self.study,
            self.pipeline_config,
            self.pipeline_step_config,
            self.dpath_pipeline_bundle,
        )

    @cached_property
    def pybids_ignore_patterns(self) -> list[str]:
        """Delegate pipeline loading to the artifact module."""
        return load_pybids_ignore_patterns(
            self.study,
            self.pipeline_config,
            self.pipeline_step_config,
            self.dpath_pipeline_bundle,
        )

    @cached_property
    def boutiques_config(self):
        """Load container execution inputs using the Boutiques module."""
        return load_boutiques_config(self.descriptor)

    def _get_pipeline_config(
        self,
        dpath_pipeline_bundle: Path,
        pipeline_name: str,
        pipeline_version: str,
        pipeline_class: type[BasePipelineConfig],
    ) -> BasePipelineConfig:
        """Delegate pipeline loading to the artifact module."""
        return load_pipeline_config(
            self.study,
            self._pipeline_type,
            self.process_template_json,
            dpath_pipeline_bundle,
            pipeline_name,
            pipeline_version,
            pipeline_class,
        )

    def process_template_json(self, template_json, **kwargs):
        """Render with the legacy workflow namespace during migration."""
        values = {
            "name": lambda: self.name,
            "dpath_root": lambda: self.dpath_root,
            "fpath_layout": lambda: self.fpath_layout,
            "verbose": lambda: self.verbose,
            "dry_run": lambda: self.dry_run,
            "pipeline_name": lambda: self.pipeline_name,
            "pipeline_version": lambda: self.pipeline_version,
            "pipeline_step": lambda: self.pipeline_step,
            "participant_id": lambda: self.participant_id,
            "session_id": lambda: self.session_id,
            "use_subcohort": lambda: self.use_subcohort,
            "write_subcohort": lambda: self.write_subcohort,
            "hpc": lambda: self.hpc,
            "n_jobs": lambda: self.n_jobs,
            "n_success": lambda: self.n_success,
            "n_total": lambda: self.n_total,
            "subcommand": lambda: self.subcommand,
            "simulate": lambda: self.simulate,
            "keep_workdir": lambda: self.keep_workdir,
            "tar": lambda: self.tar,
            "dpath_pipeline": lambda: self.dpath_pipeline,
            "dpath_pipeline_output": lambda: self.dpath_pipeline_output,
            "dpath_pipeline_work": lambda: self.dpath_pipeline_work,
            "dpath_pipeline_bids_db": lambda: self.dpath_pipeline_bids_db,
            "dpath_pipeline_bundle": lambda: self.dpath_pipeline_bundle,
            "fpath_container": lambda: self.fpath_container,
            "dpath_pipeline_idp": lambda: self.dpath_pipeline_idp,
        }
        renderer = TemplateRenderer(
            self.study.config.SUBSTITUTIONS, values, self.study.layout
        )
        return renderer.render(template_json, **kwargs)

    def set_up_bids_db(self, dpath_pybids_db, participant_id=None, session_id=None):
        """Build a filtered database using the focused PyBIDS module."""
        return build_bids_db(
            self.study.layout.dpath_bids,
            dpath_pybids_db,
            self.pybids_ignore_patterns,
            participant_id,
            session_id,
        )

    def check_pipeline_version(self):
        """Set the pipeline version based on the config if it is not given."""
        if self.pipeline_version is None:
            self.pipeline_version = get_pipeline_version(
                pipeline_name=self.pipeline_name,
                dpath_pipelines=self.study.layout.get_dpath_pipeline_store(
                    self._pipeline_type
                ),
            )
            logger.warning(
                f"Pipeline version not specified, using version {self.pipeline_version}"
            )

    def _check_pipeline_variables(self):
        """Delegate pipeline loading to the artifact module."""
        return check_pipeline_variables(
            self.study, self._pipeline_type, self.pipeline_name, self.pipeline_version
        )

    def check_pipeline_step(self):
        """Set the pipeline step name based on the config if it is not given."""
        if self.pipeline_step is None:
            self.pipeline_step = self.pipeline_step_config.NAME
            logger.warning(
                f"Pipeline step not specified, using step {self.pipeline_step}"
            )

    def run_setup(self):
        """Run pipeline setup."""
        to_return = super().run_setup()

        self.check_pipeline_version()
        self._check_pipeline_variables()
        self.check_pipeline_step()
        self._check_filter_args_compatibility()

        for dpath in self.dpaths_to_check:
            fileops.mkdir(dpath, dry_run=self.dry_run)

        return to_return

    def _run_single_wrapper(self, participant_id, session_id):
        """Run one domain item with shared error isolation."""
        return run_item(
            self.run_single,
            self.pipeline_name,
            self.pipeline_version,
            participant_id,
            session_id,
        )

    def _get_results_generator(self, participants_sessions):
        """Execute items with optional parallelism and progress reporting."""
        return iter_results(
            participants_sessions,
            self._run_single_wrapper,
            n_jobs=self.n_jobs,
            show_progress=self._show_progress,
            description=self.progress_bar_description,
            joblib_installed=JOBLIB_INSTALLED,
        )

    @staticmethod
    def apply_analysis_level(
        participants_sessions: Iterable[str],
        analysis_level: AnalysisLevelType,
    ) -> list[tuple[str, str]]:
        """Delegate cohort planning to the selection module."""
        return apply_analysis_level(participants_sessions, analysis_level)

    def _handle_execution_strategy(self, participants_sessions):
        """Handle the execution strategy based on the workflow configuration."""
        if self.write_subcohort is not None:
            self._write_subcohort_to_file(participants_sessions)
        else:
            self._run_locally(participants_sessions)

    def _check_filter_args_compatibility(self):
        """Delegate cohort planning to the selection module."""
        return check_filter_args(
            self.pipeline_step_config.ANALYSIS_LEVEL,
            self.participant_id,
            self.session_id,
        )

    def run_main(self):
        """Run the pipeline."""
        participants_sessions = self.get_participants_sessions_to_run(
            self.participant_id, self.session_id
        )

        if self.use_subcohort is not None:
            participants_sessions = self._filter_by_subcohort(participants_sessions)

        participants_sessions = self.apply_analysis_level(
            participants_sessions=participants_sessions,
            analysis_level=self.pipeline_step_config.ANALYSIS_LEVEL,
        )
        self._handle_execution_strategy(participants_sessions)

        self._log_summary_message()

    def _filter_by_subcohort(self, participants_sessions: Iterable) -> set:
        """Delegate cohort planning to the selection module."""
        return filter_by_subcohort(participants_sessions, self.use_subcohort)

    def _write_subcohort_to_file(self, participants_sessions: list) -> None:
        """Delegate cohort planning to the selection module."""
        return write_subcohort(
            participants_sessions, self.write_subcohort, self.dry_run
        )

    def _run_locally(self, participants_sessions):
        """Collect typed outcomes from local execution."""
        result = BatchResult.from_results(
            self._get_results_generator(participants_sessions)
        )
        self.n_success += result.n_success
        self.n_total += result.n_total
        self.run_single_results = result.payloads if result.n_total else []
        if self.n_total and self.n_success != self.n_total:
            self.return_code = ReturnCode.PARTIAL_SUCCESS

    def _log_summary_message(self):
        """Report the current outcome using the shared summary policy."""
        code = log_summary(
            BatchResult(self.n_success, self.n_total),
            self.write_subcohort,
            self.hpc,
            self.pipeline_step_config.ANALYSIS_LEVEL,
            self.study.layout.fpath_curation_status,
        )
        if code is not None:
            self.return_code = code

    @abstractmethod
    def get_participants_sessions_to_run(
        self, participant_id: str | None, session_id: str | None
    ):
        """
        Return participant-session pairs to loop over with run_single().

        This is an abstract method that should be defined explicitly in subclasses.
        """

    @abstractmethod
    def run_single(self, participant_id: str | None, session_id: str | None):
        """
        Run on a single participant/session.

        This is an abstract method that should be defined explicitly in subclasses.
        """

    def generate_fpath_log(
        self,
        dnames_parent: str | list[str] | None = None,
        fname_stem: str | None = None,
    ) -> Path:
        """Generate a log file path."""
        # make sure that pipeline version is not None
        self.check_pipeline_version()
        self.check_pipeline_step()
        if dnames_parent is None:
            dnames_parent = get_pipeline_tag(
                pipeline_name=self.pipeline_name,
                pipeline_version=self.pipeline_version,
                pipeline_step=self.pipeline_step,
            )
        if fname_stem is None:
            fname_stem = get_pipeline_tag(
                pipeline_name=self.pipeline_name,
                pipeline_version=self.pipeline_version,
                pipeline_step=self.pipeline_step,
                participant_id=self.participant_id,
                session_id=self.session_id,
            )
        return super().generate_fpath_log(
            dnames_parent=dnames_parent, fname_stem=fname_stem
        )
