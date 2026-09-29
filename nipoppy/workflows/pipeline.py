"""Base class for pipeline workflows."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import rich
from pydantic import ValidationError

from nipoppy.config.boutiques import (
    BoutiquesConfig,
    get_boutiques_config_from_descriptor,
)
from nipoppy.config.pipeline import (
    BasePipelineConfig,
    BIDSificationPipelineConfig,
    ExtractionPipelineConfig,
    ProcessingPipelineConfig,
)
from nipoppy.config.pipeline_step import AnalysisLevelType, ProcPipelineStepConfig
from nipoppy.config.tracker import TrackerConfig
from nipoppy.console import _INDENT, CONSOLE_STDOUT
from nipoppy.container import get_container_handler
from nipoppy.env import (
    BIDS_SESSION_PREFIX,
    BIDS_SUBJECT_PREFIX,
    FAKE_SESSION_ID,
    PipelineTypeEnum,
    StrOrPathLike,
)
from nipoppy.exceptions import (
    ConfigError,
    ContainerError,
    FileOperationError,
    InvalidArgumentError,
    ReturnCode,
    WorkflowError,
)
from nipoppy.logger import get_logger
from nipoppy.utils import fileops
from nipoppy.utils.bids import (
    add_pybids_ignore_patterns,
    check_participant_id,
    check_session_id,
    create_bids_db,
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
from nipoppy.workflows.services.templates import TemplateRenderer

if TYPE_CHECKING:
    import bids
try:
    from joblib import Parallel, delayed

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
    def fpath_container(self) -> Path:
        """Return the full path to the pipeline's container."""
        uri = self.pipeline_config.CONTAINER_INFO.URI
        fpath_container = self.pipeline_config.CONTAINER_INFO.FILE
        container_handler = get_container_handler(
            self.pipeline_step_config.CONTAINER_CONFIG,
        )

        try:
            is_downloaded = container_handler.is_image_downloaded(uri, fpath_container)
        except ContainerError as e:
            raise WorkflowError(
                f"Error in container config for pipeline {self.pipeline_name} "
                f"{self.pipeline_version}: {e}"
            ) from e

        if not is_downloaded:
            error_message = (
                f"No container image file found for pipeline"
                f" {self.pipeline_name} {self.pipeline_version}"
            )
            if uri is not None:
                pull_command = container_handler.get_pull_command(uri, fpath_container)
                error_message += (
                    ". This file can be downloaded to the appropriate path by running "
                    f"the following command:\n\n{pull_command}"
                )
            raise FileOperationError(error_message)

        return fpath_container

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
        """Get the Boutiques configuration."""
        try:
            boutiques_config = get_boutiques_config_from_descriptor(
                self.descriptor,
            )
        except ValidationError as e:
            error_message = str(e) + str(e.errors())
            raise WorkflowError(
                f"Error when loading the Boutiques config from descriptor"
                f": {error_message}"
            )
        except ConfigError as e:
            logger.debug(
                "Caught exception when trying to load Boutiques config"
                f": {type(e).__name__}: {e}"
            )
            logger.debug(
                "Assuming Boutiques config is not in descriptor. Using default"
            )
            return BoutiquesConfig()

        logger.info(f"Loaded Boutiques config from descriptor: {boutiques_config}")
        return boutiques_config

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

    def set_up_bids_db(
        self,
        dpath_pybids_db: StrOrPathLike,
        participant_id: str | None = None,
        session_id: str | None = None,
    ) -> bids.BIDSLayout:
        """Set up the BIDS database."""
        dpath_pybids_db: Path = Path(dpath_pybids_db)

        pybids_ignore_patterns = self.pybids_ignore_patterns.copy()

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

        logger.debug(f"Path to BIDS data: {self.study.layout.dpath_bids}")
        bids_layout: bids.BIDSLayout = create_bids_db(
            dpath_bids=self.study.layout.dpath_bids,
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

    def _run_single_wrapper(self, participant_id, session_id) -> bool:
        """
        Run a single participant/session and handle exceptions.

        This is a helper function for parallelization with joblib.
        Returns (True, <result>) if the run was successful, (False, None) otherwise.
        """
        try:
            # success
            return True, self.run_single(participant_id, session_id)
        except Exception as exception:
            logger.error(
                f"Error running {self.pipeline_name} {self.pipeline_version}"
                f" on participant {participant_id}, session {session_id}"
                f": {exception}"
            )

        # failure
        return False, None

    def _get_results_generator(self, participants_sessions: Iterable[tuple[str, str]]):
        participants_sessions = list(participants_sessions)
        n_total = len(participants_sessions)
        if JOBLIB_INSTALLED:
            wrapper_func = delayed(self._run_single_wrapper)
        else:
            wrapper_func = self._run_single_wrapper

        results_generator = (
            wrapper_func(participant_id, session_id)
            for participant_id, session_id in participants_sessions
        )

        if JOBLIB_INSTALLED:
            results_generator = Parallel(
                n_jobs=self.n_jobs,
                backend="threading",
                return_as="generator",
            )(results_generator)

        if self._show_progress and n_total != 0:
            results_generator = rich.progress.track(
                results_generator,
                description=f"{' ' * _INDENT}{self.progress_bar_description}",
                total=n_total,
                console=CONSOLE_STDOUT,
            )

        return results_generator

    @staticmethod
    def apply_analysis_level(
        participants_sessions: Iterable[str],
        analysis_level: AnalysisLevelType,
    ) -> list[tuple[str, str]]:
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

    def _handle_execution_strategy(self, participants_sessions):
        """Handle the execution strategy based on the workflow configuration."""
        if self.write_subcohort is not None:
            self._write_subcohort_to_file(participants_sessions)
        else:
            self._run_locally(participants_sessions)

    def _check_filter_args_compatibility(self):
        """Warn if participant/session ID filters cannot be applied."""
        analysis_level = self.pipeline_step_config.ANALYSIS_LEVEL
        ignored_flags = []
        if self.participant_id is not None and analysis_level in (
            AnalysisLevelType.group,
            AnalysisLevelType.session,
        ):
            ignored_flags.append("--participant-id")
        if self.session_id is not None and analysis_level in (
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
        """Filter participants/sessions by subcohort file."""
        try:
            df_participants_sessions = pd.read_csv(
                self.use_subcohort, header=None, sep="\t", dtype=str
            )
        except FileNotFoundError as e:
            raise FileOperationError(
                f"Subcohort file {self.use_subcohort} not found"
            ) from e
        except pd.errors.EmptyDataError as e:
            raise WorkflowError(f"Subcohort file {self.use_subcohort} is empty") from e

        return set(participants_sessions) & set(
            df_participants_sessions.itertuples(index=False, name=None)
        )

    def _write_subcohort_to_file(self, participants_sessions: list) -> None:
        """Write participants/sessions to file."""
        if not self.dry_run:
            pd.DataFrame(participants_sessions).to_csv(
                self.write_subcohort, header=False, index=False, sep="\t"
            )

    def _run_locally(self, participants_sessions: list) -> None:
        """Run pipeline locally and collect results."""
        results = list(self._get_results_generator(participants_sessions))

        if len(results) == 0:
            run_statuses = []
            run_single_results = []
        else:
            run_statuses, run_single_results = zip(*results)

        self.n_success += sum(run_statuses)
        self.n_total += len(run_statuses)
        self.run_single_results = run_single_results

        # update return code if needed
        if (self.n_success != self.n_total) and (self.n_total != 0):
            self.return_code = ReturnCode.PARTIAL_SUCCESS

    def _log_summary_message(self):
        """Log a summary message."""
        if self.write_subcohort is not None:
            logger.success(f"Wrote subcohort to {self.write_subcohort}")
        elif self.n_total == 0:
            logger.warning(
                "No participants or sessions to run. Make sure there are no mistakes "
                "in the input arguments, the dataset's manifest or config file, and/or "
                "check the curation status file at "
                f"{self.study.layout.fpath_curation_status}"
            )
            self.return_code = ReturnCode.NO_PARTICIPANTS_OR_SESSIONS_TO_RUN
        elif self.hpc is not None:
            if self.n_success == 0:
                logger.failure("Failed to submit HPC jobs")
            else:
                logger.success(f"Successfully submitted {self.n_success} HPC job(s)")
        else:
            if self.pipeline_step_config.ANALYSIS_LEVEL == AnalysisLevelType.group:
                log_msg = "Ran on the entire study"
            else:
                log_msg = (
                    f"Ran for {self.n_success} out of "
                    f"{self.n_total} participants or sessions"
                )

            if self.n_success == 0:
                logger.error(log_msg)
            elif self.n_success == self.n_total:
                logger.success(log_msg)
            else:
                logger.warning(log_msg)

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
