"""Shallow orchestration shared by the three executable pipeline workflows."""

from abc import abstractmethod
from functools import cached_property

from nipoppy.logger import get_logger
from nipoppy.pipeline_validation import check_pipeline_bundle
from nipoppy.utils import fileops
from nipoppy.utils.command import _run_command
from nipoppy.utils.utils import get_pipeline_tag
from nipoppy.workflows.base import BaseWorkflow
from nipoppy.workflows.services.boutiques import (
    execute_invocation,
    prepare_container,
    prepare_invocation,
    run_bosh_launch,
    run_bosh_simulate,
)
from nipoppy.workflows.services.cohort import (
    WorkItem,
    check_filter_args,
    plan_work_items,
    write_subcohort,
)
from nipoppy.workflows.services.dataset import generate_log_path
from nipoppy.workflows.services.execution import (
    BatchResult,
    execute_locally,
    log_summary,
)
from nipoppy.workflows.services.hpc import (
    HPCJob,
    HPCRunner,
    SubmissionPaths,
    load_hpc_config,
)

logger = get_logger()


class PipelineExecutionWorkflow(BaseWorkflow):
    """Sequence dataset setup, selection and execution through focused modules."""

    def __init__(
        self, name, dataset, pipeline, selection, execution, workspace, templates
    ):
        super().__init__(name, verbose=execution.verbose, dry_run=execution.dry_run)
        self.dataset = dataset
        self.pipeline = pipeline
        self.selection = selection
        self.execution = execution
        self.workspace = workspace
        self.templates = templates
        self.result = BatchResult()
        # Domain paths are registered explicitly and remain lazy.
        self.templates.values.update(self.template_values())

    def template_values(self):
        """Additional runtime values owned by a concrete workflow."""
        return {}

    def setup_directories(self):
        """Domain paths to create after resolving the pipeline configuration."""
        return []

    def hpc_extra_flags(self):
        """Domain flags to propagate to scheduler child commands."""
        return []

    def run_setup(self):
        """Preserve validation, logging, directory creation and raw-validation order."""
        self.dataset.study.layout.validate()
        if not self.execution.skip_logfile:
            self.pipeline.resolve_version()
            self.pipeline.resolve_step()
            parent = get_pipeline_tag(
                self.pipeline.name, self.pipeline.version, self.pipeline.step_name
            )
            stem = get_pipeline_tag(
                self.pipeline.name,
                self.pipeline.version,
                self.pipeline.step_name,
                self.selection.participant_id,
                self.selection.session_id,
            )
            logger.add_file_handler(
                generate_log_path(self.dataset.study.layout, self.name, parent, stem)
            )
        super().run_setup()
        self.pipeline.resolve_version()
        self.pipeline.check_variables()
        self.pipeline.resolve_step()
        check_filter_args(
            self.pipeline.step.ANALYSIS_LEVEL,
            self.selection.participant_id,
            self.selection.session_id,
        )
        for directory in self.setup_directories():
            fileops.mkdir(directory, dry_run=self.dry_run)
        check_pipeline_bundle(self.pipeline.bundle, strict=False)

    def run_main(self):
        """Plan items, dispatch the selected strategy and report its outcome."""
        items = plan_work_items(
            self.select_candidates(), self.selection, self.pipeline.step.ANALYSIS_LEVEL
        )
        if self.execution.write_subcohort is not None:
            write_subcohort(items, self.execution.write_subcohort, self.dry_run)
        elif self.execution.hpc is not None:
            jobs = [
                HPCJob(
                    tuple(
                        self.scheduler.generate_cli_command(
                            *item, extra_flags=self.hpc_extra_flags() or None
                        )
                    ),
                    *item,
                )
                for item in items
            ]
            name = get_pipeline_tag(
                self.pipeline.name,
                self.pipeline.version,
                self.pipeline.step_name,
                self.selection.participant_id,
                self.selection.session_id,
            )
            paths = SubmissionPaths(
                self.workspace.work, self.dataset.study.layout.dpath_logs / "hpc"
            )
            submitted = self.scheduler.submit_jobs(jobs, name, paths, self.dry_run)
            self.result = BatchResult(submitted.submitted, submitted.requested)
        else:
            self.result = execute_locally(
                items,
                lambda participant, session: self.execute_item(
                    WorkItem(participant, session)
                ),
                self.pipeline.name,
                self.pipeline.version,
                n_jobs=self.execution.n_jobs,
                show_progress=self.execution.show_progress,
            )
            self.return_code = self.result.return_code
        code = log_summary(
            self.result,
            self.execution.write_subcohort,
            self.execution.hpc,
            self.pipeline.step.ANALYSIS_LEVEL,
            self.dataset.study.layout.fpath_curation_status,
        )
        if code is not None:
            self.return_code = code

    @cached_property
    def scheduler(self):
        """Construct the scheduler from resolved configuration and CLI modes."""
        study = self.dataset.study
        return HPCRunner(
            hpc_cluster=self.execution.hpc,
            hpc_config=load_hpc_config(
                self.pipeline.step, self.pipeline.bundle, self.templates.render
            ),
            subcommand=self.name,
            dpath_root=study.layout.dpath_root,
            dpath_hpc=study.layout.dpath_hpc,
            pipeline_name=self.pipeline.name,
            pipeline_version=self.pipeline.version,
            pipeline_step=self.pipeline.step_name,
            preamble=study.config.HPC_PREAMBLE,
            queue_limit=study.config.HPC_QUEUE_LIMIT,
            keep_workdir=self.execution.keep_workdir,
            fpath_layout=self.execution.fpath_layout,
            verbose=self.verbose,
        )

    def execute_pipeline(self, item: WorkItem, bind_paths=(), **values):
        """Prepare domain binds and execute one validated Boutiques invocation."""
        handler = None
        if self.dataset.study.config.CONTAINER_CONFIG.COMMAND is not None:
            command, handler = prepare_container(
                self.pipeline.step.get_container_config(),
                self.pipeline.boutiques_config,
                self.templates.render,
                self.dataset.study.layout.dpath_root,
                *item,
                list(bind_paths),
            )
            values["container_command"] = command
        prepared = prepare_invocation(
            self.pipeline.descriptor,
            self.pipeline.invocation,
            self.templates.render,
            self.pipeline.config.CONTAINER_INFO.URI,
            lambda: self.pipeline.container_file,
            *item,
            container_handler=handler,
            verbose=self.verbose,
            **values,
        )
        runner = run_bosh_simulate if self.execution.simulate else run_bosh_launch
        execute_invocation(prepared, runner, _run_command, self.dry_run)
        return prepared.descriptor_str, prepared.invocation_str

    @abstractmethod
    def select_candidates(self):
        """Return domain-eligible participant/session pairs using selection filters."""

    @abstractmethod
    def execute_item(self, item: WorkItem):
        """Execute one domain item and return its optional payload."""
