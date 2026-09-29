"""Abstract class for workflow runners and runner utilities."""

import shlex
from abc import ABC
from functools import cached_property

from typing_extensions import override

from nipoppy.config.hpc import HpcConfig
from nipoppy.logger import get_logger
from nipoppy.pipeline_validation import check_pipeline_bundle
from nipoppy.utils.utils import get_pipeline_tag, load_json
from nipoppy.workflows.base import _run_command
from nipoppy.workflows.pipeline import BasePipelineWorkflow
from nipoppy.workflows.services.boutiques import (
    BoshRunnerCallable,
    execute_invocation,
    prepare_container,
    prepare_invocation,
    run_bosh_launch,
    run_bosh_simulate,
)
from nipoppy.workflows.services.hpc import HPCRunner

logger = get_logger()


class Runner(BasePipelineWorkflow, ABC):
    """Abstract class for workflow runners."""

    # TODO Generic type for pipeline config and pipeline step config attributes

    def __init__(
        self,
        subcommand: str,
        simulate: bool = False,
        keep_workdir: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.subcommand = subcommand
        self.simulate = simulate
        self.keep_workdir = keep_workdir

    def run_setup(self):
        """Run pipeline setup and validate the pipeline bundle."""
        to_return = super().run_setup()
        check_pipeline_bundle(self.dpath_pipeline_bundle, strict=False)
        return to_return

    @cached_property
    def hpc_runner(self) -> HPCRunner:
        """Get the HPC runner service."""
        return HPCRunner(
            hpc_cluster=self.hpc,
            hpc_config=self.hpc_config,
            subcommand=self.subcommand,
            dpath_root=self.dpath_root,
            dpath_hpc=self.study.layout.dpath_hpc,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_step=self.pipeline_step,
            preamble=self.study.config.HPC_PREAMBLE,
            queue_limit=self.study.config.HPC_QUEUE_LIMIT,
            keep_workdir=self.keep_workdir,
            fpath_layout=self.fpath_layout,
            verbose=self.verbose,
        )

    @cached_property
    def hpc_config(self) -> HpcConfig:
        """Load the pipeline step's HPC configuration."""
        fname_hpc_config = self.pipeline_step_config.HPC_CONFIG_FILE
        if fname_hpc_config is None:
            data = {}
        else:
            fpath_hpc_config = self.dpath_pipeline_bundle / fname_hpc_config
            logger.info(f"Loading HPC config from {fpath_hpc_config}")
            data = self.process_template_json(
                load_json(fpath_hpc_config, allow_json5=True)
            )
        return HpcConfig(**data)

    def _generate_cli_command_for_hpc(
        self, participant_id: str | None = None, session_id: str | None = None
    ) -> list[str]:
        """Generate the CLI command to be run on the HPC cluster."""
        return self.hpc_runner.generate_cli_command(
            participant_id=participant_id,
            session_id=session_id,
        )

    def _submit_hpc_job(self, participants_sessions):
        """Submit jobs to a HPC cluster for processing."""
        # generate the list of nipoppy commands for a shell array
        job_array_commands = []
        participant_ids = []
        session_ids = []
        for participant_id, session_id in participants_sessions:
            command = self._generate_cli_command_for_hpc(
                participant_id=participant_id, session_id=session_id
            )
            job_array_commands.append(shlex.join(command))
            participant_ids.append(participant_id)
            session_ids.append(session_id)

        job_name = get_pipeline_tag(
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_step=self.pipeline_step,
            participant_id=self.participant_id,
            session_id=self.session_id,
        )

        n_submitted_jobs = self.hpc_runner.submit(
            job_name=job_name,
            job_array_commands=job_array_commands,
            participant_ids=participant_ids,
            session_ids=session_ids,
            dpath_work=self.dpath_pipeline_work,
            dpath_hpc_logs=self.study.layout.dpath_logs / self.dname_hpc_logs,
            fname_hpc_error=self.fname_hpc_error,
            fname_job_script=self.fname_job_script,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_step=self.pipeline_step,
            dry_run=self.dry_run,
        )

        # for logging
        self.n_success += n_submitted_jobs
        self.n_total += len(job_array_commands)

    @cached_property
    def bosh_runner(self) -> BoshRunnerCallable:
        """Get the bosh exec command."""
        if self.simulate:
            return run_bosh_simulate
        else:
            return run_bosh_launch

    def launch_boutiques_run(
        self, participant_id, session_id, container_handler=None, objs=None, **kwargs
    ):
        """Prepare and execute a pipeline invocation using the Boutiques module."""
        prepared = prepare_invocation(
            self.descriptor,
            self.invocation,
            self.process_template_json,
            self.pipeline_config.CONTAINER_INFO.URI,
            lambda: self.fpath_container,
            participant_id,
            session_id,
            container_handler,
            objs,
            self.verbose,
            **kwargs,
        )
        execute_invocation(prepared, self.bosh_runner, _run_command, self.dry_run)
        return prepared.descriptor_str, prepared.invocation_str

    def process_container_config(self, participant_id, session_id, bind_paths=None):
        """Render container options and add domain bind paths."""
        return prepare_container(
            self.pipeline_step_config.get_container_config(),
            self.boutiques_config,
            self.process_template_json,
            self.study.layout.dpath_root,
            participant_id,
            session_id,
            bind_paths,
        )

    @override
    def _handle_execution_strategy(self, participants_sessions):
        """Handle the execution strategy based on the workflow configuration.

        Same as the parent, with HPC submission added as an option if configured.
        """
        if self.write_subcohort is not None:
            self._write_subcohort_to_file(participants_sessions)
        elif self.hpc is not None:
            self._submit_hpc_job(participants_sessions)
        else:
            self._run_locally(participants_sessions)
