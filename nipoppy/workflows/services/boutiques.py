"""Boutiques runner service."""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from boutiques import bosh
from pydantic import ValidationError

from nipoppy.config.boutiques import (
    BoutiquesConfig,
    get_boutiques_config_from_descriptor,
)
from nipoppy.config.container import ContainerConfig
from nipoppy.container import get_container_handler
from nipoppy.env import ContainerCommandEnum
from nipoppy.exceptions import (
    ConfigError,
    ContainerError,
    ExecutionError,
    FileOperationError,
    WorkflowError,
)
from nipoppy.logger import get_logger
from nipoppy.utils.command import CommandRunner
from nipoppy.utils.utils import TEMPLATE_REPLACE_PATTERN

logger = get_logger()


class BoshRunnerCallable(Protocol):
    """Protocol for the bosh runner callable."""

    def __call__(
        self,
        invocation_str: str,
        descriptor_str: str,
        bosh_exec_launch_args: list[str],
        run_command: CommandRunner | None,
        dry_run: bool,
    ) -> int:
        """Run the bosh command with the given arguments."""
        ...


class CommandBuilder(Protocol):
    """Protocol for a function that builds a command ro run.

    It takes the invocation, descriptor, and additional args and returns a list of
    command arguments.
    """

    def __call__(
        self,
        invocation: str,
        descriptor: str,
        args: list[str],
    ) -> list[str]:
        """Build command from invocation, descriptor, and extra args."""
        ...


class ErrorBuilder(Protocol):
    """Protocol for a function that builds an error message based on the exit code."""

    def __call__(self, exit_code: int) -> str:
        """Build error message based on the exit code."""
        ...


def _run_bosh_command(
    invocation_str: str,
    descriptor_str: str,
    run_command: CommandRunner,
    bosh_exec_launch_args: list[str] | None = None,
    dry_run: bool = False,
    *,
    mode: str,
    command_builder: CommandBuilder,
    error_message_builder: ErrorBuilder,
) -> int:
    """Execute a Boutiques command with shared error handling.

    Parameters
    ----------
    command_builder : CommandBuilder
        A function that builds the command to run based on the invocation, descriptor,
        and additional args. Returns a list of command arguments.
    error_message_builder : ErrorBuilder
        A function that builds an error message based on the exit code of a failed
        command.
    """
    if bosh_exec_launch_args is None:
        bosh_exec_launch_args = []

    command = command_builder(
        invocation_str,
        descriptor_str,
        bosh_exec_launch_args,
    )
    logger.info(f"{mode} pipeline command")

    try:
        run_command(command, quiet=True, dry_run=dry_run)
    except subprocess.CalledProcessError as exception:
        raise ExecutionError(error_message_builder(exception.returncode))

    return 0


def run_bosh_launch(
    invocation_str: str,
    descriptor_str: str,
    run_command: CommandRunner,
    bosh_exec_launch_args: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Execute a Boutiques launch command.

    Parameters
    ----------
    invocation_str : str
        The Boutiques invocation as a JSON string.
    descriptor_str : str
        The Boutiques descriptor as a JSON string.
    run_command : CommandRunner, optional
        A function to execute the command. Should act like ``runner.run_command``.
    bosh_exec_launch_args : list of str, optional
        Additional arguments for ``bosh exec launch``.
    dry_run : bool, optional
        If True, build and log the command but skip actual execution.

    Returns
    -------
    int
        The return code.
    """

    def command_builder(invocation: str, descriptor: str, args: list[str]) -> list[str]:
        return [
            "bosh",
            "exec",
            "launch",
            "--stream",
            descriptor,
            invocation,
        ] + args

    def error_message_builder(exit_code: int) -> str:
        return (
            f"Pipeline execution failed (return code: {exit_code})."
            "Hint: make sure the shell command above is correct."
        )

    return _run_bosh_command(
        invocation_str=invocation_str,
        descriptor_str=descriptor_str,
        bosh_exec_launch_args=bosh_exec_launch_args,
        run_command=run_command,
        dry_run=dry_run,
        mode="Running",
        command_builder=command_builder,
        error_message_builder=error_message_builder,
    )


def run_bosh_simulate(
    invocation_str: str,
    descriptor_str: str,
    run_command: CommandRunner,
    bosh_exec_launch_args: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Execute a Boutiques simulate command.

    Parameters
    ----------
    invocation_str : str
        The Boutiques invocation as a JSON string.
    descriptor_str : str
        The Boutiques descriptor as a JSON string.
    run_command : CommandRunner, optional
        A function to execute the command. Should act like ``runner.run_command``.
    bosh_exec_launch_args : list of str, optional
        Additional arguments for ``bosh exec launch``.
    dry_run : bool, optional
        If True, build and log the command but skip actual execution.

    Returns
    -------
    int
        The return code.
    """

    def command_builder(invocation: str, descriptor: str, args: list[str]) -> list[str]:
        return [
            "bosh",
            "exec",
            "simulate",
            "-i",
            invocation,
            descriptor,
        ]

    def error_message_builder(exit_code: int) -> str:
        return f"Pipeline simulation failed (return code: {exit_code})"

    logger.info(f"Additional launch options: {bosh_exec_launch_args}")
    rv = _run_bosh_command(
        invocation_str=invocation_str,
        descriptor_str=descriptor_str,
        bosh_exec_launch_args=bosh_exec_launch_args,
        run_command=run_command,
        dry_run=dry_run,
        mode="Simulating",
        command_builder=command_builder,
        error_message_builder=error_message_builder,
    )
    return rv


@dataclass(frozen=True)
class PreparedInvocation:
    """Validated Boutiques inputs and arguments ready for execution."""

    descriptor_str: str
    invocation_str: str
    launch_args: tuple[str, ...]


def inject_container_image(descriptor: dict, uri: str) -> dict:
    """Copy descriptor metadata and fill its container image from the bundle URI."""
    descriptor = copy.deepcopy(descriptor)
    scheme, sep, image = uri.partition("://")
    if not sep:
        logger.error(f"Failed to parse CONTAINER_INFO.URI {uri}.")
    else:
        container_type = "docker" if scheme == "docker" else "singularity"
        descriptor["container-image"] = {
            "image": image,
            "type": container_type,
        }
    return descriptor


def prepare_invocation(
    descriptor,
    invocation,
    render,
    container_uri,
    container_file,
    participant_id,
    session_id,
    container_handler=None,
    objs=None,
    verbose=False,
    **kwargs,
):
    """Launch a pipeline run using Boutiques."""
    bosh_exec_launch_args = ["--no-pull"]

    if verbose:
        bosh_exec_launch_args.append("--debug")

    # process the descriptor if it containers Nipoppy-specific placeholder
    # expressions (legacy behaviour)
    if TEMPLATE_REPLACE_PATTERN.search(descriptor["command-line"]) is not None:
        logger.info("Processing the JSON descriptor")
        descriptor_str = render(
            descriptor,
            participant_id=participant_id,
            session_id=session_id,
            objs=objs,
            **kwargs,
            return_str=True,
        )
    else:
        # if the descriptor is missing "container-image" but CONTAINER_INFO.URI is
        # set in the pipeline config, inject "container-image" so that the pipeline
        # container will still run in a container
        if descriptor.get("container-image") is None and container_uri is not None:
            logger.warning(
                "Descriptor is missing a 'container-image' field."
                " Using information from CONTAINER_INFO.URI instead."
            )

            descriptor = inject_container_image(descriptor, container_uri)

        descriptor_str = json.dumps(descriptor)
        if container_handler is None or descriptor.get("container-image") is None:
            bosh_exec_launch_args.append("--no-container")
        else:
            bosh_exec_launch_args.extend(
                [
                    "--no-automount",
                    f"--container-opts={shlex.join(container_handler.args)}",
                ]
            )
            match container_handler.command:
                case ContainerCommandEnum.SINGULARITY:
                    bosh_exec_launch_args.extend(
                        [
                            "--imagepath",
                            str(container_file()),
                            "--force-singularity",
                        ]
                    )
                case ContainerCommandEnum.APPTAINER:
                    bosh_exec_launch_args.extend(
                        [
                            "--imagepath",
                            str(container_file()),
                            "--force-apptainer",
                        ]
                    )
                case ContainerCommandEnum.DOCKER:
                    bosh_exec_launch_args.append("--force-docker")

    # validate the descriptor
    logger.debug(f"Descriptor string: {descriptor_str}")
    logger.info("Validating the JSON descriptor")
    bosh(["validate", descriptor_str])

    # process and validate the invocation
    logger.info("Processing the JSON invocation")
    invocation_str = render(
        invocation,
        participant_id=participant_id,
        session_id=session_id,
        objs=objs,
        **kwargs,
        return_str=True,
    )
    logger.debug(f"Invocation string: {invocation_str}")
    logger.info("Validating the JSON invocation")
    bosh(["invocation", "-i", invocation_str, descriptor_str])

    return PreparedInvocation(
        descriptor_str, invocation_str, tuple(bosh_exec_launch_args)
    )


def prepare_container(
    container_config,
    boutiques_config,
    render,
    dpath_root,
    participant_id,
    session_id,
    bind_paths=None,
):
    """Update container config and generate container command."""
    if bind_paths is None:
        bind_paths = []

    # always bind the dataset's root directory
    bind_paths = [dpath_root] + bind_paths

    # get and process container config
    container_config = ContainerConfig(
        **render(
            container_config.model_dump(),
            participant_id=participant_id,
            session_id=session_id,
        )
    )
    logger.debug(f"Initial container config: {container_config}")

    # get and process Boutiques config
    boutiques_config = BoutiquesConfig(
        **render(
            boutiques_config.model_dump(),
            participant_id=participant_id,
            session_id=session_id,
        )
    )

    # update container config with additional information from Boutiques config
    logger.debug(f"Boutiques config: {boutiques_config}")
    if boutiques_config != BoutiquesConfig():
        logger.debug("Updating container config with config from descriptor")
        container_config.merge(boutiques_config.get_container_config())

    container_handler = get_container_handler(container_config)

    # add bind paths
    for bind_path in bind_paths:
        if Path(bind_path).resolve() != Path.cwd().resolve():
            container_handler.add_bind_arg(bind_path)

    logger.debug(f"Using container handler: {container_handler}")

    container_command = container_handler.get_shell_command(
        subcommand=boutiques_config.CONTAINER_SUBCOMMAND,
    )

    return container_command, container_handler


def execute_invocation(
    prepared: PreparedInvocation,
    runner: BoshRunnerCallable,
    run_command: CommandRunner,
    dry_run: bool = False,
):
    """Execute the validated invocation with the chosen Boutiques strategy."""
    return runner(
        invocation_str=prepared.invocation_str,
        descriptor_str=prepared.descriptor_str,
        bosh_exec_launch_args=list(prepared.launch_args),
        run_command=run_command,
        dry_run=dry_run,
    )


def resolve_container_file(config, step):
    """Return the full path to the pipeline's container."""
    uri = config.CONTAINER_INFO.URI
    fpath_container = config.CONTAINER_INFO.FILE
    container_handler = get_container_handler(
        step.CONTAINER_CONFIG,
    )

    try:
        is_downloaded = container_handler.is_image_downloaded(uri, fpath_container)
    except ContainerError as e:
        raise WorkflowError(
            f"Error in container config for pipeline {config.NAME} "
            f"{config.VERSION}: {e}"
        ) from e

    if not is_downloaded:
        error_message = (
            f"No container image file found for pipeline {config.NAME} {config.VERSION}"
        )
        if uri is not None:
            pull_command = container_handler.get_pull_command(uri, fpath_container)
            error_message += (
                ". This file can be downloaded to the appropriate path by running "
                f"the following command:\n\n{pull_command}"
            )
        raise FileOperationError(error_message)

    return fpath_container


def load_boutiques_config(descriptor):
    """Get the Boutiques configuration."""
    try:
        boutiques_config = get_boutiques_config_from_descriptor(
            descriptor,
        )
    except ValidationError as e:
        error_message = str(e) + str(e.errors())
        raise WorkflowError(
            f"Error when loading the Boutiques config from descriptor: {error_message}"
        )
    except ConfigError as e:
        logger.debug(
            "Caught exception when trying to load Boutiques config"
            f": {type(e).__name__}: {e}"
        )
        logger.debug("Assuming Boutiques config is not in descriptor. Using default")
        return BoutiquesConfig()

    logger.info(f"Loaded Boutiques config from descriptor: {boutiques_config}")
    return boutiques_config
