"""Subprocess command execution with logging and dry-run support."""

import logging
import shlex
import subprocess
from collections.abc import Sequence
from typing import Protocol

from nipoppy.logger import get_logger

logger = get_logger()


class LogPrefix:
    """Prefixes for logging subprocess output."""

    RUN = "[RUN]"
    RUN_STDOUT = "[RUN STDOUT]"
    RUN_STDERR = "[RUN STDERR]"


def _log_command(command: str):
    """Write a command to the log with a special prefix."""
    # using extra={"markup": False} in case the command contains substrings
    # that would be interpreted as closing tags by the RichHandler
    logger.info(f"{LogPrefix.RUN} {command}", extra={"markup": False})


class CommandRunner(Protocol):
    """Protocol for functions that run commands, used for strategy injection."""

    def __call__(
        self,
        command_or_args: Sequence[str] | str,
        /,
        *,
        check: bool = True,
        quiet: bool = False,
        dry_run: bool = False,
    ) -> subprocess.Popen[str] | str:
        """Run a command in a subprocess, with logging and dry-run support."""
        ...


def _run_command(
    command_or_args: Sequence[str] | str,
    /,
    *,
    check: bool = True,
    quiet: bool = False,
    dry_run: bool = False,
    **kwargs,
) -> subprocess.Popen[str] | str:
    """Run a command in a subprocess.

    The command's stdout and stderr outputs are written to the log
    with special prefixes.

    If in "dry run" mode, the command is not executed, and the method returns
    the command string. Otherwise, the subprocess.Popen object is returned
    unless capture_output is True.

    Parameters
    ----------
    command_or_args : Sequence[str]  |  str
        The command to run.
    check : bool, optional
        If True, raise an error if the process exits with a non-zero code,
        by default True
    quiet : bool, optional
        If True, do not log the command, by default False
    **kwargs
        Passed to `subprocess.Popen`.

    Returns
    -------
    subprocess.Popen or str
    """

    def process_output(output_source, log_prefix: str, log_level=logging.INFO):
        """Consume lines from an IO stream and log them."""
        for line in output_source:
            line = line.strip("\n")
            # using extra={"markup": False} in case the output contains substrings
            # that would be interpreted as closing tags by the RichHandler
            logger.log(
                level=log_level,
                msg=f"{log_prefix} {line}",
                extra={"markup": False},
            )

    # build command string
    if not isinstance(command_or_args, str):
        args = [str(arg) for arg in command_or_args]
        command = shlex.join(args)
    else:
        command = command_or_args
        args = shlex.split(command)

    # only pass a single string if shell is True
    if not kwargs.get("shell"):
        command_or_args = args

    if not quiet:
        _log_command(command)

    if not dry_run:
        process: subprocess.Popen[str] = subprocess.Popen(
            command_or_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs,
        )

        while process.poll() is None:
            process_output(
                process.stdout,
                LogPrefix.RUN_STDOUT,
            )

            process_output(
                process.stderr,
                LogPrefix.RUN_STDERR,
                log_level=logging.ERROR,
            )

        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        run_output = process

    else:
        run_output = command

    return run_output
