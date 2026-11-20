"""Exception handling for CLI commands."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nipoppy.workflows.base import BaseWorkflow


class WorkflowHandle:
    """Track the workflow run inside :func:`exception_handler`."""

    def __init__(self) -> None:
        self.workflow: BaseWorkflow | None = None

    def run(self, workflow: BaseWorkflow) -> None:
        """Register and run a workflow."""
        self.workflow = workflow
        workflow.run()


@contextmanager
def exception_handler() -> Iterator[WorkflowHandle]:
    """Exit the CLI process with the workflow's return code.

    Errors raised by ``handle.run(workflow)`` are handled by ``BaseWorkflow.run``,
    which sets ``workflow.return_code``. Errors raised before that (e.g. while
    constructing the workflow or objects passed to it) are handled here.
    """
    handle = WorkflowHandle()
    try:
        yield handle
    except (Exception, KeyboardInterrupt) as exception:
        if handle.workflow is not None:
            # already handled by BaseWorkflow.run
            return_code = handle.workflow.return_code
        else:
            # imported lazily, like the workflows in cli.py, to keep startup fast
            from nipoppy.workflows.base import handle_exception

            return_code = handle_exception(exception)
    else:
        return_code = handle.workflow.return_code
    sys.exit(return_code)
