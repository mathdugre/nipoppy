"""Workflow utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

from nipoppy.base import Base
from nipoppy.env import StrOrPathLike
from nipoppy.exceptions import ReturnCode
from nipoppy.logger import get_logger
from nipoppy.utils.command import (  # noqa: F401
    CommandRunner,
    LogPrefix,
    _log_command,
    _run_command,
)
from nipoppy.utils.utils import (
    is_nipoppy_project,
)
from nipoppy.workflows.services.dataset import (
    DatasetAccess,
    generate_log_path,
    open_dataset,
)

logger = get_logger()


class BaseWorkflow(Base, ABC):
    """Base workflow class with logging/subprocess/filesystem utilities."""

    def __init__(self, name: str, verbose: bool = False, dry_run: bool = False):
        """Initialize the workflow instance.

        Parameters
        ----------
        name : str
            Name of the workflow, used for logging.
        verbose : bool, optional
            If True, set the logger to DEBUG level, by default False
        dry_run : bool, optional
            If True, print commands without executing them, by default False
        """
        self.name = name
        self.dry_run = dry_run
        self.verbose = verbose

        # for the CLI
        self.return_code = ReturnCode.SUCCESS

        logger.set_verbose(self.verbose)

    def run_setup(self):
        """Run the setup part of the workflow."""
        logger.debug(self)
        if self.dry_run:
            logger.info("Doing a dry run")

    @abstractmethod
    def run_main(self):
        """Run the main part of the workflow."""
        pass

    def run_cleanup(self):
        """Run the cleanup part of the workflow."""
        pass

    def run(self):
        """Run the workflow."""
        try:
            self.run_setup()
            self.run_main()
        except Exception:
            raise
        finally:
            self.run_cleanup()


class BaseDatasetWorkflow(BaseWorkflow, ABC):
    """Base workflow class with awareness of dataset layout and components."""

    def __init__(
        self,
        dpath_root: StrOrPathLike,
        name: str,
        fpath_layout: StrOrPathLike | None = None,
        verbose: bool = False,
        dry_run: bool = False,
        _skip_logfile: bool = False,
        _validate_layout: bool = True,
    ):
        """Initialize the workflow instance.

        Parameters
        ----------
        dpath_root : nipoppy.env.StrOrPathLike
            Path the the root directory of the dataset.
        name : str
            Name of the workflow, used for logging.
        fpath_layout : nipoppy.env.StrOrPathLike, optional
            Path to a custom layout file, by default None
        verbose : bool, optional
            If True, set the logger to DEBUG level, by default False
        dry_run : bool, optional
            If True, print commands without executing them, by default False
        _skip_logfile : bool, optional
            If True, do not write log to file, by default False
        _validate_layout : bool, optional
            If True, validate the layout during setup, by default True
        """
        super().__init__(name=name, verbose=verbose, dry_run=dry_run)

        # `.nipoppy` is not created by default in version 0.3.4 and below
        self.dpath_root = is_nipoppy_project(dpath_root) or Path(dpath_root)
        self.fpath_layout = fpath_layout
        self._skip_logfile = _skip_logfile
        self._validate_layout = _validate_layout

        self.study = open_dataset(self.dpath_root, self.fpath_layout)

    def generate_fpath_log(self, dnames_parent=None, fname_stem=None) -> Path:
        """Generate a log file path."""
        return generate_log_path(
            self.study.layout, self.name, dnames_parent, fname_stem
        )

    @cached_property
    def dataset(self) -> DatasetAccess:
        """Dataset access shared with composed workflows."""
        return DatasetAccess(self.study, dry_run=self.dry_run)

    def run_setup(self):
        """Run the setup part of the workflow."""
        if self._validate_layout:
            self.study.layout.validate()

        if not self._skip_logfile:
            logger.add_file_handler(self.generate_fpath_log())

        super().run_setup()

    @cached_property
    def curation_status_table(self):
        """Read the dataset table using its missing-file fallback."""
        return self.dataset.curation_status_table

    @cached_property
    def processing_status_table(self):
        """Read the dataset table using its missing-file fallback."""
        return self.dataset.processing_status_table

    @cached_property
    def dicom_dir_map(self):
        """Read the dataset table using its missing-file fallback."""
        return self.dataset.dicom_dir_map
