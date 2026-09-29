"""Dataset loading and status-table access independent of workflow inheritance."""

from functools import cached_property
from pathlib import Path

from nipoppy.env import EXT_LOG, PROGRAM_NAME, StrOrPathLike
from nipoppy.exceptions import FileOperationError
from nipoppy.layout import DatasetLayout
from nipoppy.logger import get_logger
from nipoppy.study import Study
from nipoppy.tabular.curation_status import (
    CurationStatusTable,
    generate_curation_status_table,
)
from nipoppy.tabular.dicom_dir_map import DicomDirMap
from nipoppy.tabular.processing_status import ProcessingStatusTable
from nipoppy.utils.utils import add_path_timestamp, is_nipoppy_project

logger = get_logger()


def open_dataset(
    dpath_root: StrOrPathLike, fpath_layout: StrOrPathLike | None = None
) -> Study:
    """Locate the project root and construct its lazy study reader."""
    root = is_nipoppy_project(dpath_root) or Path(dpath_root)
    return Study(DatasetLayout(dpath_root=root, fpath_config=fpath_layout))


def generate_log_path(
    layout: DatasetLayout,
    name: str,
    dnames_parent: str | list[str] | None = None,
    fname_stem: str | None = None,
) -> Path:
    """Build a timestamped log path without creating directories."""
    parents = (
        [dnames_parent] if isinstance(dnames_parent, str) else (dnames_parent or [])
    )
    directory = layout.dpath_logs / PROGRAM_NAME / name
    for parent in parents:
        directory /= parent
    stem = name if fname_stem is None else fname_stem
    return directory / add_path_timestamp(f"{stem}{EXT_LOG}")


class DatasetAccess:
    """Read study tables, providing the established missing-file fallbacks."""

    def __init__(self, study: Study, dry_run: bool = False):
        self.study = study
        self.dry_run = dry_run

    @cached_property
    def curation_status_table(self) -> CurationStatusTable:
        """
        Load the curation status file if it exists.

        Otherwise, generate a new one.
        """
        fpath_table = Path(self.study.layout.fpath_curation_status)
        try:
            return self.study.curation_status_table
        except FileNotFoundError:
            logger.warning(
                f"Curation status file not found: {fpath_table}"
                ". Generating a new one on-the-fly"
            )
            table = generate_curation_status_table(
                manifest=self.study.manifest,
                dicom_dir_map=self.dicom_dir_map,
                dpath_downloaded=self.study.layout.dpath_pre_reorg,
                dpath_organized=self.study.layout.dpath_post_reorg,
                dpath_bidsified=self.study.layout.dpath_bids,
            )

            if not self.dry_run:
                table.save_with_backup(fpath_table)
            else:
                logger.info(
                    "Not writing curation status table to "
                    f"{fpath_table} since this is a dry run"
                )

            return table

    @cached_property
    def processing_status_table(self) -> ProcessingStatusTable:
        """
        Load the processing status file it it exists.

        Otherwise, return an empty processing status table.
        """
        try:
            return self.study.processing_status_table
        except FileNotFoundError:
            return ProcessingStatusTable()

    @cached_property
    def dicom_dir_map(self) -> DicomDirMap:
        """Get the DICOM directory mapping."""
        fpath_dicom_dir_map = self.study.config.DICOM_DIR_MAP_FILE
        if fpath_dicom_dir_map is not None and not Path(fpath_dicom_dir_map).exists():
            raise FileOperationError(
                f"DICOM directory map file not found: {fpath_dicom_dir_map}"
            )

        return DicomDirMap.load_or_generate(
            manifest=self.study.manifest,
            fpath_dicom_dir_map=fpath_dicom_dir_map,
            participant_first=self.study.config.DICOM_DIR_PARTICIPANT_FIRST,
        )
