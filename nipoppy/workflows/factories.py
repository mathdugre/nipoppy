"""Explicit construction seams for pipeline CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nipoppy.env import StrOrPathLike

if TYPE_CHECKING:
    from nipoppy.workflows.bids_conversion import BIDSificationRunner
    from nipoppy.workflows.extractor import ExtractionRunner
    from nipoppy.workflows.processing_runner import ProcessingRunner


def create_processing_workflow(
    dpath_root: StrOrPathLike,
    pipeline_name: str,
    pipeline_version: str | None = None,
    pipeline_step: str | None = None,
    participant_id: str = None,
    session_id: str = None,
    use_subcohort: StrOrPathLike | None = None,
    simulate: bool = False,
    keep_workdir: bool = False,
    tar: bool = False,
    write_subcohort: StrOrPathLike | None = None,
    fpath_layout: StrOrPathLike | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    hpc: str | None = None,
) -> ProcessingRunner:
    """Construct the processing workflow from normalized CLI parameters."""
    from nipoppy.workflows.processing_runner import ProcessingRunner

    return ProcessingRunner(
        dpath_root=dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
        participant_id=participant_id,
        session_id=session_id,
        use_subcohort=use_subcohort,
        simulate=simulate,
        keep_workdir=keep_workdir,
        tar=tar,
        write_subcohort=write_subcohort,
        fpath_layout=fpath_layout,
        verbose=verbose,
        dry_run=dry_run,
        hpc=hpc,
    )


def create_bidsification_workflow(
    dpath_root: StrOrPathLike,
    pipeline_name: str,
    pipeline_version: str | None = None,
    pipeline_step: str | None = None,
    participant_id: str = None,
    session_id: str = None,
    use_subcohort: StrOrPathLike | None = None,
    simulate: bool = False,
    keep_workdir: bool = False,
    hpc: str | None = None,
    write_subcohort: StrOrPathLike | None = None,
    fpath_layout: StrOrPathLike | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> BIDSificationRunner:
    """Construct the bidsification workflow from normalized CLI parameters."""
    from nipoppy.workflows.bids_conversion import BIDSificationRunner

    return BIDSificationRunner(
        dpath_root=dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
        participant_id=participant_id,
        session_id=session_id,
        use_subcohort=use_subcohort,
        simulate=simulate,
        keep_workdir=keep_workdir,
        hpc=hpc,
        write_subcohort=write_subcohort,
        fpath_layout=fpath_layout,
        verbose=verbose,
        dry_run=dry_run,
    )


def create_extraction_workflow(
    dpath_root: StrOrPathLike,
    pipeline_name: str,
    pipeline_version: str | None = None,
    pipeline_step: str | None = None,
    participant_id: str = None,
    session_id: str = None,
    use_subcohort: StrOrPathLike | None = None,
    simulate: bool = False,
    keep_workdir: bool = False,
    hpc: str | None = None,
    write_subcohort: StrOrPathLike | None = None,
    fpath_layout: StrOrPathLike | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> ExtractionRunner:
    """Construct the extraction workflow from normalized CLI parameters."""
    from nipoppy.workflows.extractor import ExtractionRunner

    return ExtractionRunner(
        dpath_root=dpath_root,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_step=pipeline_step,
        participant_id=participant_id,
        session_id=session_id,
        use_subcohort=use_subcohort,
        simulate=simulate,
        keep_workdir=keep_workdir,
        hpc=hpc,
        write_subcohort=write_subcohort,
        fpath_layout=fpath_layout,
        verbose=verbose,
        dry_run=dry_run,
    )
