"""Explicit construction seams for pipeline CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from nipoppy.env import PipelineTypeEnum, StrOrPathLike
from nipoppy.workflows.services.cohort import SelectionOptions
from nipoppy.workflows.services.models import ExecutionOptions, PipelineRequest

if TYPE_CHECKING:
    from nipoppy.study import Study
    from nipoppy.workflows.bids_conversion import BIDSificationRunner
    from nipoppy.workflows.extractor import ExtractionRunner
    from nipoppy.workflows.processing_runner import ProcessingRunner
    from nipoppy.workflows.services.artifacts import ResolvedPipeline
    from nipoppy.workflows.services.dataset import DatasetAccess
    from nipoppy.workflows.services.paths import PipelineWorkspace
    from nipoppy.workflows.services.templates import TemplateRenderer


class PipelineDependencies(TypedDict):
    """Named dependencies accepted by PipelineExecutionWorkflow constructors."""

    name: str
    dataset: DatasetAccess
    pipeline: ResolvedPipeline
    selection: SelectionOptions
    execution: ExecutionOptions
    workspace: PipelineWorkspace
    templates: TemplateRenderer


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

    request = PipelineRequest(
        PipelineTypeEnum.PROCESSING, pipeline_name, pipeline_version, pipeline_step
    )
    selection = SelectionOptions(participant_id, session_id, use_subcohort)
    execution = ExecutionOptions(
        simulate=simulate,
        keep_workdir=keep_workdir,
        hpc=hpc,
        write_subcohort=write_subcohort,
        fpath_layout=fpath_layout,
        verbose=verbose,
        dry_run=dry_run,
    )
    return ProcessingRunner(
        tar=tar,
        **create_pipeline_dependencies(
            dpath_root, "process", request, selection, execution
        ),
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

    request = PipelineRequest(
        PipelineTypeEnum.BIDSIFICATION, pipeline_name, pipeline_version, pipeline_step
    )
    selection = SelectionOptions(participant_id, session_id, use_subcohort)
    execution = ExecutionOptions(
        simulate=simulate,
        keep_workdir=keep_workdir,
        hpc=hpc,
        write_subcohort=write_subcohort,
        fpath_layout=fpath_layout,
        verbose=verbose,
        dry_run=dry_run,
    )
    return BIDSificationRunner(
        **create_pipeline_dependencies(
            dpath_root, "bidsify", request, selection, execution
        )
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

    request = PipelineRequest(
        PipelineTypeEnum.EXTRACTION, pipeline_name, pipeline_version, pipeline_step
    )
    selection = SelectionOptions(participant_id, session_id, use_subcohort)
    execution = ExecutionOptions(
        simulate=simulate,
        keep_workdir=keep_workdir,
        hpc=hpc,
        write_subcohort=write_subcohort,
        fpath_layout=fpath_layout,
        verbose=verbose,
        dry_run=dry_run,
    )
    return ExtractionRunner(
        **create_pipeline_dependencies(
            dpath_root, "extract", request, selection, execution
        )
    )


def create_pipeline_dependencies(
    dpath_root: StrOrPathLike,
    name: str,
    request: PipelineRequest,
    selection: SelectionOptions,
    execution: ExecutionOptions,
    *,
    study: Study | None = None,
) -> PipelineDependencies:
    """Assemble explicit dependencies without loading configuration or artifacts."""
    from nipoppy.workflows.services.artifacts import ResolvedPipeline
    from nipoppy.workflows.services.dataset import DatasetAccess, open_dataset
    from nipoppy.workflows.services.paths import PipelineWorkspace
    from nipoppy.workflows.services.templates import TemplateRenderer

    if study is None:
        study = open_dataset(dpath_root, execution.fpath_layout)
    dataset = DatasetAccess(study, dry_run=execution.dry_run)
    values = {}
    templates = TemplateRenderer(
        lambda: study.config.SUBSTITUTIONS, values, study.layout
    )
    pipeline = ResolvedPipeline(study, request, templates.render)
    workspace = PipelineWorkspace(study.layout, pipeline, selection)
    values.update(
        {
            "name": lambda: name,
            "subcommand": lambda: name,
            "dpath_root": lambda: study.layout.dpath_root,
            "fpath_layout": lambda: execution.fpath_layout,
            "pipeline_name": lambda: pipeline.name,
            "pipeline_version": lambda: pipeline.version,
            "pipeline_step": lambda: pipeline.step_name,
            "participant_id": lambda: selection.participant_id,
            "session_id": lambda: selection.session_id,
            "use_subcohort": lambda: selection.use_subcohort,
            "write_subcohort": lambda: execution.write_subcohort,
            "hpc": lambda: execution.hpc,
            "verbose": lambda: execution.verbose,
            "dry_run": lambda: execution.dry_run,
            "simulate": lambda: execution.simulate,
            "keep_workdir": lambda: execution.keep_workdir,
            "n_jobs": lambda: execution.n_jobs,
            "dpath_pipeline_bundle": lambda: pipeline.bundle,
            "dpath_pipeline_work": lambda: workspace.work,
            "dpath_pipeline_bids_db": lambda: workspace.bids_db,
            "fpath_container": lambda: pipeline.container_file,
        }
    )
    return dict(
        name=name,
        dataset=dataset,
        pipeline=pipeline,
        selection=selection,
        execution=execution,
        workspace=workspace,
        templates=templates,
    )
