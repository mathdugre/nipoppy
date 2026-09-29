"""Extraction eligibility and paths to the first processing dependency."""

from functools import cached_property
from pathlib import Path

from nipoppy.config.pipeline import ProcessingPipelineConfig
from nipoppy.env import PipelineTypeEnum
from nipoppy.workflows.execution import PipelineExecutionWorkflow
from nipoppy.workflows.services.cohort import WorkItem


class ExtractionRunner(PipelineExecutionWorkflow):
    """Extract IDPs from participant/sessions completed by every dependency."""

    @cached_property
    def source(self):
        """Validate the first processing dependency, which determines data paths."""
        source = self.pipeline.config.PROC_DEPENDENCIES[0]
        bundle = self.dataset.study.layout.get_dpath_pipeline_bundle(
            PipelineTypeEnum.PROCESSING, source.NAME, source.VERSION
        )
        self.pipeline.load_config(
            bundle, source.NAME, source.VERSION, ProcessingPipelineConfig
        ).get_step_config(step_name=source.STEP)
        return source

    @cached_property
    def derivatives(self) -> Path:
        """Derivatives directory of the first processing dependency."""
        return self.dataset.study.layout.get_dpath_pipeline(
            self.source.NAME, self.source.VERSION
        )

    @cached_property
    def output(self) -> Path:
        """Processed data directory of the first dependency."""
        return self.dataset.study.layout.get_dpath_pipeline_output(
            self.source.NAME, self.source.VERSION
        )

    @cached_property
    def idp(self) -> Path:
        """IDP output directory associated with the first dependency."""
        return self.dataset.study.layout.get_dpath_pipeline_idp(
            self.source.NAME, self.source.VERSION
        )

    def template_values(self):
        """Register dependency paths without loading them during construction."""
        return {
            "dpath_pipeline": lambda: self.derivatives,
            "dpath_pipeline_output": lambda: self.output,
            "dpath_pipeline_idp": lambda: self.idp,
        }

    def setup_directories(self):
        """Create extractor work files under its own identity and IDPs under source."""
        return [self.workspace.work, self.idp]

    def select_candidates(self):
        """Intersect completion records for every processing dependency."""
        candidates = None
        for dependency in self.pipeline.config.PROC_DEPENDENCIES:
            completed = set(
                self.dataset.processing_status_table.get_completed_participants_sessions(
                    pipeline_name=dependency.NAME,
                    pipeline_version=dependency.VERSION,
                    pipeline_step=dependency.STEP,
                    participant_id=self.selection.participant_id,
                    session_id=self.selection.session_id,
                )
            )
            candidates = completed if candidates is None else candidates & completed
        return sorted(candidates)

    def execute_item(self, item: WorkItem):
        """Execute extraction with IDP and processed-data bind paths."""
        return self.execute_pipeline(item, [self.idp, self.output])
