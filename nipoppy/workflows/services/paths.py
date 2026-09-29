"""Paths to intermediate files for the requested pipeline and outer CLI filters."""

from functools import cached_property
from pathlib import Path


class PipelineWorkspace:
    """Keep intermediate path identity independent of extraction dependencies."""

    def __init__(self, layout, pipeline, selection):
        self.layout = layout
        self.pipeline = pipeline
        self.selection = selection

    def _kwargs(self):
        return {
            "pipeline_name": self.pipeline.name,
            "pipeline_version": self.pipeline.version,
            "pipeline_step": self.pipeline.step_name,
            "participant_id": self.selection.participant_id,
            "session_id": self.selection.session_id,
        }

    @cached_property
    def work(self) -> Path:
        """Working directory keyed by the outer CLI filters, not an individual item."""
        return self.layout.get_dpath_pipeline_work(**self._kwargs())

    @cached_property
    def bids_db(self) -> Path:
        """PyBIDS database directory using the same outer filter identity."""
        return self.layout.get_dpath_pybids_db(**self._kwargs())
