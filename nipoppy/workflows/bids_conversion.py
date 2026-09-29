"""BIDS conversion eligibility, bind paths and status persistence."""

from nipoppy.exceptions import WorkflowError
from nipoppy.workflows.execution import PipelineExecutionWorkflow
from nipoppy.workflows.services.cohort import WorkItem


class BIDSificationRunner(PipelineExecutionWorkflow):
    """Convert organized data to BIDS using the shared execution sequence."""

    def _unavailable_pipeline_path(self):
        raise WorkflowError(
            f'"dpath_pipeline" attribute is not available for {type(self)}'
        )

    def template_values(self):
        """Retain BIDSification's established path template semantics."""
        return {
            "dpath_pipeline": self._unavailable_pipeline_path,
            "dpath_pipeline_output": lambda: (
                self.dataset.study.layout.get_dpath_pipeline_output(
                    self.pipeline.name, self.pipeline.version
                )
            ),
        }

    def setup_directories(self):
        """BIDSification needs only its working directory created during setup."""
        return [self.workspace.work]

    def select_candidates(self):
        """Select organized participant/sessions that do not already have BIDS data."""
        table = self.dataset.curation_status_table
        filters = dict(
            participant_id=self.selection.participant_id,
            session_id=self.selection.session_id,
        )
        bidsified = set(table.get_bidsified_participants_sessions(**filters))
        return (
            pair
            for pair in table.get_organized_participants_sessions(**filters)
            if pair not in bidsified
        )

    def execute_item(self, item: WorkItem):
        """Execute conversion and update the cached table after successful execution."""
        layout = self.dataset.study.layout
        payload = self.execute_pipeline(
            item, [layout.dpath_post_reorg, layout.dpath_bids]
        )
        if self.pipeline.step.UPDATE_STATUS:
            table = self.dataset.curation_status_table
            table.set_status(
                participant_id=item.participant_id,
                session_id=item.session_id,
                col=table.col_in_bids,
                status=True,
            )
        return payload

    def _write_status_file(self):
        """Persist status after main execution unless simulation suppresses writing."""
        if self.pipeline.step.UPDATE_STATUS and not self.execution.simulate:
            self.dataset.curation_status_table.save_with_backup(
                self.dataset.study.layout.fpath_curation_status, dry_run=self.dry_run
            )

    def run_main(self):
        """Run the shared execution sequence, then persist the conversion status."""
        super().run_main()
        self._write_status_file()
