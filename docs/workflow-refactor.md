# Pipeline workflow refactor: implementation and review guide

The executable processing, BIDSification and extraction workflows now have this
inheritance chain:

```text
BaseWorkflow
└── PipelineExecutionWorkflow
    ├── ProcessingRunner
    ├── BIDSificationRunner
    └── ExtractionRunner
```

`nipoppy.base.Base` still supplies string representations. It adds no workflow
behavior. The three runners inherit two shared workflow behavior bases.

## Commit series

The starting point is `67dd5223` on `maint/flat-workflows`. Each stage has its own
local commit. Nothing in this series was pushed or submitted as a GitHub PR.

| Stage | Commit | Responsibility |
| --- | --- | --- |
| 1 | `fdb6688e` | Characterize observable CLI, execution, validation and cleanup behavior. |
| 2 | `4788f565` | Put CLI construction behind three explicit factory functions. |
| 3 | `9ab227a3` | Extract dataset discovery, table access and log path construction. |
| 4 | `e24d47cd` | Extract JSON rendering with an explicit, lazy runtime namespace. |
| 5 | `d94b8671` | Extract version resolution and runtime pipeline artifact loading. |
| 6 | `edaca90f` | Extract cohort filtering, analysis levels and work-item planning. |
| 7 | `90f2b12e` | Extract local execution, failure isolation and typed results. |
| 8 | `d852872d` | Move container and invocation mechanics into Boutiques; move subprocess utilities into `nipoppy.utils.command`. |
| 9 | `2fbe71c4` | Introduce structured scheduler jobs and submission results. |
| 10 | `08f665d5` | Introduce normalized inputs, dependency assembly and the shallow execution base. |
| 11 | `cff0393b` | Migrate BIDSification eligibility and status persistence. |
| 12 | `51d47a02` | Migrate extraction dependencies, paths and eligibility. |
| 13 | `93fac483` | Migrate processing; extract PyBIDS and archive algorithms. |
| 14 | Final commit in this series | Remove the old runner, trim tracking compatibility, migrate remaining tests and document the final architecture. |

Review a stage with `git show <commit>`. Review the complete series with
`git diff 67dd5223..HEAD`. Stages 3–9 temporarily retained delegating methods in
old bases so extraction could be reviewed independently of runner migration.
Those runner compatibility methods are removed in stage 14.

## Responsibility and dependency map

| Module | Owns | Does not receive |
| --- | --- | --- |
| `workflows/base.py` | Workflow lifecycle and logging; the dataset compatibility base used by unrelated workflows. | Pipeline execution policies. |
| `workflows/execution.py` | Setup/main sequencing, strategy dispatch, outcome application and explicit domain hooks. | Concrete runner types. |
| `workflows/factories.py` | CLI argument normalization and dependency construction. | Execution results. |
| `workflows/services/models.py` | `PipelineRequest` and `ExecutionOptions`. | Workflow instances. |
| `workflows/services/dataset.py` | Project discovery, `Study` construction, cached table fallbacks and timestamped log paths. | Workflow instances. |
| `workflows/services/artifacts.py` | Installed version resolution, `ResolvedPipeline`, runtime config loading and artifact caches. | Domain execution hooks or workflow instances. |
| `workflows/services/templates.py` | User substitution and lazy runtime replacement precedence. | An entire workflow object as the runtime namespace. |
| `workflows/services/paths.py` | Intermediate paths keyed by requested pipeline identity and outer filters. | Concrete runner types. |
| `workflows/services/cohort.py` | `SelectionOptions`, `WorkItem`, subcohort files and analysis-level collapse. | Status tables or eligibility rules. |
| `workflows/services/execution.py` | Ordered local execution, optional threading, progress, `ItemResult`, `BatchResult`, summary policy. | Dataset or concrete workflow instances. |
| `workflows/services/boutiques.py` | Container options, image checks/injection, descriptor/invocation preparation and subprocess error translation. | Concrete workflow instances. |
| `workflows/services/hpc.py` | Child CLI commands, queue capacity, aligned `HPCJob` items, filesystem preparation and `SubmissionResult`. | Concrete workflow instances. |
| `workflows/services/bids.py` | Filtered PyBIDS database construction and exact-match path injection. | Workflow instances. |
| `workflows/services/archive.py` | Archive config validation and tar validation before source removal. | Workflow instances. |
| `utils/command.py` | Logged subprocess execution and dry-run handling. | The workflow layer. |

The runtime namespace is a mapping of names to lazy value suppliers. Only names
referenced by a template are evaluated. Concrete runners register their own path
names explicitly; the renderer does not discover their attributes.

The shared execution base has two required domain hooks: `select_candidates()`
and `execute_item(WorkItem)`. Optional hooks describe setup directories, template
values and child CLI flags. Concrete runners retain their setup, persistence,
archive and cleanup decisions. No shared execution code branches on concrete
runner classes.

`BasePipelineWorkflow` remains for `PipelineTracker`. It delegates local execution
and artifact loading to the same modules. Tracking retains participant/session
items even when a step declares a broader analysis level. Dataset initialization,
reorganization, curation and pipeline-store workflows retain their dataset base.

## Lifecycle and compatibility rules

1. Construct readers lazily; configuration and descriptors are not loaded by the
   factories. `run()` still guarantees cleanup in a `finally` block.
2. Validate the dataset layout before opening the log file. Log path generation
   resolves a missing version and step. Check pipeline variables and filter
   compatibility before creating domain directories.
3. Create domain directories before raw bundle validation, matching the existing
   failure behavior. Raw validation remains in `pipeline_validation.py`; runtime
   substitution and inherited container configuration remain in artifact loading.
4. Ask the concrete runner for eligible participant/session pairs. Apply optional
   subcohort membership, then collapse to the configured analysis level.
5. Dispatch export, HPC or local execution. Local failures are isolated per item;
   successful payloads and failed placeholders remain in input order. Apply the
   existing summary and exit-code policy.
6. Run domain persistence and cleanup at the established points.

The characterization tests intentionally preserve these less obvious behaviors:

- Group analysis creates one `(None, None)` item even for an empty eligible cohort.
- User substitutions precede runtime ID replacement; explicit injected values and
  item IDs precede extra objects, runtime values and layout attributes.
- Extraction eligibility intersects **every** processing dependency. Data and IDP
  paths use the **first** dependency; work and bundle paths use the extractor.
- Processing cleanup compares successful and requested counts, including zero
  against zero after setup failure or export. Partial failure retains work files.
- BIDSification updates the cached table after success, including simulation and
  dry runs. Simulation suppresses final persistence; dry runs suppress file writes.
- HPC dry runs query capacity, remove stale scheduler errors and create the log
  directory. They report eligible slots as submitted without submitting jobs.
- HPC capacity limits keep requested and submitted counts distinct and retain the
  existing exit-code policy. Child commands omit outer simulation/dry-run/HPC/export
  flags and retain the appropriate layout, verbosity, work and archive flags.
- Archives are created after successful non-simulated execution. The original
  directory is removed only when the resulting tar file exists and validates.

These behaviors are documented for subsequent changes; this series does not fix
or redefine them. See `tests/integration/workflow-contracts.md` for the detailed
coverage matrix and the original baseline.

## Python construction

CLI-shaped Python callers should use the factory functions:

```python
from nipoppy.workflows.factories import create_processing_workflow

workflow = create_processing_workflow(
    dpath_root="/path/to/study",
    pipeline_name="fmriprep",
    pipeline_version="23.1.3",
    keep_workdir=True,
)
workflow.run()
return_code = workflow.return_code
```

BIDSification and extraction have corresponding `create_bidsification_workflow`
and `create_extraction_workflow` factories. The CLI commands, decorators, options,
aliases and exception-handler contract remain unchanged.

Direct runner constructors now accept explicit assembled dependencies. Internal
callers can supply their own dataset, pipeline reader, selection/execution options,
workspace and renderer. The concrete processing constructor additionally accepts
`tar`. The old runner constructor arguments, helper methods and `Runner` import
are intentionally removed; Python internals were permitted to change.

## Validation and remaining limits

Final validation uses Python 3.14 in the existing cached environment:

```sh
COPYFILE_DISABLE=1 uv run --no-sync python -m pytest \
  -m 'not api and not linkcheck and no_xdist' -n 0 -o addopts='' -q
COPYFILE_DISABLE=1 uv run --no-sync python -m pytest \
  -m 'not api and not linkcheck and not no_xdist' -n auto \
  --dist loadfile -o addopts='' -q
```

Results: **130 serial + 1,454 parallel = 1,584 passed**. The 131 integration cases
exercise real CLI/configuration/filesystem behavior while replacing external
Boutiques command execution and queue submission. Archive cases run actual tar.
Focused tests cover lazy rendering and normalized construction. Architecture tests
assert the two shared behavior bases, deletion of the old layer, dependency
orientation and absence of generic attribute forwarding.

Formatting, lint, AST parsing and the repository commit hooks are checked for each
stage. Other supported Python versions and live scheduler/container/API execution
have not been validated in this environment. API/linkcheck tests are excluded.

Documentation also passes a strict HTML build after generating the required inputs:

```sh
uv run --no-sync python docs/scripts/pydantic_to_jsonschema.py
uv run --no-sync python docs/scripts/process_template_json5_configs.py
uv run --no-sync sphinx-build -E -W --keep-going -b html \
  docs/source /tmp/nipoppy-spirula-docs
```

The generated schemas/examples and rendered documentation are build artifacts;
they are not included in the commit series.
