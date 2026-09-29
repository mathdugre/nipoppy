# Pipeline workflow characterization — Stage 1

Baseline: `maint/flat-workflows`, commit
`67dd5223362828a55a4f54158ff8476ddc0d318f`.

This stage adds tests around current behavior. It does not change workflows,
constructors, configuration formats, or the inheritance hierarchy. Stage 2 begins
only after the user has reviewed this stage, created its PR, and authorized
continuation.

## Test approach

`test_workflow_contracts.py` invokes the real Click commands against temporary
datasets built with the existing helpers in `tests/conftest.py`. Configuration
loading, bundle validation, eligibility, rendering, table persistence, and cleanup
are real. External Boutiques subprocesses and the scheduler adapter are replaced;
archive tests execute the local `tar` command. No container, cluster, network
service, or neuroimaging pipeline is required.

The fixture installs two versions and two steps of each target pipeline. It gives
the extractor two processing dependencies with different completed cohorts,
making both the eligibility intersection and first-dependency path selection
observable. Tests do not create subclasses of the pipeline bases or assert their
inheritance. When execution modules move, update only the external command and
scheduler patch locations, retaining these behavior assertions.

Dotenv tests use temporary files; other tests disable dotenv loading to isolate
them from the user's configuration. Paths with spaces are exercised in child HPC
commands and custom layout files. No timestamps are asserted, except a controlled
clock change that proves a previous curation-status backup survives a later write.

## Coverage

| Contract | Coverage |
| --- | --- |
| CLI options, defaults, required pipeline, hidden aliases, layout envvar | New CLI option contract and existing `tests/unit/test_cli.py` |
| Default/explicit version and step, prefixed identifiers | New CLI command and filtering tests |
| Processing/BIDSification eligibility; all extraction dependencies | New CLI command/export tests |
| Extraction paths use the first dependency | New rendered invocation tests |
| Analysis levels, incompatible filters, empty selection, group execution | New CLI analysis/filter tests |
| Subcohort intersection, deprecated export alias, empty export, dry-run export | New CLI export tests |
| Invocation/descriptor strings and launch versus simulate flags | New command capture and existing `tests/unit/workflows/test_runner.py` |
| User substitutions, pipeline variables, identifier replacement | New invocation ordering test and existing `tests/unit/workflows/test_pipeline.py` |
| Extra template values, container commands and bind paths | Existing pipeline/runner unit tests |
| Local success, partial/all failure, continuation after failure | New CLI item failure tests |
| Sequential/Joblib results, optional dependency errors | Existing pipeline unit tests |
| BIDSification status flags, successful-item updates, simulation, backups | New CLI status and backup tests |
| Processing success/failure, setup failure, export and dry-run cleanup | New lifecycle/CLI filesystem tests |
| Processing archive sequencing, simulation, archive contents | New CLI archive tests and existing processing unit tests |
| PyBIDS preparation, injection cardinality, invalid archive and missing container | Existing processing unit tests |
| HPC child flags/quoting, array alignment, capacity, failures, dry-run effects | New CLI scheduler tests and existing HPC unit tests |
| Custom layout from CLI, environment, dotenv, CLI precedence | New custom-layout tests |
| Tracker remains at participant-session level | Existing `tests/unit/workflows/test_tracker.py` |
| Cleanup runs after setup/main failure; successful `run()` returns `None` | Extended `tests/unit/workflows/test_base.py` |

## Behaviors to preserve during extraction

These are observations about the baseline, not proposed fixes.

1. **Processing cleanup after setup failure:** when no items were attempted,
   successful and attempted counts are both zero. Unless `--keep-workdir` is set,
   cleanup removes existing work and PyBIDS directories even when bundle
   validation failed. It can also remove the work directory containing a newly
   generated scheduler error file after HPC submission fails.
2. **Processing export cleanup:** exporting a cohort can remove existing work and
   PyBIDS directories. Dry-run suppresses that deletion; keep-workdir retains them.
3. **HPC capacity and exit codes:** submitting fewer jobs than requested, including
   zero when the queue is full, retains a success exit code. Processing retains its
   work directory when submitted and requested counts differ.
4. **HPC dry-run:** queries queue status, removes a stale `pysqa.err`, and creates
   the HPC log directory, but does not call scheduler submission. It reports the
   eligible count as successful submission for cleanup accounting.
5. **HPC child commands:** omit `--simulate`, `--dry-run`, and `--hpc`; propagate
   keep-workdir/verbose and processing's tar flag. Simulation on the submission
   command therefore does not become simulation in the child jobs.
6. **Group selection:** group mode executes once even when subcohort intersection
   contains no eligible pairs. Unavailable participant/session values render as
   the string `None` and emit warnings.
7. **BIDSification modes:** successful dry-run/simulated items can update the cached
   table, while dry-run suppresses disk changes and simulation suppresses the final
   save. Export and HPC mode reach the final status-save stage but leave the
   existing table unchanged.
8. **Invocation precedence:** pipeline variables are applied while loading the
   invocation, then user substitutions, then runtime identifier substitutions.
   A user substitution of an identifier placeholder can override the runtime ID.
   Study-config rendering also warns about that identifier inside `SUBSTITUTIONS`.
9. **Setup ordering:** runner work directories are created before bundle validation.
   BIDSification and extraction retain those directories if validation fails;
   processing applies its cleanup policy.
10. **Empty export:** succeeds and writes an empty file; dry-run succeeds without
    writing it. Empty local selection instead returns the no-participants code.

Do not change these policies inside extraction PRs. Any behavior fix needs its own
review and acceptance criteria.

## Checks

On macOS, `COPYFILE_DISABLE=1` prevents the system `tar` command from including
AppleDouble metadata entries that fail existing archive-content assertions. It
changes the test environment only.

```sh
COPYFILE_DISABLE=1 uv run --no-sync python -m pytest \
  tests/integration/test_workflow_contracts.py \
  tests/unit/workflows/test_base.py -n 0 -o addopts='' -q

COPYFILE_DISABLE=1 uv run --no-sync python -m pytest \
  -m "not api and not linkcheck and no_xdist" -n 0

COPYFILE_DISABLE=1 uv run --no-sync python -m pytest \
  -m "not api and not linkcheck and not no_xdist"
```

Local validation uses the existing Python 3.14 environment. The repository's CI
matrix remains responsible for Python 3.10–3.14 compatibility. No GitHub PR or
branch push is part of this stage.

Validated locally:

- 131 new CLI characterization cases and 3 new lifecycle cases.
- Focused characterization/base suite: 146 passed.
- Non-API serial suite: 130 passed.
- Non-API parallel suite: 1,441 passed; combined total: 1,571 passed.
- Ruff lint/format checks and whitespace checks passed.

The baseline focused run initially had 405 passes and two failures from macOS
AppleDouble tar entries. Both existing archive tests pass with
`COPYFILE_DISABLE=1`. Remaining suite warnings are Trogon's existing Click
deprecation warnings. Other Python versions and external API/link checks were not
run locally.
