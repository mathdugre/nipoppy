"""Runtime pipeline resolution and artifact loading.

Raw bundle validation remains in pipeline_validation: these loaders apply the
study's runtime substitutions and inherited container configuration.
"""

import re
from pathlib import Path

from packaging.version import Version

from nipoppy.config.pipeline import BasePipelineConfig
from nipoppy.config.tracker import TrackerConfig
from nipoppy.env import StrOrPathLike
from nipoppy.exceptions import ConfigError, FileOperationError, WorkflowError
from nipoppy.layout import DatasetLayout
from nipoppy.logger import get_logger
from nipoppy.utils.utils import load_json

logger = get_logger()


def get_pipeline_version(
    pipeline_name: str,
    dpath_pipelines: StrOrPathLike,
) -> str:
    """Get the latest version associated with a pipeline.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline, as specified in the config
    dpath_pipelines : nipoppy.env.StrOrPathLike
        Path to directory containing pipeline bundle subdirectories

    Returns
    -------
    str
        The pipeline version
    """
    installed_pipelines = []
    pipeline_config_latest = None
    for fpath_pipeline_config in Path(dpath_pipelines).glob(
        f"*/{DatasetLayout.fname_pipeline_config}"
    ):
        pipeline_config = BasePipelineConfig(
            **load_json(fpath_pipeline_config, allow_json5=True)
        )
        if pipeline_config.NAME == pipeline_name:
            if pipeline_config_latest is None:
                pipeline_config_latest = pipeline_config
            elif Version(pipeline_config.VERSION) > Version(
                pipeline_config_latest.VERSION
            ):
                pipeline_config_latest = pipeline_config
        installed_pipelines.append((pipeline_config.NAME, pipeline_config.VERSION))

    if pipeline_config_latest is not None:
        return pipeline_config_latest.VERSION
    else:
        raise WorkflowError(
            f"No config found for pipeline with NAME={pipeline_name}"
            ". Installed pipelines: "
            + ", ".join(f"{name} {version}" for name, version in installed_pipelines)
        )


def load_descriptor(study, config, step, bundle):
    """Load the pipeline step's Boutiques descriptor."""
    if (fname_descriptor := step.DESCRIPTOR_FILE) is None:
        raise ConfigError(
            f"No descriptor file specified for pipeline {config.NAME} {config.VERSION}"
        )
    fpath_descriptor = bundle / fname_descriptor
    logger.info(f"Loading descriptor from {fpath_descriptor}")
    descriptor = load_json(fpath_descriptor)
    descriptor = study.config.apply_pipeline_variables(
        pipeline_type=config.PIPELINE_TYPE,
        pipeline_name=config.NAME,
        pipeline_version=config.VERSION,
        json_obj=descriptor,
    )
    return descriptor


def load_invocation(study, config, step, bundle):
    """Load the pipeline step's Boutiques invocation."""
    if (fname_invocation := step.INVOCATION_FILE) is None:
        raise ConfigError(
            f"No invocation file specified for pipeline {config.NAME} {config.VERSION}"
        )
    fpath_invocation = bundle / fname_invocation
    logger.info(f"Loading invocation from {fpath_invocation}")
    invocation = load_json(fpath_invocation)

    invocation = study.config.apply_pipeline_variables(
        pipeline_type=config.PIPELINE_TYPE,
        pipeline_name=config.NAME,
        pipeline_version=config.VERSION,
        json_obj=invocation,
    )
    return invocation


def load_tracker_config(study, config, step, bundle):
    """Load the pipeline step's tracker configuration."""
    if (fname_tracker_config := step.TRACKER_CONFIG_FILE) is None:
        raise ConfigError(
            f"No tracker config file specified for pipeline {config.NAME}"
            f" {config.VERSION}"
        )
    fpath_tracker_config = bundle / fname_tracker_config
    logger.info(f"Loading tracker config from {fpath_tracker_config}")
    return TrackerConfig(**load_json(fpath_tracker_config, allow_json5=True))


def load_pybids_ignore_patterns(study, config, step, bundle):
    """
    Load the pipeline step's PyBIDS ignore pattern list.

    Note: this does not apply any substitutions, since the subject/session
    patterns are always added.
    """
    # no file specified
    if (fname_pybids_ignore := step.PYBIDS_IGNORE_FILE) is None:
        return []

    fpath_pybids_ignore = bundle / fname_pybids_ignore

    # load patterns from file
    logger.info(f"Loading PyBIDS ignore patterns from {fpath_pybids_ignore}")
    patterns = load_json(fpath_pybids_ignore, allow_json5=True)

    # validate format
    if not isinstance(patterns, list):
        raise ConfigError(
            f"Expected a list of strings in {fpath_pybids_ignore}"
            f", got {patterns} ({type(patterns)})"
        )

    return [re.compile(pattern) for pattern in patterns]


def load_pipeline_config(
    study,
    pipeline_type,
    render,
    dpath_pipeline_bundle,
    pipeline_name,
    pipeline_version,
    pipeline_class,
):
    """Get the config for a pipeline."""
    fpath_config = dpath_pipeline_bundle / study.layout.fname_pipeline_config
    if not fpath_config.exists():
        raise FileOperationError(
            f"Pipeline config file not found at {fpath_config} for "
            f"pipeline: {pipeline_name} {pipeline_version}"
        )

    # NOTE: user-defined substitutions take precedence over the pipeline variables
    pipeline_config_json = study.config.apply_pipeline_variables(
        pipeline_type=pipeline_type,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        json_obj=render(
            load_json(fpath_config, allow_json5=True),
        ),
    )

    pipeline_config = pipeline_class(**pipeline_config_json)

    # make sure the config is for the correct pipeline
    if not (
        pipeline_config.NAME == pipeline_name
        and pipeline_config.VERSION == pipeline_version
    ):
        raise WorkflowError(
            f'Expected pipeline config to have NAME="{pipeline_name}" '
            f'and VERSION="{pipeline_version}", got "{pipeline_config.NAME}" and '
            f'"{pipeline_config.VERSION}" instead'
        )

    return study.config.propagate_container_config_to_pipeline(pipeline_config)


def check_pipeline_variables(study, pipeline_type, pipeline_name, pipeline_version):
    """Check that the pipeline variables are not null in the config."""
    for name, value in study.config.PIPELINE_VARIABLES.get_variables(
        pipeline_type, pipeline_name, pipeline_version
    ).items():
        if value is None:
            raise ConfigError(
                f"Variable {name} is not set in the config for pipeline "
                f"{pipeline_name}, version {pipeline_version}. You need "
                "to set it in the PIPELINE_VARIABLES section of the config file at "
                f"{study.layout.fpath_config}"
            )
