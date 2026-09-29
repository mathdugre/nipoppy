"""Render JSON templates using an explicit, lazy runtime namespace."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from nipoppy.utils.bids import (
    participant_id_to_bids_participant_id,
    session_id_to_bids_session_id,
)
from nipoppy.utils.utils import (
    TEMPLATE_REPLACE_PATTERN,
    apply_substitutions_to_json,
    process_template_str,
)


class TemplateRenderer:
    """Resolve only referenced values; retain user-substitution and ID precedence."""

    def __init__(
        self,
        substitutions: Mapping[str, str] | Callable[[], Mapping[str, str]],
        values: Mapping[str, Callable[[], object]],
        layout,
    ):
        self.substitutions = substitutions
        self.values = values
        self.layout = layout

    def render(
        self,
        template_json,
        participant_id=None,
        session_id=None,
        bids_participant_id=None,
        bids_session_id=None,
        objs=None,
        return_str=False,
        with_substitutions=True,
        **kwargs,
    ):
        """Render a JSON value, optionally returning its serialized form."""
        if with_substitutions:
            template_json = apply_substitutions_to_json(
                template_json,
                self.substitutions()
                if callable(self.substitutions)
                else self.substitutions,
            )
        if participant_id is not None:
            kwargs["participant_id"] = participant_id
            kwargs["bids_participant_id"] = (
                bids_participant_id
                if bids_participant_id is not None
                else participant_id_to_bids_participant_id(participant_id)
            )
        if session_id is not None:
            kwargs["session_id"] = session_id
            kwargs["bids_session_id"] = (
                bids_session_id
                if bids_session_id is not None
                else session_id_to_bids_session_id(session_id)
            )
        objects = list(objs or [])
        text = json.dumps(template_json)
        for match in TEMPLATE_REPLACE_PATTERN.finditer(text):
            key = match.group(1).lower()
            if key in kwargs or any(hasattr(obj, key) for obj in objects):
                continue
            if key in self.values:
                kwargs[key] = self.values[key]()
        text = process_template_str(text, objs=[*objects, self.layout], **kwargs)
        return text if return_str else json.loads(text)
