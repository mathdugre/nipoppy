"""Tests for the explicit runtime template namespace."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nipoppy.workflows.services.templates import TemplateRenderer


def test_only_referenced_values_are_loaded():
    def unavailable():
        raise RuntimeError("must stay lazy")

    renderer = TemplateRenderer(
        {}, {"unused": unavailable, "value": lambda: "ok"}, None
    )
    assert renderer.render({"key": "[[NIPOPPY_VALUE]]"}) == {"key": "ok"}


def test_template_precedence():
    renderer = TemplateRenderer(
        {"USER_VALUE": "[[NIPOPPY_PARTICIPANT_ID]]"},
        {"participant_id": lambda: "outer", "value": lambda: "runtime"},
        SimpleNamespace(value="layout"),
    )
    assert renderer.render(
        ["USER_VALUE", "[[NIPOPPY_VALUE]]"],
        participant_id="01",
        objs=[SimpleNamespace(value="object")],
    ) == ["01", "object"]
    assert renderer.render("[[NIPOPPY_VALUE]]", value="explicit") == "explicit"


def test_paths_and_missing_values(tmp_path):
    renderer = TemplateRenderer({}, {"path": lambda: tmp_path / "x"}, None)
    assert renderer.render("[[NIPOPPY_PATH]]") == str(Path(tmp_path / "x").resolve())
    with pytest.warns(UserWarning, match="Unable to replace"):
        assert renderer.render("[[NIPOPPY_UNKNOWN]]") == "[[NIPOPPY_UNKNOWN]]"
