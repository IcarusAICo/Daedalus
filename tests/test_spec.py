"""Tests for SkillSpec validation and SkillVersion semver compat."""

from __future__ import annotations

import pytest

from daedalus.core.spec import KNOWN_SIDE_EFFECTS, SkillSpec, SkillVersion


def test_version_accepts_valid_semver():
    v = SkillVersion(raw="1.2.3")
    assert v.tuple == (1, 2, 3)


def test_version_rejects_invalid():
    with pytest.raises(Exception):
        SkillVersion(raw="not-a-version")


def test_version_caret_compatibility():
    v = SkillVersion(raw="0.3.5")
    assert v.is_compatible_with("^0.3.0")
    assert v.is_compatible_with("^0.3.5")
    assert not v.is_compatible_with("^0.4.0")
    assert not v.is_compatible_with("^1.0.0")


def test_version_exact_compatibility():
    v = SkillVersion(raw="2.0.1")
    assert v.is_compatible_with("2.0.1")
    assert not v.is_compatible_with("2.0.2")


def test_spec_id_must_be_snake_case():
    with pytest.raises(Exception):
        SkillSpec.from_dict(
            {
                "id": "BadId",
                "version": "0.1.0",
                "description": "x",
            }
        )


def test_spec_rejects_unknown_side_effect():
    with pytest.raises(Exception):
        SkillSpec.from_dict(
            {
                "id": "fine_id",
                "version": "0.1.0",
                "description": "x",
                "side_effects": ["explode_planet"],
            }
        )


def test_spec_known_side_effects_includes_input_capture():
    assert "screen_input" in KNOWN_SIDE_EFFECTS
    assert "screen_capture" in KNOWN_SIDE_EFFECTS


def test_spec_extra_fields_forbidden():
    with pytest.raises(Exception):
        SkillSpec.from_dict(
            {
                "id": "fine_id",
                "version": "0.1.0",
                "description": "x",
                "wat": True,
            }
        )
