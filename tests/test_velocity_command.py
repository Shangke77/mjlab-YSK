"""Tests for velocity command helpers."""

from mjlab.tasks.velocity.mdp.velocity_command import _symmetric_gui_limit


def test_symmetric_gui_limit_supports_zero_range() -> None:
  assert _symmetric_gui_limit((0.0, 0.0)) == 0.1


def test_symmetric_gui_limit_contains_asymmetric_range() -> None:
  assert _symmetric_gui_limit((-2.0, 0.5)) == 2.0
