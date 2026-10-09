"""Unit tests for the CLI check and entrypoint behavior."""

from app.cli import run_check


def test_cli_check_executes() -> None:
    exit_code = run_check()
    assert exit_code == 0
