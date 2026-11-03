import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fraktal", *args], capture_output=True, text=True, check=False
    )


def test_version_flag() -> None:
    result = run("--version")
    assert result.returncode == 0
    assert result.stdout.startswith("fraktal ")


def test_success_path_prints_a_parseable_envelope() -> None:
    result = run("version", "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["ok"] is True


def test_failure_path_exits_3_and_keeps_stdout_machine_readable() -> None:
    result = run("inspect", "--json")
    assert result.returncode == 3
    assert json.loads(result.stdout)["error"]["kind"] == "not_implemented"


@pytest.mark.parametrize(
    "stage", ["ingest", "generate", "render", "benchmark", "compare", "inspect"]
)
def test_every_stage_has_help(stage: str) -> None:
    result = run(stage, "--help")
    assert result.returncode == 0
    assert "--json" in result.stdout
