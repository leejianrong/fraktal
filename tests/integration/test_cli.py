import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_cli_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fraktal", "--version"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert result.stdout.startswith("fraktal ")
