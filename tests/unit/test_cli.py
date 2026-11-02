import argparse
import json

import pytest

from fraktal import __version__, cli
from fraktal.cli import ExitCode, main

STAGES = ["ingest", "generate", "render", "benchmark", "inspect"]


def test_exit_codes_are_a_stable_contract() -> None:
    assert [
        int(c) for c in (ExitCode.OK, ExitCode.USAGE, ExitCode.NOT_IMPLEMENTED, ExitCode.FAILURE)
    ] == [
        0,
        2,
        3,
        4,
    ]


def test_version_json_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version", "--json"]) == ExitCode.OK
    envelope = json.loads(capsys.readouterr().out)
    assert envelope == {"ok": True, "data": {"version": __version__}, "error": None}


def test_version_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == ExitCode.OK
    assert capsys.readouterr().out.strip() == f"fraktal {__version__}"


@pytest.mark.parametrize("stage", STAGES)
def test_unimplemented_stage_exits_3_with_json_error(
    stage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([stage, "--json"]) == ExitCode.NOT_IMPLEMENTED
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is False
    assert envelope["data"] is None
    assert envelope["error"]["kind"] == "not_implemented"
    assert stage in envelope["error"]["message"]


def test_human_errors_go_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["render"]) == ExitCode.NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not implemented" in captured.err


def test_usage_errors_exit_2() -> None:
    with pytest.raises(SystemExit) as info:
        main(["no-such-command"])
    assert info.value.code == ExitCode.USAGE
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == ExitCode.USAGE


def test_unexpected_exception_exits_4(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_args: argparse.Namespace) -> cli.Result:
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(cli.COMMANDS, "version", ("Print the version", boom))
    assert main(["version", "--json"]) == ExitCode.FAILURE
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["kind"] == "unexpected"
    assert "disk on fire" in envelope["error"]["message"]
