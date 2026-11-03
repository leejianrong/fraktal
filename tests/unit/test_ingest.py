import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from fraktal import cli
from fraktal.ingest import URL, ingest


def make_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in [
            ("AOI_5_Khartoum_Train/RGB-PanSharpen/img1.tif", b"imagery" * 5000),
            ("AOI_5_Khartoum_Train/geojson/buildings_img1.geojson", b"{}"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


class FakeTransport:
    def __init__(self, payload: bytes, *, cut_at: int | None = None) -> None:
        self.payload = payload
        self.cut_at = cut_at
        self.reads: list[int] = []
        self.sizes = 0

    def size(self, url: str) -> int:
        self.sizes += 1
        return len(self.payload)

    def read(self, url: str, start: int) -> Iterator[bytes]:
        self.reads.append(start)
        end = self.cut_at if self.cut_at is not None else len(self.payload)
        yield self.payload[start:end]


def test_first_run_extracts_imagery_and_labels(tmp_path: Path) -> None:
    result = ingest(tmp_path, transport=FakeTransport(make_tarball()))
    assert result.status == "fetched"
    assert (result.path / "AOI_5_Khartoum_Train/RGB-PanSharpen/img1.tif").is_file()
    assert (result.path / "AOI_5_Khartoum_Train/geojson/buildings_img1.geojson").is_file()
    assert not list((tmp_path / "downloads").glob("*.tar.gz"))  # archive dropped after extract


def test_second_run_is_cached_and_makes_no_network_calls(tmp_path: Path) -> None:
    ingest(tmp_path, transport=FakeTransport(make_tarball()))
    again = FakeTransport(make_tarball())
    result = ingest(tmp_path, transport=again)
    assert result.status == "cached"
    assert again.sizes == 0
    assert again.reads == []


def test_truncated_download_is_detected_then_resumed(tmp_path: Path) -> None:
    payload = make_tarball()
    with pytest.raises(OSError, match="incomplete"):
        ingest(tmp_path, transport=FakeTransport(payload, cut_at=len(payload) // 2))
    assert not (tmp_path / "spacenet2" / "khartoum").exists()
    healthy = FakeTransport(payload)
    result = ingest(tmp_path, transport=healthy)
    assert result.status == "fetched"
    assert result.resumed_from == len(payload) // 2
    assert healthy.reads == [len(payload) // 2]


def test_truncated_archive_on_disk_is_refetched(tmp_path: Path) -> None:
    payload = make_tarball()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / URL.rsplit("/", 1)[1]).write_bytes(payload[:100])
    transport = FakeTransport(payload)
    assert ingest(tmp_path, transport=transport).status == "fetched"
    assert transport.reads == [0]


def test_cli_reports_cached_in_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest(tmp_path, transport=FakeTransport(make_tarball()))

    def no_network(*_a: object, **_k: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr("fraktal.ingest.urllib.request.urlopen", no_network)
    assert cli.main(["ingest", "--what", "imagery", "--data-dir", str(tmp_path), "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["status"] == "cached"


def test_data_dir_is_git_ignored() -> None:
    root = Path(__file__).resolve().parents[2]
    assert "data/" in (root / ".gitignore").read_text().splitlines()
