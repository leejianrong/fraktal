import pytest

from fraktal.ingest import URL, HttpTransport

pytestmark = pytest.mark.integration


def test_khartoum_tarball_size_matches_adr_0008() -> None:
    size = HttpTransport().size(URL)
    assert 5.0e9 < size < 5.2e9  # ADR-0008 records 5.1 GB
