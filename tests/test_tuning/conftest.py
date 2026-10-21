from pathlib import Path

import pytest

from autoencodix.tuning import _propose

FIXTURE_ARTIFACT = Path(__file__).parent / "fixtures" / "initial_configs.json"


@pytest.fixture(autouse=True)
def mock_hf_download(monkeypatch, request):
    """Serve a trimmed local copy of the portfolio instead of hitting the Hub,
    so tests run offline. Records each call's kwargs in the returned list.
    Tests marked `network` use the real Hub instead."""
    _propose._artifact_cache.clear()
    calls = []
    if request.node.get_closest_marker("network") is None:

        def fake_download(**kwargs):
            calls.append(kwargs)
            return str(FIXTURE_ARTIFACT)

        monkeypatch.setattr(_propose, "hf_hub_download", fake_download)
    yield calls
    _propose._artifact_cache.clear()
