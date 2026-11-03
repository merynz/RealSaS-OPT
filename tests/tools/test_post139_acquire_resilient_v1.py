from __future__ import annotations

import pytest

from tools.corpus import post139_acquire_resilient_v1 as resilient


def test_csrf_token_accepts_current_itch_input_shape():
    document = '<input type="hidden" name="csrf_token" value="token-123">'
    assert resilient._csrf_token(document) == "token-123"


def test_upload_infos_extracts_standard_upload_name():
    document = '''
    <a class="download_btn" data-upload_id="12345">
      <strong class="name">Sci-Fi Essentials Kit[Standard].zip</strong>
    </a>
    <a class="download_btn" data-upload_id="99999">
      <strong class="name">Sci-Fi Essentials Kit[Pro].zip</strong>
    </a>
    '''
    assert resilient._upload_infos(document) == [
        ("12345", "Sci-Fi Essentials Kit[Standard].zip"),
        ("99999", "Sci-Fi Essentials Kit[Pro].zip"),
    ]


def test_quaternius_itch_locator_comes_from_publisher_page(monkeypatch):
    row = {
        "id": "quaternius-sci-fi-essentials",
        "acquisition": {
            "publisher_repository": "Quaternius/quaternius.github.io",
            "page_path": "packs/scifiessentialskit.html",
        },
    }
    publisher = '''
    <button onclick="window.open('https://quaternius.itch.io/sci-fi-essentials-kit', '_blank')">Download</button>
    <script>Itch.attachBuyButton(null, {"user":"quaternius","game":"sci-fi-essentials-kit"});</script>
    '''
    monkeypatch.setattr(resilient.source_inventory, "fetch_text", lambda _url: publisher)
    url, page = resilient._itch_game_url_from_quaternius_page(row)
    assert url == "https://quaternius.itch.io/sci-fi-essentials-kit"
    assert page.endswith("/packs/scifiessentialskit.html")


def test_kenney_pin_is_only_fallback_for_expected_resolution_failure(monkeypatch):
    row = {"id": "kenney-mini-characters", "acquisition": {"page_url": "https://kenney.nl/assets/mini-characters"}}
    monkeypatch.setattr(
        resilient,
        "_ACTIVE_PINS",
        {"kenney_direct_zip_by_source_id": {"kenney-mini-characters": "https://kenney.nl/media/pages/assets/mini-characters/hash/archive.zip"}},
    )
    def fail(_row):
        raise ValueError("POST139_KENNEY_ZIP_RESOLUTION:kenney-mini-characters:[]")
    monkeypatch.setattr(resilient, "_ORIGINAL_KENNEY_OFFICIAL_ZIP", fail)
    resolved, provenance = resilient._resilient_kenney_official_zip(row)
    assert resolved.endswith("/archive.zip")
    assert provenance == row["acquisition"]["page_url"]


def test_kenney_unexpected_failure_is_not_hidden(monkeypatch):
    row = {"id": "kenney-mini-characters", "acquisition": {"page_url": "https://kenney.nl/assets/mini-characters"}}
    monkeypatch.setattr(resilient, "_ORIGINAL_KENNEY_OFFICIAL_ZIP", lambda _row: (_ for _ in ()).throw(ValueError("OTHER_FAILURE")))
    with pytest.raises(ValueError, match="OTHER_FAILURE"):
        resilient._resilient_kenney_official_zip(row)
