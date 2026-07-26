"""Offline, deterministic tests for the CT Plant ID FastAPI app.

No real model load (the fake in conftest replaces PlantModel) and no network.
Tests assert the documented response SHAPES and the not_sure behavior, driven by
the fake model's known outputs.
"""
import importlib
import io

import pytest
from PIL import Image

from tests.conftest import FakePlantModel

REQUIRED_CANDIDATE_KEYS = {"species", "common_name", "prob", "status", "is_weed"}


def _jpeg_bytes(color=(34, 139, 34), size=(16, 16)):
    """A tiny in-memory RGB JPEG — enough for the decode path, no disk, no net."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def test_health_reports_ok_and_species_count(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # Species count mirrors the fake model's class list length (5).
    assert body["species"] == len(FakePlantModel.classes)
    assert isinstance(body["species"], int)


def test_identify_returns_documented_shape(client):
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Top-level shape.
    assert isinstance(body["top_species"], str)
    assert isinstance(body["not_sure"], bool)
    assert isinstance(body["candidates"], list)

    # k=5 candidates (fake has 5 classes).
    candidates = body["candidates"]
    assert len(candidates) == 5

    # Every candidate carries all required keys with sane types.
    for c in candidates:
        assert REQUIRED_CANDIDATE_KEYS.issubset(c.keys())
        assert isinstance(c["species"], str)
        assert isinstance(c["prob"], float)
        assert isinstance(c["status"], str)
        assert isinstance(c["is_weed"], bool)
        assert c["common_name"] is None or isinstance(c["common_name"], str)

    # Probabilities are sorted descending.
    probs = [c["prob"] for c in candidates]
    assert probs == sorted(probs, reverse=True)

    # top_species matches the leading candidate.
    assert body["top_species"] == candidates[0]["species"]


def test_not_sure_false_when_confident(client):
    # Top-1 comfortably above the threshold -> confident. Reference the actual
    # threshold so this tracks config changes instead of hardcoding a number.
    from app import main
    FakePlantModel.probs = [main.NOT_SURE_THRESHOLD + 0.20, 0.05, 0.04, 0.03, 0.02]
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["not_sure"] is False


def test_not_sure_true_when_low_confidence(client):
    # Dial the top-1 prob just below the (data-driven) threshold -> not sure.
    from app import main
    low = max(main.NOT_SURE_THRESHOLD - 0.05, 0.01)
    FakePlantModel.probs = [low, 0.04, 0.03, 0.02, 0.01]
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["not_sure"] is True
    # Candidates still come back for the ambiguous case.
    assert len(body["candidates"]) == 5


def test_confidence_label_bands(client):
    # Probabilities are temperature-calibrated, so the label bands read on the
    # honest number: >=0.80 strong, >=0.60 likely, >=threshold possible, else uncertain.
    from app import main
    assert main.confidence_label(0.94) == "Strong match"
    assert main.confidence_label(0.70) == "Likely match"
    assert main.confidence_label(main.NOT_SURE_THRESHOLD + 0.05) == "Possible match"
    assert main.confidence_label(main.NOT_SURE_THRESHOLD - 0.05) == "Uncertain"


def test_identify_includes_confidence_label(client):
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    label = resp.json()["confidence_label"]
    assert label in {"Strong match", "Likely match", "Possible match", "Uncertain"}


def test_show_status_true_on_confident_match(client):
    # "Likely" (>=0.60) or "Strong" -> we trust the ID enough to show its status.
    FakePlantModel.probs = [0.75, 0.05, 0.04, 0.03, 0.02]
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.json()["show_status"] is True


def test_show_status_false_on_weak_match(client):
    # "Possible"/"Uncertain" -> hide the native/invasive/weed flag (safety).
    FakePlantModel.probs = [0.45, 0.20, 0.15, 0.10, 0.05]
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.json()["show_status"] is False


def test_out_of_scope_suppresses_status_even_when_confident(client):
    # The confident-OOD case (e.g. a garden hydrangea landing on a wrong CT
    # species at high prob): out_of_scope must win and hide the status/weed flag,
    # which the confidence gate alone can't do.
    FakePlantModel.out_of_scope = True
    FakePlantModel.probs = [0.85, 0.07, 0.04, 0.03, 0.01]
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["out_of_scope"] is True
    assert body["show_status"] is False


def test_in_scope_default_reports_out_of_scope_false(client):
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.json()["out_of_scope"] is False


def test_genus_fallback_leads_when_species_uncertain_but_genus_is_not(client):
    # The case both evaluations found: torn between congeners, so no single
    # species clears the bar while the genus total is high. Lead with the genus.
    FakePlantModel.probs = [0.35, 0.30, 0.20, 0.03, 0.02]
    FakePlantModel.genus_prob = 0.85
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    body = resp.json()
    assert body["lead_with_genus"] is True
    assert body["genus"] == "Rosa"
    assert body["genus_prob"] == 0.85


def test_genus_fallback_off_when_species_is_confident(client):
    FakePlantModel.probs = [0.88, 0.05, 0.03, 0.02, 0.01]
    FakePlantModel.genus_prob = 0.95
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.json()["lead_with_genus"] is False


def test_genus_fallback_off_when_genus_also_uncertain(client):
    # Species AND genus both unsure -> nothing to fall back to; stay uncertain.
    FakePlantModel.probs = [0.20, 0.15, 0.10, 0.05, 0.03]
    FakePlantModel.genus_prob = 0.25
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.json()["lead_with_genus"] is False


def test_out_of_scope_suppresses_genus_lead(client):
    FakePlantModel.out_of_scope = True
    FakePlantModel.probs = [0.35, 0.30, 0.20, 0.03, 0.02]
    FakePlantModel.genus_prob = 0.85
    resp = client.post(
        "/api/identify",
        files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.json()["lead_with_genus"] is False


def test_hazard_surfaces_for_poison_ivy():
    # Perfect ID + "Status unknown" was the field-test gap: the status vocabulary
    # had no way to say "this will give you a rash".
    attributes = pytest.importorskip("app.attributes")
    result = attributes.annotate("Toxicodendron radicans")
    assert result["hazard"]
    assert "rash" in result["hazard"].lower()


def test_hazard_is_none_for_harmless_plant():
    attributes = pytest.importorskip("app.attributes")
    assert attributes.annotate("Acer rubrum")["hazard"] is None


def test_genus_label_uses_group_noun_when_species_agree():
    attributes = pytest.importorskip("app.attributes")
    assert attributes.genus_label("Quercus") == "oak"
    assert attributes.genus_label("Acer") == "maple"


def test_genus_label_falls_back_to_scientific_name():
    attributes = pytest.importorskip("app.attributes")
    # A genus with no common-name data can't invent a group noun.
    assert attributes.genus_label("Zzyzxia") == "Zzyzxia"


def test_garbage_upload_is_4xx_not_500(client):
    resp = client.post(
        "/api/identify",
        files={"image": ("junk.jpg", b"this is not an image", "image/jpeg")},
    )
    assert 400 <= resp.status_code < 500, resp.status_code
    assert resp.status_code != 500


def test_empty_upload_is_4xx_not_500(client):
    resp = client.post(
        "/api/identify",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert 400 <= resp.status_code < 500, resp.status_code


def test_missing_image_field_is_422(client):
    # No multipart file at all -> FastAPI validation error, never a 500.
    resp = client.post("/api/identify")
    assert resp.status_code == 422


# --- attributes.annotate(), only if that module has landed ------------------
# importorskip is called INSIDE each test (function scope) so a missing
# attributes.py skips only these three tests, not the whole module.


def test_annotate_always_returns_the_documented_keys():
    attributes = pytest.importorskip("app.attributes")
    result = attributes.annotate("Acer rubrum")
    assert set(result.keys()) == {"common_name", "status", "is_weed", "hazard"}


def test_annotate_invasive_species():
    attributes = pytest.importorskip("app.attributes")
    # Multiflora rose is a well-known CT invasive/weed.
    result = attributes.annotate("Rosa multiflora")
    assert result["status"] == "invasive"
    assert result["is_weed"] is True


def test_annotate_ornamental_species():
    # Workstream B added cultivated garden species; they should carry the
    # "ornamental" status (a known fact from the cultivated scope), a common
    # name, and must never be flagged a weed.
    attributes = pytest.importorskip("app.attributes")
    result = attributes.annotate("Hydrangea paniculata")
    assert result["status"] == "ornamental"
    assert result["is_weed"] is False
    assert result["common_name"]


def test_annotate_unknown_species_safe_defaults():
    attributes = pytest.importorskip("app.attributes")
    # A typo / unheard-of binomial must not raise and must fall back safely.
    result = attributes.annotate("Zzyzx nonexistus")
    assert set(result.keys()) == {"common_name", "status", "is_weed", "hazard"}
    assert result["status"] in {"native", "introduced", "invasive", "ornamental", "unknown"}
    assert isinstance(result["is_weed"], bool)
    assert result["hazard"] is None
