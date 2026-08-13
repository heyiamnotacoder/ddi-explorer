"""Normalization tests — offline parts only (no network)."""
from app.pipeline.normalize import _lookup_indian, split_components


def test_indian_brand_exact():
    comps = _lookup_indian("dolo 650 tablet")
    assert comps and any("paracetamol" in c for c in comps)


def test_indian_brand_fuzzy():
    comps = _lookup_indian("Augmentin 625")
    assert comps and any("amoxycillin" in c for c in comps)


def test_fdc_has_two_components():
    comps = _lookup_indian("augmentin 625 duo tablet")
    assert comps and len(comps) == 2


def test_unknown_returns_none():
    assert _lookup_indian("xyznotadrug12345") is None


def test_split_components_strips_strengths():
    assert split_components("amoxycillin (500mg) / clavulanic acid (125mg)") == [
        "amoxycillin", "clavulanic acid"]


def test_split_components_single():
    assert split_components("telmisartan (40mg)") == ["telmisartan"]
