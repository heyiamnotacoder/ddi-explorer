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


def test_split_components_does_not_treat_ml_as_drug():
    assert split_components("paracetamol (100mg/ml)") == ["paracetamol"]
    assert split_components("ambroxol (30mg/5ml)") == ["ambroxol"]
    assert split_components("insulin glulisine (100iu/ml)") == ["insulin glulisine"]


def test_bare_dolo_is_paracetamol_not_drops_ml():
    comps = _lookup_indian("dolo")
    assert comps
    flat = [c for comp in comps for c in split_components(comp)]
    assert "paracetamol" in " ".join(flat)
    assert "ml" not in flat


def test_generic_amlodipine_is_not_an_fdc():
    comps = _lookup_indian("amlodipine")
    assert comps
    flat = [c for comp in comps for c in split_components(comp)]
    assert flat == ["amlodipine"]


def test_dolo_does_not_match_unrelated_dolo_prefix_brands():
    comps = _lookup_indian("dolo")
    assert comps
    blob = " ".join(comps).lower()
    assert "piroxicam" not in blob
    assert "aceclofenac" not in blob


def test_five_plain_names_make_five_singleton_components():
    """The screenshot case: 5 drugs must not explode into 7 components."""
    names = ["dolo", "pento", "amlodipine", "atorvastatin"]
    flats = []
    for n in names:
        comps = _lookup_indian(n)
        assert comps, n
        flat = [c for comp in comps for c in split_components(comp)]
        assert len(flat) == 1, (n, flat)
        flats.append(flat[0])
    assert "ml" not in flats
    assert "atenolol" not in flats
