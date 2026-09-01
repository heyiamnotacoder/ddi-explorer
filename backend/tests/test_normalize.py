"""Normalization tests — offline parts only (no network)."""
from app.pipeline.normalize import (
    _bare_name,
    _lookup_indian,
    _pick_rxnav_candidate,
    split_components,
)


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


def test_bare_name_strips_strength_keeps_brand_numbers():
    assert _bare_name("amilodipine 20mg") == "amilodipine"
    assert _bare_name("levoceterizine 5 mg") == "levoceterizine"
    assert _bare_name("warfar 100 mg") == "warfar"
    assert _bare_name("omeprazole 5 mg") == "omeprazole"
    # Indian brands often encode strength without a unit
    assert _bare_name("dolo 650") == "dolo 650"
    assert _bare_name("telma 40") == "telma 40"


def test_screenshot_typos_with_doses_are_single_generics():
    """Typed 'amilodipine 20mg, levoceterizine 5 mg, warfar 100 mg, omeprazole 5 mg'.

    Strength on the name must not: (a) turn amlodipine into an FDC, or
    (b) leave warfarin unresolved. Brand names with a number (dolo 650)
    still resolve via the full string.
    """
    cases = {
        "amilodipine 20mg": ["amlodipine"],
        "amlodipine 20mg": ["amlodipine"],
        "levoceterizine 5 mg": ["levocetirizine"],
        "warfar 100 mg": ["warfarin"],
        "omeprazole 5 mg": ["omeprazole"],
        "dolo 650": None,  # filled below — must stay paracetamol, not a generic miss
    }
    for raw, expected in cases.items():
        comps = _lookup_indian(raw)
        assert comps, raw
        flat = [c for comp in comps for c in split_components(comp)]
        if expected is None:
            assert "paracetamol" in " ".join(flat)
            assert len(flat) == 1
        else:
            assert flat == expected, (raw, flat)


def test_rxnav_picker_skips_combo_when_query_is_a_single_drug():
    rows = [
        ("898356", "amlodipine 5 MG / benazepril hydrochloride 20 MG Oral Capsule", 7.3),
        ("17767", "amlodipine", 7.1),
    ]
    rxcui, name = _pick_rxnav_candidate("amilodipine 20mg", rows)
    assert rxcui == "17767"
    assert name == "amlodipine"


def test_rxnav_picker_skips_nameless_hits():
    rows = [
        ("241757", "", 6.0),
        ("11289", "warfarin", 5.9),
    ]
    rxcui, name = _pick_rxnav_candidate("warfar 100 mg", rows)
    assert rxcui == "11289"
    assert name == "warfarin"
