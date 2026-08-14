"""Offline substitution ranking — no network, no LLM."""
from app.agent.alternatives import (
    fallback_strategy,
    importance_of,
    is_actionable,
    is_safer,
    keep_list,
    pair_burden,
    pick_candidates,
    timing_notes,
)
from app.models import Category, Grade, Importance, NormalizedDrug, PairResult


def _drug(name: str, *comps: str) -> NormalizedDrug:
    parts = list(comps) or [name]
    return NormalizedDrug(input_name=name, generic_name=", ".join(parts),
                          components=parts)


def _pair(a: str, b: str, *, grade: Grade | None = Grade.A,
          category: Category = Category.INTERACTION,
          severity: str | None = "major") -> PairResult:
    return PairResult(drugs=(a, b), grade=grade, category=category,
                      severity=severity, summary=f"{a} + {b}")


def test_paracetamol_is_adjuvant():
    assert importance_of("paracetamol") == Importance.ADJUVANT
    assert importance_of("acetaminophen") == Importance.ADJUVANT


def test_insulin_and_warfarin_are_anchors():
    assert importance_of("insulin glargine") == Importance.ANCHOR
    assert importance_of("warfarin") == Importance.ANCHOR
    assert importance_of("levothyroxine") == Importance.ANCHOR


def test_telmisartan_is_controller():
    assert importance_of("telmisartan") == Importance.CONTROLLER


def test_unknown_defaults_to_controller():
    assert importance_of("xyznotadrug123") == Importance.CONTROLLER


def test_iron_token_does_not_match_spironolactone():
    assert importance_of("spironolactone") == Importance.CONTROLLER


def test_timing_pair_is_not_actionable():
    p = _pair("levothyroxine", "calcium", grade=None, category=Category.TIMING)
    assert not is_actionable(p)


def test_grade_c_is_not_actionable():
    p = _pair("diclofenac", "telmisartan", grade=Grade.C, severity="moderate")
    assert not is_actionable(p)


def test_grade_b_is_actionable():
    p = _pair("warfarin", "amiodarone", grade=Grade.B, severity="major")
    assert is_actionable(p)


def test_grade_c_only_yields_no_candidates():
    drugs = [_drug("Diclofenac 50", "diclofenac"), _drug("Telma 40", "telmisartan")]
    pairs = [_pair("diclofenac", "telmisartan", grade=Grade.C, severity="moderate")]
    assert pick_candidates(drugs, pairs) == []
    text = fallback_strategy([], [], pairs)
    assert "grade c" in text.lower()
    assert "do not change" in text.lower()


def test_grade_c_does_not_trigger_change_when_mixed_with_none():
    drugs = [_drug("Dolo 650", "paracetamol"), _drug("Telma 40", "telmisartan")]
    pairs = [_pair("paracetamol", "telmisartan", grade=Grade.C, severity="minor")]
    assert pick_candidates(drugs, pairs) == []


def test_mixed_a_and_c_still_picks_the_a_pair_drug():
    """Grade C must not block a change that a Grade A pair already justifies."""
    drugs = [_drug("Warfarin 5", "warfarin"),
             _drug("Diclofenac 50", "diclofenac"),
             _drug("Telma 40", "telmisartan")]
    pairs = [
        _pair("warfarin", "diclofenac"),
        _pair("diclofenac", "telmisartan", grade=Grade.C, severity="moderate"),
    ]
    picked = pick_candidates(drugs, pairs)
    assert [c.name for c in picked] == ["diclofenac"]


def test_pick_prefers_adjuvant_over_anchor():
    drugs = [_drug("Warfarin 5", "warfarin"),
             _drug("Diclofenac 50", "diclofenac"),
             _drug("Telma 40", "telmisartan")]
    pairs = [
        _pair("warfarin", "diclofenac"),
        _pair("warfarin", "telmisartan", grade=None, category=Category.NONE,
              severity=None),
    ]
    picked = pick_candidates(drugs, pairs)
    assert [c.name for c in picked] == ["diclofenac"]
    assert picked[0].importance == Importance.ADJUVANT


def test_timing_only_yields_no_candidates():
    drugs = [_drug("Thyronorm", "levothyroxine"), _drug("Calcium", "calcium")]
    pairs = [_pair("levothyroxine", "calcium", grade=None, category=Category.TIMING)]
    assert pick_candidates(drugs, pairs) == []
    notes = timing_notes(pairs)
    assert notes and "separate" in notes[0].lower()


def test_controller_only_if_major_or_contra():
    drugs = [_drug("Warfarin 5", "warfarin"), _drug("Ecosprin 75", "aspirin")]
    minor = [_pair("warfarin", "aspirin", grade=Grade.C, severity="minor")]
    assert pick_candidates(drugs, minor) == []
    major = [_pair("warfarin", "aspirin", grade=Grade.A, severity="major")]
    picked = pick_candidates(drugs, major)
    assert [c.name for c in picked] == ["aspirin"]


def test_keep_lists_anchor():
    drugs = [_drug("Warfarin 5", "warfarin"), _drug("Dolo 650", "paracetamol")]
    pairs = [_pair("warfarin", "paracetamol")]
    cands = pick_candidates(drugs, pairs)
    keep = keep_list(drugs, pairs, cands)
    assert any(k.name == "warfarin" and k.importance == Importance.ANCHOR for k in keep)


def test_safer_rejects_new_contraindication():
    old = [_pair("warfarin", "diclofenac")]
    new = [_pair("warfarin", "ketorolac", category=Category.CONTRAINDICATED)]
    assert not is_safer(old, new)


def test_safer_when_burden_drops():
    old = [_pair("warfarin", "diclofenac")]
    new: list[PairResult] = []
    assert is_safer(old, new)
    assert pair_burden(new) < pair_burden(old)


def test_same_grade_a_is_not_safer():
    """Topical NSAID still Grade A with warfarin — do not badge as safer."""
    old = [
        _pair("warfarin", "diclofenac"),
        _pair("diclofenac", "telmisartan", grade=Grade.C, severity="moderate"),
    ]
    new = [
        _pair("warfarin", "diclofenac topical"),
        _pair("diclofenac topical", "telmisartan", grade=Grade.C, severity="moderate"),
    ]
    assert not is_safer(old, new)


def test_fallback_explains_timing_only():
    text = fallback_strategy([], ["Keep levothyroxine and calcium; separate."])
    assert "separating" in text.lower()
