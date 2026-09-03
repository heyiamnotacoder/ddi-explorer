"""Pre-filter accepts the same NormalizedDrug shape as a fresh check."""
import pytest

from app import service
from app.models import Category, Grade, NormalizedDrug
from app.pipeline.prefilter import check_known_pairs


def _drug(name: str, *comps: str, rxcui: str | None = None,
          component_rxcuis: dict[str, str] | None = None) -> NormalizedDrug:
    parts = list(comps) or [name]
    cuis = component_rxcuis or {}
    if rxcui and len(parts) == 1 and parts[0] not in cuis:
        cuis = {parts[0]: rxcui, **cuis}
    return NormalizedDrug(
        input_name=name, generic_name=", ".join(parts),
        rxcui=rxcui, components=parts, component_rxcuis=cuis,
    )


def _rxnav_payload(*pairs: tuple[str, str, str, str, str]) -> dict:
    """(cui_a, name_a, cui_b, name_b, description) → RxNav list.json body."""
    interaction_pairs = []
    for cui_a, name_a, cui_b, name_b, desc in pairs:
        interaction_pairs.append({
            "description": desc,
            "severity": "moderate",
            "interactionConcept": [
                {"minConceptItem": {"rxcui": cui_a, "name": name_a}},
                {"minConceptItem": {"rxcui": cui_b, "name": name_b}},
            ],
        })
    return {
        "fullInteractionTypeGroup": [{
            "fullInteractionType": [{"interactionPair": interaction_pairs}],
        }],
    }


def _patch_rxnav(monkeypatch, payload: dict, captured: dict) -> None:
    class FakeClient:
        def __init__(self, * _a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, url, params=None, timeout=None):
            captured["rxcuis"] = (params or {}).get("rxcuis", "")
            captured["url"] = url

            class Resp:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return payload

            return Resp()

    monkeypatch.setattr("app.pipeline.prefilter.httpx.AsyncClient", FakeClient)


@pytest.mark.asyncio
async def test_check_known_pairs_takes_normalized_drugs():
    resolved, unknown = await check_known_pairs([
        _drug("warfarin", "warfarin"),
        _drug("omeprazole", "omeprazole"),
    ])
    assert resolved == []
    assert unknown == [("omeprazole", "warfarin")]


@pytest.mark.asyncio
async def test_fdc_pairs_with_leftover_components():
    """Recheck leftover: FDC minus one component still uses NormalizedDrug."""
    leftover = _drug("Augmentin", "amoxycillin", "clavulanic acid").model_copy(
        update={"components": ["amoxycillin"]}
    )
    alt = _drug("azithromycin", "azithromycin")
    resolved, unknown = await check_known_pairs([leftover, alt])
    assert resolved == []
    assert unknown == [("amoxycillin", "azithromycin")]


@pytest.mark.asyncio
async def test_recheck_passes_normalized_drugs(monkeypatch, patch_seams):
    seen: dict = {}

    async def fake_norm(names):
        return [_drug(names[0], "pantoprazole", rxcui="40790")], []

    async def fake_prefilter(drugs):
        seen["types"] = [type(d).__name__ for d in drugs]
        return [], []

    async def fake_waterfall(*_a, **_k):
        return []

    patch_seams(normalize=fake_norm, prefilter=fake_prefilter,
                waterfall=fake_waterfall)

    leftover = _drug("warfarin", "warfarin", rxcui="11289")
    await service._recheck_against_rest("pantoprazole", [leftover], None)
    assert seen["types"] == ["NormalizedDrug", "NormalizedDrug"]


def _augmentin_vs_warfarin() -> list[NormalizedDrug]:
    return [
        _drug(
            "Augmentin", "amoxycillin", "clavulanic acid",
            rxcui="723",
            component_rxcuis={"amoxycillin": "723", "clavulanic acid": "21212"},
        ),
        _drug("warfarin", "warfarin", rxcui="11289"),
    ]


@pytest.mark.asyncio
async def test_fdc_either_ingredient_can_be_local_grade_a(monkeypatch):
    """Second (or first) FDC ingredient can hit RxNav without borrowing a sibling CUI."""
    captured: dict = {}
    payload = _rxnav_payload(
        ("21212", "Clavulanate", "11289", "Warfarin",
         "Clavulanate may increase the INR of warfarin."),
        ("723", "Amoxicillin", "11289", "Warfarin",
         "Amoxicillin may increase the anticoagulant activities of Warfarin."),
    )
    _patch_rxnav(monkeypatch, payload, captured)

    resolved, unknown = await check_known_pairs(_augmentin_vs_warfarin())
    requested = set((captured.get("rxcuis") or "").split())
    assert requested == {"723", "21212", "11289"}

    keys = {tuple(sorted(p.drugs)) for p in resolved}
    assert keys == {
        ("amoxycillin", "warfarin"),
        ("clavulanic acid", "warfarin"),
    }
    assert all(p.grade == Grade.A and p.source_tier == "local" for p in resolved)
    assert all(p.category == Category.INTERACTION for p in resolved)
    assert unknown == []


@pytest.mark.asyncio
async def test_fdc_second_ingredient_hit_skips_waterfall(monkeypatch):
    captured: dict = {}
    payload = _rxnav_payload(
        ("21212", "Clavulanate", "11289", "Warfarin",
         "Clavulanate may increase the INR of warfarin."),
    )
    _patch_rxnav(monkeypatch, payload, captured)

    resolved, unknown = await check_known_pairs(_augmentin_vs_warfarin())
    assert [tuple(sorted(p.drugs)) for p in resolved] == [
        ("clavulanic acid", "warfarin"),
    ]
    assert resolved[0].grade == Grade.A
    assert resolved[0].source_tier == "local"
    # first ingredient still goes to the waterfall; the hit does not
    assert ("amoxycillin", "warfarin") in {tuple(sorted(p)) for p in unknown}
    assert ("clavulanic acid", "warfarin") not in {tuple(sorted(p)) for p in unknown}


@pytest.mark.asyncio
async def test_leftover_fdc_keeps_remaining_component_cui(monkeypatch):
    captured: dict = {}
    _patch_rxnav(monkeypatch, {"fullInteractionTypeGroup": []}, captured)
    fdc = _augmentin_vs_warfarin()[0]
    leftover = fdc.model_copy(update={
        "components": ["clavulanic acid"],
        "rxcui": fdc.rxcui_for("clavulanic acid"),
    })
    warfarin = _drug("warfarin", "warfarin", rxcui="11289")
    resolved, unknown = await check_known_pairs([leftover, warfarin])
    requested = set((captured.get("rxcuis") or "").split())
    assert requested == {"21212", "11289"}
    assert "723" not in requested
    assert resolved == []
    assert unknown == [("clavulanic acid", "warfarin")]
