"""Pre-filter accepts the same NormalizedDrug shape as a fresh check."""
import pytest

from app import service
from app.models import NormalizedDrug
from app.pipeline.prefilter import check_known_pairs


def _drug(name: str, *comps: str, rxcui: str | None = None) -> NormalizedDrug:
    parts = list(comps) or [name]
    return NormalizedDrug(
        input_name=name, generic_name=", ".join(parts),
        rxcui=rxcui, components=parts,
    )


@pytest.mark.asyncio
async def test_check_known_pairs_takes_normalized_drugs():
    resolved, unknown = await check_known_pairs([
        _drug("warfarin", "warfarin"),
        _drug("omeprazole", "omeprazole"),
    ])
    assert resolved == []
    assert unknown == [("warfarin", "omeprazole")]


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
async def test_recheck_passes_normalized_drugs(monkeypatch):
    seen: dict = {}

    async def fake_norm(names):
        return [_drug(names[0], "pantoprazole", rxcui="40790")], []

    async def fake_prefilter(drugs):
        seen["types"] = [type(d).__name__ for d in drugs]
        return [], []

    async def fake_waterfall(*_a, **_k):
        return []

    monkeypatch.setattr(service.norm, "normalize_drugs", fake_norm)
    monkeypatch.setattr(service.prefilter, "check_known_pairs", fake_prefilter)
    monkeypatch.setattr(service.waterfall, "evaluate_pairs", fake_waterfall)

    leftover = _drug("warfarin", "warfarin", rxcui="11289")
    await service._recheck_against_rest("pantoprazole", [leftover], None)
    assert seen["types"] == ["NormalizedDrug", "NormalizedDrug"]
