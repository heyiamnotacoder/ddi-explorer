"""Scrubber tests — the privacy guarantees of the whole app live here."""
from app.pipeline.scrubber import for_reasoning_llm, scrub_text


def test_patient_identifiers_removed():
    r = scrub_text("Patient Ramesh Kumar, MRN: AIIMS-20451, DOB: 12/05/1968. Age 56, CrCl 42.")
    assert "Ramesh" not in r.text
    assert "AIIMS-20451" not in r.text
    assert "12/05/1968" not in r.text
    # clinical context must survive
    assert "Age 56" in r.text and "CrCl 42" in r.text


def test_phone_and_email_removed():
    r = scrub_text("Contact 9876543210 or dr.mehta@hospital.in for history.")
    assert "9876543210" not in r.text
    assert "dr.mehta@hospital.in" not in r.text


def test_aadhaar_and_abha_removed():
    r = scrub_text("Aadhaar 1234 5678 9012, ABHA 91-2345-6789-0123")
    assert "1234 5678 9012" not in r.text
    assert "91-2345-6789-0123" not in r.text


def test_drug_names_never_redacted():
    """Regression: NER flagged Indian brands ('Telma', 'Dolo') as PERSON."""
    r = scrub_text("Tab Telma 40 1-0-0, Tab Dolo 650 1-0-1, Cap Augmentin 625 1-1-1")
    assert "Telma" in r.text
    assert "Dolo" in r.text
    assert "Augmentin" in r.text


def test_dosing_schedule_survives():
    r = scrub_text("Tab Metformin 500 1-0-1 for 3 months")
    assert "1-0-1" in r.text
    assert "3 months" in r.text  # relative time is dosing info, not PHI


def test_empty_and_clean_text():
    assert scrub_text("").text == ""
    r = scrub_text("warfarin 5 mg + aspirin 75 mg")
    assert r.text == "warfarin 5 mg + aspirin 75 mg"


def test_for_reasoning_llm_strips_phone_and_mrn():
    out = for_reasoning_llm(
        "Call 9876543210, MRN: AIIMS-20451. Age 56, CrCl 42.")
    assert out is not None
    assert "9876543210" not in out
    assert "AIIMS-20451" not in out
    assert "Age 56" in out and "CrCl 42" in out
    assert "[PHONE_" in out and "[MRN_" in out


def test_for_reasoning_llm_blank_is_none():
    assert for_reasoning_llm(None) is None
    assert for_reasoning_llm("") is None
    assert for_reasoning_llm("   ") is None
