#!/usr/bin/env python3
"""Live full POST /api/check eval. Not default pytest. Needs backend/.env keys."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
REQUIRED_IDS = {"warfarin-aspirin", "maoi-ssri", "screenshot-quartet"}
NAME_ALIASES = {
    "amoxycillin": "amoxicillin",
    "amoxicillin": "amoxycillin",
    "rifampicin": "rifampin",
    "rifampin": "rifampicin",
    "paracetamol": "acetaminophen",
    "acetaminophen": "paracetamol",
}
CASE_KEYS = {"id", "text", "patient_context", "timing", "expect"}
EXPECT_KEYS = {
    "min_normalized",
    "max_normalized",
    "generics_include",
    "unresolved_empty",
    "pair",
    "grade",
    "grade_in",
    "category",
    "banner",
    "avoid_substance",
    "components_singleton",
    "fdc_min_components",
    "patient_note",
    "dose_condition",
    "no_contraindication",
    "pairs_not_grade_a",
}


def load_cases() -> list[dict]:
    blob = json.loads(CASES_PATH.read_text())
    cases = blob.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"no cases in {CASES_PATH}")
    return cases


def validate_cases(cases: list[dict], *, require_anchors: bool = True) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for i, case in enumerate(cases):
        loc = case.get("id") or f"index {i}"
        extra = set(case) - CASE_KEYS
        if extra:
            errors.append(f"{loc}: unknown case keys {sorted(extra)}")
        if not case.get("id"):
            errors.append(f"index {i}: missing id")
        elif case["id"] in ids:
            errors.append(f"{loc}: duplicate id")
        else:
            ids.add(case["id"])
        if not str(case.get("text") or "").strip():
            errors.append(f"{loc}: empty text")
        expect = case.get("expect")
        if not isinstance(expect, dict):
            errors.append(f"{loc}: expect must be an object")
            continue
        bad = set(expect) - EXPECT_KEYS
        if bad:
            errors.append(f"{loc}: unknown expect keys {sorted(bad)}")
        pair = expect.get("pair")
        if pair is not None and (
            not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(x, str) and x.strip() for x in pair)
        ):
            errors.append(f"{loc}: pair must be two names")
        if expect.get("grade") and not pair:
            errors.append(f"{loc}: grade requires pair")
        if expect.get("grade_in") and not pair:
            errors.append(f"{loc}: grade_in requires pair")
        if expect.get("category") and not pair:
            errors.append(f"{loc}: category requires pair")
        not_a = expect.get("pairs_not_grade_a")
        if not_a is not None:
            if not isinstance(not_a, list) or not not_a:
                errors.append(f"{loc}: pairs_not_grade_a must be a non-empty list")
            else:
                for item in not_a:
                    if (
                        not isinstance(item, list) or len(item) != 2
                        or not all(isinstance(x, str) and x.strip() for x in item)
                    ):
                        errors.append(f"{loc}: pairs_not_grade_a items must be two names")
    if require_anchors:
        if len(cases) < 30:
            errors.append(f"need ~30 cases, found {len(cases)}")
        missing_anchors = REQUIRED_IDS - {c.get("id") for c in cases}
        if missing_anchors:
            errors.append(f"missing anchors {sorted(missing_anchors)}")
    elif not cases:
        errors.append("need at least 1 case")
    return errors


def _names_of(drug: dict) -> list[str]:
    out: list[str] = []
    for key in ("generic_name", "input_name"):
        v = drug.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v)
    for c in drug.get("components") or []:
        if isinstance(c, str) and c.strip():
            out.append(c)
    return out


def _all_names(resp: dict) -> list[str]:
    names: list[str] = []
    for d in resp.get("normalized_drugs") or []:
        names.extend(_names_of(d))
    return names


def _has_name(names: list[str], want: str) -> bool:
    needles = {want.lower()}
    alias = NAME_ALIASES.get(want.lower())
    if alias:
        needles.add(alias)
    for n in names:
        low = n.lower()
        for w in needles:
            if w == low or w in low or low in w:
                return True
    return False


def _find_pair(resp: dict, a: str, b: str) -> dict | None:
    for p in resp.get("pairs") or []:
        drugs = [str(x) for x in (p.get("drugs") or [])]
        if _has_name(drugs, a) and _has_name(drugs, b):
            return p
    return None


def _fail(failures: list[str], msg: str) -> None:
    failures.append(msg)


def judge(case: dict, resp: dict) -> list[str]:
    failures: list[str] = []
    expect = case.get("expect") or {}
    disclaimer = resp.get("disclaimer") or ""
    if "decision-support" not in disclaimer.lower():
        _fail(failures, "missing disclaimer")

    drugs = resp.get("normalized_drugs") or []
    if expect.get("min_normalized") is not None and len(drugs) < expect["min_normalized"]:
        _fail(failures, f"normalized {len(drugs)} < {expect['min_normalized']}")
    if expect.get("max_normalized") is not None and len(drugs) > expect["max_normalized"]:
        _fail(failures, f"normalized {len(drugs)} > {expect['max_normalized']}")

    names = _all_names(resp)
    for g in expect.get("generics_include") or []:
        if not _has_name(names, g):
            _fail(failures, f"missing generic {g!r} in {names}")

    if expect.get("unresolved_empty") and (resp.get("unresolved_drugs") or []):
        _fail(failures, f"unresolved {resp.get('unresolved_drugs')}")

    if expect.get("components_singleton"):
        for d in drugs:
            comps = d.get("components") or []
            if len(comps) != 1:
                _fail(failures, f"{d.get('input_name')!r} has components {comps}")

    if expect.get("fdc_min_components"):
        n = expect["fdc_min_components"]
        if not any(len(d.get("components") or []) >= n for d in drugs):
            _fail(failures, f"no FDC with >={n} components")

    pair_spec = expect.get("pair")
    found = None
    if pair_spec:
        found = _find_pair(resp, pair_spec[0], pair_spec[1])
        if found is None:
            shown = [p.get("drugs") for p in (resp.get("pairs") or [])]
            _fail(failures, f"pair {pair_spec} not checked; pairs={shown}")
        else:
            grade = found.get("grade")
            if expect.get("grade") and grade != expect["grade"]:
                _fail(failures, f"grade {grade!r} != {expect['grade']!r}")
            if expect.get("grade_in") and grade not in expect["grade_in"]:
                _fail(failures, f"grade {grade!r} not in {expect['grade_in']}")
            if expect.get("category") and found.get("category") != expect["category"]:
                _fail(
                    failures,
                    f"category {found.get('category')!r} != {expect['category']!r}",
                )
            if expect.get("dose_condition") and not found.get("dose_condition"):
                _fail(failures, "missing dose_condition")
            if expect.get("patient_note") and not found.get("patient_specific_note"):
                _fail(failures, "missing patient_specific_note")

    for p in resp.get("pairs") or []:
        if p.get("grade") in ("A", "B", "C") and not (p.get("citations") or []):
            _fail(failures, f"graded {p.get('drugs')} has no citations")

    banner = resp.get("contraindicated_banner") or []
    if expect.get("banner") is True and not banner:
        _fail(failures, "expected contraindication banner")
    if expect.get("banner") is False and banner:
        _fail(failures, f"unexpected banner { [b.get('drugs') for b in banner] }")

    if expect.get("no_contraindication"):
        if banner:
            _fail(failures, f"unexpected banner {[b.get('drugs') for b in banner]}")
        ci = [
            p.get("drugs")
            for p in (resp.get("pairs") or [])
            if p.get("category") == "contraindicated"
        ]
        if ci:
            _fail(failures, f"contraindicated pairs {ci}")

    for pair_spec in expect.get("pairs_not_grade_a") or []:
        found_na = _find_pair(resp, pair_spec[0], pair_spec[1])
        if found_na is not None and found_na.get("grade") == "A":
            _fail(
                failures,
                f"pair {pair_spec} should not be Grade A "
                f"(grade={found_na.get('grade')!r} category={found_na.get('category')!r})",
            )

    if expect.get("avoid_substance"):
        want = expect["avoid_substance"].lower()
        items = resp.get("avoid_with_medications") or []
        hit = any(want in str(it.get("substance") or "").lower() for it in items)
        if not hit:
            _fail(failures, f"avoid-with missing {want!r}; got {items}")

    return failures


async def run_case(client, case: dict, timeout: float) -> dict:
    payload = {"text": case["text"]}
    if case.get("patient_context"):
        payload["patient_context"] = case["patient_context"]
    if case.get("timing"):
        payload["timing"] = case["timing"]
    t0 = time.monotonic()
    try:
        r = await client.post("/api/check", json=payload, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — eval must record the miss
        return {
            "id": case["id"],
            "ok": False,
            "seconds": round(time.monotonic() - t0, 2),
            "failures": [f"request error: {type(e).__name__}: {e}"],
        }
    seconds = round(time.monotonic() - t0, 2)
    if r.status_code != 200:
        return {
            "id": case["id"],
            "ok": False,
            "seconds": seconds,
            "failures": [f"HTTP {r.status_code}: {r.text[:500]}"],
        }
    try:
        resp = r.json()
    except Exception as e:  # noqa: BLE001
        return {
            "id": case["id"],
            "ok": False,
            "seconds": seconds,
            "failures": [f"invalid JSON: {e}"],
        }
    failures = judge(case, resp)
    return {
        "id": case["id"],
        "ok": not failures,
        "seconds": seconds,
        "failures": failures,
        "normalized": [
            {
                "input": d.get("input_name"),
                "generic": d.get("generic_name"),
                "components": d.get("components"),
            }
            for d in (resp.get("normalized_drugs") or [])
        ],
        "pairs": [
            {
                "drugs": p.get("drugs"),
                "grade": p.get("grade"),
                "category": p.get("category"),
                "tier": p.get("source_tier"),
            }
            for p in (resp.get("pairs") or [])
        ],
    }


def matcher_self_check() -> list[str]:
    """Offline checks that the judge enforces the three anchors."""
    errors: list[str] = []
    a_ok = {
        "disclaimer": "clinical decision-support aid. It does not replace clinical judgment.",
        "normalized_drugs": [
            {"input_name": "warfarin 5 mg", "generic_name": "warfarin", "components": ["warfarin"]},
            {"input_name": "aspirin 75 mg", "generic_name": "aspirin", "components": ["aspirin"]},
        ],
        "pairs": [{
            "drugs": ["warfarin", "aspirin"],
            "grade": "A",
            "category": "interaction",
            "citations": [{"source": "openfda", "title": "label"}],
        }],
        "contraindicated_banner": [],
        "avoid_with_medications": [],
        "unresolved_drugs": [],
    }
    f = judge(
        {"id": "warfarin-aspirin", "expect": {"pair": ["warfarin", "aspirin"], "grade": "A"}},
        a_ok,
    )
    if f:
        errors.append(f"warfarin-aspirin should pass: {f}")
    f = judge(
        {"id": "warfarin-aspirin", "expect": {"pair": ["warfarin", "aspirin"], "grade": "A"}},
        {**a_ok, "pairs": [{**a_ok["pairs"][0], "grade": "C"}]},
    )
    if not f:
        errors.append("warfarin-aspirin Grade C should fail")

    banner_ok = {
        **a_ok,
        "normalized_drugs": [
            {"input_name": "phenelzine", "generic_name": "phenelzine", "components": ["phenelzine"]},
            {"input_name": "fluoxetine", "generic_name": "fluoxetine", "components": ["fluoxetine"]},
        ],
        "pairs": [{
            "drugs": ["phenelzine", "fluoxetine"],
            "grade": "A",
            "category": "contraindicated",
            "citations": [{"source": "openfda", "title": "label"}],
        }],
        "contraindicated_banner": [{
            "drugs": ["phenelzine", "fluoxetine"],
            "category": "contraindicated",
        }],
    }
    f = judge({"id": "maoi-ssri", "expect": {"banner": True}}, banner_ok)
    if f:
        errors.append(f"maoi-ssri should pass: {f}")
    f = judge({"id": "maoi-ssri", "expect": {"banner": True}}, {**banner_ok, "contraindicated_banner": []})
    if not f:
        errors.append("maoi-ssri without banner should fail")

    shot = {
        "disclaimer": "clinical decision-support aid.",
        "normalized_drugs": [
            {"input_name": "amilodipine 20mg", "generic_name": "amlodipine", "components": ["amlodipine"]},
            {"input_name": "levoceterizine 5 mg", "generic_name": "levocetirizine", "components": ["levocetirizine"]},
            {"input_name": "warfar 100 mg", "generic_name": "warfarin", "components": ["warfarin"]},
            {"input_name": "omeprazole 5 mg", "generic_name": "omeprazole", "components": ["omeprazole"]},
        ],
        "pairs": [{
            "drugs": ["warfarin", "omeprazole"],
            "grade": "A",
            "citations": [{"source": "openfda", "title": "label"}],
        }],
        "contraindicated_banner": [],
        "avoid_with_medications": [],
        "unresolved_drugs": [],
    }
    f = judge({
        "id": "screenshot-quartet",
        "expect": {
            "min_normalized": 4,
            "max_normalized": 4,
            "components_singleton": True,
            "generics_include": ["amlodipine", "levocetirizine", "warfarin", "omeprazole"],
            "pair": ["warfarin", "omeprazole"],
        },
    }, shot)
    if f:
        errors.append(f"screenshot-quartet should pass: {f}")
    f = judge({
        "id": "screenshot-quartet",
        "expect": {"pair": ["warfarin", "omeprazole"], "min_normalized": 4},
    }, {**shot, "pairs": []})
    if not f:
        errors.append("screenshot-quartet without warfarin-omeprazole should fail")

    noci = {
        **a_ok,
        "pairs": [{
            "drugs": ["metformin", "sitagliptin"],
            "grade": None,
            "category": "none",
            "citations": [],
        }],
        "contraindicated_banner": [],
    }
    f = judge(
        {"id": "t2dm-oral-triple", "expect": {
            "no_contraindication": True,
            "pairs_not_grade_a": [["metformin", "sitagliptin"]],
        }},
        noci,
    )
    if f:
        errors.append(f"no-CI colist should pass: {f}")
    f = judge(
        {"id": "t2dm-oral-triple", "expect": {
            "no_contraindication": True,
            "pairs_not_grade_a": [["metformin", "sitagliptin"]],
        }},
        {**noci, "pairs": [{
            "drugs": ["metformin", "sitagliptin"],
            "grade": "A",
            "category": "contraindicated",
            "citations": [{"source": "openfda", "title": "label"}],
        }], "contraindicated_banner": [{
            "drugs": ["metformin", "sitagliptin"],
            "category": "contraindicated",
        }]},
    )
    if not f:
        errors.append("no-CI colist Grade A banner should fail")
    return errors


def require_keys() -> None:
    from app.config import get_settings

    s = get_settings()
    if not (s.anthropic_api_key or s.deepseek_api_key or s.openai_api_key):
        raise SystemExit(
            "No LLM key in backend/.env (ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / "
            "OPENAI_API_KEY). Live eval will not start."
        )


async def run_live(cases: list[dict], timeout: float, base_url: str | None) -> list[dict]:
    import httpx

    if base_url:
        client_cm = httpx.AsyncClient(base_url=base_url.rstrip("/"))
    else:
        require_keys()
        from httpx import ASGITransport

        from app.main import app

        client_cm = httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://eval.local",
        )

    results: list[dict] = []
    async with client_cm as client:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case['id']} ...", flush=True)
            row = await run_case(client, case, timeout)
            mark = "ok" if row["ok"] else "FAIL"
            extra = "" if row["ok"] else " — " + "; ".join(row["failures"])
            print(f"  {mark} {row['seconds']}s{extra}", flush=True)
            results.append(row)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="validate cases.json only")
    p.add_argument("--only", action="append", default=[], help="run these case ids")
    p.add_argument("--limit", type=int, default=0, help="run at most N cases")
    p.add_argument("--timeout", type=float, default=180.0, help="seconds per case")
    p.add_argument(
        "--base-url",
        default=None,
        help="hit a running server (e.g. http://127.0.0.1:8000); default is in-process ASGI",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="cases JSON path (default: eval/cases.json)",
    )
    args = p.parse_args(argv)

    cases_path = args.cases.resolve() if args.cases else CASES_PATH
    blob = json.loads(cases_path.read_text())
    cases = blob.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"no cases in {cases_path}")
    schema_errors = validate_cases(
        cases, require_anchors=cases_path.resolve() == CASES_PATH.resolve()
    )
    if schema_errors:
        print("cases.json invalid:", file=sys.stderr)
        for e in schema_errors:
            print(f"  {e}", file=sys.stderr)
        return 2

    if args.only:
        want = set(args.only)
        cases = [c for c in cases if c["id"] in want]
        missing = want - {c["id"] for c in cases}
        if missing:
            print(f"unknown ids: {sorted(missing)}", file=sys.stderr)
            return 2
    if args.limit:
        cases = cases[: args.limit]

    print(f"{len(cases)} cases from {cases_path}")
    if args.dry_run:
        if cases_path.resolve() == CASES_PATH.resolve():
            self_errors = matcher_self_check()
            if self_errors:
                print("matcher self-check failed:", file=sys.stderr)
                for e in self_errors:
                    print(f"  {e}", file=sys.stderr)
                return 2
        print("dry-run ok")
        return 0

    results = asyncio.run(run_live(cases, args.timeout, args.base_url))
    passed = sum(1 for r in results if r["ok"])
    report = {
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}")
    print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
