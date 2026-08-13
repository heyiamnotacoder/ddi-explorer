"""Download + compact the Indian medicine dataset into app/data/indian_drugs.json.

Source: github.com/junioralive/Indian-Medicine-Dataset (~250K Indian brands
with compositions, scraped from 1mg). We keep only brand_name -> [generics].

Usage: python scripts/fetch_indian_dataset.py
"""
import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

CSV_URL = ("https://raw.githubusercontent.com/junioralive/Indian-Medicine-Dataset/"
           "main/DATA/indian_medicine_data.csv")
OUT = Path(__file__).parent.parent / "app" / "data" / "indian_drugs.json"


def clean(comp: str) -> str | None:
    comp = comp.strip().lower()
    if not comp or comp in ("na", "none", "null"):
        return None
    return comp


def main() -> int:
    print("Downloading dataset (one-time, ~30-60MB)...")
    try:
        with urllib.request.urlopen(CSV_URL, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}\nYou can manually place the CSV and re-run.")
        return 1

    index: dict[str, list[str]] = {}
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        name = (row.get("name") or "").strip().lower()
        if not name or (row.get("Is_discontinued") or "").strip().upper() == "TRUE":
            continue
        comps = [c for c in (clean(row.get("short_composition1", "")),
                             clean(row.get("short_composition2", ""))) if c]
        if comps:
            index[name] = comps

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"Wrote {len(index)} brand entries -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
