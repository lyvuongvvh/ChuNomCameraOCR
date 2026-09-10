"""Phase 3, Stage 1: build a best-effort Han-Nom character -> Vietnamese reading dictionary,
combining two sources - deliberately NOT claimed to be complete (see README.md's coverage
numbers and "Known simplifications"):

1. Unicode's Unihan database (`kVietnamese` field) - authoritative Sino-Vietnamese readings for
   standard Han characters, permissively licensed (Unicode Character Database terms), but only
   ~50% coverage against this project's own vocabulary (`Patches/All.txt`) - and notably
   incomplete even for common classical Han characters (e.g. 以, 為, 而, 者 have no kVietnamese
   entry), not just genuinely Nom-specific ones.
2. `pearapple123/rime-chunom`'s `chu_nom.dict.yaml` single-character section - a small (~1,000
   entries), community-curated Rime IME dictionary sourced from chunom.org, covering exactly the
   kind of high-frequency Nom-invented function words (e.g. 𧵑 "của", 㐌 "đã") Unihan has no
   readings for at all. No explicit license found on that repository - used here for research/
   prototyping only, flagged as an open question for any production use (see README.md).

Characters covered by both sources keep all reading variants (deduplicated). Characters in
neither source are left uncovered - `translate_lib/reading.py` passes them through unchanged
rather than guessing, and Stage 2 (LLM) is expected to use surrounding context for those.

Usage:
    python scripts/build_reading_dict.py --vocab-labels ../NomNaOCR/Patches/All.txt \
        --out data/reading_dict.json
"""
import argparse
import io
import json
import pathlib
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))

from nomnaocr_lib.vocab import build_vocab  # noqa: E402

UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
RIME_CHUNOM_URL = "https://raw.githubusercontent.com/pearapple123/rime-chunom/master/chu_nom.dict.yaml"


def fetch_unihan_kvietnamese() -> dict:
    data = urllib.request.urlopen(UNIHAN_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    text = z.read("Unihan_Readings.txt").decode("utf-8")
    readings = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        cp, field, value = parts
        if field == "kVietnamese":
            # kVietnamese's own delimiter is space - a single line can list multiple candidate
            # readings (e.g. "bac phao phao" for 砲), not one multi-word reading.
            ch = chr(int(cp[2:], 16))
            readings.setdefault(ch, []).extend(value.strip().lower().split())
    return readings


def fetch_rime_chunom_single_chars() -> dict:
    text = urllib.request.urlopen(RIME_CHUNOM_URL, timeout=30).read().decode("utf-8")
    lines = text.splitlines()
    compounds_idx = next(i for i, l in enumerate(lines) if l.strip() == "#Compounds")
    readings = {}
    for line in lines[:compounds_idx]:
        if "\t" not in line:
            continue
        ch, reading = line.split("\t", 1)
        ch, reading = ch.strip(), reading.strip().lower()
        if len(ch) == 1 and reading:
            readings.setdefault(ch, []).append(reading)
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-labels", required=True, type=pathlib.Path,
                         help="Label file to report coverage against (e.g. NomNaOCR's All.txt)")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    print("Fetching Unihan kVietnamese readings...")
    unihan = fetch_unihan_kvietnamese()
    print(f"  {len(unihan)} characters")

    print("Fetching rime-chunom single-character readings...")
    rime = fetch_rime_chunom_single_chars()
    print(f"  {len(rime)} characters")

    combined = {}
    for ch, vals in unihan.items():
        combined.setdefault(ch, {"readings": [], "sources": []})
        combined[ch]["readings"].extend(v for v in vals if v not in combined[ch]["readings"])
        combined[ch]["sources"].append("unihan")
    for ch, vals in rime.items():
        combined.setdefault(ch, {"readings": [], "sources": []})
        combined[ch]["readings"].extend(v for v in vals if v not in combined[ch]["readings"])
        combined[ch]["sources"].append("rime-chunom")

    vocab = build_vocab(str(args.vocab_labels), min_length=1)
    covered = [c for c in vocab if c in combined]
    print(f"\nProject vocab: {len(vocab)} characters")
    print(f"Combined dictionary covers: {len(covered)} ({100 * len(covered) / len(vocab):.1f}%)")
    print(f"Unihan-only would cover: {sum(1 for c in vocab if c in unihan)} "
          f"({100 * sum(1 for c in vocab if c in unihan) / len(vocab):.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(combined)} entries to {args.out}")


if __name__ == "__main__":
    main()
