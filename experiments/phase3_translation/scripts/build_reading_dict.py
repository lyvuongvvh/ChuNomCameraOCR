"""Phase 3, Stage 1: build a best-effort Han-Nom character -> Vietnamese reading dictionary,
combining three sources - deliberately NOT claimed to be complete (see README.md's coverage
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
3. VNPF's digitized "Bang tra chu Nom" (Ho Le, chief editor; Institute of Linguistics, Hanoi,
   1976), queried live at nomfoundation.org/nom-tools/BTCN - only for characters the two sources
   above don't already cover, since it's a small nonprofit's server, not a bulk-download dataset.
   Empirically NOT a superset of Unihan (e.g. resolves 以 and 北 but not 為 or 何 - it catalogs
   characters used to write Nom, not a general Han-Vietnamese dictionary), so it's a genuine
   complement rather than a replacement. Used here per nomfoundation.org's terms of use, which
   permit non-commercial/research use with attribution and restrict only commercial
   redistribution.

Characters covered by more than one source keep all reading variants (deduplicated). Characters
in none of the three are left uncovered - `translate_lib/reading.py` passes them through
unchanged rather than guessing, and Stage 2 (LLM) is expected to use surrounding context for
those.

Usage:
    python scripts/build_reading_dict.py --vocab-labels ../NomNaOCR/Patches/All.txt \
        --out data/reading_dict.json
"""
import argparse
import io
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))

from nomnaocr_lib.vocab import build_vocab  # noqa: E402

UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
RIME_CHUNOM_URL = "https://raw.githubusercontent.com/pearapple123/rime-chunom/master/chu_nom.dict.yaml"
BTCN_URL = "https://nomfoundation.org/nom-tools/BTCN/BTCN?uiLang=vn"

# The tool renders one query per character as a delimited section starting with this literal
# header row (confirmed against the tool's real HTML, not guessed) - splitting on it is far more
# robust than trying to match every tag in its dated, occasionally-unclosed markup.
BTCN_SECTION_DELIM = "<tr><th style='width:10%'></th><th style='width:20%'></th><th style='width:70%'></th></tr>"
BTCN_CHAR_RE = re.compile(r"<U>(.)</B></U>")
# A reading row is either "<i>READING</i>" optionally followed by its own example cell (used when
# a character has exactly one reading), or a plain "READING" cell + example cell (used for each
# additional reading when a character has several) - both confirmed against real responses for
# characters with 1, 2, and 5 readings.
BTCN_READING_RE = re.compile(
    r"<td></td><td><i>([^<]+)</i></td>(?:<td>\s*([^<]*)</td></tr>)?"
    r"|<td></td><td>([^<]+)</td><td>\s*([^<]*)</td></tr>"
)


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


def parse_btcn_response(html: str) -> dict:
    """Parses one response from the Bang tra chu Nom lookup tool, which accepts several
    characters concatenated in a single query and returns one section per character (confirmed
    empirically - not documented anywhere). Returns {char: [reading, ...]} for characters the
    dictionary resolved; a character with no entry (its section just says "Khong tim thay!") is
    simply absent from the result, same convention the other two fetchers use."""
    readings = {}
    for section in html.split(BTCN_SECTION_DELIM)[1:]:
        char_match = BTCN_CHAR_RE.search(section)
        if not char_match:
            continue
        ch = char_match.group(1)
        for m in BTCN_READING_RE.finditer(section):
            reading = m.group(1) or m.group(3)
            if reading:
                readings.setdefault(ch, []).append(reading.strip().lower())
    return readings


def fetch_btcn_readings(chars: list, batch_size: int = 80, delay: float = 1.0) -> dict:
    """Queries nomfoundation.org's Bang tra chu Nom for `chars`, batching multiple characters per
    request (confirmed to work, not just single-character queries) and pausing between requests -
    this is a small nonprofit's server, not a CDN, so the goal is a handful of requests for
    exactly the characters we actually need (see main() - only vocab characters Unihan/rime-chunom
    didn't already cover), not one request per character or a full-dictionary crawl."""
    readings = {}
    chars = list(dict.fromkeys(chars))  # de-dupe, keep deterministic order
    for i in range(0, len(chars), batch_size):
        batch = chars[i:i + batch_size]
        body = urllib.parse.urlencode(
            {"inputText": "".join(batch), "includeCitations": "", "hidden1": "no"}
        ).encode("utf-8")
        req = urllib.request.Request(
            BTCN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        readings.update(parse_btcn_response(html))
        done = min(i + batch_size, len(chars))
        print(f"  queried {done}/{len(chars)} characters, {len(readings)} resolved so far")
        if done < len(chars):
            time.sleep(delay)
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-labels", required=True, type=pathlib.Path,
                         help="Label file to report coverage against (e.g. NomNaOCR's All.txt)")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--skip-btcn", action="store_true",
                         help="Skip the live BTCN lookup (Unihan+rime-chunom only) - for fast "
                              "dev iteration without depending on nomfoundation.org's server.")
    args = parser.parse_args()

    print("Fetching Unihan kVietnamese readings...")
    unihan = fetch_unihan_kvietnamese()
    print(f"  {len(unihan)} characters")

    print("Fetching rime-chunom single-character readings...")
    rime = fetch_rime_chunom_single_chars()
    print(f"  {len(rime)} characters")

    vocab = build_vocab(str(args.vocab_labels), min_length=1)
    unihan_or_rime_covered = sum(1 for c in vocab if c in unihan or c in rime)

    btcn = {}
    if not args.skip_btcn:
        uncovered = [c for c in vocab if c not in unihan and c not in rime]
        print(f"\nQuerying VNPF's Bang tra chu Nom for the {len(uncovered)} project-vocab "
              f"characters Unihan/rime-chunom don't cover...")
        btcn = fetch_btcn_readings(uncovered)
        print(f"  resolved {len(btcn)} of {len(uncovered)} previously-uncovered characters")

    combined = {}
    for ch, vals in unihan.items():
        combined.setdefault(ch, {"readings": [], "sources": []})
        combined[ch]["readings"].extend(v for v in vals if v not in combined[ch]["readings"])
        combined[ch]["sources"].append("unihan")
    for ch, vals in rime.items():
        combined.setdefault(ch, {"readings": [], "sources": []})
        combined[ch]["readings"].extend(v for v in vals if v not in combined[ch]["readings"])
        combined[ch]["sources"].append("rime-chunom")
    for ch, vals in btcn.items():
        combined.setdefault(ch, {"readings": [], "sources": []})
        combined[ch]["readings"].extend(v for v in vals if v not in combined[ch]["readings"])
        combined[ch]["sources"].append("btcn")

    covered = [c for c in vocab if c in combined]
    print(f"\nProject vocab: {len(vocab)} characters")
    print(f"Combined dictionary covers: {len(covered)} ({100 * len(covered) / len(vocab):.1f}%)")
    print(f"Unihan-only would cover: {sum(1 for c in vocab if c in unihan)} "
          f"({100 * sum(1 for c in vocab if c in unihan) / len(vocab):.1f}%)")
    print(f"Unihan+rime-chunom (previous state) would cover: {unihan_or_rime_covered} "
          f"({100 * unihan_or_rime_covered / len(vocab):.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(combined)} entries to {args.out}")


if __name__ == "__main__":
    main()
