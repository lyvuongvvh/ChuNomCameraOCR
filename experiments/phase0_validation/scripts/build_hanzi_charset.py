"""One-time setup: build a plain-text Chu Han (Chinese-derived) character charset from the
Unicode Unihan database, used to classify ground-truth characters as Chu Han vs. likely
Chu Nom-proper for the Phase 0 comparison (see compare_hanzi.py).

Heuristic (flagged per CLAUDE.md as an untested assumption, not ground truth): a character
with a `kMandarin` reading in Unihan is treated as Chu Han (i.e. it's attested as a real
Chinese character); one with no entry is treated as likely Chu Nom-proper. Some genuine
Nom-only characters were unified with rare/dialectal Chinese ideographs and would be
misclassified as Han by this heuristic - this affects Phase 0's numbers, not the recognizers
being compared, since both models are scored against the same classification.

Runs on the host directly (stdlib only, no kraken/tensorflow needed). Requires internet access.

Usage:
    python scripts/build_hanzi_charset.py --out data/hanzi_charset.txt
"""
import argparse
import io
import pathlib
import urllib.request
import zipfile

UNIHAN_ZIP_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    print(f"Downloading {UNIHAN_ZIP_URL} ...")
    with urllib.request.urlopen(UNIHAN_ZIP_URL) as resp:
        zip_bytes = resp.read()

    hanzi = set()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open("Unihan_Readings.txt") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.startswith("#") or not line.strip():
                    continue
                codepoint, field, _value = line.rstrip("\n").split("\t")
                if field == "kMandarin":
                    # codepoint looks like "U+4E00"
                    hanzi.add(chr(int(codepoint[2:], 16)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(sorted(hanzi)), encoding="utf-8")
    print(f"Wrote {len(hanzi)} Chu Han characters to {args.out}")


if __name__ == "__main__":
    main()
