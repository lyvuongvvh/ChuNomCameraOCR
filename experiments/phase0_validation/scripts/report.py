"""Phase 0 validation: turn compare_hanzi.py's per-page output into the final
results.md report - the numbers CLAUDE.md's Phase 0 step needs to decide whether
Phase 1 (fine-tuning pipeline) is worth building.

Usage:
    python scripts/report.py --comparison data/comparison.json --out results.md
"""
import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="data/comparison.json", type=pathlib.Path)
    parser.add_argument("--out", default="results.md", type=pathlib.Path)
    args = parser.parse_args()

    per_page = json.loads(args.comparison.read_text(encoding="utf-8"))

    total_hanzi = sum(p["total_hanzi"] for p in per_page.values())
    total_chat = sum(p["chat_correct"] for p in per_page.values())
    total_nomnaocr = sum(p["nomnaocr_correct"] for p in per_page.values())

    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:.1f}%" if d else "n/a"

    lines = [
        "# Phase 0 Validation Results",
        "",
        "Chu Han (Chinese-derived) character recognition, CHAT (pretrained, as-is) vs. "
        "NomNaOCR (pretrained CRNNxCTC, as-is), on the same sample pages.",
        "",
        "**Classification of Chu Han vs. Chu Nom-proper is a Unihan-kMandarin heuristic, "
        "not verified ground truth - see build_hanzi_charset.py.**",
        "",
        "## Aggregate",
        "",
        "| Model | Correct Han chars | Total Han chars | Accuracy |",
        "|---|---|---|---|",
        f"| CHAT (pretrained) | {total_chat} | {total_hanzi} | {pct(total_chat, total_hanzi)} |",
        f"| NomNaOCR (CRNNxCTC, pretrained) | {total_nomnaocr} | {total_hanzi} | {pct(total_nomnaocr, total_hanzi)} |",
        "",
        "## Per page",
        "",
        "| Page | Total Han | CHAT correct | NomNaOCR correct |",
        "|---|---|---|---|",
    ]
    for page_id, p in per_page.items():
        lines.append(f"| {page_id} | {p['total_hanzi']} | {p['chat_correct']} | {p['nomnaocr_correct']} |")

    lines += [
        "",
        "## Reading this result",
        "",
        "- Scoring is a bag-of-characters comparison per page (see compare_hanzi.py docstring), "
        "not full Sequence/Character Accuracy or CER - those come in Phase 2 for the fine-tuned model.",
        "- This is Phase 0 only: no fine-tuning has happened. If CHAT does not meaningfully beat "
        "NomNaOCR here, per CLAUDE.md the hybrid fine-tuning approach (Phase 1) should not be pursued "
        "without revisiting the hypothesis.",
    ]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"CHAT: {total_chat}/{total_hanzi} ({pct(total_chat, total_hanzi)})")
    print(f"NomNaOCR: {total_nomnaocr}/{total_hanzi} ({pct(total_nomnaocr, total_hanzi)})")


if __name__ == "__main__":
    main()
