"""Phase 0 validation: compare CHAT's and NomNaOCR's predictions against ground truth,
counting correct Chu Han (Chinese-derived) characters per CLAUDE.md's Phase 0 requirement.

Runs on the host directly (stdlib only). Needs:
  - data/manifest.json: maps each sample page to its ground-truth text and the patch
    filenames that make it up, in reading order. See README.md for the expected format and
    how to build it once the dataset is downloaded.
  - data/hanzi_charset.txt: from build_hanzi_charset.py
  - data/predictions_chat.json: from run_chat.py (keyed by page image filename)
  - data/predictions_nomnaocr.json: from run_nomnaocr.py (keyed by patch filename)

Scoring is deliberately simple for Phase 0: per page, for each Chu Han character in the
ground truth, count how many times it also appears in the model's predicted text (capped at
the ground-truth count), i.e. a multiset/bag-of-characters intersection. This sidesteps
sequence-alignment and reading-order issues (CHAT's own README notes a reading-order model
isn't published yet) that would otherwise contaminate a strict positional comparison. It is
NOT the Sequence Accuracy / Character Accuracy / CER metrics NomNaOCR reports - those apply
properly in Phase 2, once there's a fine-tuned model to evaluate against a held-out test set
with consistent tokenization.
"""
import argparse
import json
import pathlib
from collections import Counter


def load_charset(path: pathlib.Path) -> set[str]:
    return set(path.read_text(encoding="utf-8"))


def bag_correct(predicted_text: str, ground_truth_text: str, charset: set[str]) -> tuple[int, int]:
    """Return (correct, total) Chu Han characters, via multiset intersection."""
    gt_hanzi = Counter(c for c in ground_truth_text if c in charset)
    pred_hanzi = Counter(c for c in predicted_text if c in charset)
    correct = sum(min(count, pred_hanzi[c]) for c, count in gt_hanzi.items())
    total = sum(gt_hanzi.values())
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.json", type=pathlib.Path)
    parser.add_argument("--charset", default="data/hanzi_charset.txt", type=pathlib.Path)
    parser.add_argument("--chat-predictions", default="data/predictions_chat.json", type=pathlib.Path)
    parser.add_argument("--nomnaocr-predictions", default="data/predictions_nomnaocr.json", type=pathlib.Path)
    parser.add_argument("--finetuned-predictions", default=None, type=pathlib.Path,
                         help="Optional: predictions_chat.json-shaped file from a fine-tuned "
                              "CHAT checkpoint (run_chat.py --rec-model ...), for the fine-tuning "
                              "trial's three-way comparison. Omit for the standard Phase 0 two-way run.")
    parser.add_argument("--out", default="data/comparison.json", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    charset = load_charset(args.charset)
    chat_preds = json.loads(args.chat_predictions.read_text(encoding="utf-8"))
    nomnaocr_preds = json.loads(args.nomnaocr_predictions.read_text(encoding="utf-8"))
    finetuned_preds = (
        json.loads(args.finetuned_predictions.read_text(encoding="utf-8"))
        if args.finetuned_predictions else None
    )

    per_page = {}
    for page_id, page in manifest.items():
        ground_truth = page["ground_truth"]

        chat_text = chat_preds.get(page["page_image"], {}).get("full_text", "")
        nomnaocr_text = "".join(nomnaocr_preds.get(patch, "") for patch in page["patches"])

        chat_correct, total = bag_correct(chat_text, ground_truth, charset)
        nomnaocr_correct, total_check = bag_correct(nomnaocr_text, ground_truth, charset)
        assert total == total_check

        per_page[page_id] = {
            "total_hanzi": total,
            "chat_correct": chat_correct,
            "nomnaocr_correct": nomnaocr_correct,
        }

        if finetuned_preds is not None:
            finetuned_text = finetuned_preds.get(page["page_image"], {}).get("full_text", "")
            finetuned_correct, total_check = bag_correct(finetuned_text, ground_truth, charset)
            assert total == total_check
            per_page[page_id]["finetuned_correct"] = finetuned_correct

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(per_page, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote per-page comparison to {args.out}")


if __name__ == "__main__":
    main()
