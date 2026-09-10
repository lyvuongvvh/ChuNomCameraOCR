"""Phase 3, Stage 2: run the real LLM translation pipeline over recognized Han-Nom text.

Defaults to Phase 2b's post-corrected predictions as input, not the raw baseline - results.md
found post-correction recovers ~2.5x more translation-relevant errors than fine-tuning, making it
the stronger source text for this stage. Override with --predictions to use a different source
(e.g. Phase 2's baseline or Phase 2c's fine-tuned output).

COST WARNING: this calls the real Anthropic API, billed per token on your own API key. Defaults
to --max-lines 10 so a first run is cheap and fast to sanity-check; pass --all explicitly to run
the complete input set (thousands of lines - meaningful real cost, not a default to reach for
without deciding to).

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=...   (from https://console.anthropic.com/settings/keys)

Usage:
    python scripts/translate.py \
        --predictions ../phase2b_postcorrection/data/corrected_predictions.json \
        --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --reading-dict data/reading_dict.json \
        --out data/translations.json --max-lines 10
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from translate_lib.reading import apply_reading_dict  # noqa: E402
from translate_lib.llm_translate import translate_line  # noqa: E402

# Sonnet 5 rates (USD per million tokens) as of this writing - check https://claude.com/pricing
# for current rates before trusting this for real budgeting; prices can change.
APPROX_PRICE_PER_MTOK = {"input": 2.0, "output": 10.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=pathlib.Path,
                         help="img_name -> recognized text JSON (defaults recommend Phase 2b's "
                              "corrected_predictions.json - see module docstring)")
    parser.add_argument("--manifest", required=True, type=pathlib.Path,
                         help="Phase 2's manifest.json - just used to get an ordered img_name list")
    parser.add_argument("--reading-dict", required=True, type=pathlib.Path)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--max-lines", type=int, default=10,
                         help="Cost control: cap how many lines to translate. Ignored if --all is set.")
    parser.add_argument("--all", action="store_true",
                         help="Translate every line in --predictions - real cost, not a casual default.")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8,
                         help="Concurrent API requests. The test batch ran at ~3.5s/line "
                              "sequentially - impractical for thousands of lines, so this "
                              "parallelizes across independent lines. 8 is a moderate default; "
                              "lower it if you hit rate-limit (429) errors on your API tier.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Get a key from "
            "https://console.anthropic.com/settings/keys and `export ANTHROPIC_API_KEY=...` "
            "before running this script. This is a separate, billed credential - not available "
            "from a Claude Code session's own authentication."
        )

    try:
        import anthropic
    except ImportError:
        raise SystemExit("The `anthropic` package isn't installed - run `pip install anthropic` first.")

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reading_dict = json.loads(args.reading_dict.read_text(encoding="utf-8"))

    img_names = [e["img_name"] for e in manifest if e["img_name"] in predictions]
    if not args.all:
        img_names = img_names[:args.max_lines]
    print(f"Translating {len(img_names)} lines"
          + ("" if args.all else f" (--max-lines {args.max_lines}, pass --all for the full set)"))

    results = {}
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"--resume: {len(results)} translations already in {args.out}, skipping those")

    todo = [name for name in img_names if name not in results]
    if not todo:
        print("Nothing to do.")
        return

    # One client shared across threads: the anthropic SDK's client is documented as thread-safe
    # (it's a thin wrapper over an httpx client, which supports concurrent requests).
    client = anthropic.Anthropic(api_key=api_key)

    lock = threading.Lock()
    total_input_tok, total_output_tok, n_errors = 0, 0, 0
    start = time.monotonic()

    def worker(img_name: str):
        text = predictions[img_name]
        reading = apply_reading_dict(text, reading_dict)
        translation, usage = translate_line(client, text, reading, model=args.model)
        return img_name, text, reading, translation, usage

    print(f"Using {args.workers} concurrent workers")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, name): name for name in todo}
        completed = 0
        for future in as_completed(futures):
            img_name = futures[future]
            completed += 1
            try:
                img_name, text, reading, translation, usage = future.result()
                with lock:
                    results[img_name] = {"text": text, "reading": reading, "translation": translation}
                    total_input_tok += usage["input_tokens"]
                    total_output_tok += usage["output_tokens"]
            except Exception as e:
                # One bad line (network blip, rate limit exhausting retries, etc.) shouldn't sink
                # a multi-hour batch - log it, skip it, and let --resume pick it up on a rerun
                # (it won't appear in `results`, so --resume will retry it, not silently drop it).
                n_errors += 1
                print(f"ERROR translating {img_name}: {e!r}")

            if completed % args.checkpoint_every == 0 or completed == len(todo):
                elapsed = time.monotonic() - start
                with lock:
                    est_cost = (total_input_tok / 1e6 * APPROX_PRICE_PER_MTOK["input"]
                                + total_output_tok / 1e6 * APPROX_PRICE_PER_MTOK["output"])
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[{completed}/{len(todo)}] {elapsed:.0f}s elapsed, "
                      f"{total_input_tok}+{total_output_tok} tokens, "
                      f"~${est_cost:.4f} (rough estimate, verify against current pricing), "
                      f"{n_errors} errors so far")

    print(f"Wrote {len(results)} translations to {args.out} ({n_errors} lines failed - rerun with "
          f"--resume to retry them)")


if __name__ == "__main__":
    main()
