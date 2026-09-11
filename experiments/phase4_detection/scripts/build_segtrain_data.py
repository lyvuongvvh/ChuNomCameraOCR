"""Phase 4 detection, Rung 1: converts NomNaOCR's real gts/*.txt column quads into PageXML for
`ketos segtrain`, reusing the synthetic-baseline geometry already proven in
experiments/phase0_validation/scripts/build_finetune_data.py (top-edge midpoint -> bottom-edge
midpoint, matching vertical-rl top-to-bottom reading) - just applied at whole-page granularity
instead of per-patch, since segtrain needs real page layouts, not isolated line crops.

Excludes any page present in --exclude-manifest (the pages score_detection.py evaluates against)
from the training set entirely - mirrors Phase 0's own documented Train/Validate contamination
fix (build_finetune_data.py's docstring): the pages used to judge whether Rung 1 clears the bar
must never have been seen during Rung 1's own training.

Usage:
    python scripts/build_segtrain_data.py \
        --manifest data/manifest.json \
        --exclude-manifest data/manifest_sample15.json \
        --out-dir data/segtrain_pagexml \
        --limit-pages 50
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from xml.sax.saxutils import escape

from PIL import Image

PAGE_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="{image_filename}" imageWidth="{width}" imageHeight="{height}" readingDirection="top-to-bottom">
    <TextRegion id="r1">
      <Coords points="0,0 {width},0 {width},{height} 0,{height}"/>
{lines}
    </TextRegion>
  </Page>
</PcGts>
"""

LINE_TEMPLATE = """      <TextLine id="{line_id}">
        <Coords points="{boundary}"/>
        <Baseline points="{baseline}"/>
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>
      </TextLine>"""


def sanitize(work: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", work)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--exclude-manifest", required=True, type=pathlib.Path,
                     help="Pages in this manifest are held out - never written as training data.")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--limit-pages", type=int, default=None,
                     help="Cap total pages written - for a fast sanity trial, not the full run.")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    exclude = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
    exclude_keys = {(e["work"], e["page_name"]) for e in exclude}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_lines = 0
    skipped_excluded = 0
    skipped_no_lines = 0
    for entry in manifest:
        if args.limit_pages is not None and total_pages >= args.limit_pages:
            break
        key = (entry["work"], entry["page_name"])
        if key in exclude_keys:
            skipped_excluded += 1
            continue
        quads = entry["gt_quads"]
        if not quads:
            skipped_no_lines += 1
            continue

        img_path = pathlib.Path(entry["image_path"])
        with Image.open(img_path) as img:
            width, height = img.size

        out_stem = f"{sanitize(entry['work'])}_{entry['page_name']}"
        line_xmls = []
        for i, q in enumerate(quads):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = q["points"]
            top_mid = ((x1 + x2) / 2, (y1 + y2) / 2)
            bottom_mid = ((x3 + x4) / 2, (y3 + y4) / 2)
            boundary_str = " ".join(f"{round(x)},{round(y)}" for x, y in
                                     [(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            baseline_str = f"{round(top_mid[0])},{round(top_mid[1])} {round(bottom_mid[0])},{round(bottom_mid[1])}"
            line_xmls.append(LINE_TEMPLATE.format(
                line_id=f"l{i}", boundary=boundary_str, baseline=baseline_str,
                text=escape(q.get("transcript", "")),
            ))
            total_lines += 1

        xml_content = PAGE_XML_TEMPLATE.format(
            image_filename=str(img_path.resolve()), width=width, height=height,
            lines="\n".join(line_xmls),
        )
        (args.out_dir / f"{out_stem}.xml").write_text(xml_content, encoding="utf-8")
        total_pages += 1

    print(f"Wrote {total_pages} PageXML files ({total_lines} lines) to {args.out_dir}")
    print(f"Excluded {skipped_excluded} held-out eval pages, skipped {skipped_no_lines} pages with no GT lines")


if __name__ == "__main__":
    main()
