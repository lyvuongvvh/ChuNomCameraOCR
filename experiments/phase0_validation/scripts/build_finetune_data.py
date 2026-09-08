"""Builds kraken PageXML training data for fine-tuning CHAT's recognition model on NomNaOCR,
using only patches from NomNaOCR's own *training* split (Patches/Train.txt) - the held-out
Patches/Validate.txt patches used by the Phase 0 benchmark (see build_manifest.py) must never
appear here, so the before/after fine-tuning comparison stays honest.

Why PageXML instead of kraken's simpler `path` format (image + .gt.txt pairs): kraken's `path`
mode assumes images are already oriented as horizontal line strips. CHAT's recognizer instead
expects input that has been through kraken's own baseline-driven dewarping
(`kraken.lib.segmentation.extract_polygons`), which is what correctly normalizes vertical
Chinese/Nom columns at inference time. Feeding raw vertical column crops through `path` mode
would train on the wrong orientation. NomNaOCR's `gts/<page>.txt` files give each patch's
bounding quadrilateral within its source page; this script derives a synthetic baseline from
it (the vertical line from the top edge's midpoint to the bottom edge's midpoint, matching
vertical-rl top-to-bottom reading) and writes standard PageXML, so `ketos train -f page` runs
the exact same dewarping code path kraken uses everywhere else - no guessing about rotation
direction.

Usage (run on the host, no Docker needed):
    python scripts/build_finetune_data.py --dataset-root /path/to/NomNaOCR \
        --out-dir data/finetune_data --max-patches 2000
"""
import argparse
import json
import pathlib
import re
from xml.sax.saxutils import escape

from PIL import Image

IMAGE_EXT = ".jpg"
MIN_LONG_SIDE = 1600  # matches run_chat.py's MIN_LONG_SIDE - see that file's comment for why.

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


def load_labels(labels_path: pathlib.Path) -> dict[str, str]:
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            img_name, text = line.split("\t")
            labels[img_name] = text
    return labels


def upscale_if_small(img: Image.Image, target_long_side: int = MIN_LONG_SIDE) -> tuple[Image.Image, float]:
    """Same policy as run_chat.py's _upscale_if_small - never downscales. Returns (image, scale)
    so caller can scale the quad coordinates to match."""
    long_side = max(img.size)
    if long_side >= target_long_side:
        return img, 1.0
    scale = target_long_side / long_side
    new_size = (round(img.width * scale), round(img.height * scale))
    return img.resize(new_size, Image.LANCZOS), scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--max-patches", type=int, default=2000,
                         help="Total training-line budget across all pages (a 'quick trial' subset, not the full ~34K-line training split).")
    args = parser.parse_args()

    train_txt = args.dataset_root / "Patches" / "Train.txt"
    assert train_txt.exists(), f"missing {train_txt}"
    train_labels = load_labels(train_txt)
    train_keys = set(train_labels)

    pages_dir = args.dataset_root / "Pages"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_pages = 0
    works = sorted(p for p in pages_dir.iterdir() if p.is_dir())
    for work_dir in works:
        if total_lines >= args.max_patches:
            break
        work = work_dir.name
        gts_dir = work_dir / "gts"
        imgs_dir = work_dir / "imgs"
        if not gts_dir.is_dir() or not imgs_dir.is_dir():
            continue
        for gts_file in sorted(gts_dir.glob("*.txt")):
            if total_lines >= args.max_patches:
                break
            page_stem = gts_file.stem
            page_img_path = imgs_dir / f"{page_stem}{IMAGE_EXT}"
            if not page_img_path.exists():
                continue

            raw_lines = [l for l in gts_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            eligible = []
            for i, raw_line in enumerate(raw_lines):
                patch_key = f"{work}/{page_stem}_{i}{IMAGE_EXT}"
                if patch_key not in train_keys:
                    continue  # not in NomNaOCR's training split - skip (may be held-out/validate)
                parts = raw_line.split(",", 8)
                coords = [float(c) for c in parts[:8]]
                text = train_labels[patch_key]  # use the split file's own text, not gts's raw field
                eligible.append((coords, text))
            if not eligible:
                continue

            img = Image.open(page_img_path)
            img, scale = upscale_if_small(img)
            key = f"{sanitize(work)}_{page_stem}"
            out_img_name = f"{key}.png"
            img.save(args.out_dir / out_img_name)

            line_xmls = []
            for i, (coords, text) in enumerate(eligible):
                coords = [c * scale for c in coords]
                x1, y1, x2, y2, x3, y3, x4, y4 = coords
                top_mid = ((x1 + x2) / 2, (y1 + y2) / 2)
                bottom_mid = ((x3 + x4) / 2, (y3 + y4) / 2)
                boundary_str = " ".join(f"{round(x)},{round(y)}" for x, y in
                                         [(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
                baseline_str = f"{round(top_mid[0])},{round(top_mid[1])} {round(bottom_mid[0])},{round(bottom_mid[1])}"
                line_xmls.append(LINE_TEMPLATE.format(
                    line_id=f"l{i}", boundary=boundary_str, baseline=baseline_str,
                    text=escape(text),
                ))
                total_lines += 1

            xml_content = PAGE_XML_TEMPLATE.format(
                image_filename=out_img_name, width=img.width, height=img.height,
                lines="\n".join(line_xmls),
            )
            (args.out_dir / f"{key}.xml").write_text(xml_content, encoding="utf-8")
            total_pages += 1

    print(f"Wrote {total_pages} PageXML files ({total_lines} training lines) to {args.out_dir}")
    manifest_note = {"total_pages": total_pages, "total_lines": total_lines, "max_patches": args.max_patches}
    (args.out_dir / "_build_info.json").write_text(json.dumps(manifest_note, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
