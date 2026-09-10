"""Ground-truth alignment for Truyen Kieu: matches each NomNaOCR patch (one Nom line, already
recognized and read via Stage 1/2) to its verse number and modern Quoc Ngu text in the real,
complete 3254-verse poem - so Phase 3's translations can finally be scored against a genuine
parallel corpus instead of judged qualitatively (see results.md, "no ground truth" limitation).

Source of the reference text: Vietnamese Wikisource's "Truyen Kieu" page
(https://vi.wikisource.org/wiki/Truyen_Kieu), same underlying work as NomNaOCR's three digitized
editions (1866/1871/1872). Public domain (author died 1820; page tagged {{PD-old}}).

Why alignment is needed at all (not just "manifest order == verse order"): each edition's
manifest only covers a few hundred of the 3254 verses (spot-digitized pages, not the whole book),
so OCR line N is NOT verse N - it could be any verse, as long as the *order* is preserved within
an edition (pages are scanned/patch-indexed in reading order, never shuffled). That turns this
into a subsequence alignment problem: find the best strictly-increasing mapping of OCR lines onto
verses, skipping over undigitized verses, minimizing total text distance.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def _load_edit_distance():
    """Phase 2's eval_lib.metrics has the exact Levenshtein implementation we want to reuse, but
    it lives in a sibling experiment folder that also happens to be named "eval_lib" - a plain
    sys.path-based package import would collide with (and shadow) THIS package's own name in
    sys.modules. Loading the file directly by path sidesteps that collision entirely."""
    metrics_path = Path(__file__).resolve().parent.parent.parent / "phase2_nomnaocr_baseline" / "eval_lib" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_phase2_metrics", metrics_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass field introspection needs this registered first
    spec.loader.exec_module(module)
    return module.edit_distance


edit_distance = _load_edit_distance()

VERSE_MARKER_RE = re.compile(r"\{\{số\|(\d+)\}\}")
IMG_NAME_RE = re.compile(r"page0*(\d+)([ab]?)_0*(\d+)")


def parse_wikisource_poem(raw_wikitext: str) -> list[tuple[int, str]]:
    """Parses the <poem>...</poem> block of Vietnamese Wikisource's raw wikitext for Truyen
    Kieu into an ordered list of (verse_number, text), 1-indexed. Every 5th line carries an
    explicit {{so|N}} (accented "số") marker - used to anchor numbering rather than assuming
    no lines were ever dropped/merged upstream; un-marked lines are numbered by simple
    continuation from the last marker.
    """
    match = re.search(r"<poem>(.*?)</poem>", raw_wikitext, re.DOTALL)
    if not match:
        raise ValueError("no <poem>...</poem> block found in wikitext")
    body = match.group(1)

    verses = []
    next_number = 1
    for line in body.split("\n"):
        line = line.strip("\t")
        if not line.strip():
            continue
        marker = VERSE_MARKER_RE.search(line)
        if marker:
            next_number = int(marker.group(1))
            line = VERSE_MARKER_RE.sub("", line)
        text = line.strip().strip("\t ")
        if not text:
            continue
        verses.append((next_number, text))
        next_number += 1
    return verses


UNRESOLVED_CHAR_RE = re.compile(r"\[.\]")


def normalize(text: str) -> str:
    """Lowercases, strips Vietnamese diacritics/tone marks and all punctuation/whitespace, for
    coarse comparison between a Stage-1 phonetic reading (built from a lossy, gap-ridden
    dictionary - see README.md's 66.1% coverage caveat) and Wikisource's modern spelling. Tone
    marks in particular are exactly the kind of detail Stage 1's dictionary is least reliable
    about, so comparing with them included would penalize correct alignments for wrong reasons.

    Also drops apply_reading_dict's own "[X]" bracket markers for characters it couldn't
    resolve (reading.py) - X is the original Han/Nom character, which Python's str.isalnum()
    happily counts as alphanumeric (Unicode "Letter, Other"), so left in place it would get
    compared directly against Latin Quoc Ngu text and inflate distance for reasons that have
    nothing to do with whether the alignment itself is correct.
    """
    text = UNRESOLVED_CHAR_RE.sub("", text)
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in stripped.lower() if ch.isascii() and ch.isalnum())


def parse_img_sort_key(img_name: str) -> tuple[int, str, int]:
    """Sort key reproducing each edition's true page/reading order from its filename.
    Handles both naming conventions seen across the three Kieu editions: "pageNNN_M" (1871,
    no recto/verso suffix) and "pageNNa_M"/"pageNNb_M" (1866/1872, 'a'=recto before 'b'=verso).
    """
    m = IMG_NAME_RE.search(img_name)
    if not m:
        raise ValueError(f"unrecognized image name format: {img_name}")
    page_num, side, patch_idx = m.groups()
    return (int(page_num), side, int(patch_idx))


@dataclass
class AlignedLine:
    img_name: str
    verse_number: int
    reference_text: str
    ocr_reading: str
    distance: int
    normalized_len: int

    @property
    def similarity(self) -> float:
        """1.0 = identical after normalization, 0.0 = completely different. Guards against
        divide-by-zero on an empty reference line (shouldn't happen, but cheap to guard)."""
        if self.normalized_len == 0:
            return 0.0
        return 1.0 - self.distance / self.normalized_len


NGRAM_SIZE = 3
NGRAM_TOP_K = 40
LENGTH_SLACK = 2
SENTINEL = 1_000_000  # cost for any (row, verse) pair with no computed real distance


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[k:k + n] for k in range(len(text) - n + 1)}


def build_ngram_index(norm_verses: list[str], n: int = NGRAM_SIZE) -> dict[str, list[int]]:
    index = defaultdict(list)
    for vidx, text in enumerate(norm_verses):
        for gram in _ngrams(text, n):
            index[gram].append(vidx)
    return index


def candidate_verses(norm_reading: str, index: dict, len_verse: list[int], n: int = NGRAM_SIZE,
                      top_k: int = NGRAM_TOP_K) -> set[int]:
    """Which verse indices are worth computing a real edit_distance against for this one OCR
    reading. Primarily n-gram seeded (shared 3-character substrings, tallied by frequency, top-K
    kept) - this is what makes candidate generation robust to Stage-1's dictionary gaps: a
    reading missing a third of its characters to "[X]" brackets (see normalize()) still shares
    plenty of 3-grams with its true verse from whatever it DID resolve, regardless of how much
    shorter that leaves the overall string. A pure length-based filter fails exactly on these
    cases - verified on a real line where dropped brackets shortened the normalized reading by
    12 characters, enough to exclude the true verse from any reasonable length window.

    A length window is used only as a LAST-RESORT fallback, when n-grams find nothing at all
    (e.g. a reading that's almost entirely unresolved brackets) - and even then a tight one:
    Kieu's normalized verse lengths turn out to be densely clustered (measured: hundreds of the
    3254 verses land on any single length value around 20-28 chars), so a loose length window
    that looked "clearly narrower than the whole poem" was actually barely filtering anything -
    that's what made an earlier version of this function pull in most of the poem per row and
    blow up alignment time to minutes per edition.
    """
    counts = Counter()
    for gram in _ngrams(norm_reading, n):
        for vidx in index.get(gram, ()):
            counts[vidx] += 1
    candidates = {vidx for vidx, _ in counts.most_common(top_k)}
    if not candidates:
        target_len = len(norm_reading)
        candidates.update(vidx for vidx, vlen in enumerate(len_verse) if abs(vlen - target_len) <= LENGTH_SLACK)
    return candidates


def align_edition(ocr_rows: list[dict], verses: list[tuple[int, str]]) -> list[AlignedLine]:
    """ocr_rows: manifest rows (each with 'img_name' and a 'reading' field already computed by
    Stage 1/2) for ONE edition, in the order they should be matched (caller must have already
    sorted by parse_img_sort_key). verses: full ordered (verse_number, text) list for the whole
    poem.

    Finds the lowest-total-edit-distance mapping of ocr_rows[i] onto some strictly increasing
    subsequence of verses (skipping verses the edition didn't digitize) - a standard weighted
    subsequence-alignment DP, O(n * M) cells where n = len(ocr_rows), M = len(verses) (a few
    hundred by ~3254).

    Computing real edit_distance for every one of the n * M cells is correct but far too slow in
    plain Python (measured: ~210s for a single 702-line edition) - and a couple of cheaper
    per-cell proxies tried first (multiset "bag" distance, then a plain normalized-length window)
    both turned out to pick the WRONG verse on real data, not just approximate it (see
    candidate_verses' docstring and this project's results.md for the concrete failing case).
    The fix keeps real edit_distance as the only cost function - so the DP's decisions stay
    exact - but only evaluates it for each row's n-gram-seeded candidate verses (typically a few
    dozen, not 3254); every other cell gets a fixed SENTINEL, which is always worse than any
    real match so the DP correctly prefers "skip" over an unevaluated cell.
    """
    n, m = len(ocr_rows), len(verses)
    norm_ocr = [normalize(r["reading"]) for r in ocr_rows]
    norm_verse = [normalize(v[1]) for v in verses]
    len_verse = [len(s) for s in norm_verse]
    ngram_index = build_ngram_index(norm_verse)
    row_candidates = [
        {vidx: edit_distance(oi, norm_verse[vidx])
         for vidx in candidate_verses(oi, ngram_index, len_verse)}
        for oi in norm_ocr
    ]

    # dp[i][j] = min total distance aligning ocr_rows[:i] using verses[:j]
    # choice[i][j] records whether the best path matched (i-1 <-> j-1) or skipped verse j-1
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    choice = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0
    for i in range(1, n + 1):
        row, prev_row = dp[i], dp[i - 1]
        crow = choice[i]
        costs = row_candidates[i - 1]
        for j in range(i, m + 1):
            best, best_choice = row[j - 1], "skip"  # skip verse j-1, leave ocr_rows[i-1] unmatched-so-far
            match_cost = prev_row[j - 1] + costs.get(j - 1, SENTINEL)
            if match_cost < best:
                best, best_choice = match_cost, "match"
            row[j] = best
            crow[j] = best_choice

    # backtrack from (n, m) to recover which verse each ocr_row matched
    matches = [None] * n
    i, j = n, m
    while i > 0:
        if choice[i][j] == "match":
            matches[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1

    aligned = []
    for idx, row in enumerate(ocr_rows):
        vidx = matches[idx]
        verse_number, verse_text = verses[vidx]
        dist = edit_distance(norm_ocr[idx], norm_verse[vidx])
        aligned.append(AlignedLine(
            img_name=row["img_name"],
            verse_number=verse_number,
            reference_text=verse_text,
            ocr_reading=row["reading"],
            distance=dist,
            normalized_len=len(norm_verse[vidx]),
        ))
    return aligned
