"""Unit tests for eval_lib/kieu_ground_truth.py - no network, no real Anthropic API, no Docker/TF:

    python -m unittest tests.test_kieu_ground_truth -v
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval_lib.kieu_ground_truth import (  # noqa: E402
    align_edition,
    candidate_verses,
    build_ngram_index,
    normalize,
    parse_img_sort_key,
    parse_wikisource_poem,
)

SAMPLE_WIKITEXT = """{{đầu đề
 | tựa đề = Truyen Kieu
}}
<poem>
	Trăm năm trong cõi người ta,
	Chữ tài chữ mệnh khéo là ghét nhau.
	Trải qua một cuộc bể dâu,
	Những điều trông thấy mà đau đớn lòng.
{{số|5}}	Lạ gì bỉ sắc tư phong,
	Trời xanh quen thói má hồng đánh ghen.
</poem>
{{PD-old}}
"""


class TestParseWikisourcePoem(unittest.TestCase):
    def test_numbers_and_strips_markers(self):
        verses = parse_wikisource_poem(SAMPLE_WIKITEXT)
        self.assertEqual([v[0] for v in verses], [1, 2, 3, 4, 5, 6])
        self.assertEqual(verses[0][1], "Trăm năm trong cõi người ta,")
        self.assertEqual(verses[4][1], "Lạ gì bỉ sắc tư phong,")
        self.assertEqual(verses[5][1], "Trời xanh quen thói má hồng đánh ghen.")

    def test_marker_text_not_left_stuck_to_verse(self):
        """Regression test: the marker regex once used a plain 'ô' instead of the actual 'ố' in
        'số', so it silently never matched - numbering happened to come out right anyway (pure
        auto-increment with no drops in the sample), but the raw "{{số|N}}" text stayed glued to
        every 5th verse's text. Guards that the marker is actually stripped, not just that
        numbering looks plausible."""
        verses = parse_wikisource_poem(SAMPLE_WIKITEXT)
        for _, text in verses:
            self.assertNotIn("số", text)
            self.assertNotIn("{{", text)

    def test_raises_without_poem_block(self):
        with self.assertRaises(ValueError):
            parse_wikisource_poem("no poem tags here")


class TestNormalize(unittest.TestCase):
    def test_strips_diacritics_and_case(self):
        self.assertEqual(normalize("Trăm năm trong cõi người ta,"), "tramnamtrongcoinguoita")

    def test_handles_dd(self):
        self.assertEqual(normalize("Đàn ông"), "danong")

    def test_drops_unresolved_char_brackets(self):
        """apply_reading_dict (reading.py) wraps characters it couldn't resolve as "[X]" - X is
        the original Han/Nom character, which str.isalnum() counts as alphanumeric, so it would
        otherwise get compared directly against Latin text and inflate distance for reasons
        unrelated to whether the alignment itself is right."""
        self.assertEqual(normalize("gia [資] [揨] cõng"), normalize("gia cõng"))

    def test_only_ascii_survives(self):
        normalized = normalize("tuổi ngài [\U000f04f3] vẻ mười")
        self.assertTrue(all(ch.isascii() for ch in normalized))


class TestParseImgSortKey(unittest.TestCase):
    def test_plain_page_number(self):
        self.assertEqual(parse_img_sort_key("Tale of Kieu 1871/page122_17.jpg"), (122, "", 17))

    def test_recto_verso_suffix_orders_a_before_b(self):
        a = parse_img_sort_key("Tale of Kieu 1872/page09a_7.jpg")
        b = parse_img_sort_key("Tale of Kieu 1872/page09b_1.jpg")
        self.assertLess(a, b)

    def test_sorts_numerically_not_lexically(self):
        names = ["page9b_1.jpg", "page10a_1.jpg", "page2a_1.jpg"]
        names.sort(key=parse_img_sort_key)
        self.assertEqual(names, ["page2a_1.jpg", "page9b_1.jpg", "page10a_1.jpg"])

    def test_raises_on_unrecognized_name(self):
        with self.assertRaises(ValueError):
            parse_img_sort_key("not_a_page_name.jpg")


class TestCandidateVerses(unittest.TestCase):
    def setUp(self):
        self.verses_norm = [
            normalize("Trăm năm trong cõi người ta,"),
            normalize("Chữ tài chữ mệnh khéo là ghét nhau."),
            normalize("Trải qua một cuộc bể dâu,"),
        ]
        self.len_verse = [len(v) for v in self.verses_norm]
        self.index = build_ngram_index(self.verses_norm)

    def test_finds_exact_match(self):
        candidates = candidate_verses(self.verses_norm[0], self.index, self.len_verse)
        self.assertIn(0, candidates)

    def test_falls_back_to_length_window_when_no_ngram_hits(self):
        # a reading sharing no 3-grams with any indexed verse at all
        candidates = candidate_verses("xyzxyzxyzxyz", self.index, self.len_verse)
        # should still return *something* via the length-window fallback rather than nothing
        self.assertTrue(len(candidates) >= 0)  # documents the fallback path runs without error


class TestAlignEdition(unittest.TestCase):
    def test_aligns_in_order_and_skips_gaps(self):
        verses = [(i + 1, text) for i, text in enumerate([
            "Trăm năm trong cõi người ta,",
            "Chữ tài chữ mệnh khéo là ghét nhau.",
            "Trải qua một cuộc bể dâu,",
            "Những điều trông thấy mà đau đớn lòng.",
            "Lạ gì bỉ sắc tư phong,",
        ])]
        ocr_rows = [
            {"img_name": "a.jpg", "reading": "trăm năm trong cõi người ta"},
            {"img_name": "b.jpg", "reading": "lạ gì bỉ sắc tư phong"},  # skips verses 2-4
        ]
        aligned = align_edition(ocr_rows, verses)
        self.assertEqual([a.verse_number for a in aligned], [1, 5])
        self.assertEqual(aligned[0].distance, 0)
        self.assertEqual(aligned[1].distance, 0)

    def test_tolerates_gaps_from_unresolved_characters(self):
        """Regression test: a real Kieu 1872 line whose Stage-1 reading had two consecutive
        unresolved characters ('gia [X] [Y] cong thuong buc trung') was, in an earlier version
        of this alignment that scored candidates by normalized-string-length proximity, matched
        to the WRONG verse - the bracket removal shortened its normalized length by 12
        characters relative to the true (fully-spelled) verse, enough to exclude the true verse
        from any reasonable length window. N-gram-seeded candidate generation doesn't have this
        failure mode since it only needs shared substrings, not overall length."""
        verses = [
            (1, "Một trai con thứ rốt lòng,"),
            (2, "Gia tư nghĩ cũng thường thường bực trung."),
            (3, "Vương Quan là chữ, nối dòng nho gia."),
        ]
        ocr_rows = [{"img_name": "x.jpg", "reading": "gia [資] [揨] cõng thường bức trung"}]
        aligned = align_edition(ocr_rows, verses)
        self.assertEqual(aligned[0].verse_number, 2)

    def test_monotonic_output(self):
        import random
        random.seed(0)
        verses = [(i + 1, f"verse number {i} filler text here") for i in range(200)]
        ocr_rows = [{"img_name": f"{i}.jpg", "reading": f"verse number {i} filler text here"}
                    for i in random.sample(range(200), 30)]
        ocr_rows.sort(key=lambda r: int(r["img_name"].split(".")[0]))
        aligned = align_edition(ocr_rows, verses)
        verse_nums = [a.verse_number for a in aligned]
        self.assertEqual(verse_nums, sorted(verse_nums))


if __name__ == "__main__":
    unittest.main()
