"""Unit tests for eval_lib/dvsktt_ground_truth.py - no network, no real Anthropic API, no
Docker/TF:

    python -m unittest tests.test_dvsktt_ground_truth -v
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval_lib.dvsktt_ground_truth import (  # noqa: E402
    parse_img_key,
    parse_translation_pages,
)

SAMPLE_TEXT = """2


Đại Việt Sử Ký Toàn Thư - Ngoại Kỷ - Quyển I

Đại Việt Sử Ký Ngoại Kỷ Toàn Thư

[la] Triều Liệt Đại Phu, Quốc Tử Giám Tư Nghiệp,
Kiêm Sử Quan Tu Soạn, Thần Ngô Sĩ Liên Biên
Xét: Thời Hoàng Đế dựng muôn nước.

Đại Việt Sử Ký Toàn Thư - Ngoại Kỷ - Quyển I

[lb] Tên húy là Lộc Tục, con cháu họ Thần Nông.
Nhâm Tuất, năm thứ 1.

Đại Việt Sử Ký Toàn Thư - Ngoại Kỷ - Quyển I

đồ đảng quan tước theo thứ bậc[2a] khác nhau, khi ấy đồ đảng của Thân Lợi chỉ hơn nghìn người.
"""


class TestParseImgKey(unittest.TestCase):
    def test_parses_ban_toan(self):
        self.assertEqual(
            parse_img_key("DVSKTT-3 Ban ky toan thu/DVSKTT_ban_toan_V_30a_9.jpg"),
            ("ban_toan", "V", 30, "a"),
        )

    def test_parses_ngoai(self):
        self.assertEqual(
            parse_img_key("DVSKTT-2 Ngoai ky toan thu/DVSKTT_ngoai_III_10b_0.jpg"),
            ("ngoai", "III", 10, "b"),
        )

    def test_parses_ban_thuc_and_ban_tuc(self):
        self.assertEqual(
            parse_img_key("DVSKTT_ban_thuc_XIII_10a_0.jpg"),
            ("ban_thuc", "XIII", 10, "a"),
        )
        self.assertEqual(
            parse_img_key("DVSKTT_ban_tuc_XIX_14a_2.jpg"),
            ("ban_tuc", "XIX", 14, "a"),
        )

    def test_raises_on_unrecognized_name(self):
        with self.assertRaises(ValueError):
            parse_img_key("not_a_dvsktt_name.jpg")


class TestParseTranslationPages(unittest.TestCase):
    def setUp(self):
        self.pages = parse_translation_pages(SAMPLE_TEXT)

    def test_extracts_leaf_content_by_structural_key(self):
        self.assertIn(("Ngoại Kỷ", "I", 1, "a"), self.pages)
        self.assertIn("Xét: Thời Hoàng Đế", self.pages[("Ngoại Kỷ", "I", 1, "a")])

    def test_ocr_garbled_marker_normalized_to_leaf_1(self):
        """Regression test: the raw OCR consistently misreads "1" as lowercase "l" in these
        bracket markers ("[la]" for "[1a]", verified against the real downloaded text) - this
        must resolve to leaf number 1, not be dropped or misparsed as leaf "l"."""
        self.assertIn(("Ngoại Kỷ", "I", 1, "b"), self.pages)

    def test_marker_mid_line_still_splits_correctly(self):
        """A real line has a leaf marker appearing MID-SENTENCE, not at the start (the printed
        leaf boundary falls inside a word/clause, not conveniently between them) - text before
        the marker belongs to the previous leaf, text after belongs to the new one."""
        self.assertIn("khác nhau", self.pages[("Ngoại Kỷ", "I", 2, "a")])
        self.assertNotIn("khác nhau", self.pages[("Ngoại Kỷ", "I", 1, "b")])

    def test_repeated_running_header_does_not_break_tracking(self):
        """The book's own running header ("Đại Việt Sử Ký Toàn Thư - Ngoại Kỷ - Quyển I")
        repeats on every physical page - re-seeing the SAME section/quyen must be a no-op, not
        reset or duplicate the leaf currently being accumulated."""
        # if the repeated header broke tracking, leaf 1b's content would be lost or split
        self.assertIn("Nhâm Tuất", self.pages[("Ngoại Kỷ", "I", 1, "b")])

    def test_bare_page_number_lines_dropped_as_noise(self):
        """The standalone "2" at the very top of the file is the 1993 book's own printed page
        number (unrelated to the original woodblock leaf numbers) and must not leak into any
        leaf's extracted text."""
        for text in self.pages.values():
            self.assertNotIn("2", text.split())  # "2" as a standalone token, not e.g. inside "1968"

    def test_no_leaf_key_before_any_header_seen(self):
        text_with_leading_marker = "[1a] some content before any header\n" + SAMPLE_TEXT
        pages = parse_translation_pages(text_with_leading_marker)
        self.assertNotIn("some content before any header", " ".join(pages.values()))


if __name__ == "__main__":
    unittest.main()
