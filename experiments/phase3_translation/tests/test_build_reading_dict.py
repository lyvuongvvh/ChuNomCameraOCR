"""Unit tests for scripts/build_reading_dict.py's BTCN and Digitizing Vietnam response parsers,
run on the host (stdlib only, no network):

    python -m unittest tests.test_build_reading_dict -v
"""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from build_reading_dict import (  # noqa: E402
    parse_btcn_response, BTCN_SECTION_DELIM, parse_dvn_response,
)

# Real response fragments captured from a live query for "以者北為" (2025-09) - kept verbatim
# (including the tool's own inconsistent tag closing) so the parser is tested against the actual
# dated markup, not an idealized version of it.
FIXTURE_RESPONSE = BTCN_SECTION_DELIM + """
<tr><td class="hnText" valign=middle align="center" bgcolor=#CCFFCC><B>Tìm</td><td bgcolor=#CCFFCC><font color=#0000FF><U>以</B></U></font></td><td bgcolor=#CCFFCC></td></tr><tr><td><b>186</b></td><td class=hnText align=left valign=middle bgcolor=#FFFFEE><font size="+2"><a href=/common/showBTCN.php?detail=186&uiLang=vn target=Detail>以</a></font></span></td><td></td></tr><tr><span class=hnText>

<tr><td></td><td><i>DĨ</i></td><td>
dĩ nhiên</td></tr>

</span><tr></tr></td></tr>""" + BTCN_SECTION_DELIM + """
<tr><td class="hnText" valign=middle align="center" bgcolor=#CCFFCC><B>Tìm</td><td bgcolor=#CCFFCC><font color=#0000FF><U>者</B></U></font></td><td bgcolor=#CCFFCC></td></tr><tr><td><b>1275</b></td><td class=hnText align=left valign=middle bgcolor=#FFFFEE><font size="+2"><a href=/common/showBTCN.php?detail=1275&uiLang=vn target=Detail>者</a></font></span></td><td></td></tr><tr><span class=hnText>

<tr><td></td><td><i>GIẢ</i></td><td>
giả giả</td></tr>


<tr><td></td><td>TRẢ</td><td>
trả ơn</td></tr>

</span><tr></tr></td></tr>""" + BTCN_SECTION_DELIM + """
<tr><td class="hnText" valign=middle align="center" bgcolor=#CCFFCC><B>Tìm</td><td bgcolor=#CCFFCC><font color=#0000FF><U>北</B></U></font></td><td bgcolor=#CCFFCC></td></tr><tr><td><b>201</b></td><td class=hnText align=left valign=middle bgcolor=#FFFFEE><font size="+2"><a href=/common/showBTCN.php?detail=201&uiLang=vn target=Detail>北</a></font></span></td><td></td></tr><tr><span class=hnText>

<tr><td></td><td><i>BẮC</i></td><td>
phương bắc</td></tr>


<tr><td></td><td>BÁC</td><td>
chú bác</td></tr>


<tr><td></td><td>BẤC</td><td>
gió bấc</td></tr>


<tr><td></td><td>BẬC</td><td>
bậc cửa</td></tr>


<tr><td></td><td>BƯỚC</td><td>
bước tới</td></tr>

</span><tr></tr></td></tr>""" + BTCN_SECTION_DELIM + """
<tr><td class="hnText" valign=middle align="center" bgcolor=#CCFFCC><B>Tìm</td><td bgcolor=#CCFFCC><font color=#0000FF><U>為</B></U></font></td><td bgcolor=#CCFFCC></td></tr><tr><td bgcolor=#ffcccc ></td><td	bgcolor=#ffcccc ></td><td bgcolor=#ffcccc >Không tìm thấy!</td></tr>"""


class TestParseBtcnResponse(unittest.TestCase):
    def test_single_reading_character(self):
        result = parse_btcn_response(FIXTURE_RESPONSE)
        self.assertEqual(result["以"], ["dĩ"])

    def test_two_reading_character(self):
        result = parse_btcn_response(FIXTURE_RESPONSE)
        self.assertEqual(result["者"], ["giả", "trả"])

    def test_five_reading_character(self):
        result = parse_btcn_response(FIXTURE_RESPONSE)
        self.assertEqual(result["北"], ["bắc", "bác", "bấc", "bậc", "bước"])

    def test_not_found_character_is_absent(self):
        result = parse_btcn_response(FIXTURE_RESPONSE)
        self.assertNotIn("為", result)

    def test_readings_are_lowercased(self):
        result = parse_btcn_response(FIXTURE_RESPONSE)
        self.assertTrue(all(r == r.lower() for r in result["北"]))

    def test_no_sections_returns_empty(self):
        self.assertEqual(parse_btcn_response("<html>no results table here</html>"), {})


def _dvn_chunk(inner_text: str) -> str:
    """Builds a `self.__next_f.push([1, "..."])` script tag the way Digitizing Vietnam's Next.js
    app actually emits one - `inner_text` is the flight-payload content the tool ships *before*
    the outer JSON-string escaping React applies to smuggle it through a script tag. This is a
    reconstruction that matches the confirmed real format (see build_reading_dict.py's module
    docstring), not a byte-for-byte capture like the BTCN fixture above - the real payload is a
    multi-KB blob not worth hand-transcribing, and the parser only cares about the "hn"/"qn"
    pairs inside it, which this preserves exactly."""
    return f'<script>self.__next_f.push([1,{json.dumps(inner_text)}])</script>'


class TestParseDvnResponse(unittest.TestCase):
    def test_single_entry(self):
        html = _dvn_chunk('1e:{"entry":{"tdcndg":{"defs":[{"hn":"以","qn":"dĩ"}]}}}')
        self.assertEqual(parse_dvn_response(html, "以"), ["dĩ"])

    def test_multiple_readings_for_one_character(self):
        html = _dvn_chunk(
            '1e:{"entry":{"tdcndg":{"defs":['
            '{"hn":"為","qn":"vài"},{"hn":"為","qn":"vay"},{"hn":"為","qn":"vì"}'
            ']}}}'
        )
        self.assertEqual(parse_dvn_response(html, "為"), ["vài", "vay", "vì"])

    def test_ignores_entries_for_other_characters(self):
        html = _dvn_chunk(
            '1e:{"entry":{"tdcndg":{"defs":[{"hn":"為","qn":"vài"},{"hn":"何","qn":"hà"}]}}}'
        )
        self.assertEqual(parse_dvn_response(html, "為"), ["vài"])

    def test_no_matching_entry_returns_empty_list(self):
        html = _dvn_chunk('1e:{"entry":{"tdcndg":{"defs":[]}}}')
        self.assertEqual(parse_dvn_response(html, "何"), [])

    def test_readings_are_lowercased_and_deduped(self):
        html = _dvn_chunk(
            '1e:{"entry":{"tdcndg":{"defs":[{"hn":"為","qn":"VÌ"},{"hn":"為","qn":"vì"}]}}}'
        )
        self.assertEqual(parse_dvn_response(html, "為"), ["vì"])

    def test_reads_across_multiple_script_chunks(self):
        html = (_dvn_chunk('1d:{"other":"chunk"}')
                + _dvn_chunk('1e:{"entry":{"tdcndg":{"defs":[{"hn":"何","qn":"hà"}]}}}'))
        self.assertEqual(parse_dvn_response(html, "何"), ["hà"])

    def test_no_chunks_returns_empty(self):
        self.assertEqual(parse_dvn_response("<html>no flight payload here</html>", "何"), [])


if __name__ == "__main__":
    unittest.main()
