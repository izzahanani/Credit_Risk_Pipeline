"""Offline parser test using HTML shaped like the Wikipedia table."""
import pytest

from src.scraper import ScrapeError, parse_interest_rate

SAMPLE_HTML = """
<table class="wikitable sortable">
<tr><th>Country or currency union</th>
    <th>Central bank interest rate (%)</th><th>Change</th>
    <th>Effective date of last change</th>
    <th>Average inflation rate 2017–2021 (%)<sup>[1]</sup></th>
    <th>Central bank interest rate minus average inflation rate</th></tr>
<tr><td><span class="flagicon"><img alt=""></span>&nbsp;<a>Japan</a></td>
    <td>−0.10</td><td>0.10</td><td>1 Feb 2016</td><td>0.3</td><td>-0.4</td></tr>
<tr><td><span class="flagicon"><img alt=""></span>&nbsp;<a>Malaysia</a></td>
    <td>2.75<sup>[88]</sup></td><td>0.25</td><td>9 July 2025<sup>[88]</sup></td>
    <td>1.5</td><td>1.25</td></tr>
</table>
"""


def test_parses_target_country():
    assert parse_interest_rate(SAMPLE_HTML, "Malaysia") == 2.75


def test_handles_unicode_minus():
    assert parse_interest_rate(SAMPLE_HTML, "Japan") == -0.10


def test_missing_country_raises():
    with pytest.raises(ScrapeError):
        parse_interest_rate(SAMPLE_HTML, "Atlantis")
