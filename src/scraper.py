"""Scrape the current central bank policy rate with BeautifulSoup.

Source: Wikipedia's "List of sovereign states by central bank interest rates".
The page holds one sortable wikitable whose first column is the country and
whose rate column header contains "interest rate (%)".
"""
from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

from src import config

logger = logging.getLogger(__name__)

HEADERS = {
    # Wikipedia rejects requests without a descriptive User-Agent.
    "User-Agent": "credit-risk-prototype/1.0 (educational technical assessment)"
}


class ScrapeError(RuntimeError):
    """Raised when the rate cannot be located in the page."""


def fetch_html(url: str = config.MACRO_SOURCE_URL) -> str:
    """Download the raw HTML of the source page."""
    response = requests.get(
        url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.text


def _clean_cell(text: str) -> str:
    """Strip footnote markers such as [12] and surrounding whitespace."""
    return re.sub(r"\[[^\]]*\]", "", text).replace("\xa0", " ").strip()


def _to_float(text: str) -> float:
    """Convert a cell like '2.75', '2.75%' or '−0.10' to a float."""
    normalised = _clean_cell(text).replace("−", "-").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", normalised)
    if not match:
        raise ScrapeError(f"No numeric value found in cell: {text!r}")
    return float(match.group())


def parse_interest_rate(html: str, country: str = config.MACRO_COUNTRY) -> float:
    """Extract the policy rate (%) for `country` from the page HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for table in soup.find_all("table", class_="wikitable"):
        header_row = table.find("tr")
        if header_row is None:
            continue
        headers = [
            _clean_cell(th.get_text(" ")).lower()
            for th in header_row.find_all(["th", "td"])
        ]
        # Locate the rate column by name, not by position, so the parser
        # survives columns being added or reordered.
        rate_idx = next(
            (
                i
                for i, h in enumerate(headers)
                if "interest rate" in h and "minus" not in h
            ),
            None,
        )
        if rate_idx is None:
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) <= rate_idx:
                continue
            name = _clean_cell(cells[0].get_text(" "))
            if name.lower() == country.lower():
                return _to_float(cells[rate_idx].get_text(" "))

    raise ScrapeError(f"Could not find an interest rate row for {country!r}")


def get_macro_interest_rate(
    country: str = config.MACRO_COUNTRY,
    fallback: float | None = config.FALLBACK_INTEREST_RATE,
) -> tuple[float, str]:
    """Return (rate, source) where source is 'scraped' or 'fallback'.

    A scoring pipeline should not die because a public web page is briefly
    unreachable, so a documented fallback is used and loudly logged.
    Pass fallback=None to make failures fatal instead.
    """
    try:
        rate = parse_interest_rate(fetch_html(), country)
        logger.info("Scraped %s policy rate: %.2f%%", country, rate)
        return rate, "scraped"
    except (requests.RequestException, ScrapeError) as exc:
        if fallback is None:
            raise
        logger.warning(
            "Scrape failed (%s). Using fallback rate %.2f%%.", exc, fallback
        )
        return fallback, "fallback"
