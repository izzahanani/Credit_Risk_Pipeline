"""Task 1 entry point: scrape the macro rate, build customers, export CSV.

Usage (from the project root):
    python run_task1.py
"""
import logging

from src import config
from src.data_generator import generate_customer_profiles, validate_profiles
from src.scraper import get_macro_interest_rate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("task1")


def main() -> None:
    rate, source = get_macro_interest_rate()
    logger.info("Macro interest rate: %.2f%% (%s)", rate, source)

    profiles = generate_customer_profiles(macro_interest_rate=rate)
    validate_profiles(profiles)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(config.CUSTOMER_PROFILES_CSV, index=False)
    logger.info("Wrote %d rows to %s", len(profiles), config.CUSTOMER_PROFILES_CSV)
    logger.info("Summary:\n%s", profiles.describe().round(2).to_string())


if __name__ == "__main__":
    main()
