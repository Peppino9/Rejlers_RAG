"""
src/scrape_playwright_loop.py
------------------------------
Entry point wrapper for the StoryMaps Playwright scraper.

The actual scraping logic lives in:
  scrapers/storymaps_playwright/scrape_storymaps_loop.py
so it's easy to navigate and keep the scraper-related code isolated.
"""

from __future__ import annotations

import os

from scrapers.storymaps_playwright.scrape_storymaps_loop import scrape_items


def main() -> None:
    # Requirement: headless=False initially so you can visually confirm it works.
    # Override with: HEADLESS=true python -m src.scrape_playwright_loop
    headless_env = os.getenv("HEADLESS", "false").strip().lower()
    headless = headless_env in {"1", "true", "yes", "y"}
    scrape_items(start=1, end_inclusive=8, headless=headless)


if __name__ == "__main__":
    main()

