"""
scrape_storymaps_loop.py
------------------------
Playwright sync scraper for ArcGIS StoryMaps.

Scrapes:
  ?item=1..8
from the provided base URL pattern, waits for JS rendering, then extracts visible text.

Outputs:
  ./data/sweco_bollebygd_item_{i}.md

All configuration is kept here so `src/scrape_playwright_loop.py` stays small.
"""

from __future__ import annotations

import re
import os
import time
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = (
    "https://samrad.sweco.se/portal/apps/storymaps/collections/"
    "53cd4fa4b8a7484fbf9b6e58101c78e3?item={i}"
)

MIN_TEXT_LEN = 1200  # heuristic: story text should be much longer than nav-only text
OCR_FALLBACK_ENABLED = os.getenv("SCRAPE_OCR_FALLBACK", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

NAV_NOISE_SUBSTRINGS = [
    "Gå till innehåll",
    "Så använder du samrådsportalen",
    "Om projektet",
    "Förutsättningar",
    "Utredning av alternativ",
    "Utvärdering och slutsatser",
    "Fortsatt arbete",
    "Lämna synpunkter",
    "Interaktiv karta",
    "Järnvägsplan Bollebygd",
]


def _resolve_chromium_executable_path() -> str | None:
    """
    Resolve an existing Chromium executable path from the Playwright browser cache.

    This is needed because PLAYWRIGHT_BROWSERS_PATH can be shared across environments
    with different architectures (arm64 vs x64). If the expected architecture binary
    isn't present, Playwright throws:
      "Executable doesn't exist at <.../chrome-mac-arm64/...>"

    We probe the shared cache and return the first executable path that exists.
    """
    cache_root = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
    if not cache_root:
        return None

    cache_root_path = Path(cache_root)
    if not cache_root_path.exists():
        return None

    # Prefer arm64 if present, otherwise x64.
    candidates: list[Path] = []
    candidates += list(
        cache_root_path.glob(
            "chromium-*/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
        )
    )
    candidates += list(
        cache_root_path.glob(
            "chromium-*/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
        )
    )

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _extract_candidate_text(page) -> list[str]:
    """
    Extract candidate text blocks from likely StoryMaps containers.
    Returns a list of texts (may include nav-only text).
    """
    candidates: list[str] = []
    try:
        for sel in [
            "main",
            "[role='main']",
            "article",
            "[class*='storymap' i]",
            "[class*='story' i]",
            "body",
        ]:
            locator = page.locator(sel)
            if locator.count() == 0:
                continue
            # Use first match for structural containers to reduce duplicate nav text.
            txt = locator.first.inner_text(timeout=5_000)
            if txt and txt.strip():
                candidates.append(txt)
    except Exception:
        # Last-resort fallback
        try:
            candidates.append(page.inner_text("body"))
        except Exception:
            pass

    return candidates


def _extract_text_from_iframes(page) -> list[str]:
    """
    ArcGIS StoryMaps often renders content inside iframes.
    Extract from all frames' body to capture the story content.
    """
    texts: list[str] = []
    try:
        for frame in page.frames:
            # Skip the main frame (we already extracted from the page) and about:blank frames.
            if frame.parent is None:
                continue
            url = getattr(frame, "url", "") or ""
            if not url or url.startswith("about:"):
                continue
            try:
                txt = frame.locator("body").inner_text(timeout=5_000)
                if txt and txt.strip():
                    texts.append(txt)
            except Exception:
                continue
    except Exception:
        return texts
    return texts


def _extract_text_from_jsonld(page) -> list[str]:
    """
    Extract story text from JSON-LD script tags.

    StoryMaps pages often embed article text/metadata inside:
      <script type="application/ld+json"> ... </script>
    """
    texts: list[str] = []
    try:
        loc = page.locator('script[type="application/ld+json"]')
        count = loc.count()
        for i in range(count):
            raw = loc.nth(i).inner_text(timeout=3_000).strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            # Recursively collect string leaves.
            stack = [data]
            found: list[str] = []
            while stack:
                cur = stack.pop()
                if isinstance(cur, str):
                    s = cur.strip()
                    if len(s) >= 80:
                        found.append(s)
                    continue
                if isinstance(cur, list):
                    for it in cur:
                        stack.append(it)
                    continue
                if isinstance(cur, dict):
                    for _, v in cur.items():
                        stack.append(v)
                    continue
            texts.extend(found)
    except Exception:
        return texts

    # Remove obvious non-story boilerplate
    cleaned = []
    for t in texts:
        if "unsupported-browser" in t.lower():
            continue
        cleaned.append(t)
    return cleaned


def _extract_visible_text(page) -> str:
    """
    Extract visible story text from:
      1) best matching page containers
      2) iframes (common for StoryMaps)

    Uses a heuristic to avoid returning nav-only text.
    """
    candidates = _extract_candidate_text(page)
    iframe_texts = _extract_text_from_iframes(page)
    candidates.extend(iframe_texts)
    candidates.extend(_extract_text_from_jsonld(page))

    # Prefer the largest / richest text candidates first.
    candidates = [c.strip() for c in candidates if c and c.strip()]
    candidates.sort(key=len, reverse=True)
    if candidates:
        if len(candidates[0]) >= MIN_TEXT_LEN:
            return candidates[0]

    # If everything looks short, concatenate all candidates.
    return "\n\n".join(candidates).strip()


def _extract_visible_text_with_retries(page, timeout_sec: float = 50.0) -> str:
    """
    Repeatedly trigger lazy loading and extract text until we get a long enough
    story body (or the timeout expires).
    """
    deadline = time.time() + timeout_sec
    best = ""
    while time.time() < deadline:
        try:
            _trigger_lazy_loading(page)
        except Exception:
            pass

        try:
            txt = _extract_visible_text(page)
            txt = _clean_text(txt)
            if txt:
                if len(txt) > len(best):
                    best = txt
                if len(txt) >= MIN_TEXT_LEN:
                    return txt
        except Exception:
            pass

        page.wait_for_timeout(2_000)

    return best


def _trigger_lazy_loading(page) -> None:
    """
    StoryMaps loads content lazily (scroll-triggered).
    Trigger rendering by paging down / scrolling a few times.
    """
    # StoryMaps often gates the actual story text behind an initial "Start" step.
    # Best-effort: click anything whose visible textContent equals "Start".
    try:
        page.evaluate(
            """
            () => {
              const els = Array.from(document.querySelectorAll('*'));
              const target = els.find(el => {
                const t = (el.innerText || el.textContent || '').trim();
                return t === 'Start';
              });
              if (target && typeof target.click === 'function') {
                target.click();
              }
            }
            """
        )
        page.wait_for_timeout(1500)
    except Exception:
        pass

    # Give JS a moment before interaction.
    page.wait_for_timeout(1_000)
    for _ in range(10):
        try:
            page.keyboard.press("PageDown")
        except Exception:
            # If PageDown doesn't work, try scroll-to-bottom in JS.
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        page.wait_for_timeout(900)

    # Final scroll to bottom (sometimes triggers the last chunk of text).
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
    except Exception:
        pass

    # If StoryMaps content lives inside iframes, scrolling the top document
    # may not trigger rendering there. Scroll child frames too (best-effort).
    try:
        for frame in page.frames:
            if frame.parent is None:
                continue
            try:
                frame.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            except Exception:
                continue
    except Exception:
        pass


def _clean_text(text: str) -> str:
    # Normalize whitespace while keeping paragraph structure.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cleanup_ocr_noise(text: str) -> str:
    """
    Remove common navigation/header noise from OCR text.
    Story text should remain.
    """
    if not text:
        return text
    lines = [ln.strip() for ln in text.splitlines()]
    kept: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if any(ns.lower() in ln.lower() for ns in NAV_NOISE_SUBSTRINGS):
            continue
        kept.append(ln)
    # Re-join to preserve paragraphs reasonably.
    return "\n".join(kept).strip()


def _try_ocr_easyocr(reader, image_path: str) -> str | None:
    """
    Run OCR via easyocr.Reader on the screenshot.
    Returns cleaned OCR text, or None on failure.
    """
    try:
        results = reader.readtext(image_path, detail=0, paragraph=True)
        if not results:
            return None
        # results is a list[str] when detail=0.
        text = "\n".join([str(x) for x in results if str(x).strip()])
        text = _clean_text(text)
        text = _cleanup_ocr_noise(text)
        return text if text else None
    except Exception:
        return None


def scrape_items(
    start: int = 1,
    end_inclusive: int = 8,
    headless: bool = False,
) -> None:
    """
    Scrape items from `start..end_inclusive` and write markdown files into ./data.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ocr_reader = None
    if OCR_FALLBACK_ENABLED:
        try:
            import easyocr  # type: ignore

            # Swedish model is mainly English alphabets + Swedish diacritics; use 'sv'.
            # gpu=False keeps it CPU-only and more portable for PoC.
            ocr_reader = easyocr.Reader(["sv"], gpu=False)
            print("OCR fallback enabled (easyocr).", flush=True)
        except Exception as e:
            print(f"OCR fallback disabled: easyocr not available ({e}).", flush=True)
            ocr_reader = None

    with sync_playwright() as p:
        total = end_inclusive
        executable_path = _resolve_chromium_executable_path()
        for i in range(start, end_inclusive + 1):
            url = BASE_URL.format(i=i)
            out_path = data_dir / f"sweco_bollebygd_item_{i}.md"

            print(f"Scraping Item {i}/{total}...", flush=True)

            launch_kwargs = {"headless": headless}
            if executable_path:
                launch_kwargs["executable_path"] = executable_path
            browser = p.chromium.launch(**launch_kwargs)
            try:
                # StoryMaps may be served with a certificate Playwright can't validate
                # (ERR_CERT_AUTHORITY_INVALID). For PoC scraping, we ignore HTTPS errors.
                # Use a higher device scale factor for better OCR quality on text rendered
                # as images (StoryMaps).
                context = browser.new_context(
                    ignore_https_errors=True,
                    viewport={"width": 1920, "height": 1080},
                    device_scale_factor=2,
                )
                page = context.new_page()

                print(f"  Navigating to: {url}", flush=True)
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)

                # Crucial: wait for network idle so JS and lazy-loaded content can finish.
                try:
                    page.wait_for_load_state("networkidle", timeout=60_000)
                except Exception as e:
                    # networkidle can be flaky for some SPAs; continue after fallback wait.
                    print(f"  Warning: networkidle wait timed out/failed: {e}", flush=True)

                print("  Extracting visible text (with retries)...", flush=True)
                text = _extract_visible_text_with_retries(page, timeout_sec=60.0)
                # If extraction looks too small, OCR the rendered page.
                if len(text) < MIN_TEXT_LEN:
                    html_path = out_path.with_suffix(".html")
                    png_path = out_path.with_suffix(".png")
                    try:
                        html_path.write_text(page.content(), encoding="utf-8")
                        print(f"  Debug: saved {html_path.name} (HTML dump)", flush=True)
                    except Exception:
                        pass
                    try:
                        # Zoom in to make text easier for OCR when the story body is
                        # rendered as images/canvas.
                        try:
                            page.evaluate("document.body.style.zoom = '200%';")
                            page.wait_for_timeout(1500)
                        except Exception:
                            pass
                        page.screenshot(path=str(png_path), full_page=True)
                        print(f"  Debug: saved {png_path.name} (full-page screenshot)", flush=True)
                    except Exception as e:
                        print(f"  Debug: screenshot failed ({e})", flush=True)

                    if ocr_reader is not None:
                        print("  OCR fallback: extracting text from screenshot...", flush=True)
                        ocr_text = _try_ocr_easyocr(ocr_reader, str(png_path))
                        if ocr_text:
                            print(f"  OCR text length: {len(ocr_text)} (replacing DOM text)", flush=True)
                            text = ocr_text

                md = f"# Sweco StoryMaps item {i}\n\n{text}\n"
                out_path.write_text(md, encoding="utf-8")
                print(f"  Saved {out_path.name}", flush=True)
            except Exception as e:
                print(f"  Error scraping item {i}: {e}", flush=True)
            finally:
                print("  Closing browser...", flush=True)
                browser.close()


if __name__ == "__main__":
    # Start with headless=False so you can visually confirm.
    # Flip to True when you want unattended runs.
    scrape_items(headless=False)

