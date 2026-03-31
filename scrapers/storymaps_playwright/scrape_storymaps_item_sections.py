from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse very large whitespace runs, but keep paragraph breaks reasonably.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_body_text(page, scroll_rounds: int = 10) -> str:
    """
    Best-effort extraction for StoryMaps story pages.

    We scroll a bit to trigger lazy-loaded text, then read visible body text.
    """
    page.wait_for_load_state("domcontentloaded", timeout=60_000)
    page.wait_for_timeout(2_000)

    for _ in range(scroll_rounds):
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        except Exception:
            pass
        try:
            page.keyboard.press("PageDown")
        except Exception:
            pass
        page.wait_for_timeout(900)

    try:
        txt = page.locator("body").inner_text(timeout=30_000)
        return _clean_text(txt)
    except Exception as e:
        _log(f"  Warning: body.inner_text() failed ({type(e).__name__}: {e}). Falling back to page.content().")
        # Fallback: sometimes `body.inner_text()` is blocked, but `page.content()` works.
        try:
            html = page.content()
            return _clean_text(re.sub(r"<[^>]+>", "\n", html))
        except Exception:
            return ""


def _extract_active_iframe_story_url(collection_page) -> str:
    """
    From a StoryMaps collection view, return the `src` of the active story iframe.
    """
    # In the captured HTML, the active tab has `div.content-wrapper.active`
    candidates = [
        "div.content-wrapper.active iframe",
        "iframe.active-item-iframe",
    ]
    for sel in candidates:
        try:
            loc = collection_page.locator(sel)
            if loc.count() > 0:
                src = loc.first.get_attribute("src")
                if src:
                    return src
        except Exception:
            continue

    # Last resort: return the first story iframe on the page.
    try:
        loc = collection_page.locator("iframe.content-frame")
        if loc.count() > 0:
            src = loc.first.get_attribute("src")
            if src:
                return src
    except Exception:
        pass

    raise RuntimeError("Could not find active StoryMaps iframe story URL.")


def _slice_sections_by_headings(full_text: str, headings: list[str]) -> dict[str, str]:
    headings_set = {h.strip() for h in headings}
    # Map heading -> list of lines
    buckets: dict[str, list[str]] = {h: [] for h in headings}
    order_found: list[str] = []

    current: str | None = None
    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            if current is not None:
                buckets[current].append("")  # paragraph break
            continue

        # Normalize a couple of common heading variants.
        normalized = line.rstrip(":").rstrip(" -").strip()
        if normalized in headings_set:
            current = normalized
            if current not in order_found:
                order_found.append(current)
            continue

        if current is not None:
            buckets[current].append(raw_line)

    # Join + clean.
    out: dict[str, str] = {}
    for h in headings:
        content = "\n".join(buckets.get(h, []))
        content = _clean_text(content)
        out[h] = content
    return out


def scrape_storymaps_item_sections(
    collection_url: str,
    headings: list[str],
    out_dir: Path,
    headless: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(ignore_https_errors=True)

        try:
            page = context.new_page()
            _log(f"Loading collection URL: {collection_url}")
            page.goto(collection_url, wait_until="domcontentloaded", timeout=60_000)
            # Give JS a moment to apply ?item=... state and mark the active tab.
            page.wait_for_timeout(3_000)

            _log("Extracting active iframe story URL...")
            story_url = _extract_active_iframe_story_url(page)
            _log(f"Active story URL: {story_url}")

            story_page = context.new_page()
            _log("Loading story page...")
            story_page.goto(story_url, wait_until="domcontentloaded", timeout=60_000)
            _log("Extracting story text (scroll + body text)...")
            full_text = _extract_body_text(story_page, scroll_rounds=10)
            _log(f"Full text extracted. chars={len(full_text)}")

            sections = _slice_sections_by_headings(full_text, headings)

            # Write outputs
            md_path = out_dir / "sections.md"
            meta_path = out_dir / "meta.json"

            _log(f"Writing output to: {out_dir}")
            with md_path.open("w", encoding="utf-8") as f:
                f.write("# Sweco StoryMaps - Extracted Sections\n\n")
                f.write(f"Collection URL: {collection_url}\n\n")
                f.write(f"Active story URL: {story_url}\n\n")
                f.write("Extracted headings:\n\n")
                for h in headings:
                    content = sections.get(h, "").strip()
                    if content:
                        f.write(f"## {h}\n\n{content}\n\n")
                    else:
                        f.write(f"## {h}\n\n*(Not found / empty after extraction.)*\n\n")

            meta = {
                "collection_url": collection_url,
                "active_story_url": story_url,
                "headings": headings,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "full_text_char_len": len(full_text),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            context.close()
            browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-url", required=True)
    ap.add_argument(
        "--headings",
        nargs="+",
        default=["Inledning", "Bakgrund", "Planläggningsprocessen", "Mål för projektet"],
    )
    ap.add_argument(
        "--out-subfolder",
        default="sweco_storymaps_53cd4fa4b8a7484fbf9b6e58101c78e3_item_3",
        help="Subfolder name inside ./data",
    )
    ap.add_argument("--headless", default="true", help="true|false")
    args = ap.parse_args()

    headless = str(args.headless).strip().lower() in {"1", "true", "yes", "y"}

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    out_dir = data_dir / args.out_subfolder

    scrape_storymaps_item_sections(
        collection_url=args.collection_url,
        headings=args.headings,
        out_dir=out_dir,
        headless=headless,
    )


if __name__ == "__main__":
    main()

