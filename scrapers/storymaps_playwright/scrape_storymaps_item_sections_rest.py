from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.request
from urllib.error import HTTPError
from datetime import datetime, timezone
from pathlib import Path


def _http_get_text(url: str) -> str:
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        raw = resp.read()
    return raw.decode("utf-8", "ignore")


def _strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_origin(collection_url: str) -> str:
    m = re.match(r"^(https?://[^/]+)", collection_url.strip())
    if not m:
        raise ValueError(f"Could not determine origin from URL: {collection_url}")
    return m.group(1)


def _extract_active_story_url_from_collection_html(html: str) -> str:
    # Primary: active story iframe has class `active-item-iframe`
    m = re.search(
        r'<iframe[^>]*class="[^"]*active-item-iframe[^"]*"[^>]*src="([^"]+)"',
        html,
        flags=re.I,
    )
    if m:
        return m.group(1)

    # Fallback: find an iframe inside the active tab wrapper.
    # We look for a `content-wrapper active` block and capture its first iframe src.
    m = re.search(
        r'content-wrapper[^"]*\bactive\b[^"]*.*?<iframe[^>]*src="([^"]+)"',
        html,
        flags=re.I | re.S,
    )
    if m:
        return m.group(1)

    raise RuntimeError("Could not find active story iframe URL in collection HTML.")


def _story_id_from_story_url(story_url: str) -> str:
    return story_url.rstrip("/").split("/")[-1]


def _parse_collection_id_and_item_param(collection_url: str) -> tuple[str, int]:
    collection_url = collection_url.strip()
    m_coll = re.search(r"/collections/([a-fA-F0-9]{32})", collection_url)
    if not m_coll:
        raise ValueError(f"Could not parse collection id from URL: {collection_url}")
    collection_id = m_coll.group(1)

    m_item = re.search(r"[?&]item=(\d+)", collection_url)
    if not m_item:
        raise ValueError(f"Could not parse item query param from URL: {collection_url}")
    item_param = int(m_item.group(1))
    if item_param < 1:
        raise ValueError("item query param must be >= 1.")

    return collection_id, item_param


def _fetch_collection_tab_story_id(collection_data: dict, item_param: int) -> str:
    """
    In the collection /data payload, a node of type `collection-ui` contains `data.items`
    (ordered tabs). Each entry references a `resourceId`, which is mapped in
    the top-level `resources` to a story item id.
    """
    nodes = collection_data.get("nodes") or {}
    resources = collection_data.get("resources") or {}

    items_list = None
    for _, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") != "collection-ui":
            continue
        d = node.get("data")
        if isinstance(d, dict) and isinstance(d.get("items"), list) and len(d["items"]) >= item_param:
            items_list = d["items"]
            break

    if not items_list:
        raise RuntimeError("Could not find tab items list in collection payload.")

    tab_index = item_param - 1
    tab_entry = items_list[tab_index]
    if not isinstance(tab_entry, dict) or "resourceId" not in tab_entry:
        raise RuntimeError("Unexpected tab entry format in collection payload.")

    resource_id = tab_entry["resourceId"]
    resource = resources.get(resource_id)
    if not isinstance(resource, dict):
        raise RuntimeError(f"Could not find resource {resource_id} in collection payload.")
    data = resource.get("data")
    if not isinstance(data, dict) or "itemId" not in data:
        raise RuntimeError(f"Resource {resource_id} does not contain data.itemId.")

    return str(data["itemId"])


def _fetch_collection_data(collection_id: str, origin: str) -> dict:
    url = f"{origin}/portal/sharing/rest/content/items/{collection_id}/data?f=json"
    raw = _http_get_text(url)
    return json.loads(raw)


def _fetch_story_data(story_id: str, origin: str) -> dict:
    url = f"{origin}/portal/sharing/rest/content/items/{story_id}/data?f=json"
    raw = _http_get_text(url)
    return json.loads(raw)


def _download_binary(url: str, dest_path: Path) -> int:
    """
    Download a binary file with urllib and write it to dest_path.
    Returns bytes written.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        # Best-effort to avoid fully buffering large files in memory.
        with dest_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
    return dest_path.stat().st_size


def _download_images_from_story_data(
    story_id: str,
    origin: str,
    story_data: dict,
    images_dir: Path,
) -> list[dict]:
    nodes = story_data.get("nodes") or {}
    resources = story_data.get("resources") or {}

    image_nodes: list[tuple[str, dict]] = []
    for nid, node in nodes.items():
        if isinstance(node, dict) and node.get("type") == "image":
            image_nodes.append((nid, node))

    assets: list[dict] = []
    for node_id, node in image_nodes:
        data = node.get("data") or {}
        resource_ref = data.get("image")
        if not resource_ref:
            continue
        res_obj = resources.get(resource_ref)
        if not isinstance(res_obj, dict):
            continue
        if res_obj.get("type") != "image":
            continue
        res_data = res_obj.get("data") or {}
        resource_file = res_data.get("resourceId")
        if not isinstance(resource_file, str) or not resource_file:
            continue

        # StoryMaps stores image bytes as item resources.
        img_url = f"{origin}/portal/sharing/rest/content/items/{story_id}/resources/{resource_file}"
        dest_path = images_dir / resource_file

        # Skip if already downloaded.
        bytes_written = 0
        if dest_path.exists():
            bytes_written = dest_path.stat().st_size
        else:
            try:
                bytes_written = _download_binary(img_url, dest_path)
            except HTTPError:
                # Some resources might not be publicly accessible; keep going.
                continue

        assets.append(
            {
                "node_id": node_id,
                "alt": data.get("alt"),
                "caption": data.get("caption"),
                "resource_ref": resource_ref,
                "resource_file": resource_file,
                "bytes": bytes_written,
            }
        )

    # Stable ordering for diffs.
    assets.sort(key=lambda x: (str(x.get("node_id")), str(x.get("resource_file"))))
    return assets


def _fetch_webmap_item_json(origin: str, webmap_item_id: str) -> dict:
    url = f"{origin}/portal/sharing/rest/content/items/{webmap_item_id}/data?f=json"
    raw = _http_get_text(url)
    return json.loads(raw)


def _download_webmaps_from_story_data(
    origin: str,
    story_data: dict,
    webmaps_dir: Path,
) -> list[dict]:
    nodes = story_data.get("nodes") or {}
    resources = story_data.get("resources") or {}

    webmap_nodes: list[tuple[str, dict]] = []
    for nid, node in nodes.items():
        if isinstance(node, dict) and node.get("type") == "webmap":
            webmap_nodes.append((nid, node))

    assets: list[dict] = []
    for node_id, node in webmap_nodes:
        data = node.get("data") or {}
        map_ref = data.get("map")
        if not map_ref:
            continue
        map_res_obj = resources.get(map_ref)
        if not isinstance(map_res_obj, dict):
            continue
        if map_res_obj.get("type") != "webmap":
            continue
        map_res_data = map_res_obj.get("data") or {}
        webmap_item_id = map_res_data.get("itemId")
        if not isinstance(webmap_item_id, str) or not webmap_item_id:
            continue

        webmap_json = _fetch_webmap_item_json(origin=origin, webmap_item_id=webmap_item_id)

        dest_path = webmaps_dir / f"{webmap_item_id}.json"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if not dest_path.exists():
            dest_path.write_text(json.dumps(webmap_json, ensure_ascii=False, indent=2), encoding="utf-8")

        # Extract URLs for convenience (WMS/service endpoints, etc.)
        s = json.dumps(webmap_json, ensure_ascii=False)
        urls = sorted(set(re.findall(r"https?://[^\\\"\\']+", s)))

        assets.append(
            {
                "node_id": node_id,
                "webmap_item_id": webmap_item_id,
                "map_ref": map_ref,
                "caption": data.get("caption"),
                "visible_layers_urls": urls,
            }
        )

    # Stable ordering.
    assets.sort(key=lambda x: str(x.get("webmap_item_id")))
    return assets


def _find_heading_node_ids(story_data: dict, headings: list[str]) -> dict[str, str]:
    nodes = story_data.get("nodes") or {}
    out: dict[str, str] = {}
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") != "text":
            continue
        d = node.get("data") or {}
        if d.get("type") == "h2" and d.get("text") in headings:
            out[d["text"]] = nid
    return out


def _find_parent_panel_id(nodes: dict, child_id: str) -> str | None:
    for pid, pnode in nodes.items():
        if not isinstance(pnode, dict):
            continue
        if pnode.get("type") != "immersive-narrative-panel":
            continue
        children = pnode.get("children")
        if isinstance(children, list) and child_id in children:
            return pid
    return None


def _render_bullet_list_from_node_text(raw_html: str) -> list[str]:
    # Example format from node data:
    # "<li>Tillföra ...</li><li>...</li>"
    items = re.findall(r"<li>(.*?)</li>", raw_html, flags=re.S | re.I)
    if not items:
        cleaned = _strip_html_tags(raw_html)
        return [cleaned] if cleaned else []
    return [_strip_html_tags(i) for i in items if _strip_html_tags(i)]


def _extract_sections_from_story_data(story_data: dict, headings: list[str]) -> dict[str, str]:
    nodes = story_data.get("nodes") or {}

    heading_node_ids = _find_heading_node_ids(story_data, headings)
    out: dict[str, str] = {h: "" for h in headings}

    for heading in headings:
        h_node_id = heading_node_ids.get(heading)
        if not h_node_id:
            continue

        panel_id = _find_parent_panel_id(nodes, h_node_id)
        if not panel_id:
            continue

        panel = nodes[panel_id]
        children = panel.get("children") if isinstance(panel, dict) else None
        if not isinstance(children, list):
            continue

        # Collect everything after the heading node within this panel.
        try:
            start_idx = children.index(h_node_id)
        except ValueError:
            continue

        blocks: list[str] = []
        for cid in children[start_idx + 1 :]:
            cnode = nodes.get(cid)
            if not isinstance(cnode, dict):
                continue

            if cnode.get("type") != "text":
                continue

            d = cnode.get("data") or {}
            text = d.get("text", "")
            node_type = d.get("type")
            if not isinstance(text, str) or not text.strip():
                continue

            if node_type == "paragraph":
                blocks.append(_strip_html_tags(text))
            elif node_type == "h3":
                blocks.append(f"### {_strip_html_tags(text)}")
            elif node_type == "bullet-list":
                bullets = _render_bullet_list_from_node_text(text)
                for b in bullets:
                    blocks.append(f"- {b}")
            else:
                # Keep any other text node types as raw, but stripped of HTML.
                blocks.append(_strip_html_tags(text))

        # Join blocks with paragraph spacing.
        content = "\n\n".join([b for b in blocks if b.strip()]).strip()
        out[heading] = content

    return out


def _detect_h2_headings_from_story_data(story_data: dict) -> list[str]:
    """
    Detect all `h2` headings present in a StoryMaps story `/data` payload.

    We preserve the encounter order as found in the JSON payload.
    """
    nodes = story_data.get("nodes") or {}
    out: list[str] = []
    seen: set[str] = set()
    for _, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") != "text":
            continue
        d = node.get("data") or {}
        if d.get("type") == "h2":
            txt = d.get("text")
            if isinstance(txt, str) and txt.strip() and txt not in seen:
                out.append(txt.strip())
                seen.add(txt.strip())
    return out


def scrape_storymaps_item_sections_rest(
    collection_url: str,
    headings: list[str],
    out_dir: Path,
    download_images: bool,
    download_webmaps: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    origin = _extract_origin(collection_url)
    collection_id, item_param = _parse_collection_id_and_item_param(collection_url)
    collection_data = _fetch_collection_data(collection_id=collection_id, origin=origin)
    story_id = _fetch_collection_tab_story_id(collection_data=collection_data, item_param=item_param)
    story_url = f"{origin}/portal/apps/storymaps/stories/{story_id}"

    story_data = _fetch_story_data(story_id=story_id, origin=origin)
    if not headings:
        headings = _detect_h2_headings_from_story_data(story_data=story_data)
    sections = _extract_sections_from_story_data(story_data=story_data, headings=headings)

    md_path = out_dir / "sections.md"
    meta_path = out_dir / "meta.json"

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Sweco StoryMaps - Extracted Sections (REST)\n\n")
        f.write(f"Collection URL: {collection_url}\n\n")
        f.write(f"Active story URL: {story_url}\n\n")
        for h in headings:
            content = sections.get(h, "").strip()
            if content:
                f.write(f"## {h}\n\n{content}\n\n")
            else:
                f.write(f"## {h}\n\n*(Not found / empty after extraction.)*\n\n")

    meta = {
        "collection_url": collection_url,
        "active_story_url": f"{origin}/portal/apps/storymaps/stories/{story_id}",
        "active_story_id": story_id,
        "headings": headings,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "sections_present": {h: bool(sections.get(h, "").strip()) for h in headings},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional asset downloads.
    assets_summary: dict = {}
    if download_images:
        images_dir = out_dir / "images"
        assets_summary["images"] = _download_images_from_story_data(
            story_id=story_id,
            origin=origin,
            story_data=story_data,
            images_dir=images_dir,
        )
    if download_webmaps:
        webmaps_dir = out_dir / "webmaps"
        assets_summary["webmaps"] = _download_webmaps_from_story_data(
            origin=origin,
            story_data=story_data,
            webmaps_dir=webmaps_dir,
        )

    if assets_summary:
        (out_dir / "assets_manifest.json").write_text(
            json.dumps(assets_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-url", required=True)
    ap.add_argument(
        "--headings",
        nargs="*",
        default=None,
    )
    ap.add_argument(
        "--out-subfolder",
        default="sweco_storymaps_53cd4fa4b8a7484fbf9b6e58101c78e3_item_3",
        help="Subfolder name inside ./data",
    )
    ap.add_argument(
        "--download-images",
        action="store_true",
        help="Download image nodes referenced by the story payload into ./images/",
    )
    ap.add_argument(
        "--download-webmaps",
        action="store_true",
        help="Save referenced webmap item JSON + extracted service URLs into ./webmaps/",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    out_dir = data_dir / args.out_subfolder

    scrape_storymaps_item_sections_rest(
        collection_url=args.collection_url,
        headings=(args.headings or []),
        out_dir=out_dir,
        download_images=bool(args.download_images),
        download_webmaps=bool(args.download_webmaps),
    )


if __name__ == "__main__":
    main()

