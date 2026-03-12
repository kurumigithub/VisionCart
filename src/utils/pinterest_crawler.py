"""
Pinterest Board Crawler

Crawls Pinterest to collect board names and image URLs, optionally downloading
images to disk. Supports two modes: direct board URL or search-by-prompt.

What it does
------------
- Board URL mode: Given a Pinterest board URL, fetches the board name and
  image URLs from that board.
- Search mode: Given a short prompt (e.g. "spring garden ideas"), uses Selenium
  to load Pinterest's board search page, extracts board URLs from the rendered
  HTML, then crawls each board for images.
- Optionally downloads images into data/YYYY-MM-DD_HH-MM-SS/board_name/.

Inputs
------
- input_value: Either a Pinterest board URL (e.g. https://www.pinterest.com/user/board/)
  or a short search prompt (e.g. "modern living room").
- num_boards: Number of boards to crawl when using a search prompt (default: 1).
- max_images_per_board: Maximum images to collect per board (default: 20).
- --download: If set, downloads images to data/YYYY-MM-DD_HH-MM-SS.
- --json: If set, prints results as JSON instead of human-readable text.

Outputs
-------
- List[BoardInfo]: Each BoardInfo has `name`, `url`, and `image_urls`.
- With --download: Image files saved under data/YYYY-MM-DD_HH-MM-SS/board_name/.
- With --json: JSON array of {name, url, image_urls}.

CLI usage
---------
  python -m pinterest_crawler "spring garden" -n 3 -m 15 --download
  python -m pinterest_crawler "https://www.pinterest.com/user/board/" -m 30 -d
  python -m pinterest_crawler "interior design" -n 2 --json
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

# Optional Selenium imports (for JS-rendered search pages)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:  # pragma: no cover - selenium is optional
    webdriver = None  # type: ignore[assignment]
    ChromeOptions = None  # type: ignore[assignment]
    WebDriverWait = None  # type: ignore[assignment]


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class BoardInfo:
    name: str
    url: str
    image_urls: List[str]


def _fetch(url: str, *, delay: float = 0.0) -> Optional[str]:
    """Fetch a URL and return the HTML text, or None on error."""
    if delay:
        time.sleep(delay)

    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _extract_pws_data(html: str) -> Optional[dict]:
    """Extract Pinterest's embedded JSON from the __PWS_DATA__ script tag."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__PWS_DATA__")
    if not script or not script.string:
        return None

    try:
        return json.loads(script.string)
    except Exception:
        return None


def _guess_board_name_from_html(html: str) -> Optional[str]:
    """Fallback: try to guess board name from <title> or <meta> tags."""
    soup = BeautifulSoup(html, "html.parser")

    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if "on Pinterest" in title:
            title = title.replace("on Pinterest", "").strip(" -")
        return title

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    return None


def _pick_highest_res_from_srcset(srcset: str) -> Optional[str]:
    """
    Parse srcset (e.g. "url1 236w, url2 474w" or "url1 1x, url2 2x")
    and return the highest-resolution URL.
    """
    best_url, best_val = None, -1
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.rsplit(maxsplit=1)
        url = tokens[0].strip()
        if len(tokens) == 1:
            return url
        desc = tokens[1].lower()
        if desc.endswith("w"):
            try:
                val = int(desc[:-1])
            except ValueError:
                val = 0
        elif desc.endswith("x"):
            try:
                val = int(float(desc[:-1]) * 1000)
            except ValueError:
                val = 0
        else:
            val = 0
        if val > best_val:
            best_val, best_url = val, url
    return best_url


def _extract_image_urls_from_html(html: str, max_images: int) -> List[str]:
    """
    Fallback: extract image URLs from standard <img> tags on the page.

    Pinterest image URLs encode size in the path, e.g.:
    - https://i.pinimg.com/75x75_RS/...
    - https://i.pinimg.com/236x/...
    - https://i.pinimg.com/videos/thumbnails/...

    We:
    - Include only `i.pinimg.com` URLs.
    - Exclude small 75x75 icons.
    - Include typical pin images (236x) and video thumbnails.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []

    for img in soup.find_all("img"):
        # Prefer srcset/data-srcset to get highest-resolution URL (like currentSrc)
        srcset = img.get("data-srcset") or img.get("srcset")
        if srcset:
            src = _pick_highest_res_from_srcset(srcset)
        else:
            src = img.get("data-src") or img.get("src")
        if not src:
            continue
        # Handle srcset that returns a single URL without descriptors
        if " " in src:
            src = src.split()[0]
        if "https://i.pinimg.com/" not in src:
            continue

        # Exclude 75x75 icons
        if "/75x75" in src:
            continue

        if src not in urls:
            urls.append(src)
        if len(urls) >= max_images:
            break

    return urls


def _extract_board_from_pws_data(data: dict, board_url: str, max_images: int) -> BoardInfo:
    """
    Try to extract board name and image URLs from Pinterest's internal JSON.
    This is best-effort and may break if Pinterest changes their schema.
    """
    name: Optional[str] = None
    image_urls: List[str] = []

    props = data.get("props") or {}
    redux = props.get("initialReduxState") or {}

    boards = redux.get("boards", {})
    by_id = boards.get("byId", {})
    for board_id, board_obj in by_id.items():
        board_name = board_obj.get("name")
        if board_name:
            name = board_name
            break

    SIZE_ORDER = ("originals", "736x", "474x", "564x", "236x")  # prefer highest res
    pins = redux.get("pins", {}).get("byId", {})
    for pin_id, pin_obj in pins.items():
        images_obj = pin_obj.get("images") or {}
        best_url = None
        for size_key in SIZE_ORDER:
            if size_key in images_obj:
                best_url = images_obj[size_key].get("url")
                break
        if not best_url:
            for img_variant in images_obj.values():
                u = img_variant.get("url")
                if u:
                    best_url = u
                    break
        if best_url and best_url not in image_urls:
            image_urls.append(best_url)
        if len(image_urls) >= max_images:
            break

    if not name:
        name = board_url.rstrip("/").split("/")[-1] or board_url

    return BoardInfo(name=name, url=board_url, image_urls=image_urls)


def crawl_board(board_url: str, *, max_images: int = 50) -> Optional[BoardInfo]:
    """
    Crawl a single Pinterest board URL.

    Returns a BoardInfo object, or None if the page could not be parsed.
    """
    html = _fetch(board_url)
    if not html:
        return None

    data = _extract_pws_data(html)
    if data:
        board = _extract_board_from_pws_data(data, board_url, max_images)
        if board.image_urls:
            if not board.name:
                board.name = _guess_board_name_from_html(html) or board.name
            return board

    fallback_name = _guess_board_name_from_html(html) or board_url
    image_urls = _extract_image_urls_from_html(html, max_images)
    if not image_urls:
        return None

    return BoardInfo(name=fallback_name, url=board_url, image_urls=image_urls)


def _extract_board_urls_from_anchors(html: str, max_boards: int, seen: List[str]) -> List[str]:
    """
    Extract board URLs from <a href> tags in Selenium-rendered search page HTML.
    Only adds URLs not already in `seen`.
    """
    soup = BeautifulSoup(html, "html.parser")
    board_urls = list(seen)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            full = f"https://www.pinterest.com{href}"
        elif href.startswith("http"):
            if "pinterest.com" not in href:
                continue
            full = href
        else:
            continue

        path = full.split("pinterest.com")[-1].split("?")[0]
        if any(seg in path for seg in ("/pin/", "/ideas/", "/search/", "/login/", "/signup/")):
            continue
        # Board URLs are typically /username/boardname/
        if path.count("/") < 2:
            continue

        if full not in board_urls:
            board_urls.append(full)
        if len(board_urls) >= max_boards:
            break

    return board_urls


def _extract_board_urls_from_search_html(html: str, max_boards: int) -> List[str]:
    """
    Extract candidate board URLs from a Pinterest board search page.

    Tries PWS JSON first; if that yields nothing (common on search pages),
    falls back to parsing <a href> tags from the Selenium-rendered HTML.
    """
    board_urls: List[str] = []

    # Try Pinterest's embedded JSON (may be empty on search pages)
    data = _extract_pws_data(html)
    if data:
        props = data.get("props") or {}
        redux = props.get("initialReduxState") or {}
        boards_state = redux.get("boards") or {}
        by_id = boards_state.get("byId") or {}

        for board_obj in by_id.values():
            url = board_obj.get("url")
            if not url:
                owner = board_obj.get("owner") or {}
                username = owner.get("username")
                slug = board_obj.get("slug") or board_obj.get("name")
                if username and slug:
                    url = f"/{username}/{slug}/"

            if not url:
                continue

            if url.startswith("/"):
                full = f"https://www.pinterest.com{url}"
            elif url.startswith("http"):
                full = url
            else:
                full = f"https://www.pinterest.com/{url.lstrip('/')}"

            if full not in board_urls:
                board_urls.append(full)
            if len(board_urls) >= max_boards:
                return board_urls

    # Fallback: extract from <a href> in rendered HTML (Selenium page_source)
    board_urls = _extract_board_urls_from_anchors(html, max_boards, board_urls)
    return board_urls


def search_boards_by_prompt(
    prompt: str,
    *,
    num_boards: int = 5,
    max_images_per_board: int = 20,
    delay_between_requests: float = 1.0,
) -> List[BoardInfo]:
    """
    Legacy HTTP-only implementation for board search.

    Pinterest's search results are now heavily JavaScript-driven, so this
    approach is unreliable. Use `search_boards_by_prompt_selenium` instead.
    """
    raise RuntimeError(
        "HTTP-only board search is disabled. "
        "Use `search_boards_by_prompt_selenium` (Selenium-based search) instead."
    )


def _create_default_selenium_driver():
    """
    Create a headless Chrome driver for Selenium.

    Raises RuntimeError if Selenium is not installed.
    """
    if webdriver is None or ChromeOptions is None:
        raise RuntimeError(
            "Selenium is not available. Install it with `pip install selenium` "
            "and ensure a Chrome/Chromium driver is installed."
        )

    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    return webdriver.Chrome(options=options)


def search_boards_by_prompt_selenium(
    prompt: str,
    *,
    num_boards: int = 5,
    max_images_per_board: int = 20,
    delay_between_requests: float = 1.0,
    wait_timeout: float = 10.0,
    driver=None,
) -> List[BoardInfo]:
    """
    Search Pinterest for boards using Selenium to render JavaScript.

    - Opens a real browser (headless Chrome by default).
    - Loads the boards search page for the prompt.
    - Uses the rendered HTML (`page_source`) with `_extract_board_urls_from_search_html`.

    You can pass an existing Selenium `driver` instance if you want to
    manage its lifecycle yourself; otherwise a temporary driver is created
    and closed.
    """
    from urllib.parse import quote_plus

    if webdriver is None:
        raise RuntimeError(
            "Selenium is not available. Install it with `pip install selenium` "
            "and ensure a Chrome/Chromium driver is installed."
        )

    created_driver = False
    if driver is None:
        driver = _create_default_selenium_driver()
        created_driver = True

    try:
        q = quote_plus(prompt)
        print(f"Searching for boards with prompt (Selenium): {q}")
        search_url = f"https://www.pinterest.com/search/boards/?q={q}"
        driver.get(search_url)

        # Give the page some time to execute JS and render results.
        if WebDriverWait is not None:
            print(f"Waiting for page to load")
            WebDriverWait(driver, wait_timeout).until(
                lambda d: "__PWS_DATA__" in d.page_source
                or "enable-javascript.com" not in d.page_source
            )
        else:
            time.sleep(3)

        html = driver.page_source
        # print(f"HTML: {html}")
        board_urls = _extract_board_urls_from_search_html(html, num_boards)

        results: List[BoardInfo] = []
        print(f"Found {len(board_urls)} board URLs via Selenium")

        for idx, url in enumerate(board_urls):
            board = crawl_board(url, max_images=max_images_per_board)
            if board:
                results.append(board)
            if idx < len(board_urls) - 1 and delay_between_requests > 0:
                time.sleep(delay_between_requests)

        return results
    finally:
        if created_driver:
            try:
                driver.quit()
            except Exception:
                pass


def crawl_pinterest(
    input_value: str,
    *,
    num_boards: int = 1,
    max_images_per_board: int = 20,
) -> List[BoardInfo]:
    """
    High-level entry point.

    - If `input_value` looks like a URL, treat it as a single board URL.
    - Otherwise, treat it as a short prompt and search for boards.
    """
    input_value = input_value.strip()

    if input_value.startswith("http://") or input_value.startswith("https://"):
        board = crawl_board(input_value, max_images=max_images_per_board)
        return [board] if board else []

    # Use Selenium-based search; HTTP-only scraping of search results is
    # unreliable due to Pinterest's heavy use of JavaScript.
    return search_boards_by_prompt_selenium(
        input_value,
        num_boards=num_boards,
        max_images_per_board=max_images_per_board,
    )


def _to_serializable(result: List[BoardInfo]):
    return [asdict(b) for b in result]


def _get_download_dir(base: str = "data") -> Path:
    """Create and return a download folder: base/YYYY-MM-DD_HH-MM-SS."""
    folder = Path(base) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _safe_filename(url: str, index: int) -> str:
    """Generate a safe filename from URL or index."""
    parsed = url.split("/")[-1].split("?")[0]
    if parsed and re.match(r"^[\w\-\.]+$", parsed):
        base, ext = os.path.splitext(parsed)
        return f"{base}{ext}" if ext else f"{base}.jpg"
    return f"image_{index:04d}.jpg"


def download_images(boards: List[BoardInfo], base_dir: str = "data") -> str:
    """
    Download all images from the given boards into base_dir/YYYY-MM-DD_HH-MM-SS.
    Returns the absolute path of the download folder.
    """
    folder = _get_download_dir(base_dir)
    folder = folder.resolve()

    for board in boards:
        board_subdir = re.sub(r"[^\w\-]", "_", board.name)[:50].strip("_")
        if not board_subdir:
            board_subdir = "board"
        target = folder / board_subdir
        target.mkdir(parents=True, exist_ok=True)

        for idx, url in enumerate(board.image_urls):
            try:
                resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
                resp.raise_for_status()
                fname = _safe_filename(url, idx)
                path = target / fname
                path.write_bytes(resp.content)
            except Exception as e:
                print(f"  Skipped {url[:60]}... ({e})")

    return str(folder)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl Pinterest boards to collect board names and image URLs."
    )

    parser.add_argument(
        "input_value",
        help="Either a Pinterest board URL or a short text prompt.",
    )
    parser.add_argument(
        "-n",
        "--num-boards",
        type=int,
        default=1,
        help="Number of boards to retrieve when using a text prompt.",
    )
    parser.add_argument(
        "-m",
        "--max-images-per-board",
        type=int,
        default=20,
        help="Maximum number of images to collect per board.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download images to data/YYYY-MM-DD_HH-MM-SS.",
    )

    return parser


def _print_human_readable(boards: List[BoardInfo]) -> None:
    if not boards:
        print("No boards found or failed to crawl.")
        return

    for idx, board in enumerate(boards, start=1):
        print(f"[{idx}] Board: {board.name}")
        print(f"    URL: {board.url}")
        print(f"    Images ({len(board.image_urls)}):")
        for img in board.image_urls:
            print(f"        {img}")
        print()


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = crawl_pinterest(
        args.input_value,
        num_boards=args.num_boards,
        max_images_per_board=args.max_images_per_board,
    )

    if args.download and results:
        folder = download_images(results, base_dir="data")
        print(f"Downloaded images to: {folder}")

    if args.json:
        print(json.dumps(_to_serializable(results), indent=2))
    else:
        _print_human_readable(results)

