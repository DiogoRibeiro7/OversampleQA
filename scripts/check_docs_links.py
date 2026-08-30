"""Check stable external links in repository documentation.

MkDocs validates internal documentation links during ``mkdocs build``. This
script covers a small allowlist of external hosts that should be stable enough
to gate in CI without turning documentation builds into a flaky internet crawl.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CHECKED_HOSTS = frozenset(
    {
        "diogoribeiro7.github.io",
        "docs.astral.sh",
        "doi.org",
        "github.com",
        "img.shields.io",
        "keepachangelog.com",
        "pypi.org",
        "semver.org",
        "www.python.org",
        "zenodo.org",
    }
)
DOC_ROOTS = (
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("ROADMAP.md"),
    Path("CITATION.cff"),
    Path("docs"),
)
HTTP_OK = range(200, 400)
MARKDOWN_URL = re.compile(r"(?<!git\+)https?://[^\s<>)\"']+")


def documentation_files(paths: tuple[Path, ...] = DOC_ROOTS) -> list[Path]:
    """Return documentation files that may contain stable external links."""
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in {".md", ".cff"}:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
    return sorted(files)


def strip_url_punctuation(url: str) -> str:
    """Remove punctuation commonly captured at the end of Markdown links."""
    return url.rstrip(".,;:)}]")


def iter_checked_urls(path: Path) -> set[str]:
    """Extract checked external URLs from a documentation file."""
    urls: set[str] = set()
    for match in MARKDOWN_URL.finditer(path.read_text(encoding="utf-8")):
        url = strip_url_punctuation(match.group(0))
        if urlparse(url).netloc.lower() in CHECKED_HOSTS:
            urls.add(url)
    return urls


def request_url(url: str, timeout: float) -> int:
    """Return the HTTP status for a URL, using GET if HEAD is unsupported."""
    headers = {"User-Agent": "oversampleqa-docs-link-check/1.0"}
    request = Request(url, method="HEAD", headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status
    except HTTPError as exc:
        if exc.code not in {403, 405}:
            raise
    request = Request(url, method="GET", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.status


def check_url(url: str, retries: int, timeout: float) -> str | None:
    """Return an error string when a URL cannot be reached successfully."""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            status = request_url(url, timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = str(exc)
        else:
            if status in HTTP_OK:
                return None
            last_error = f"HTTP {status}"
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    return last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check stable external links in repository documentation."
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    failures: list[str] = []
    checked: dict[str, set[Path]] = {}
    for path in documentation_files():
        for url in iter_checked_urls(path):
            checked.setdefault(url, set()).add(path)

    for url in sorted(checked):
        error = check_url(url, retries=args.retries, timeout=args.timeout)
        if error:
            locations = ", ".join(str(path) for path in sorted(checked[url]))
            failures.append(f"{url} ({locations}): {error}")

    if failures:
        print("External documentation link check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Checked {len(checked)} stable external documentation links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
