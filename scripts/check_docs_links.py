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
#: Statuses that mean the target is genuinely gone and somebody must edit a
#: document. Everything else that is not OK -- 5xx, 429, connection failures,
#: timeouts, and the 401/403 a host returns when it dislikes a robot -- says
#: the host could not answer, which is not evidence the link is wrong.
HTTP_GONE = frozenset({404, 410})
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


def check_url(url: str, retries: int, timeout: float) -> tuple[str, str] | None:
    """Check one URL.

    Returns ``None`` when the URL is fine, otherwise ``(kind, detail)`` where
    kind is ``"gone"`` or ``"unavailable"``.

    The distinction is the point. A 404 means a document points somewhere that
    no longer exists and someone has to fix it. A timeout or a 502 means the
    host was having a bad day, which nobody here can act on -- and reporting
    the two identically is how a report earns its way into being ignored. On
    2026-08-31 doi.org returned five timeouts and a 502 for six perfectly valid
    DOIs; under the old behaviour that read exactly like six dead links.
    """
    last: tuple[str, str] = ("unavailable", "no attempt made")
    for attempt in range(retries + 1):
        try:
            status = request_url(url, timeout)
        except HTTPError as exc:
            kind = "gone" if exc.code in HTTP_GONE else "unavailable"
            last = (kind, str(exc))
            if kind == "gone":
                # A 404 will not become a 200 on retry.
                return last
        except (URLError, TimeoutError) as exc:
            last = ("unavailable", str(exc))
        else:
            if status in HTTP_OK:
                return None
            kind = "gone" if status in HTTP_GONE else "unavailable"
            last = (kind, f"HTTP {status}")
            if kind == "gone":
                return last
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    return last


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check stable external links in repository documentation."
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    gone: list[str] = []
    unavailable: list[str] = []
    checked: dict[str, set[Path]] = {}
    for path in documentation_files():
        for url in iter_checked_urls(path):
            checked.setdefault(url, set()).add(path)

    for url in sorted(checked):
        result = check_url(url, retries=args.retries, timeout=args.timeout)
        if result is None:
            continue
        kind, detail = result
        locations = ", ".join(str(path) for path in sorted(checked[url]))
        (gone if kind == "gone" else unavailable).append(
            f"{url} ({locations}): {detail}"
        )

    if unavailable:
        # Reported, not fatal. Nothing in this repository can fix someone
        # else's outage, and failing on it trains readers to skip the output.
        print("Hosts that could not be reached (not treated as failures):")
        for entry in unavailable:
            print(f"- {entry}")

    if gone:
        print("Documentation links that are gone:", file=sys.stderr)
        for entry in gone:
            print(f"- {entry}", file=sys.stderr)
        return 1

    print(
        f"Checked {len(checked)} stable external documentation links; "
        f"{len(checked) - len(unavailable)} reachable, {len(unavailable)} "
        "unreachable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
