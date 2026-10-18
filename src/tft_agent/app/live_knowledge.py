"""Version-verified web evidence, immutable revisions and bounded daily refresh.

Search discovers URLs only. Only fetched, allowlisted article bodies can become evidence.
All blocking I/O runs outside the UI/event loop; one lock deduplicates concurrent lookups
across the desktop's per-request event loops and its background refresh thread.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from urllib.robotparser import RobotFileParser

from tft_agent.app.pipeline import EvidenceRecord
from tft_agent.core.config import Settings


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str
    body: str
    published_at: str | None = None
    modified_at: str | None = None
    links: tuple[tuple[str, str], ...] = ()


class SearchAdapter(Protocol):
    def search(self, query: str) -> list[str]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> WebPage: ...


class BraveSearch:
    """Brave Web Search API; never promotes result descriptions to evidence."""

    def __init__(self, key: str, timeout: float, domains: list[str]) -> None:
        self.key, self.timeout, self.domains = key, timeout, domains

    def search(self, query: str) -> list[str]:
        sites = " OR ".join(f"site:{domain}" for domain in self.domains)
        params = urlencode({"q": f"{query} ({sites})" if sites else query, "count": 6})
        request = Request(
            "https://api.search.brave.com/res/v1/web/search?" + params,
            headers={"X-Subscription-Token": self.key, "Accept": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read(1_000_000))
        return [
            str(row["url"])
            for row in payload.get("web", {}).get("results", [])
            if isinstance(row, dict) and isinstance(row.get("url"), str)
        ][:6]


def allowed_url(url: str, domains: list[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    try:
        return (
            parsed.scheme == "https"
            and parsed.port in {None, 443}
            and not parsed.username
            and not parsed.password
            and any(host == d or host.endswith("." + d) for d in domains if d)
        )
    except ValueError:
        return False


class _Redirects(HTTPRedirectHandler):
    def __init__(self, domains: list[str]) -> None:
        self.domains = domains

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if not allowed_url(newurl, self.domains):
            raise ValueError("redirect outside trusted sources")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ArticleParser(HTMLParser):
    """Keep visible article text; scripts/navigation cannot supply patch evidence."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.visible: list[str] = []
        self.article: list[str] = []
        self.title: list[str] = []
        self.heading: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.anchor: str | None = None
        self.published: str | None = None
        self.modified: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a":
            self.anchor = attributes.get("href")
        if tag == "meta":
            name = attributes.get("property") or attributes.get("name")
            if name in {"article:published_time", "datePublished"}:
                self.published = attributes.get("content")
            if name in {"article:modified_time", "dateModified"}:
                self.modified = attributes.get("content")
        if tag not in {"meta", "link", "img", "br", "hr", "input", "source", "wbr"}:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.anchor = None
        if tag in self.stack:
            last = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[last:]

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if not value or any(
            t in self.stack
            for t in (
                "script",
                "style",
                "nav",
                "footer",
                "noscript",
                "aside",
            )
        ):
            return
        if "title" in self.stack:
            self.title.append(value)
        elif "head" not in self.stack:
            if "h1" in self.stack:
                self.heading.append(value)
            if self.anchor:
                self.links.append((self.anchor, value))
            self.visible.append(value)
            if "article" in self.stack or "main" in self.stack:
                self.article.append(value)


class TrustedPageFetcher:
    def __init__(self, domains: list[str], timeout: float) -> None:
        self.domains, self.timeout = domains, timeout
        self.opener = build_opener(_Redirects(domains))

    def _read(self, url: str, limit: int, *, html_only: bool = False) -> tuple[str, str]:
        if not allowed_url(url, self.domains):
            raise ValueError("source is not allowlisted")
        request = Request(url, headers={"User-Agent": "TFTKnowledgeBot/1.0"})
        with self.opener.open(request, timeout=self.timeout) as response:
            if html_only and response.headers.get_content_type() not in {
                "text/html",
                "application/xhtml+xml",
            }:
                raise ValueError("source did not return an HTML article")
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("page too large")
            return decode_page(raw, response.headers.get_content_charset()), str(response.geturl())

    def fetch(self, url: str) -> WebPage:
        if not allowed_url(url, self.domains):
            raise ValueError("source is not allowlisted")
        origin = urlparse(url)
        robots_url = f"https://{origin.netloc}/robots.txt"
        try:
            robots_text, _ = self._read(robots_url, 256_000)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            robots_text = "User-agent: *\nAllow: /"
        robots = RobotFileParser()
        robots.parse(robots_text.splitlines())
        if not robots.can_fetch("TFTKnowledgeBot", url):
            raise ValueError("robots disallows article retrieval")
        html, final_url = self._read(url, 2_000_000, html_only=True)
        parser = ArticleParser()
        parser.feed(html)
        return WebPage(
            final_url,
            " ".join(parser.heading or parser.title)[:300],
            "\n".join(parser.article or parser.visible),
            parser.published,
            parser.modified,
            tuple((urljoin(final_url, href), label) for href, label in parser.links),
        )


def decode_page(raw: bytes, header_charset: str | None = None) -> str:
    """Chinese sources commonly declare GBK in HTML rather than HTTP headers."""
    declared = re.search(
        rb"<meta\b[^>]*charset\s*=\s*['\"]?\s*([a-zA-Z0-9_-]+)",
        raw[:8192],
        re.I,
    )
    encoding = header_charset or (declared.group(1).decode("ascii") if declared else "utf-8")
    if encoding.lower() in {"gb2312", "gbk", "gb_2312-80"}:
        encoding = "gb18030"
    try:
        return raw.decode(encoding)
    except (UnicodeError, LookupError) as exc:
        raise ValueError("source body has an invalid or unsupported encoding") from exc


def page_patches(page: WebPage) -> set[str]:
    """Read version claims from the article, never from the search query."""
    number = r"(?<![\d.])(\d{1,2}\.\d{1,2}[a-z]?)(?![\d.a-z])"
    markers = re.findall(number, page.title, re.I)
    markers.extend(re.findall(r"(?:patch\s*|版本\s*[:：]?\s*)" + number, page.body[:4000], re.I))
    markers.extend(re.findall(number + r"\s*(?:版本|补丁)", page.body[:4000], re.I))
    return {value.lower() for value in markers}


def verified_patch(page: WebPage, patch: str) -> bool:
    """Exact explicit article patch only; no query-derived or inferred relabelling."""
    return page_patches(page) == {patch.lower()}


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    except ValueError:
        return None


@dataclass
class WebKnowledgeResult:
    evidence: list[EvidenceRecord] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    status: str = "cache"
    rejections: list[dict[str, str]] = field(default_factory=list)


class LiveKnowledge:
    def __init__(
        self,
        settings: Settings,
        search: SearchAdapter | None = None,
        fetcher: PageFetcher | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.path: Path = settings.database_path.resolve()
        key = settings.knowledge_search_api_key
        self.search = search or (
            BraveSearch(
                key.get_secret_value(),
                settings.knowledge_web_timeout_seconds,
                settings.knowledge_allowed_domains,
            )
            if key and key.get_secret_value().strip() and settings.knowledge_web_enabled
            else None
        )
        self.fetcher = fetcher or TrustedPageFetcher(
            settings.knowledge_allowed_domains,
            settings.knowledge_web_timeout_seconds,
        )
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS web_knowledge_revisions (
                revision_id TEXT PRIMARY KEY, url TEXT NOT NULL, patch TEXT NOT NULL,
                title TEXT NOT NULL, body TEXT NOT NULL, published_at TEXT,
                modified_at TEXT, retrieved_at TEXT NOT NULL, checked_at TEXT NOT NULL,
                content_hash TEXT NOT NULL, evidence_kind TEXT NOT NULL,
                UNIQUE(url, patch, content_hash)
            );
            CREATE TABLE IF NOT EXISTS web_knowledge_queries (
                query_key TEXT PRIMARY KEY, query_text TEXT NOT NULL, patch TEXT NOT NULL,
                terms_json TEXT NOT NULL, checked_at TEXT NOT NULL, status TEXT NOT NULL,
                urls_json TEXT NOT NULL
            );
            """)
            with connection:
                yield connection
        finally:
            connection.close()

    def lookup(
        self,
        query: str,
        patch: str,
        terms: list[str],
        *,
        needs_search: bool = True,
        concepts: list[str] | None = None,
    ) -> WebKnowledgeResult:
        with self.lock:
            return self._lookup(
                query, patch, terms, needs_search=needs_search, concepts=concepts or []
            )

    def _lookup(
        self,
        query: str,
        patch: str,
        terms: list[str],
        *,
        needs_search: bool,
        concepts: list[str],
    ) -> WebKnowledgeResult:
        now = self.clock()
        key = hashlib.sha256((patch + "\n" + query.strip()).encode()).hexdigest()
        with self._connect() as db:
            previous = db.execute(
                "SELECT * FROM web_knowledge_queries WHERE query_key=?",
                (key,),
            ).fetchone()
            cached_urls = json.loads(previous["urls_json"]) if previous else []
            rows = db.execute(
                "SELECT * FROM web_knowledge_revisions WHERE patch=? "
                "ORDER BY checked_at DESC, retrieved_at DESC, rowid DESC",
                (patch,),
            ).fetchall()
        selected: dict[str, sqlite3.Row] = {}
        seen: set[str] = set()
        for row in rows:
            if row["url"] in seen:
                continue
            seen.add(row["url"])
            text = row["title"] + "\n" + row["body"]
            if (
                (row["url"] in cached_urls and (not terms or all(t in text for t in terms)))
                or (terms and all(t in text for t in terms))
                or (not terms and any(t in text for t in concepts if len(t) >= 2))
            ):
                selected.setdefault(row["url"], row)
        hours = self.settings.knowledge_refresh_hours
        fresh = previous and now - datetime.fromisoformat(previous["checked_at"]) < timedelta(
            hours=hours if previous["status"] == "ok" else 0.25,
        )
        result = WebKnowledgeResult()
        if needs_search and not fresh:
            if self.search is None or not self.settings.knowledge_web_enabled:
                result.status = "unconfigured"
                result.gaps.append("联网补查未配置或已关闭；以下仅依据本地已验证资料。")
            else:
                result.status = "ok"
                accepted: list[str] = []
                try:
                    urls = self.search.search(f"云顶之弈 国服 {patch} {query}")
                    for url in list(dict.fromkeys([*urls, *cached_urls]))[:3]:
                        try:
                            if not allowed_url(url, self.settings.knowledge_allowed_domains):
                                continue
                            page = self.fetcher.fetch(url)
                            text = page.title + "\n" + page.body
                            if not verified_patch(page, patch):
                                versions = sorted(page_patches(page))[:5]
                                result.rejections.append(
                                    {
                                        "url": page.url,
                                        "reason": "patch_mismatch_or_unknown",
                                        "source_patches": ",".join(versions),
                                    }
                                )
                                if len(versions) == 1 and versions[0].startswith(
                                    patch.rstrip("abcdefghijklmnopqrstuvwxyz")
                                ):
                                    result.gaps.append(
                                        f"发现来源标注版本 {versions[0]}，本地启用版本为 {patch}；"
                                        "未将不同补丁资料混用。"
                                    )
                                continue
                            if (
                                not allowed_url(page.url, self.settings.knowledge_allowed_domains)
                                or len(page.body) < 80
                                or (terms and not all(t in text for t in terms))
                            ):
                                result.rejections.append(
                                    {"url": page.url, "reason": "body_or_topic_unverified"}
                                )
                                continue
                            published, modified = _date(page.published_at), _date(page.modified_at)
                            if any(
                                d and d > now + timedelta(days=1) for d in (published, modified)
                            ):
                                continue
                            self._store(page, patch, now)
                            accepted.append(page.url)
                        except (OSError, ValueError, TimeoutError) as exc:
                            result.rejections.append(
                                {
                                    "url": url,
                                    "reason": type(exc).__name__,
                                    "http_status": str(getattr(exc, "code", "")),
                                }
                            )
                            continue
                    if not accepted:
                        result.status = "no_verified_source"
                        result.gaps.append("未找到版本与玩法均可核验的新攻略；保留已有资料。")
                except HTTPError as exc:
                    result.status = "unavailable"
                    try:
                        code = json.loads(exc.read(4096)).get("error", {}).get("code")
                    except (ValueError, AttributeError):
                        code = None
                    if exc.code in {401, 403} or code == "SUBSCRIPTION_TOKEN_INVALID":
                        result.status = "authentication_failed"
                        result.gaps.append(
                            "搜索服务鉴权失败，请检查搜索 API 密钥，不能使用模型密钥。"
                        )
                    elif exc.code == 429:
                        result.status = "rate_limited"
                        result.gaps.append("搜索服务暂时限流；保留已有资料，稍后再补查。")
                    else:
                        result.gaps.append("联网补查暂时不可用；保留已有资料，不保证最新强度。")
                except (OSError, ValueError, TimeoutError, KeyError):
                    result.status = "unavailable"
                    result.gaps.append("联网补查暂时不可用；保留已有资料，不保证最新强度。")
                with self._connect() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO web_knowledge_queries VALUES (?,?,?,?,?,?,?)",
                        (
                            key,
                            query,
                            patch,
                            json.dumps(terms, ensure_ascii=False),
                            now.isoformat(),
                            result.status,
                            json.dumps(list(dict.fromkeys([*accepted, *cached_urls]))),
                        ),
                    )
                refreshed = self._lookup(query, patch, terms, needs_search=False, concepts=concepts)
                result.evidence = refreshed.evidence
                result.gaps.extend(refreshed.gaps)
                return result
        elif fresh and previous["status"] != "ok" and needs_search:
            result.status = str(previous["status"])
            result.gaps.append("最近一次补查未取得可验证的新资料，稍后会重试。")
        for row in list(selected.values())[:6]:
            age = now - datetime.fromisoformat(row["checked_at"])
            content_date = _date(row["modified_at"]) or _date(row["published_at"])
            date_known = bool(content_date)
            freshness = (
                "stale"
                if (
                    age >= timedelta(hours=hours)
                    or (content_date and now - content_date > timedelta(hours=hours * 2))
                )
                else ("checked" if date_known else "publication_unknown")
            )
            result.evidence.append(
                EvidenceRecord(
                    evidence_id="web:" + row["revision_id"],
                    source=urlparse(row["url"]).hostname or "web",
                    requested_patch=patch,
                    source_patch=patch,
                    patch_match="exact",
                    entity_type="strategy_document",
                    canonical_entity_id=row["revision_id"],
                    display_name=row["title"],
                    section="web_strategy",
                    semantic_content=row["body"][:12000],
                    relevance_score=0.8,
                    url=row["url"],
                    published_at=row["published_at"],
                    modified_at=row["modified_at"],
                    retrieved_at=row["checked_at"],
                    source_revision=row["revision_id"],
                    evidence_kind=row["evidence_kind"],
                    freshness=freshness,
                )
            )
            if freshness != "checked":
                result.gaps.append("部分攻略已过检查周期或未标明发布时间，不能据此确认最新强度。")
        return result

    def _store(self, page: WebPage, patch: str, now: datetime) -> None:
        content_hash = hashlib.sha256(
            json.dumps(
                [
                    page.title,
                    page.body,
                    page.published_at,
                    page.modified_at,
                ],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        revision = hashlib.sha256((page.url + patch + content_hash).encode()).hexdigest()
        host = urlparse(page.url).hostname or ""
        kind = (
            "mechanic_fact"
            if host
            in {
                "teamfighttactics.leagueoflegends.com",
                "lol.qq.com",
            }
            else "author_advice"
        )
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO web_knowledge_revisions VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(url,patch,content_hash) DO UPDATE SET checked_at=excluded.checked_at
            """,
                (
                    revision,
                    page.url,
                    patch,
                    page.title,
                    page.body,
                    page.published_at,
                    page.modified_at,
                    now.isoformat(),
                    now.isoformat(),
                    content_hash,
                    kind,
                ),
            )

    def refresh_due(self, patch: str) -> None:
        """Revisit tracked questions plus a discovery query; active patch only."""
        if self.search is None or not self.settings.knowledge_web_enabled:
            return
        with self.lock, self._connect() as db:
            queries = db.execute(
                "SELECT query_text,terms_json FROM web_knowledge_queries WHERE patch=? "
                "ORDER BY checked_at ASC LIMIT 12",
                (patch,),
            ).fetchall()
        jobs = [(row["query_text"], json.loads(row["terms_json"])) for row in queries]
        jobs.append(("最新阵容攻略 运营 转型", []))
        for query, terms in jobs:
            if self.stop_event.is_set():
                break
            self.lookup(query, patch, terms)

    def start(self, active_patch: Callable[[], str]) -> None:
        if (
            self.thread is not None
            or self.search is None
            or not self.settings.knowledge_web_enabled
        ):
            return
        self.stop_event.clear()

        def run() -> None:
            while not self.stop_event.is_set():
                try:
                    self.refresh_due(active_patch())
                except (OSError, ValueError, sqlite3.Error):
                    # Failed refresh must never affect the desktop or existing evidence.
                    pass
                self.stop_event.wait(3600)

        self.thread = threading.Thread(target=run, name="tft-knowledge-refresh", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
