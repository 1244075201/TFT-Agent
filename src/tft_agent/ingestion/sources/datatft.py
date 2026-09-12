"""Polite, snapshot-oriented DataTFT adapter with no runtime web access."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from tft_agent.tft_mcp_server.schemas import CompStatsRecord, DataSnapshotMetadata

SOURCE = "datatft"
API_BASE_URL = "https://api.datatft.com"
VERSION_URL = "https://www.datatft.com/explorer"
USER_AGENT = "TFT-Agent/0.2 (+local personal research; low-frequency snapshot)"
MAX_VERSION_ASSETS = 16

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawHTTPResponse:
    endpoint_type: str
    request_url: str
    request_key: str | None
    retrieved_at: datetime
    http_status: int
    etag: str | None
    content_type: str | None
    body: str


class DataTFTAccessDenied(RuntimeError):
    pass


class DataTFTSource:
    """Fetch public pages only after a conservative robots.txt check."""

    def __init__(self, url: str, server: str = "cn", timeout_seconds: float = 30) -> None:
        self.url = url
        self.server = server
        self.timeout_seconds = timeout_seconds

    def fetch(self, etag: str | None = None) -> tuple[str, str | None]:
        self._assert_robots_allows()
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}
        if etag:
            headers["If-None-Match"] = etag
        request = Request(self.url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8"), response.headers.get("ETag")
        except HTTPError as exc:
            if exc.code in {401, 403, 429}:
                raise DataTFTAccessDenied(
                    f"DataTFT refused automated access: HTTP {exc.code}"
                ) from exc
            raise RuntimeError(f"DataTFT request failed: HTTP {exc.code}") from exc
        except (URLError, TimeoutError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"DataTFT request failed: {exc}") from exc

    def fetch_current(
        self,
        expected_game_patch: str | None = None,
        request_interval_seconds: float = 0.75,
    ) -> tuple[str, str | None]:
        """Fetch the same public composition data used by the DataTFT web frontend."""

        self._assert_robots_allows()
        game_patch = self.discover_current_patch()
        if expected_game_patch is not None and expected_game_patch != game_patch:
            raise ValueError(
                f"Configured patch {expected_game_patch!r} does not match DataTFT {game_patch!r}"
            )
        listing = self._post_json("/team/comps", {"season": ""})
        raw_list = listing.get("data", {}).get("list")
        if not isinstance(raw_list, list) or not raw_list:
            raise ValueError("DataTFT composition endpoint returned no list")
        raw_ids = [
            str(item.get("id"))
            for item in raw_list
            if isinstance(item, dict) and item.get("id") is not None
        ]
        if len(raw_ids) != len(raw_list):
            raise ValueError("DataTFT composition list contains an entry without an ID")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("DataTFT composition list contains duplicate IDs")
        comps: list[dict[str, Any]] = []
        for raw_comp in raw_list:
            if not isinstance(raw_comp, dict) or raw_comp.get("id") is None:
                continue
            time.sleep(request_interval_seconds)
            response = self._post_json("/team/strategy", {"id": raw_comp["id"]})
            detail = response.get("data")
            comps.append(self._canonical_live_comp(raw_comp, detail))
        payload = json.dumps(
            {
                "game_patch": game_patch,
                "sample_scope": (
                    "DataTFT CN current composition tier and strategy; "
                    "performance rates are not exposed by these endpoints"
                ),
                "comps": comps,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return payload, None

    def fetch_current_with_raw(
        self,
        responses: list[RawHTTPResponse],
        expected_game_patch: str | None = None,
        request_interval_seconds: float = 0.75,
    ) -> tuple[str, str | None]:
        """Fetch a canonical snapshot while retaining every source HTTP response."""

        self._assert_robots_allows()
        game_patch = self.discover_current_patch(responses)
        if expected_game_patch is not None and expected_game_patch != game_patch:
            raise ValueError(
                f"Configured patch {expected_game_patch!r} does not match DataTFT {game_patch!r}"
            )
        listing = self._post_json(
            "/team/comps", {"season": ""}, responses, "COMPOSITION_LIST"
        )
        raw_list = listing.get("data", {}).get("list")
        if not isinstance(raw_list, list) or not raw_list:
            raise ValueError("DataTFT composition endpoint returned no list")
        raw_ids = [
            str(item.get("id"))
            for item in raw_list
            if isinstance(item, dict) and item.get("id") is not None
        ]
        if len(raw_ids) != len(raw_list):
            raise ValueError("DataTFT composition list contains an entry without an ID")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("DataTFT composition list contains duplicate IDs")
        comps: list[dict[str, Any]] = []
        for raw_comp in raw_list:
            if not isinstance(raw_comp, dict) or raw_comp.get("id") is None:
                continue
            time.sleep(request_interval_seconds)
            response = self._post_json(
                "/team/strategy", {"id": raw_comp["id"]}, responses,
                "COMPOSITION_DETAIL", str(raw_comp["id"]),
            )
            detail = response.get("data")
            if not isinstance(detail, dict):
                raise ValueError(f"DataTFT detail {raw_comp['id']} returned no object")
            comps.append(self._canonical_live_comp(raw_comp, detail))
        payload = json.dumps(
            {
                "game_patch": game_patch,
                "sample_scope": (
                    "DataTFT CN current composition tier and strategy; "
                    "performance rates are not exposed by these endpoints"
                ),
                "comps": comps,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return payload, None

    def discover_current_patch(
        self, responses: list[RawHTTPResponse] | None = None
    ) -> str:
        """Read DataTFT's structured frontend version, with legacy text fallbacks."""

        request = Request(VERSION_URL, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                html = response.read().decode("utf-8")
                if responses is not None:
                    responses.append(
                        _raw_response("VERSION_PAGE", VERSION_URL, None, response, html)
                    )
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Failed to read DataTFT version page: {exc}") from exc
        direct, source = _extract_patch_with_source(html)
        if direct is not None:
            logger.info("Detected DataTFT patch %s from %s", direct, source)
            return direct

        pending = _javascript_urls(html, VERSION_URL, html=True)
        visited: set[str] = set()
        while pending and len(visited) < MAX_VERSION_ASSETS:
            script_url = pending.pop(0)
            if script_url in visited:
                continue
            visited.add(script_url)
            try:
                with urlopen(
                    Request(script_url, headers={"User-Agent": USER_AGENT}),
                    timeout=self.timeout_seconds,
                ) as response:
                    script = response.read().decode("utf-8")
                    if responses is not None:
                        responses.append(
                            _raw_response("VERSION_PAGE", script_url, None, response, script)
                        )
            except (HTTPError, URLError, TimeoutError, UnicodeDecodeError):
                continue
            detected, source = _extract_patch_with_source(script)
            if detected is not None:
                logger.info(
                    "Detected DataTFT patch %s from %s at %s", detected, source, script_url
                )
                return detected
            for dependency in _javascript_urls(script, script_url):
                if dependency not in visited and dependency not in pending:
                    pending.append(dependency)
        raise ValueError("DataTFT frontend did not expose a recognizable current patch")

    def _post_json(
        self, path: str, body: dict[str, Any],
        responses: list[RawHTTPResponse] | None = None,
        endpoint_type: str = "UNKNOWN", request_key: str | None = None,
    ) -> dict[str, Any]:
        url = API_BASE_URL + path
        encoded = json.dumps(body).encode("utf-8")
        request = Request(
            url,
            data=encoded,
            method="POST",
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Origin": "https://www.datatft.com",
                "Referer": self.url,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                if responses is not None:
                    responses.append(
                        _raw_response(endpoint_type, url, request_key, response, raw_body)
                    )
                value = json.loads(raw_body)
        except HTTPError as exc:
            if responses is not None:
                error_body = exc.read().decode("utf-8", errors="replace")
                responses.append(RawHTTPResponse(
                    endpoint_type=endpoint_type, request_url=url,
                    request_key=request_key, retrieved_at=datetime.now(UTC),
                    http_status=exc.code, etag=exc.headers.get("ETag"),
                    content_type=exc.headers.get("Content-Type"), body=error_body,
                ))
            if exc.code in {401, 403, 429}:
                raise DataTFTAccessDenied(
                    f"DataTFT API refused public frontend request: HTTP {exc.code}"
                ) from exc
            raise RuntimeError(f"DataTFT API request failed: HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"DataTFT API request failed: {exc}") from exc
        if not isinstance(value, dict) or value.get("code") != 1:
            raise ValueError("DataTFT API returned an unsuccessful response")
        return value

    @staticmethod
    def _canonical_live_comp(summary: dict[str, Any], detail: Any) -> dict[str, Any]:
        detail = detail if isinstance(detail, dict) else {}
        equips = detail.get("equips") if isinstance(detail.get("equips"), list) else []
        core_units = [
            str(item["heroName"]).strip()
            for item in equips
            if isinstance(item, dict) and item.get("heroName")
        ]
        item_plans: list[str] = []
        for item in equips:
            if not isinstance(item, dict) or not item.get("heroName"):
                continue
            raw_sets = item.get("equips")
            sets: list[Any] = raw_sets if isinstance(raw_sets, list) else []
            for position, identifiers in enumerate(sets, start=1):
                if isinstance(identifiers, list):
                    item_plans.append(
                        f"{item['heroName']}方案{position}:" + "/".join(map(str, identifiers))
                    )
        level_plan = [
            f"{label}:{detail[key]}"
            for label, key in (("前期", "earlyDesc"), ("中期", "midDesc"), ("成型", "finalDesc"))
            if detail.get(key)
        ]
        return {
            "id": str(summary["id"]),
            "name": str(detail.get("title") or summary.get("title") or summary["id"]),
            "tier": summary.get("tier"),
            "core_units": core_units,
            "recommended_items": item_plans,
            # ``equipId`` is an equipment/lineup presentation field, not an augment.
            # Keep this empty unless DataTFT exposes an explicit recommendation field.
            "augments": [],
            "level_plan": [
                *level_plan,
                *([f"核心:{detail['core']}"] if detail.get("core") else []),
                *([f"开局条件:{detail['open']}"] if detail.get("open") else []),
            ],
        }

    def parse(
        self,
        payload: str,
        source_url: str | None = None,
        etag: str | None = None,
        retrieved_at: datetime | None = None,
    ) -> tuple[DataSnapshotMetadata, list[CompStatsRecord]]:
        raw = self._extract_json(payload)
        patch = str(raw.get("game_patch") or raw.get("patch") or "").strip()
        comps = raw.get("comps") or raw.get("data")
        if not patch or not isinstance(comps, list):
            raise ValueError("DataTFT snapshot requires game_patch and a comps array")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        timestamp = retrieved_at or datetime.now(UTC)
        metadata = DataSnapshotMetadata(
            snapshot_id=f"datatft:{self.server}:{patch}:{digest[:16]}",
            server=self.server,
            game_patch=patch,
            sample_scope=_optional_text(raw.get("sample_scope")),
            source_url=source_url or self.url,
            retrieved_at=timestamp,
            content_hash=digest,
            etag=etag,
        )
        records = [self._normalize_comp(item, metadata) for item in comps]
        if not records:
            raise ValueError("DataTFT snapshot contains no compositions")
        return metadata, records

    def _assert_robots_allows(self) -> None:
        parsed = urlparse(self.url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            request = Request(robots_url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=10) as response:
                robots = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 404:
                return
            raise DataTFTAccessDenied("Cannot verify DataTFT robots policy") from exc
        except (URLError, TimeoutError) as exc:
            raise DataTFTAccessDenied("Cannot verify DataTFT robots policy") from exc
        path = parsed.path or "/"
        active = False
        for line in robots.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                active = value in {"*", "tft-agent"}
            elif active and key == "disallow" and value and path.startswith(value):
                raise DataTFTAccessDenied(f"DataTFT robots.txt disallows {path}")

    @staticmethod
    def _extract_json(payload: str) -> dict[str, Any]:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            match = re.search(
                r'<script[^>]+id=["\'](?:__DATATFT_DATA__|__NEXT_DATA__)["\'][^>]*>(.*?)</script>',
                payload,
                re.DOTALL | re.IGNORECASE,
            )
            if match is None:
                raise ValueError(
                    "DataTFT page has no supported public snapshot; use --input with an "
                    "exported JSON file"
                ) from None
            value = json.loads(unescape(match.group(1)))
        if not isinstance(value, dict):
            raise ValueError("DataTFT snapshot root must be an object")
        if "props" in value:
            value = value.get("props", {}).get("pageProps", value)
        return dict(value)

    @staticmethod
    def _normalize_comp(item: Any, metadata: DataSnapshotMetadata) -> CompStatsRecord:
        if not isinstance(item, dict):
            raise ValueError("each DataTFT comp must be an object")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError("each DataTFT comp requires a name")
        comp_id = str(item.get("id") or hashlib.sha256(name.encode()).hexdigest()[:16])
        return CompStatsRecord(
            comp_id=comp_id,
            name=name,
            tier=_optional_text(item.get("tier")),
            average_place=_optional_float(item.get("average_place")),
            play_rate=_rate(item.get("play_rate")),
            top4_rate=_rate(item.get("top4_rate")),
            win_rate=_rate(item.get("win_rate")),
            core_units=_strings(item.get("core_units")),
            recommended_items=_strings(item.get("recommended_items")),
            augments=_strings(item.get("augments")),
            level_plan=_strings(item.get("level_plan")),
            snapshot=metadata,
        )


def _optional_text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value not in {None, ""} else None


def _rate(value: Any) -> float | None:
    percent = isinstance(value, str) and value.strip().endswith("%")
    normalized = value.strip().removesuffix("%") if isinstance(value, str) else value
    number = _optional_float(normalized)
    if number is not None and (percent or number > 1):
        number /= 100
    return number


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in value] if isinstance(value, list) else []


def _extract_current_patch(content: str) -> str | None:
    return _extract_patch_with_source(content)[0]


def _extract_patch_with_source(content: str) -> tuple[str | None, str | None]:
    structured = re.search(
        r'(?:const|let|var)\s+[^;\r\n]{0,256}?\b\w+="([0-9]+\.[0-9]+[a-z]?)",'
        r'\w+=\[\{value:"\1"\}',
        content,
        re.IGNORECASE,
    )
    if structured is not None:
        return structured.group(1), "js_bundle_structured_versions"
    for pattern in (
        r"Ver:\s*([0-9]+\.[0-9]+[a-z]?)",
        r"数据版本\s*([0-9]+\.[0-9]+[a-z]?)",
    ):
        match = re.search(pattern, content, re.IGNORECASE)
        if match is not None:
            return match.group(1), "legacy_pattern"
    return None, None


def _javascript_urls(content: str, base_url: str, html: bool = False) -> list[str]:
    if html:
        sources = re.findall(
            r'<script[^>]+src=["\']([^"\']+\.js)["\']', content, re.IGNORECASE
        )
    else:
        sources = re.findall(
            r'["\']((?:\.?\.?/|/)?assets/[A-Za-z0-9_.-]+\.js|\./[A-Za-z0-9_.-]+\.js)["\']',
            content,
        )
    origin = urlparse(VERSION_URL)
    urls: list[str] = []
    for source in sources:
        normalized = "/" + source if source.startswith("assets/") else source
        candidate = urljoin(base_url, normalized)
        parsed = urlparse(candidate)
        if parsed.scheme == origin.scheme and parsed.netloc == origin.netloc:
            if candidate not in urls:
                urls.append(candidate)
    return urls


def _raw_response(
    endpoint_type: str, request_url: str, request_key: str | None,
    response: Any, body: str,
) -> RawHTTPResponse:
    return RawHTTPResponse(
        endpoint_type=endpoint_type,
        request_url=request_url,
        request_key=request_key,
        retrieved_at=datetime.now(UTC),
        http_status=int(response.status),
        etag=response.headers.get("ETag"),
        content_type=response.headers.get("Content-Type"),
        body=body,
    )
