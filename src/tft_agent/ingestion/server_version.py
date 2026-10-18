"""Conservative CN standard-mode version discovery from official article bodies."""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

from tft_agent.app.live_knowledge import (
    BraveSearch,
    PageFetcher,
    SearchAdapter,
    TrustedPageFetcher,
    WebPage,
    allowed_url,
)
from tft_agent.core.config import Settings
from tft_agent.core.update_state import ServerVersion, UpdateState, patch_order


def parse_announcement(page: WebPage, now: datetime) -> ServerVersion:
    result = ServerVersion(checked_at=now, source_url=page.url)
    if not allowed_url(page.url, ["lol.qq.com"]):
        return result
    text = page.title + "\n" + page.body
    if "云顶之弈" not in text or any(word in page.title for word in (
        "金铲铲", "测试服", "PBE", "预告", "抢先", "返场", "狂暴", "双人",
    )):
        return result
    versions = set(re.findall(
        r"(?<![\d.])(\d{1,2}\.\d{1,2}[a-z]?)\s*(?:版本|补丁)", text, re.I,
    ))
    versions.update(re.findall(r"(?:版本|补丁)\s*[:：]?\s*(\d+\.\d+[a-z]?)", text, re.I))
    # A single explicit TFT identity is required; League patch/build numbers aren't aliases.
    title_versions = re.findall(r"云顶之弈[^\d\n]{0,12}(\d+\.\d+[a-z]?)", page.title, re.I)
    title_versions.extend(re.findall(r"(\d+\.\d+[a-z]?)\s*云顶之弈", page.title, re.I))
    if title_versions:
        versions = set(title_versions)
    if len(versions) != 1:
        result.reason = "官方公告未明确唯一云顶补丁（热修复编号不得推测）"
        return result
    patch = versions.pop().lower()
    when = re.search(
        r"(?:已于|将于|将在|于)\s*(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"
        r"\s*(凌晨|上午|下午|晚上)?\s*(\d{1,2})[:：点时](\d{1,2})?", text,
    )
    effective: datetime | None = None
    if when:
        publication = re.search(r"(\d{4})-\d\d-\d\d[ T]\d\d:\d\d", page.body)
        year_text = when[1] or ((page.published_at or "")[:4]) or (
            publication[1] if publication else ""
        )
        if not year_text.isdigit():
            return result
        year, month, day, hour = map(int, (year_text, when[2], when[3], when[5]))
        if when[4] in {"下午", "晚上"} and hour < 12:
            hour += 12
        try:
            effective = datetime(year, month, day, hour, int(when[6] or 0),
                                 tzinfo=timezone(timedelta(hours=8))).astimezone(UTC)
        except ValueError:
            return result
    if effective is None and any(word in text for word in ("现已上线", "更新完成", "已更新")):
        try:
            effective = datetime.fromisoformat((page.published_at or "").replace("Z", "+00:00"))
            if effective.tzinfo is None:
                effective = None
        except ValueError:
            pass
    if effective is None or effective > now or now - effective > timedelta(days=45):
        result.reason = "公告生效时间未核验或尚未生效"
        return result
    if ("停机" in text and "不停机" not in text
            and not any(word in text for word in ("更新完成", "现已上线", "维护结束"))):
        result.reason = "停机更新尚无完成依据"
        return result
    return result.model_copy(update={"game_patch": patch, "status": "confirmed",
                                     "effective_at": effective, "reason": "国服官方生效公告"})


class OfficialServerVersionResolver:
    def __init__(self, settings: Settings, search: SearchAdapter | None = None,
                 fetcher: PageFetcher | None = None) -> None:
        key = settings.knowledge_search_api_key
        self.search = search or (BraveSearch(key.get_secret_value(), 10, ["lol.qq.com"])
                                 if key and key.get_secret_value() else None)
        self.fetcher = fetcher or TrustedPageFetcher(["lol.qq.com"], 10)
        self.state = UpdateState(settings.database_path)
        self.urls = settings.server_version_urls

    def resolve(self) -> ServerVersion:
        now = datetime.now(UTC)
        urls = list(self.urls)
        # The official category page is a discovery index, never version evidence.
        try:
            index = self.fetcher.fetch("https://lol.qq.com/gicp/news/663/2/1983/1.html")
            urls.extend(url for url, label in index.links
                        if "云顶" in label and ("版本" in label or "不停机" in label))
        except (OSError, ValueError, TimeoutError):
            pass
        try:
            if self.search:
                urls.extend(self.search.search(
                    f"云顶之弈 {now.year}年{now.month}月 国服 版本 不停机 更新公告"
                ))
        except (OSError, ValueError, TimeoutError):
            pass
        verified: list[ServerVersion] = []
        rejected: list[ServerVersion] = []
        for url in list(dict.fromkeys(urls))[:6]:
            try:
                if not allowed_url(url, ["lol.qq.com"]):
                    continue
                page = self.fetcher.fetch(url)
                result = parse_announcement(page, now)
                self.state.put("announcement:" + url, {
                    "title": page.title, "body": page.body, "url": page.url,
                    "checked_at": now.isoformat(), "result": result.model_dump(mode="json"),
                })
                if result.status == "confirmed":
                    verified.append(result)
                else:
                    rejected.append(result)
            except (OSError, ValueError, TimeoutError):
                continue
        if not verified:
            if rejected:
                return rejected[0].model_copy(update={
                    "reason": "已自动检查官方公告；" + rejected[0].reason,
                })
            return ServerVersion(
                checked_at=now, reason="已尝试自动核验，但未获取到可核验的官方公告",
            )
        return max(verified, key=lambda value: patch_order(value.game_patch or ""))
