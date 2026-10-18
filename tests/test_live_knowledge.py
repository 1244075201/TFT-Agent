from __future__ import annotations

import io
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tft_agent.app.live_knowledge import (
    ArticleParser,
    LiveKnowledge,
    WebPage,
    allowed_url,
    decode_page,
    verified_patch,
)
from tft_agent.core.config import Settings

URL = "https://datatft.com/guides/emerging"
BODY = "18.2版本 地狱火艾希玩法。" + "若获得合适条件，可以按作者建议过渡，注意经济和装备。" * 8


@pytest.mark.parametrize(
    "meta",
    [
        '<meta charset="gbk">',
        '<meta http-equiv="Content-Type" content="text/html; charset=gb2312">',
    ],
)
def test_chinese_page_encoding_from_meta(meta: str) -> None:
    content = meta + "<article>地狱火艾希攻略</article>"
    assert "地狱火艾希攻略" in decode_page(content.encode("gb18030"))


def test_invalid_encoding_is_not_silently_replaced_with_garbled_evidence() -> None:
    with pytest.raises(ValueError):
        decode_page(b"\xff\xfeinvalid", "utf-8")


class Search:
    calls = 0
    failure = False

    def search(self, query: str) -> list[str]:
        self.calls += 1
        if self.failure:
            raise OSError("unavailable")
        return [URL]


class Fetcher:
    page = WebPage(URL, "18.2 地狱火艾希攻略", BODY, "2026-09-20T00:00:00Z")

    def fetch(self, url: str) -> WebPage:
        return self.page


def live(tmp_path: Path) -> tuple[LiveKnowledge, Search, Fetcher, list[datetime]]:
    search, fetcher = Search(), Fetcher()
    now = [datetime(2026, 9, 21, tzinfo=UTC)]
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "knowledge.sqlite3",
        knowledge_search_api_key=None,
        knowledge_web_enabled=True,
    )
    return LiveKnowledge(settings, search, fetcher, lambda: now[0]), search, fetcher, now


def test_new_combination_cache_reuse_and_singleflight(tmp_path: Path) -> None:
    client, search, _, _ = live(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: client.lookup("地狱火艾希怎么玩", "18.2", ["地狱火", "艾希"]),
                range(4),
            )
        )
    assert search.calls == 1
    assert all(len(r.evidence) == 1 for r in results)
    evidence = results[0].evidence[0]
    assert evidence.source_patch == "18.2"
    assert evidence.evidence_kind == "author_advice"
    assert evidence.url == URL
    other = client.lookup("换一种问法", "18.2", ["地狱火", "艾希"], needs_search=False)
    assert other.evidence[0].source_revision == evidence.source_revision


def test_daily_refresh_immutable_revision_and_failure_preservation(tmp_path: Path) -> None:
    client, search, fetcher, now = live(tmp_path)
    first = client.lookup("地狱火艾希", "18.2", ["艾希"])
    now[0] += timedelta(days=1)
    client.lookup("地狱火艾希", "18.2", ["艾希"])
    with sqlite3.connect(client.path) as db:
        assert db.execute("SELECT count(*) FROM web_knowledge_revisions").fetchone()[0] == 1
    fetcher.page = WebPage(URL, fetcher.page.title, BODY + "新版建议转型", "2026-09-22")
    now[0] += timedelta(days=1)
    second = client.lookup("地狱火艾希", "18.2", ["艾希"])
    assert first.evidence[0].source_revision != second.evidence[0].source_revision
    with sqlite3.connect(client.path) as db:
        assert db.execute("SELECT count(*) FROM web_knowledge_revisions").fetchone()[0] == 2
    search.failure = True
    now[0] += timedelta(days=1)
    failed = client.lookup("地狱火艾希", "18.2", ["艾希"])
    assert failed.status == "unavailable"
    assert failed.evidence[0].source_revision == second.evidence[0].source_revision
    assert failed.evidence[0].freshness == "stale"


@pytest.mark.parametrize(
    "page",
    [
        WebPage(URL, "18.1 攻略", BODY.replace("18.2", "18.1")),
        WebPage(URL, "新玩法", BODY.replace("18.2版本", "")),
        WebPage(URL, "18.2", "搜索摘要 地狱火艾希"),
        WebPage("https://untrusted.example/guide", "18.2", BODY),
        WebPage(URL, "18.2", BODY.replace("艾希", "其他英雄")),
        WebPage(URL, "18.2", BODY, "2099-01-01"),
    ],
)
def test_rejects_unverified_sources(tmp_path: Path, page: WebPage) -> None:
    client, _, fetcher, _ = live(tmp_path)
    fetcher.page = page
    result = client.lookup("地狱火艾希", "18.2", ["地狱火", "艾希"])
    assert result.evidence == []
    assert result.status == "no_verified_source"


def test_unknown_publication_is_explicit_and_not_inferred(tmp_path: Path) -> None:
    client, _, fetcher, _ = live(tmp_path)
    fetcher.page = WebPage(URL, "18.2", BODY)
    result = client.lookup("地狱火艾希", "18.2", ["艾希"])
    assert result.evidence[0].published_at is None
    assert result.evidence[0].freshness == "publication_unknown"
    assert result.gaps


def test_no_provider_is_honest(tmp_path: Path) -> None:
    client = LiveKnowledge(
        Settings(
            _env_file=None,
            database_path=tmp_path / "empty.sqlite3",
            knowledge_search_api_key=None,
        )
    )
    result = client.lookup("新玩法", "18.2", [])
    assert result.status == "unconfigured"
    assert result.gaps


def test_search_authentication_error_is_actionable_without_exposing_key(tmp_path: Path) -> None:
    client, _, _, _ = live(tmp_path)

    class InvalidKey:
        def search(self, query: str) -> list[str]:
            raise HTTPError(
                "https://api.search.brave.com/res/v1/web/search",
                422,
                "Unprocessable",
                Message(),
                io.BytesIO(b'{"error":{"code":"SUBSCRIPTION_TOKEN_INVALID"}}'),
            )

    client.search = InvalidKey()
    result = client.lookup("地狱火艾希", "18.2", ["艾希"])
    assert result.status == "authentication_failed"
    assert any("不能使用模型密钥" in gap for gap in result.gaps)


def test_hotfix_mismatch_is_explicit_without_relabelling(tmp_path: Path) -> None:
    client, _, fetcher, _ = live(tmp_path)
    fetcher.page = WebPage(URL, "18.2b 地狱火艾希", BODY.replace("18.2", "18.2b"))
    result = client.lookup("地狱火艾希", "18.2", ["艾希"])
    assert not result.evidence
    assert any("18.2b" in gap and "未将不同补丁资料混用" in gap for gap in result.gaps)
    assert result.rejections[0]["reason"] == "patch_mismatch_or_unknown"


def test_article_body_not_scripts_or_navigation_supplies_patch() -> None:
    parser = ArticleParser()
    parser.feed(
        "<title>攻略</title><nav>18.2版本</nav><script>18.2版本</script>"
        "<article>18.1版本 老攻略</article>"
    )
    page = WebPage(URL, " ".join(parser.title), " ".join(parser.article))
    assert not verified_patch(page, "18.2")
    assert verified_patch(page, "18.1")


def test_combat_decimals_do_not_become_patch_numbers() -> None:
    assert verified_patch(WebPage(URL, "18.2攻略", "攻击速度1.5，18.2版本攻略"), "18.2")
    assert not verified_patch(WebPage(URL, "18.2b攻略", "18.2b版本攻略"), "18.2")


def test_conflicting_sources_remain_separate(tmp_path: Path) -> None:
    client, _, _, _ = live(tmp_path)

    class ConflictingSources:
        def search(self, query: str) -> list[str]:
            return [URL, "https://tftacademy.com/guide"]

        def fetch(self, url: str) -> WebPage:
            conclusion = "建议优先尝试" if url == URL else "不建议无条件使用"
            return WebPage(url, "18.2 地狱火艾希", BODY + conclusion, "2026-09-20")

    provider = ConflictingSources()
    client.search = provider
    client.fetcher = provider
    result = client.lookup("地狱火艾希", "18.2", ["地狱火", "艾希"])
    assert len(result.evidence) == 2
    assert len({item.url for item in result.evidence}) == 2
    assert any("不建议无条件使用" in str(item.semantic_content) for item in result.evidence)


def test_daily_refresh_rechecks_questions_and_discovers_without_duplicate_calls(
    tmp_path: Path,
) -> None:
    client, search, _, now = live(tmp_path)
    client.lookup("地狱火艾希", "18.2", ["艾希"])
    client.refresh_due("18.2")
    assert search.calls == 2  # Existing question is fresh; discovery is new.
    client.refresh_due("18.2")
    assert search.calls == 2
    now[0] += timedelta(days=1)
    client.refresh_due("18.2")
    assert search.calls == 4
    # Data from the old patch is never used for the new patch.
    assert client.lookup("地狱火艾希", "18.3", ["艾希"]).evidence == []


def test_oldest_revision_cannot_reappear_when_new_revision_changes_topic(tmp_path: Path) -> None:
    client, _, fetcher, now = live(tmp_path)
    client.lookup("地狱火艾希", "18.2", ["艾希"])
    now[0] += timedelta(days=1)
    fetcher.page = WebPage(URL, "18.2其他玩法", BODY.replace("艾希", "其他英雄"))
    client.lookup("其他英雄", "18.2", ["其他英雄"])
    assert client.lookup("艾希追问", "18.2", ["艾希"], needs_search=False).evidence == []


@pytest.mark.parametrize(
    "url",
    [
        "http://datatft.com/a",
        "https://datatft.com.evil.example/a",
        "https://evil.example/a",
        "https://user:password@datatft.com/a",
        "https://datatft.com:8080/a",
    ],
)
def test_source_boundary(url: str) -> None:
    assert not allowed_url(url, ["datatft.com"])
