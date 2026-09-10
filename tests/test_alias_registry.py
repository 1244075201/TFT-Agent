from __future__ import annotations

from tft_agent.app.aliases import AliasEntry, AliasRegistry, EntityTextNormalizer


def test_normalizer_handles_nfkc_case_space_and_punctuation() -> None:
    assert EntityTextNormalizer.normalize(" 蓝 Ｂｕｆｆ！") == "蓝buff"
    assert EntityTextNormalizer.normalize("蓝 Buff") == "蓝buff"
    assert EntityTextNormalizer.normalize("蓝BUFF") == "蓝buff"


def test_registry_materializes_scope_and_rejects_missing_conflicts() -> None:
    entries = [
        AliasEntry(aliases=["奶妈"], entity_type="champion", canonical_name="索拉卡",
                   scope="global"),
        AliasEntry(aliases=["天使"], entity_type="composition", canonical_name="森林天使",
                   canonical_id="comp", scope="set", set_number=18),
        AliasEntry(aliases=["奶妈"], entity_type="composition", canonical_name="奶妈阵容",
                   canonical_id="other", scope="set", set_number=18),
        AliasEntry(aliases=["大头"], entity_type="champion", canonical_name="黑默丁格",
                   scope="global"),
    ]
    registry = AliasRegistry(entries)
    accepted = registry.materialize([
        ("unit", "champion", "索拉卡", "test"),
        ("comp", "composition", "森林天使", "test"),
        ("other", "composition", "奶妈阵容", "test"),
    ], 18, "18.1d")
    assert [(item.alias, item.canonical_id) for item in accepted] == [("天使", "comp")]
    assert {item.reason for item in registry.rejections()} == {
        "alias_conflict", "target_missing",
    }


def test_registry_default_contains_only_reviewed_effective_initial_set() -> None:
    registry = AliasRegistry.load_default()
    approved = [entry for entry in registry.entries if entry.review_status == "approved"]
    assert sum(entry.entity_type == "champion" for entry in approved) == 10
    assert sum(entry.entity_type == "item" for entry in approved) == 18
    assert sum(entry.entity_type == "composition" for entry in approved) == 3
    assert not any("generated_prefix" in entry.notes or "generated_suffix" in entry.notes
                   for entry in registry.entries)
