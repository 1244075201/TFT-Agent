import pytest

from tft_agent.core.schemas import NumericClaim, NumericClaimStatus, NumericDerivation
from tft_agent.llm.numeric_claims import NumericClaimValidator


def test_direct_probability_percent_normalization_is_supported() -> None:
    claim = NumericClaim(
        value=10,
        unit="percent",
        meaning="shop odds",
        evidence_ref="tool.call-1.data.odds.cost_4",
    )

    result = (
        NumericClaimValidator().validate([claim], {"tool.call-1.data.odds.cost_4": 0.10}).results[0]
    )

    assert result.status == NumericClaimStatus.SUPPORTED


def test_mismatched_and_invalid_references_are_distinct() -> None:
    mismatched = NumericClaim(
        value=15,
        unit="percent",
        meaning="wrong odds",
        evidence_ref="tool.call-1.data.odds.cost_4",
    )
    invalid = NumericClaim(
        value=10,
        unit="percent",
        meaning="invented call",
        evidence_ref="tool.nonexistent.data.odds.cost_4",
    )

    results = (
        NumericClaimValidator()
        .validate([mismatched, invalid], {"tool.call-1.data.odds.cost_4": 0.10})
        .results
    )

    assert results[0].status == NumericClaimStatus.MISMATCHED
    assert results[1].status == NumericClaimStatus.INVALID_REFERENCE


def test_unsupported_claim_and_whitelisted_derivation() -> None:
    unsupported = NumericClaim(value=20, unit="gold", meaning="roll threshold")
    derived = NumericClaim(
        value=8,
        unit="level",
        meaning="next level",
        derivation=NumericDerivation(operation="add", inputs=["game_state.level", 1]),
    )

    results = (
        NumericClaimValidator().validate([unsupported, derived], {"game_state.level": 7}).results
    )

    assert results[0].status == NumericClaimStatus.UNSUPPORTED
    assert results[1].status == NumericClaimStatus.SUPPORTED


def test_answer_numeric_literals_must_be_covered_by_declared_claims() -> None:
    supported = NumericClaim(
        value=35,
        unit="hp",
        meaning="current hp",
        evidence_ref="game_state.hp",
    )

    report = NumericClaimValidator().validate_answer(
        "You have 35 HP; do not roll to 20 gold.",
        [supported],
        {"game_state.hp": 35},
    )

    assert report.results[0].status == NumericClaimStatus.SUPPORTED
    assert report.results[1].claim.value == "20"
    assert report.results[1].status == NumericClaimStatus.UNSUPPORTED


def test_percent_claim_covers_percent_and_decimal_renderings() -> None:
    claim = NumericClaim(
        value=10,
        unit="percent",
        meaning="shop odds",
        evidence_ref="tool.call-1.data.odds.cost_4",
    )

    report = NumericClaimValidator().validate_answer(
        "The probability is 10% (0.10).",
        [claim],
        {"tool.call-1.data.odds.cost_4": 0.10},
    )

    assert report.valid is True


@pytest.mark.parametrize("text", ["还能撑三五轮", "建议留十来金币", "大约一两轮"])
def test_unicode_numeric_words_require_declared_provenance(text: str) -> None:
    report = NumericClaimValidator().validate_answer(text, [], {})

    assert not report.valid
    assert all(result.status == NumericClaimStatus.UNSUPPORTED for result in report.results)
