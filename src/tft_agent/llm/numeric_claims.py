"""Deterministic numeric-claim provenance validation and safe repair inputs."""

import re
import unicodedata
from functools import reduce
from math import isclose
from operator import add, mul, sub, truediv

from pydantic import Field

from tft_agent.core.schemas import (
    NumericClaim,
    NumericClaimStatus,
    StrictModel,
)
from tft_agent.llm.evidence import EvidenceRegistry


class NumericClaimValidation(StrictModel):
    """Validation result for one provider-declared claim."""

    claim: NumericClaim
    status: NumericClaimStatus
    evidence_value: int | float | str | None = None


class NumericValidationReport(StrictModel):
    """Aggregate deterministic validation report for one generated answer."""

    results: list[NumericClaimValidation] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(result.status == NumericClaimStatus.SUPPORTED for result in self.results)

    @property
    def failures(self) -> list[NumericClaimValidation]:
        return [result for result in self.results if result.status != NumericClaimStatus.SUPPORTED]


class NumericClaimValidator:
    """Validate references and arithmetic without understanding TFT meanings."""

    def validate(
        self,
        claims: list[NumericClaim],
        registry: EvidenceRegistry,
    ) -> NumericValidationReport:
        return NumericValidationReport(
            results=[self._validate_claim(claim, registry) for claim in claims]
        )

    def validate_answer(
        self,
        answer: str,
        claims: list[NumericClaim],
        registry: EvidenceRegistry,
    ) -> NumericValidationReport:
        """Validate provenance plus generic numeric-literal declaration completeness."""

        report = self.validate(claims, registry)
        declared = {form for claim in claims for form in _claim_literal_forms(claim)}
        for token in _numeric_literals(answer):
            if _canonical_literal(token) in declared:
                continue
            report.results.append(
                NumericClaimValidation(
                    claim=NumericClaim(
                        value=token.removesuffix("%"),
                        unit="percent" if token.endswith("%") else None,
                        meaning="undeclared numeric literal",
                    ),
                    status=NumericClaimStatus.UNSUPPORTED,
                )
            )
        return report

    def _validate_claim(
        self,
        claim: NumericClaim,
        registry: EvidenceRegistry,
    ) -> NumericClaimValidation:
        if claim.evidence_ref is not None:
            if claim.evidence_ref not in registry:
                return NumericClaimValidation(
                    claim=claim, status=NumericClaimStatus.INVALID_REFERENCE
                )
            evidence = registry[claim.evidence_ref]
            status = (
                NumericClaimStatus.SUPPORTED
                if _values_match(claim.value, evidence, claim.unit)
                else NumericClaimStatus.MISMATCHED
            )
            return NumericClaimValidation(claim=claim, status=status, evidence_value=evidence)
        if claim.derivation is not None:
            try:
                value = _evaluate_derivation(
                    claim.derivation.operation,
                    claim.derivation.inputs,
                    registry,
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return NumericClaimValidation(
                    claim=claim, status=NumericClaimStatus.INVALID_REFERENCE
                )
            status = (
                NumericClaimStatus.SUPPORTED
                if _values_match(claim.value, value, claim.unit)
                else NumericClaimStatus.MISMATCHED
            )
            return NumericClaimValidation(claim=claim, status=status, evidence_value=value)
        return NumericClaimValidation(claim=claim, status=NumericClaimStatus.UNSUPPORTED)


def _values_match(
    claim_value: int | float | str,
    evidence_value: int | float | str,
    unit: str | None,
) -> bool:
    if isinstance(claim_value, str) or isinstance(evidence_value, str):
        return str(claim_value) == str(evidence_value)
    normalized_claim = float(claim_value)
    if unit == "percent" and 0 <= float(evidence_value) <= 1:
        normalized_claim /= 100
    return isclose(normalized_claim, float(evidence_value), rel_tol=1e-9, abs_tol=1e-9)


def _evaluate_derivation(
    operation: str,
    inputs: list[str | int | float],
    registry: EvidenceRegistry,
) -> float:
    values = [
        float(registry[item]) if isinstance(item, str) and item in registry else _literal(item)
        for item in inputs
    ]
    operations = {"add": add, "subtract": sub, "multiply": mul, "divide": truediv}
    if operation not in operations:
        raise ValueError("unsupported derivation operation")
    return reduce(operations[operation], values)


def _literal(value: str | int | float) -> float:
    if isinstance(value, str):
        raise KeyError(value)
    return float(value)


def _numeric_literals(answer: str) -> set[str]:
    literals = set(re.findall(r"(?<![\w])\d+(?:\.\d+)*(?:%)?", answer))
    unicode_run = ""
    for character in answer:
        if not character.isascii() and _has_numeric_property(character):
            unicode_run += character
        elif unicode_run:
            literals.add(unicode_run)
            unicode_run = ""
    if unicode_run:
        literals.add(unicode_run)
    return literals


def _has_numeric_property(character: str) -> bool:
    try:
        unicodedata.numeric(character)
    except (TypeError, ValueError):
        return False
    return True


def _claim_literal_forms(claim: NumericClaim) -> set[str]:
    forms = {_canonical_literal(str(claim.value))}
    if isinstance(claim.value, (int, float)) and claim.unit == "percent":
        forms.add(_canonical_literal(str(float(claim.value) / 100)))
    if isinstance(claim.value, (int, float)) and claim.unit == "probability":
        forms.add(_canonical_literal(str(float(claim.value) * 100)))
    return forms


def _canonical_literal(value: str) -> str:
    value = value.removesuffix("%")
    if value.count(".") <= 1:
        try:
            return f"{float(value):.12g}"
        except ValueError:
            pass
    return value
