"""Provider-neutral provenance validation for proposed tool arguments."""

from dataclasses import dataclass
from typing import Any

from tft_agent.core.evidence import EvidenceRegistry
from tft_agent.core.schemas import (
    NumericClaimStatus,
    NumericDerivation,
    ToolArgumentClaim,
    ToolCall,
)


@dataclass(frozen=True)
class ToolArgumentValidation:
    """Validation result for one required argument value."""

    claim: ToolArgumentClaim
    status: NumericClaimStatus


def construct_argument_claims(
    arguments: dict[str, Any], registry: EvidenceRegistry
) -> list[ToolArgumentClaim]:
    """Construct provenance only for exact facts or explicit substring extraction."""

    claims: list[ToolArgumentClaim] = []
    for name, value in arguments.items():
        exact_ref = next(
            (
                ref
                for ref, known in registry.items()
                if ref.rsplit(".", 1)[-1] == name and _equal(value, known)
            ),
            None,
        )
        if exact_ref is not None:
            claims.append(
                ToolArgumentClaim(argument_name=name, value=value, evidence_ref=exact_ref)
            )
            continue
        derived_ref = next(
            (
                ref
                for ref, known in registry.items()
                if ref.rsplit(".", 1)[-1] == name
                and isinstance(value, (int, float))
                and isinstance(known, (int, float))
                and not isinstance(value, bool)
                and not isinstance(known, bool)
                and value == known + 1
            ),
            None,
        )
        if derived_ref is not None:
            claims.append(
                ToolArgumentClaim(
                    argument_name=name,
                    value=value,
                    derivation=NumericDerivation(operation="add", inputs=[derived_ref, 1]),
                )
            )
            continue
        question_ref = (
            "game_state.user_question" if "game_state.user_question" in registry else "task.goal"
        )
        question = registry.get(question_ref)
        if isinstance(value, str) and isinstance(question, str) and value in question:
            claims.append(
                ToolArgumentClaim(
                    argument_name=name,
                    value=value,
                    derivation=NumericDerivation(
                        operation="substring", inputs=[question_ref, value]
                    ),
                )
            )
    return claims


class ToolArgumentValidator:
    """Validate every proposed argument against current-task evidence."""

    def validate(
        self, tool_call: ToolCall, registry: EvidenceRegistry
    ) -> list[ToolArgumentValidation]:
        claims = {claim.argument_name: claim for claim in tool_call.argument_claims}
        results: list[ToolArgumentValidation] = []
        for name, value in tool_call.arguments.items():
            claim = claims.get(name)
            if claim is None:
                claim = ToolArgumentClaim(argument_name=name, value=value)
                results.append(ToolArgumentValidation(claim, NumericClaimStatus.UNSUPPORTED))
                continue
            if not _equal(value, claim.value):
                results.append(ToolArgumentValidation(claim, NumericClaimStatus.MISMATCHED))
                continue
            results.append(ToolArgumentValidation(claim, self._validate_source(claim, registry)))
        return results

    @staticmethod
    def _validate_source(
        claim: ToolArgumentClaim, registry: EvidenceRegistry
    ) -> NumericClaimStatus:
        if claim.evidence_ref is not None:
            if claim.evidence_ref not in registry:
                return NumericClaimStatus.INVALID_REFERENCE
            if claim.evidence_ref.rsplit(".", 1)[-1] != claim.argument_name:
                return NumericClaimStatus.MISMATCHED
            return (
                NumericClaimStatus.SUPPORTED
                if _equal(claim.value, registry[claim.evidence_ref])
                else NumericClaimStatus.MISMATCHED
            )
        derivation = claim.derivation
        if derivation is None:
            return NumericClaimStatus.UNSUPPORTED
        if len(derivation.inputs) != 2:
            return NumericClaimStatus.UNSUPPORTED
        source_ref, expected = derivation.inputs
        if not isinstance(source_ref, str) or source_ref not in registry:
            return NumericClaimStatus.INVALID_REFERENCE
        source = registry[source_ref]
        if derivation.operation in {"add", "subtract", "multiply", "divide"}:
            if not isinstance(source, (int, float)) or not isinstance(expected, (int, float)):
                return NumericClaimStatus.MISMATCHED
            try:
                if derivation.operation == "add":
                    derived = source + expected
                elif derivation.operation == "subtract":
                    derived = source - expected
                elif derivation.operation == "multiply":
                    derived = source * expected
                else:
                    derived = source / expected
            except ZeroDivisionError:
                return NumericClaimStatus.MISMATCHED
            return (
                NumericClaimStatus.SUPPORTED
                if _equal(claim.value, derived)
                else NumericClaimStatus.MISMATCHED
            )
        if derivation.operation != "substring":
            return NumericClaimStatus.UNSUPPORTED
        if not isinstance(source, str) or not isinstance(expected, str):
            return NumericClaimStatus.MISMATCHED
        return (
            NumericClaimStatus.SUPPORTED
            if expected == claim.value and expected in source
            else NumericClaimStatus.MISMATCHED
        )


def _equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right
