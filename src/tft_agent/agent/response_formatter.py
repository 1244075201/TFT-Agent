"""Formatting boundary for user-facing final responses."""

from tft_agent.core.schemas import FinalResponse


class ResponseFormatter:
    """Render a final response without changing its semantics."""

    def format_text(self, response: FinalResponse) -> str:
        sections = [response.recommendation]
        if response.reasoning:
            sections.append("Reasoning:\n- " + "\n- ".join(response.reasoning))
        if response.next_steps:
            sections.append("Next steps:\n- " + "\n- ".join(response.next_steps))
        if response.sources:
            sections.append("Sources:\n- " + "\n- ".join(response.sources))
        return "\n\n".join(sections)
