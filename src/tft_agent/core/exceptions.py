"""Domain-level exceptions that may cross implementation boundaries."""


class TFTAgentError(Exception):
    """Base exception for expected application failures."""


class LLMClientError(TFTAgentError):
    """Raised when an LLM adapter cannot produce a decision."""


class MCPClientError(TFTAgentError):
    """Raised when the MCP transport or remote server cannot complete a request."""


class ToolNotFoundError(TFTAgentError):
    """Raised when a requested tool has not been registered."""


class ConfigurationError(TFTAgentError):
    """Raised when required application configuration is absent or invalid."""
