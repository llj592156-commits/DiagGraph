"""Typed provider failures; only exhausted transient requests are chunk-local."""


class ExtractionRequestError(RuntimeError):
    """Carry a sanitized request failure without provider bodies or credentials."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ExtractionRetriesExhausted(RuntimeError):
    """Allow business ingestion to retain and skip one unavailable source chunk."""

    def __init__(self, error: ExtractionRequestError, attempts: int):
        if not error.retryable:
            raise ValueError("Only transient extraction failures may be skipped")
        super().__init__(f"{error}; exhausted {attempts} attempts")
        self.details = {
            "kind": "request_failure",
            "message": str(error),
            "status": error.status,
            "attempts": attempts,
        }
