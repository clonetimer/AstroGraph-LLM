from __future__ import annotations

from typing import Any, Iterable


class AstroGraphError(Exception):
    code = "ASTROGRAPH_INTERNAL_ERROR"
    http_status = 500
    default_recoverable = False
    default_recommended_action = "CONTACT_MAINTAINER"
    default_requires_human_review = True
    default_claim_boundary = "The requested operation did not produce a verified result."

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
        recommended_action: str | None = None,
        missing_fields: Iterable[str] | None = None,
        requires_human_review: bool | None = None,
        claim_boundary: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = self.default_recoverable if recoverable is None else bool(recoverable)
        self.recommended_action = recommended_action or self.default_recommended_action
        self.missing_fields = sorted(set(str(item) for item in (missing_fields or []) if str(item)))
        self.requires_human_review = (
            self.default_requires_human_review if requires_human_review is None else bool(requires_human_review)
        )
        self.claim_boundary = claim_boundary or self.default_claim_boundary

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "http_status": self.http_status,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "recommended_action": self.recommended_action,
            "missing_fields": self.missing_fields,
            "requires_human_review": self.requires_human_review,
            "claim_boundary": self.claim_boundary,
        }


class IdempotencyConflictError(AstroGraphError):
    code = "ASTROGRAPH_IDEMPOTENCY_CONFLICT"
    http_status = 409
    default_recoverable = True
    default_recommended_action = "REUSE_OR_REPLACE_IDEMPOTENCY_KEY"


class IdempotencyLeaseLostError(AstroGraphError):
    code = "ASTROGRAPH_IDEMPOTENCY_LEASE_LOST"
    http_status = 409
    default_recoverable = True
    default_recommended_action = "RETRY_WITH_THE_SAME_IDEMPOTENCY_KEY"
    default_claim_boundary = (
        "The producer completed without owning the current fencing epoch, so its response was not committed."
    )


class ResourceNotFoundError(AstroGraphError):
    code = "ASTROGRAPH_RESOURCE_NOT_FOUND"
    http_status = 404
    default_recoverable = True
    default_recommended_action = "VERIFY_RESOURCE_IDENTIFIER"


class KnowledgeUnavailableError(AstroGraphError):
    code = "ASTROGRAPH_KNOWLEDGE_UNAVAILABLE"
    http_status = 503
    default_recoverable = True
    default_recommended_action = "RETRY_OR_USE_EVIDENCE_ONLY_MODE"


class StateConflictError(AstroGraphError):
    code = "ASTROGRAPH_STATE_CONFLICT"
    http_status = 409
    default_recoverable = True
    default_recommended_action = "REFRESH_TASK_STATE"


class InvalidOperationError(AstroGraphError):
    code = "ASTROGRAPH_INVALID_OPERATION"
    http_status = 422
    default_recoverable = True
    default_recommended_action = "CORRECT_INPUT_OR_OPERATION"


class UnknownFaultError(AstroGraphError):
    code = "ASTROGRAPH_UNKNOWN_FAULT"
    http_status = 422
    default_recoverable = True
    default_recommended_action = "REQUEST_ADDITIONAL_TELEMETRY_OR_EXPERT_REVIEW"
    default_requires_human_review = True
    default_claim_boundary = "No known-catalog diagnosis can be confirmed from the supplied telemetry."


class InsufficientEvidenceError(AstroGraphError):
    code = "ASTROGRAPH_INSUFFICIENT_EVIDENCE"
    http_status = 422
    default_recoverable = True
    default_recommended_action = "SUPPLY_REQUIRED_TELEMETRY_CHANNELS"
    default_requires_human_review = False
    default_claim_boundary = "Diagnosis was not attempted because required evidence was incomplete."


class OutOfScopeError(AstroGraphError):
    code = "ASTROGRAPH_OUT_OF_SCOPE"
    http_status = 422
    default_recoverable = False
    default_recommended_action = "SELECT_SUPPORTED_BACKEND_OR_MODEL"
    default_requires_human_review = True
    default_claim_boundary = "The input is outside this model's declared backend and catalog scope."
