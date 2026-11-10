"""Exceptions for the backends subsystem."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspect.backends._execution import JobReference


class BackendError(Exception):
    """Base exception for all backend-related errors."""


class PartialSubmissionError(BackendError):
    """A logical batch failed after one or more remote jobs were created."""

    def __init__(
        self,
        message: str,
        *,
        submitted_references: Sequence[JobReference],
        cancellation_failures: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.submitted_references = tuple(submitted_references)
        self.submitted_job_ids = tuple(
            job_id
            for reference in self.submitted_references
            for job_id in (reference.child_job_ids or (reference.job_id,))
        )
        self.cancellation_failures = tuple(str(item) for item in cancellation_failures)


class AuthenticationError(BackendError):
    """Raised when authentication with a cloud platform fails."""


class TranslationError(BackendError):
    """Raised when circuit translation to a platform-specific format fails."""


class JobError(BackendError):
    """Raised when a submitted job fails or encounters an error.

    Attributes:
        error_type: A short classification of the error when one is known
            (the provider's own error type, or a QuSpect value such as
            ``"not_found"``); ``None`` otherwise.
        job_id: The provider's job identifier, when known.
        result_pending: Whether the job can still deliver its result.
            ``False`` here: the job failed, was cancelled, is unknown to the
            provider, or returned a result that cannot be decoded.  Only
            :class:`JobTimeoutError` sets it to ``True``.
    """

    #: ``True`` only on :class:`JobTimeoutError`: reading again later can succeed.
    result_pending: bool = False

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.job_id = job_id


class JobTimeoutError(JobError):
    """Raised when a job's result is not available within the wait.

    The job has not failed.  It was still queued or running, or the provider
    had not published its result yet, when the timeout expired; calling
    ``result()`` again later can succeed.  Do not treat it as a failed job
    or as an undecodable result.

    This class subclasses :class:`JobError` for backward compatibility, so a
    handler for ``JobError`` also receives it.  Such a handler must either
    catch ``JobTimeoutError`` first or test ``exc.result_pending``, which is
    ``True`` here and ``False`` on every other ``JobError``.
    """

    result_pending = True


class BackendNotFoundError(BackendError):
    """Raised when a requested backend name is not available."""


class OptionalDependencyError(BackendError):
    """Raised when an optional third-party SDK is not installed."""
