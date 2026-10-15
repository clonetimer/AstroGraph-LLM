"""Shared wall-clock and physical-attempt budgets (not model-supplied).

The POSIX alarm is an interruptible-Python fallback, NOT a C/GPU sandbox.
Production Mission CLI runs in a separately supervised process as well.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import signal
import threading
import time


class ExecutionDeadlineExceeded(TimeoutError):
    pass


class AttemptBudgetExceeded(RuntimeError):
    pass


@dataclass
class ExecutionBudget:
    timeout_s: float
    max_http_attempts: int = 6
    deadline: float = field(init=False)
    http_attempts: int = 0
    attempts: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.timeout_s <= 0 or self.max_http_attempts < 0:
            raise ValueError("invalid host execution budget")
        self.deadline = time.monotonic() + self.timeout_s

    def remaining(self, cap: float | None = None) -> float:
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise ExecutionDeadlineExceeded("TASK_DEADLINE_EXCEEDED")
        return min(left, cap) if cap is not None else left

    def consume_http(self, provider: str, strategy: str):
        self.remaining()
        if self.http_attempts >= self.max_http_attempts:
            raise AttemptBudgetExceeded("LLM_PHYSICAL_ATTEMPT_BUDGET_EXHAUSTED")
        self.http_attempts += 1
        self.attempts.append({"attempt": self.http_attempts, "provider": provider, "strategy": strategy})

    def snapshot(self):
        return {"timeout_s":self.timeout_s,"max_http_attempts":self.max_http_attempts,
                "http_attempts":self.http_attempts,"attempts":list(self.attempts)}


_CURRENT: ContextVar[ExecutionBudget | None] = ContextVar("astrograph_execution_budget", default=None)

def current_budget():
    return _CURRENT.get()


def remaining_time(cap: float | None = None):
    budget = current_budget()
    return budget.remaining(cap) if budget is not None else cap


@contextmanager
def execution_scope(budget: ExecutionBudget):
    token = _CURRENT.set(budget)
    try:
        yield budget
    finally:
        _CURRENT.reset(token)


@contextmanager
def interruptible_timeout(seconds: float, code: str = "CALL_DEADLINE_EXCEEDED"):
    """Interrupt Python/sleep on POSIX main thread; nested timers keep earliest deadline.

    Other contexts require the outer process supervisor for hard termination.
    Never advertises an arbitrary thread or C extension as killable.
    """
    seconds = remaining_time(seconds)
    if seconds is None or seconds <= 0:
        raise ExecutionDeadlineExceeded(code)
    supported = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not supported:
        yield False
        remaining_time()
        return
    start = time.monotonic()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise ExecutionDeadlineExceeded(code)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, old_delay) if old_delay > 0 else seconds)
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, old_delay - (time.monotonic() - start)), old_interval)
