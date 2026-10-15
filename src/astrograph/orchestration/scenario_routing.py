from __future__ import annotations

import time
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class ScenarioClass(str, Enum):
    CRITICAL = 'CRITICAL'
    STANDARD = 'STANDARD'
    BENIGN = 'BENIGN'


class RoutePolicy(str, Enum):
    FAST_DETERMINISTIC_FORCE_HUMAN = 'FAST_DETERMINISTIC_FORCE_HUMAN'
    DEEP_KG_LLM = 'DEEP_KG_LLM'
    FAST_RULE_ONLY = 'FAST_RULE_ONLY'


class ScenarioRouter:
    """B5: scenario-aware dual-path routing (critical -> fast deterministic + forced human)."""

    schema_version = 'astrograph-scenario-router.v1'

    # Domain keywords belong to a scenario adapter/configuration, not this generic router.
    CRITICAL_KEYWORDS = ('critical', 'safety_critical')
    BENIGN_KEYWORDS = ('nominal', 'healthy', 'no_fault')

    def __init__(
        self,
        critical_keywords: Sequence[str] | None = None,
        benign_keywords: Sequence[str] | None = None,
    ) -> None:
        self.critical_keywords = tuple(critical_keywords or self.CRITICAL_KEYWORDS)
        self.benign_keywords = tuple(benign_keywords or self.BENIGN_KEYWORDS)

    def classify(self, fault_id: str, *, subsystem: str = '') -> ScenarioClass:
        haystack = (fault_id + ' ' + subsystem).lower()
        if any(k in haystack for k in self.benign_keywords):
            return ScenarioClass.BENIGN
        if any(k in haystack for k in self.critical_keywords):
            return ScenarioClass.CRITICAL
        return ScenarioClass.STANDARD

    def route(self, fault_id: str, *, subsystem: str = '') -> dict:
        """Return the route policy + metadata for a scenario."""
        cls = self.classify(fault_id, subsystem=subsystem)
        if cls == ScenarioClass.CRITICAL:
            policy = RoutePolicy.FAST_DETERMINISTIC_FORCE_HUMAN
            requires_human_gate = True
            depth = 'fast'
        elif cls == ScenarioClass.STANDARD:
            policy = RoutePolicy.DEEP_KG_LLM
            requires_human_gate = False
            depth = 'deep'
        else:
            policy = RoutePolicy.FAST_RULE_ONLY
            requires_human_gate = False
            depth = 'fast'
        return {
            'scenario_class': cls.value,
            'route_policy': policy.value,
            'requires_human_gate': requires_human_gate,
            'depth': depth,
            'routed_at': time.strftime('%Y-%m-%dT%H:%M:%S+08:00'),
        }


__all__ = ['ScenarioClass', 'RoutePolicy', 'ScenarioRouter']
