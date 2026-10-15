from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from astrograph.schemas import AnomalyEvidence


class AnomalyDetector(Protocol):
    detector_id: str

    def detect_run(self, run_id: str) -> list[AnomalyEvidence]: ...


class BaseAnomalyDetector(ABC):
    detector_id: str

    @abstractmethod
    def detect_run(self, run_id: str) -> list[AnomalyEvidence]:
        raise NotImplementedError
