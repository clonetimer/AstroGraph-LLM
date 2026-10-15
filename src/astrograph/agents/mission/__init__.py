"""Task-oriented Agent facade over AstroGraph's existing diagnosis and tools.

The facade does not replace the fusion kernel, introduce a new product DAG,
or authorize model/knowledge publication.
"""
from .models import AgentRequest, AgentSettings, AgentResult

__all__ = ["AgentRequest", "AgentSettings", "AgentResult"]
from .diagnostic_scientist import (
    DiagnosticHypothesisProposal,
    DiagnosticScientistAgent,
    DiagnosticScientistResult,
    EvidenceRequest,
    ExperimentEvidence,
    ExperimentPlan,
    ScientistStatus,
)

__all__ += [
    "DiagnosticHypothesisProposal",
    "DiagnosticScientistAgent",
    "DiagnosticScientistResult",
    "EvidenceRequest",
    "ExperimentEvidence",
    "ExperimentPlan",
    "ScientistStatus",
]
