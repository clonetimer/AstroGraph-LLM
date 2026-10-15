"""Deterministic causal-motif matching for multievent diagnostic subgraphs.

The matcher is structural: it never infers a physical causal edge from semantic
similarity alone.  A motif match therefore means topological consistency with a
declared pattern, not proof of causality.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

class EdgeType(str, Enum):
    CAUSAL="CAUSAL"; TEMPORAL="TEMPORAL"; COMMAND_RESPONSE="COMMAND_RESPONSE"; COMMON_MODE="COMMON_MODE"

@dataclass(frozen=True)
class MotifEdge:
    source_role:str; target_role:str; edge_type:EdgeType

@dataclass(frozen=True)
class CausalMotif:
    motif_id:str; roles:tuple[str,...]; required_edges:tuple[MotifEdge,...]; description:str
    def __post_init__(self):
        if not self.motif_id or not self.roles: raise ValueError("motif id/roles required")
        if len(set(self.roles))!=len(self.roles): raise ValueError("duplicate motif roles")
        declared=set(self.roles)
        if any(e.source_role not in declared or e.target_role not in declared for e in self.required_edges): raise ValueError("edge references undeclared role")

@dataclass(frozen=True)
class ObservedEdge:
    source_node:str; target_node:str; edge_type:EdgeType


def match_motif(motif:CausalMotif, role_to_node:dict[str,str], observed_edges:Iterable[ObservedEdge])->bool:
    if set(role_to_node)!=set(motif.roles): return False
    observed={(e.source_node,e.target_node,e.edge_type) for e in observed_edges}
    return all((role_to_node[e.source_role],role_to_node[e.target_role],e.edge_type) in observed for e in motif.required_edges)


def default_spacecraft_motifs()->tuple[CausalMotif,...]:
    return (
        CausalMotif("CHAIN",("cause","intermediate","effect"),(MotifEdge("cause","intermediate",EdgeType.CAUSAL),MotifEdge("intermediate","effect",EdgeType.CAUSAL)),"hierarchical causal chain"),
        CausalMotif("FORK",("cause","effect_a","effect_b"),(MotifEdge("cause","effect_a",EdgeType.CAUSAL),MotifEdge("cause","effect_b",EdgeType.CAUSAL)),"one cause with parallel effects"),
        CausalMotif("COLLIDER",("cause_a","cause_b","effect"),(MotifEdge("cause_a","effect",EdgeType.CAUSAL),MotifEdge("cause_b","effect",EdgeType.CAUSAL)),"multiple candidate causes converging on one effect"),
        CausalMotif("COMMAND_RESPONSE_FAILURE",("command","component","response"),(MotifEdge("command","component",EdgeType.COMMAND_RESPONSE),MotifEdge("component","response",EdgeType.CAUSAL)),"commanded actuation with failed/degraded response"),
        CausalMotif("COMMON_MODE",("common_cause","component_a","component_b"),(MotifEdge("common_cause","component_a",EdgeType.COMMON_MODE),MotifEdge("common_cause","component_b",EdgeType.COMMON_MODE)),"shared upstream cause affecting multiple components"),
        CausalMotif("SEQUENTIAL_SYMPTOMS",("symptom_a","symptom_b"),(MotifEdge("symptom_a","symptom_b",EdgeType.TEMPORAL),),"ordered symptoms without asserting physical causality"),
    )


def motif_coverage(matched_motif_ids_by_group:dict[str,set[str]])->dict:
    if not matched_motif_ids_by_group: raise ValueError("no independent groups")
    covered=sum(bool(v) for v in matched_motif_ids_by_group.values())
    return {"group_count":len(matched_motif_ids_by_group),"covered_group_count":covered,"motif_coverage":covered/len(matched_motif_ids_by_group),"unmatched_groups":sorted(g for g,v in matched_motif_ids_by_group.items() if not v)}

__all__=["EdgeType","MotifEdge","CausalMotif","ObservedEdge","match_motif","default_spacecraft_motifs","motif_coverage"]
