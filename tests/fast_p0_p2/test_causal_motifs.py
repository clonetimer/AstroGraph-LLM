from astrograph.knowledge.causal_motifs import *
def test_topology_and_direction_are_required():
 motif=default_spacecraft_motifs()[0]
 mapping={"cause":"a","intermediate":"b","effect":"c"}
 assert match_motif(motif,mapping,[ObservedEdge("a","b",EdgeType.CAUSAL),ObservedEdge("b","c",EdgeType.CAUSAL)])
 assert not match_motif(motif,mapping,[ObservedEdge("b","a",EdgeType.CAUSAL),ObservedEdge("b","c",EdgeType.CAUSAL)])
def test_motif_coverage_is_group_level():
 r=motif_coverage({"g1":{"CHAIN"},"g2":set()}); assert r["motif_coverage"]==.5
