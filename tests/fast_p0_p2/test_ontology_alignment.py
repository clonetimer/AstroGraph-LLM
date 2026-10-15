from astrograph.assurance.ontology_alignment import *
def test_semantic_strict_and_executable_are_separate():
 r=build_alignment_report(dataset_id="x",dataset_classes=["a","b","c"],mechanism_bindings={"a":"m","b":"m"},strict_mappings={"a":"F"},executable_fault_ids=set())
 assert r.counts=={"dataset":3,"semantic":2,"strict_registered":1,"executable":0,"fusion_eligible":0}
