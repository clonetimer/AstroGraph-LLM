
import pytest
from astrograph.experiments.kg_coverage_protocol import (
    GroupOutcome, NonInferiorityMargins, cluster_bootstrap_difference, select_development_policy,
)
M = NonInferiorityMargins(.02,.02,.02,.02,.02)
def row(p,g,a,r=.1,n=.2,b=.1,e=.05,u=.12): return GroupOutcome(p,g,a,r,n,b,e,u)

def test_selection_uses_exact_independent_development_groups():
    rows = [row("A0",3,.90),row("A0",8,.91),row("P",3,.92),row("P",8,.93)]
    selected, summaries = select_development_policy(rows, baseline_policy_id="A0", development_groups=(3,8), forbidden_groups=(4,9), margins=M)
    assert selected == "P"
    assert summaries["P"].group_count == 2

def test_groups_4_9_fail_closed():
    rows = [row("A0",4,.9), row("A0",9,.9)]
    with pytest.raises(ValueError):
        select_development_policy(rows, baseline_policy_id="A0", development_groups=(4,9), forbidden_groups=(4,9), margins=M)

def test_window_like_duplicate_group_is_rejected():
    rows = [row("A0",3,.9), row("A0",3,.91)]
    with pytest.raises(ValueError):
        select_development_policy(rows, baseline_policy_id="A0", development_groups=(3,), forbidden_groups=(4,9), margins=M)

def test_cluster_bootstrap_is_paired_by_group():
    c=[row("P",3,.92),row("P",8,.93)]
    b=[row("A0",3,.90),row("A0",8,.91)]
    out=cluster_bootstrap_difference(c,b,metric="identification_accuracy",replicates=200,seed_sha256="d"*64)
    assert out["ci95_low"] > 0


def test_each_policy_must_cover_each_development_group_exactly_once():
    rows=[row("A0",3,.9),row("A0",8,.9),row("P",3,.91)]
    with pytest.raises(ValueError):
        select_development_policy(rows,baseline_policy_id="A0",development_groups=(3,8),forbidden_groups=(4,9),margins=M)
