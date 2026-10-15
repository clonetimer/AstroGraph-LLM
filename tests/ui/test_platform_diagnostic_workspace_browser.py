from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src/astrograph/resources"
CSS = (RESOURCES / "platform_ui.css").read_text(encoding="utf-8")


def _classic_bundle() -> str:
    chunks: list[str] = []
    for name in (
        "platform_core.js",
        "platform_operations.js",
        "platform_diagnostic_workspace.js",
    ):
        source = (RESOURCES / name).read_text(encoding="utf-8")
        source = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
        if "\nexport {" in source:
            source = source.rsplit("\nexport {", 1)[0]
        chunks.append(source)
    return "\n".join(chunks)


JS = _classic_bundle()


def _chromium_executable() -> str | None:
    candidates = [
        os.environ.get("ASTROGRAPH_CHROMIUM_EXECUTABLE"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


@pytest.fixture()
def page():
    with sync_playwright() as playwright:
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        executable = _chromium_executable()
        if executable:
            kwargs["executable_path"] = executable
        browser = playwright.chromium.launch(**kwargs)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        page.set_content('<main id="result-content"></main>')
        page.add_style_tag(content=CSS)
        page.evaluate(
            """Object.defineProperty(window, 'localStorage', {
              configurable: true,
              value: {
                _data: {}, getItem(k) { return this._data[k] ?? null; },
                setItem(k, v) { this._data[k] = String(v); },
                removeItem(k) { delete this._data[k]; }
              }
            })"""
        )
        page.add_script_tag(content=JS)
        page.evaluate("() => setupOperationsHooks({renderDiagnosticWorkspace})")
        yield page
        browser.close()


def _render(page, *, graph: bool = True, simulation: bool = True, evidence: bool = True) -> None:
    page.evaluate(
        """([hasGraph,hasSimulation,hasEvidence]) => {
          window.__report = {
            job_id:'JOB-R34',
            agent_execution:{investigation:{
              findings:[{id:'step-1',probe_id:'COMMAND_RESPONSE_RESIDUAL',channel:'flow',component_id:'PMP420'}],
              required_observations:[]
            }}
          };
          window.__maintenance = {
            component:'PMP420', fault_id:'PMP420:FlowBlocked',
            diagnosis_disposition:'SUPPORTED', release_status:'FORMAL_RELEASE_BLOCKED',
            requires_human_review:false,
            alternative_hypotheses:[{fault_id:'PMP420:FailedOff',label:'PMP420:FailedOff'}],
            evidence_summary:hasEvidence?{status:'AVAILABLE',supporting_items:[{
              id:'ev-flow',label:'FT420 flow collapse',evidence_definition_id:'FLOW_COLLAPSE',
              channel:'flow',component_id:'PMP420'
            }],contradicting_items:[],missing_items:[]}:null,
            mechanism_graph:hasGraph?{status:'AVAILABLE',provenance_sha256:'3333333333333333333333333333333333333333333333333333333333333333',nodes:[
              {id:'PMP420',label:'PMP420',related_channels:['flow']},
              {id:'FT420',label:'FT420',related_channels:['flow']}
            ],edges:[{from:'PMP420',to:'FT420',relation:'affects'}]}:{status:'UNAVAILABLE'},
            simulation_review:hasSimulation?{numerical_traces:[{
              id:'sim-flow-blocked',label:'FlowBlocked',channel:'flow',
              points:[{t:0,v:10},{t:1,v:4},{t:2,v:1}]
            }]}:{status:'MECHANISM_ONLY',fit_score:0.92}
          };
          window.__visualization = {channels:[{name:'flow',label:'FT420 flow',unit:'kg/s',points:[
            {t:0,v:10},{t:1,v:5},{t:2,v:2}
          ]}]};
          window.__workspace = {
            schema_version:'astrograph-diagnostic-workspace-presentation.v1',
            job_id:'JOB-R34', presentation_sha256:'1111111111111111111111111111111111111111111111111111111111111111',
            source_bundle_status:'VERIFIED', source_bundle_sha256:'2222222222222222222222222222222222222222222222222222222222222222',
            authority_snapshot:{diagnosis_disposition:'SUPPORTED',release_status:'FORMAL_RELEASE_BLOCKED',evidence_bundle_sha256:'4444444444444444444444444444444444444444444444444444444444444444'},
            candidate_context:{selected_fault_id:'PMP420:FlowBlocked',component_id:'PMP420',subsystem_id:null,alternatives:[{candidate_id:'PMP420:FailedOff',label:'PMP420:FailedOff'}]},
            mechanism_graph:hasGraph?{status:'AVAILABLE',provenance_sha256:'3333333333333333333333333333333333333333333333333333333333333333',nodes:[
              {id:'component:PMP420',label:'PMP420',related_channels:['flow']},
              {id:'component:FT420',label:'FT420',related_channels:['flow']}
            ],edges:[{from:'component:PMP420',to:'component:FT420',relation:'CONNECTED_TO'}]}:{status:'UNAVAILABLE',nodes:[],edges:[]},
            observed_traces:[{trace_id:'observed:flow',label:'FT420 flow',channel:'flow',unit:'kg/s',samples:[{time_s:0,value:10},{time_s:1,value:5},{time_s:2,value:2}]}],
            simulation_traces:hasSimulation?[{trace_id:'sim-flow-blocked',provider_id:'WATERPUMP_REDUCED_ORDER_V1',diagnosis_id:'DXC-DIAG-1',candidate_id:'PMP420:FlowBlocked',source_trace_sha256:'5555555555555555555555555555555555555555555555555555555555555555',channel:'flow',unit:'kg/s',semantics:'SIMULATED_TRAJECTORY_NOT_PROBABILITY',samples:[{time_s:0,value:10},{time_s:1,value:4},{time_s:2,value:1}]}]:[],
            evidence_links:hasEvidence?[{evidence_id:'ev-flow',label:'FT420 flow collapse',role:'SUPPORTING',channel:'flow',component_id:'PMP420'}]:[],
            agent_links:[{link_id:'step-1',probe_id:'COMMAND_RESPONSE_RESIDUAL',status:'COMPLETED',channel:'flow',component_id:'PMP420'}]
          };
          enhanceOperationsResult(window.__report,window.__maintenance,{execution_status:'SUCCEEDED'},window.__visualization,window.__workspace);
        }""",
        [graph, simulation, evidence],
    )


def test_selection_links_panels_without_changing_formal_authority(page) -> None:
    _render(page)
    formal_before = page.locator("#ops-result-navigation").inner_text()
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    assert page.evaluate("() => workspaceSnapshot().candidateId") == "PMP420:FailedOff"
    page.locator('[data-dw-evidence="ev-flow"]').click()
    selection = page.evaluate("() => workspaceSnapshot()")

    assert selection["evidenceId"] == "ev-flow"
    assert selection["graphNodeId"] == "component:PMP420"
    assert selection["observedTraceId"] == "flow"
    assert page.locator("#ops-result-navigation").inner_text() == formal_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_graph_and_overlay_render_only_structured_numeric_inputs(page) -> None:
    _render(page)
    assert page.locator(".dw-overlay-chart path.observed").count() == 1
    assert page.locator(".dw-overlay-chart path.simulated").count() == 1
    assert "fault probability" in page.locator("#dw-main").inner_text()
    assert "% probability" not in page.locator("#diagnostic-workspace").inner_text()

    page.locator('[data-dw-main="graph"]').click()
    assert page.locator(".dw-node").count() == 2
    page.locator('[data-dw-node="component:FT420"]').click()
    assert page.evaluate("() => workspaceSnapshot().graphNodeId") == "component:FT420"

    _render(page, graph=False, simulation=False)
    assert "当前诊断没有可叠加的数值仿真轨迹" in page.locator("#dw-main").inner_text()
    assert page.locator(".dw-overlay-chart path.simulated").count() == 0
    page.locator('[data-dw-main="graph"]').click()
    assert "当前报告没有可交互图谱数据。" in page.locator("#dw-main").inner_text()
    assert page.locator(".dw-node").count() == 0


def test_agent_step_is_navigation_only_and_missing_evidence_fails_closed(page) -> None:
    page.evaluate("() => { window.__fetchCount=0; window.fetch=()=>{ window.__fetchCount+=1; throw new Error('unexpected fetch') } }")
    _render(page, evidence=False)
    assert "正式证据摘要不可用。" in page.locator("#dw-support").inner_text()
    page.locator('[data-dw-agent-step="step-1"]').click()
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => workspaceSnapshot().graphNodeId") == "component:PMP420"
    assert page.locator('input[type="number"], input[name*="cypher" i], textarea[name*="cypher" i]').count() == 0


def test_simulation_selector_requires_a_channel_compatible_trace(page) -> None:
    _render(page)
    page.evaluate("""() => {
      window.__workspace.simulation_traces[0].channel='current';
      enhanceOperationsResult(window.__report,window.__maintenance,{execution_status:'SUCCEEDED'},window.__visualization,window.__workspace);
    }""")
    assert "当前遥测通道没有匹配的仿真轨迹" in page.locator("#dw-main").inner_text()
    assert page.locator("#dw-simulation-select").count() == 0
    assert page.locator(".dw-overlay-chart path.simulated").count() == 0



def test_r35_overview_selection_summary_and_reset_are_read_only(page) -> None:
    _render(page)
    overview = page.locator("#dw-overview").inner_text()
    assert "图谱" in overview and "2 个节点" in overview
    assert "遥测" in overview and "1 条观测" in overview
    assert "数值仿真" in overview and "1 条轨迹" in overview
    assert "正式 Evidence" in overview and "1 项" in overview
    assert "Agent 回执" in overview and "1 项" in overview

    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    page.locator('[data-dw-main="graph"]').click()
    page.locator('[data-dw-node="component:FT420"]').click()
    assert "PMP420:FailedOff" in page.locator("#dw-selection").inner_text()
    assert "仍来自并绑定冻结候选" in page.locator(".dw-selection-note").inner_text()
    assert "FT420" in page.locator(".dw-focus-detail").inner_text()
    assert "1 条显式关联" in page.locator(".dw-focus-detail").inner_text()

    page.locator("#dw-reset-context").click()
    selection = page.evaluate("() => workspaceSnapshot()")
    assert selection == {
        "candidateId": None, "componentId": None, "evidenceId": None,
        "graphNodeId": None, "observedTraceId": None, "simulationTraceId": None,
        "mainView": "overlay",
    }
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_trace_context_and_escape_reset(page) -> None:
    _render(page)
    trace_meta = page.locator(".dw-trace-meta").inner_text()
    assert "通道" in trace_meta and "flow" in trace_meta
    assert "单位" in trace_meta and "kg/s" in trace_meta
    assert "时间范围" in trace_meta and "0–2 s" in trace_meta
    assert "时间 (s)" in (page.locator(".dw-overlay-chart").text_content() or "")

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    page.locator("#diagnostic-workspace").press("Escape")
    assert page.evaluate("() => workspaceSnapshot().candidateId") is None
    assert page.evaluate("() => workspaceSnapshot().mainView") == "overlay"

def test_workspace_has_no_page_level_horizontal_overflow_at_160_percent(page) -> None:
    _render(page)
    page.evaluate("""() => {
      document.documentElement.dataset.fontBand='xlarge';
      document.documentElement.style.setProperty('--font-scale','1.6');
    }""")
    assert page.evaluate("() => document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_r35_1_review_identity_and_handoff_copy_preserve_authority(page) -> None:
    page.evaluate("""() => {
      Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async(text)=>{window.__handoffClipboard=text;}}});
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    identity = page.locator("#dw-review-identity").inner_text()
    assert "JOB-R34" in identity
    assert "工作区投影 SHA256" in identity and "1111111111111111" in identity
    assert "来源包状态" in identity and "VERIFIED" in identity
    assert "图谱 provenance SHA256" in identity and "3333333333333333" in identity
    assert "EvidenceBundle SHA256" in identity and "4444444444444444" in identity

    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Frozen candidate: PMP420:FlowBlocked" in handoff
    assert "Candidate view: PMP420:FlowBlocked [FROZEN_REPORT_CANDIDATE]" in handoff
    assert "provider=WATERPUMP_REDUCED_ORDER_V1" in handoff
    assert "source_trace_sha256=" + "5" * 64 in handoff

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    updated = page.locator("#dw-handoff-text").inner_text()
    assert "Frozen candidate: PMP420:FlowBlocked" in updated
    assert "Candidate view: PMP420:FailedOff [TRANSIENT_ALTERNATIVE_BROWSE_ONLY]" in updated
    page.locator("#dw-copy-handoff").click()
    page.wait_for_function("() => Boolean(window.__handoffClipboard)")
    assert page.evaluate("() => window.__handoffClipboard") == page.locator("#dw-handoff-text").inner_text()
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_2_review_session_is_ephemeral_and_handoff_tracks_review_state(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    assert "0/6" in page.locator("#dw-review-session").inner_text()
    assert "当前投影与页面上下文未出现需要注意的显式材料信号" in page.locator("#dw-review-session").inner_text()

    page.locator('[data-dw-review-area="identity"]').click()
    page.locator('[data-dw-review-area="evidence"]').click()
    assert page.evaluate("() => reviewSessionSnapshot()") == {
        "identity": True, "candidate": False, "evidence": True,
        "graph": False, "traces": False, "agent": False,
    }
    assert "2/6" in page.locator("#dw-review-session").inner_text()
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Reviewed areas: 来源身份, Evidence" in handoff
    assert "Unreviewed areas: 候选上下文, 机理图谱, 遥测与仿真, Agent 回执" in handoff

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    session_text = page.locator("#dw-review-session").inner_text()
    assert "TRANSIENT_CANDIDATE_BROWSE" in session_text
    assert "PMP420:FailedOff" in session_text
    assert "Attention signals: TRANSIENT_CANDIDATE_BROWSE" in page.locator("#dw-handoff-text").inner_text()

    page.locator("#dw-reset-review-session").click()
    assert not any(page.evaluate("() => Object.values(reviewSessionSnapshot())"))
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_2_attention_signals_only_reflect_explicit_projection_state(page) -> None:
    _render(page)
    page.evaluate("""() => {
      window.__workspace.source_bundle_status='STALE';
      window.__workspace.evidence_links.push(
        {evidence_id:'ev-contradict',label:'Explicit contradiction',role:'CONTRADICTING'},
        {evidence_id:'ev-missing',label:'Declared missing item',role:'MISSING'}
      );
      window.__workspace.agent_links[0].status='PENDING';
      window.__workspace.mechanism_graph={status:'UNAVAILABLE',nodes:[],edges:[]};
      window.__workspace.observed_traces=[];
      window.__workspace.simulation_traces=[];
      enhanceOperationsResult(window.__report,window.__maintenance,{execution_status:'SUCCEEDED'},window.__visualization,window.__workspace);
    }""")
    text = page.locator("#dw-review-session").inner_text()
    for code in (
        "SOURCE_BUNDLE_NOT_VERIFIED",
        "CONTRADICTING_EVIDENCE_PRESENT",
        "MISSING_EVIDENCE_DECLARED",
        "AGENT_STEP_INCOMPLETE",
        "MECHANISM_GRAPH_UNAVAILABLE",
        "OBSERVED_TELEMETRY_UNAVAILABLE",
        "NUMERIC_SIMULATION_UNAVAILABLE",
    ):
        assert code in text
    assert "不生成新的故障判断、EvidenceFact、概率或资格结论" in text

def test_r35_3_review_closure_tracks_coverage_and_attention_without_authority_mutation(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    closure = page.evaluate("() => reviewClosureState(window.__workspace)")
    assert closure["status"] == "REVIEW_INCOMPLETE"
    assert closure["reviewedCount"] == 0
    assert len(closure["unreviewedAreas"]) == 6
    assert page.locator('[data-dw-closure-status="REVIEW_INCOMPLETE"]').count() == 1

    for area in ("identity", "candidate", "evidence", "graph", "traces", "agent"):
        page.locator(f'[data-dw-review-area="{area}"]').click()
    closure = page.evaluate("() => reviewClosureState(window.__workspace)")
    assert closure["status"] == "REVIEWED_NO_EXPLICIT_ATTENTION"
    assert closure["unreviewedAreas"] == []
    assert closure["attentionSignals"] == []
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Review closure status: REVIEWED_NO_EXPLICIT_ATTENTION" in handoff
    assert "Open review areas: none" in handoff
    assert "Open attention signal codes: none" in handoff
    assert "Review closure meaning: SESSION_TRACE_ONLY_NOT_APPROVAL" in handoff

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    closure = page.evaluate("() => reviewClosureState(window.__workspace)")
    assert closure["status"] == "REVIEWED_WITH_ATTENTION"
    assert [signal["code"] for signal in closure["attentionSignals"]] == ["TRANSIENT_CANDIDATE_BROWSE", "REVIEW_COVERAGE_CONTEXT_CHANGED"]
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Review closure status: REVIEWED_WITH_ATTENTION" in handoff
    assert "Open attention signal codes: TRANSIENT_CANDIDATE_BROWSE, REVIEW_COVERAGE_CONTEXT_CHANGED" in handoff

    page.locator('[data-dw-review-area="agent"]').click()
    closure = page.evaluate("() => reviewClosureState(window.__workspace)")
    assert closure["status"] == "REVIEW_INCOMPLETE"
    assert closure["unreviewedAreas"] == ["Agent 回执"]
    assert "Open review areas: Agent 回执" in page.locator("#dw-handoff-text").inner_text()
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before



def test_r35_4_unreviewed_area_navigation_preserves_review_and_authority_state(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    page.locator('[data-dw-review-nav-area="graph"]').click()
    page.wait_for_function("() => workspaceSnapshot().mainView === 'graph'")
    page.wait_for_function("() => document.activeElement?.dataset?.dwMain === 'graph'")
    assert page.evaluate("() => reviewSessionSnapshot().graph") is False
    assert "故障传播图" in page.locator("#dw-main").inner_text()

    page.locator('[data-dw-review-nav-area="evidence"]').click()
    page.wait_for_function("() => workspaceSnapshot().evidenceId === 'ev-flow'")
    page.wait_for_function("() => document.activeElement?.dataset?.dwEvidence === 'ev-flow'")
    assert page.evaluate("() => reviewSessionSnapshot().evidence") is False
    assert page.locator('[data-dw-evidence="ev-flow"].selected').count() == 1

    assert not any(page.evaluate("() => Object.values(reviewSessionSnapshot())"))
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_4_attention_signal_drilldown_targets_explicit_material_without_clearing_signal(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")
    page.evaluate("""() => {
      window.__workspace.evidence_links.push(
        {evidence_id:'ev-contradict',label:'Explicit contradiction',role:'CONTRADICTING',channel:'flow',component_id:'PMP420'},
        {evidence_id:'ev-missing',label:'Declared missing item',role:'MISSING',channel:'flow',component_id:'PMP420'}
      );
      window.__workspace.agent_links[0].status='PENDING';
      enhanceOperationsResult(window.__report,window.__maintenance,{execution_status:'SUCCEEDED'},window.__visualization,window.__workspace);
    }""")

    page.locator('[data-dw-review-nav-signal="CONTRADICTING_EVIDENCE_PRESENT"]').click()
    page.wait_for_function("() => workspaceSnapshot().evidenceId === 'ev-contradict'")
    page.wait_for_function("() => document.activeElement?.dataset?.dwEvidence === 'ev-contradict'")
    assert "CONTRADICTING_EVIDENCE_PRESENT" in page.locator("#dw-review-session").inner_text()

    page.locator('[data-dw-review-nav-signal="AGENT_STEP_INCOMPLETE"]').click()
    page.wait_for_function("() => document.activeElement?.dataset?.dwAgentStep === 'step-1'")
    assert "AGENT_STEP_INCOMPLETE" in page.locator("#dw-review-session").inner_text()

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    page.locator('[data-dw-review-nav-signal="TRANSIENT_CANDIDATE_BROWSE"]').click()
    page.wait_for_function("() => document.activeElement?.dataset?.dwCandidate === 'PMP420:FailedOff'")
    assert page.evaluate("() => workspaceSnapshot().candidateId") == "PMP420:FailedOff"
    assert "TRANSIENT_CANDIDATE_BROWSE" in page.locator("#dw-review-session").inner_text()

    assert not any(page.evaluate("() => Object.values(reviewSessionSnapshot())"))
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_5_review_coverage_records_capture_explicit_material_scope_for_handoff(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    for area in ("identity", "evidence", "traces"):
        page.locator(f'[data-dw-review-area="{area}"]').click()

    coverage = page.evaluate("() => reviewCoverageSnapshot()")
    assert coverage["identity"]["presentationSha256"] == "1" * 64
    assert "source_bundle=" + "2" * 64 in coverage["identity"]["coverageKey"]
    assert "evidence_items=SUPPORTING:ev-flow" in coverage["evidence"]["coverageKey"]
    assert "sim-flow-blocked:WATERPUMP_REDUCED_ORDER_V1:PMP420:FlowBlocked:" in coverage["traces"]["coverageKey"]

    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Review coverage provenance (ephemeral):" in handoff
    assert "Coverage contract: BROWSER_SESSION_SCOPE_V1" in handoff
    assert "Coverage identity (来源身份):" in handoff
    assert "Coverage evidence (Evidence):" in handoff
    assert "Coverage traces (遥测与仿真):" in handoff
    assert "Coverage candidate (候选上下文): NOT_CAPTURED" in handoff
    assert "Coverage drift areas: none" in handoff
    assert "MATERIAL_SCOPE_TRACE_ONLY_NOT_AUDIT_OR_APPROVAL" in handoff

    assert page.locator('[data-dw-review-coverage-area="identity"]').count() == 1
    assert page.locator('[data-dw-review-coverage-area="evidence"]').count() == 1
    assert page.locator('[data-dw-review-coverage-area="traces"]').count() == 1
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_5_review_coverage_drift_is_visible_without_mutating_review_or_authority(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    for area in ("identity", "candidate", "evidence", "graph", "traces", "agent"):
        page.locator(f'[data-dw-review-area="{area}"]').click()
    assert page.evaluate("() => reviewClosureState(window.__workspace).status") == "REVIEWED_NO_EXPLICIT_ATTENTION"
    assert page.evaluate("() => reviewCoverageDrift(window.__workspace).length") == 0

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    assert page.evaluate("() => reviewSessionSnapshot().candidate") is True
    drift = page.evaluate("() => reviewCoverageDrift(window.__workspace).map(x => x.area)")
    assert drift == ["candidate"]
    signals = page.evaluate("() => reviewAttentionSignals(window.__workspace).map(x => x.code)")
    assert "TRANSIENT_CANDIDATE_BROWSE" in signals
    assert "REVIEW_COVERAGE_CONTEXT_CHANGED" in signals
    assert page.evaluate("() => reviewClosureState(window.__workspace).status") == "REVIEWED_WITH_ATTENTION"
    assert page.locator('[data-dw-review-coverage-area="candidate"].changed').count() == 1
    assert "候选上下文" in page.locator("#dw-handoff-text").inner_text().split("Coverage drift areas:", 1)[1]

    page.locator('[data-dw-review-nav-signal="REVIEW_COVERAGE_CONTEXT_CHANGED"]').click()
    page.wait_for_function("() => document.activeElement?.dataset?.dwReviewCoverageArea === 'candidate'")
    assert page.evaluate("() => reviewSessionSnapshot().candidate") is True

    page.locator("#dw-reset-context").click()
    assert page.evaluate("() => reviewCoverageDrift(window.__workspace).length") == 0
    assert "REVIEW_COVERAGE_CONTEXT_CHANGED" not in page.evaluate("() => reviewAttentionSignals(window.__workspace).map(x => x.code)")
    assert page.evaluate("() => reviewClosureState(window.__workspace).status") == "REVIEWED_NO_EXPLICIT_ATTENTION"

    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_6_incoming_replay_matches_captured_scope_without_mutating_current_session(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    for area in ("identity", "candidate", "graph"):
        page.locator(f'[data-dw-review-area="{area}"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    assert not any(page.evaluate("() => Object.values(reviewSessionSnapshot())"))

    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    comparison = page.evaluate("() => Object.fromEntries(reviewCoverageReplayComparison(window.__workspace).map(x => [x.area,x.status]))")
    assert comparison == {
        "identity": "MATCHED", "candidate": "MATCHED", "evidence": "NOT_PROVIDED",
        "graph": "MATCHED", "traces": "NOT_PROVIDED", "agent": "NOT_PROVIDED",
    }
    assert page.evaluate("() => incomingReviewCoverageSnapshot().contract") == "BROWSER_REVIEW_COVERAGE_REPLAY_V1"
    assert page.evaluate("() => reviewClosureState(window.__workspace).status") == "REVIEW_INCOMPLETE"
    assert not any(page.evaluate("() => Object.values(reviewSessionSnapshot())"))
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Incoming scope comparison: identity=MATCHED, candidate=MATCHED" in handoff
    assert "INCOMING_SCOPE_COMPARISON_ONLY_NOT_AUDIT_OR_APPROVAL" in handoff
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_6_replay_distinguishes_changed_not_reproduced_and_legacy_handoff(page) -> None:
    _render(page)
    for area in ("candidate", "evidence", "graph", "traces"):
        page.locator(f'[data-dw-review-area="{area}"]').click()
    legacy_handoff = page.evaluate("() => handoffSummaryText(window.__workspace).split('Outgoing review coverage replay manifest:')[0]")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")

    page.locator("#dw-incoming-coverage-manifest").fill(legacy_handoff)
    page.locator("#dw-load-incoming-coverage").click()
    assert page.evaluate("() => incomingReviewCoverageSnapshot().import_format") == "R35_5_HANDOFF_COVERAGE_LINES"
    initial = page.evaluate("() => Object.fromEntries(reviewCoverageReplayComparison(window.__workspace).map(x => [x.area,x.status]))")
    assert initial["candidate"] == "MATCHED"
    assert initial["evidence"] == "MATCHED"
    assert initial["graph"] == "MATCHED"
    assert initial["traces"] == "MATCHED"
    assert initial["identity"] == "NOT_PROVIDED"

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    changed = page.evaluate("() => Object.fromEntries(reviewCoverageReplayComparison(window.__workspace).map(x => [x.area,x.status]))")
    assert changed["candidate"] == "CHANGED"
    assert "INCOMING_REVIEW" not in page.evaluate("() => reviewAttentionSignals(window.__workspace).map(x => x.code).join(',')")

    page.locator("#dw-reset-context").click()
    page.evaluate("""() => {
      window.__workspace.mechanism_graph={status:'UNAVAILABLE',nodes:[],edges:[]};
      renderWorkspacePanels();
    }""")
    current = page.evaluate("() => Object.fromEntries(reviewCoverageReplayComparison(window.__workspace).map(x => [x.area,x.status]))")
    assert current["candidate"] == "MATCHED"
    assert current["graph"] == "NOT_REPRODUCED"
    page.locator('[data-dw-replay-nav-area="graph"]').click()
    page.wait_for_function("() => document.activeElement?.dataset?.dwMain === 'graph'")
    assert page.evaluate("() => incomingReviewCoverageSnapshot() !== null") is True


def test_r35_6_invalid_replay_fails_closed_and_clear_removes_browser_local_input(page) -> None:
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")
    invalid = '{"contract":"BROWSER_REVIEW_COVERAGE_REPLAY_V1","areas":{"identity":{"basis":["presentation=abc"],"coverage_key":"tampered"}}}'
    page.locator("#dw-incoming-coverage-manifest").fill(invalid)
    page.locator("#dw-load-incoming-coverage").click()
    assert page.evaluate("() => incomingReviewCoverageSnapshot()") is None
    assert "coverage_key 与 basis 不一致" in page.locator(".dw-review-replay-error").inner_text()
    assert page.locator(".dw-review-replay-row").count() == 0
    assert page.evaluate("() => reviewCoverageReplayComparison(window.__workspace).length") == 0

    page.locator("#dw-clear-incoming-coverage").click()
    assert page.locator("#dw-incoming-coverage-manifest").input_value() == ""
    assert page.locator(".dw-review-replay-error").count() == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_7_delta_queue_collects_only_nonmatched_replay_items_and_revisit_is_trace_only(page) -> None:
    page.evaluate("""() => {
      window.__fetchCount=0;
      window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')};
    }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    for area in ("candidate", "graph"):
        page.locator(f'[data-dw-review-area="{area}"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")

    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    page.evaluate("""() => {
      window.__workspace.mechanism_graph={status:'UNAVAILABLE',nodes:[],edges:[]};
      renderWorkspacePanels();
    }""")

    queue = page.evaluate("() => Object.fromEntries(reviewReplayDeltaQueueSnapshot(window.__workspace).map(x => [x.area,{status:x.status,revisited:x.revisited}]))")
    assert queue["candidate"] == {"status": "CHANGED", "revisited": False}
    assert queue["graph"] == {"status": "NOT_REPRODUCED", "revisited": False}
    for area in ("identity", "evidence", "traces", "agent"):
        assert queue[area]["status"] == "NOT_PROVIDED"
    assert "MATCHED" not in {item["status"] for item in queue.values()}

    closure_before = page.evaluate("() => reviewClosureState(window.__workspace).status")
    signals_before = page.evaluate("() => reviewAttentionSignals(window.__workspace).map(x => x.code)")
    page.locator('[data-dw-rereview-toggle-area="candidate"]').click()
    candidate = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'candidate')")
    assert candidate["status"] == "CHANGED"
    assert candidate["revisited"] is True
    assert page.evaluate("() => reviewCoverageReplayComparison(window.__workspace).find(x => x.area === 'candidate').status") == "CHANGED"
    assert page.evaluate("() => reviewSessionSnapshot().candidate") is False
    assert page.evaluate("() => reviewClosureState(window.__workspace).status") == closure_before
    assert page.evaluate("() => reviewAttentionSignals(window.__workspace).map(x => x.code)") == signals_before
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_7_revisit_acknowledgement_is_bound_to_exact_current_comparison_fingerprint(page) -> None:
    _render(page)
    page.locator('[data-dw-review-area="graph"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()

    page.locator('[data-dw-main="graph"]').click()
    page.locator('[data-dw-node="component:PMP420"]').click()
    assert page.evaluate("() => reviewCoverageReplayComparison(window.__workspace).find(x => x.area === 'graph').status") == "CHANGED"
    first = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'graph')")
    page.locator('[data-dw-rereview-toggle-area="graph"]').click()
    revisited = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'graph')")
    assert revisited["revisited"] is True
    assert revisited["fingerprint"] == first["fingerprint"]

    page.locator('[data-dw-node="component:FT420"]').click()
    changed_again = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'graph')")
    assert changed_again["status"] == "CHANGED"
    assert changed_again["fingerprint"] != revisited["fingerprint"]
    assert changed_again["revisited"] is False
    assert page.locator('[data-dw-rereview-area="graph"].pending').count() == 1


def test_r35_7_guided_queue_navigation_handoff_and_clear_remain_browser_local(page) -> None:
    _render(page)
    page.locator('[data-dw-review-area="candidate"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()

    snapshot = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace)")
    assert snapshot[0]["area"] == "identity"  # NOT_PROVIDED retains canonical review-area order.
    active_before = page.locator(".dw-rereview-row.active").get_attribute("data-dw-rereview-area")
    page.locator("#dw-rereview-next").click()
    active_after = page.locator(".dw-rereview-row.active").get_attribute("data-dw-rereview-area")
    assert active_after != active_before

    page.locator('[data-dw-rereview-nav-area="candidate"]').click()
    page.wait_for_function("() => document.activeElement?.dataset?.dwCandidate === 'PMP420:FailedOff'")
    page.locator('[data-dw-rereview-toggle-area="candidate"]').click()
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Incoming re-review queue:" in handoff
    assert "candidate=CHANGED/REVISITED" in handoff
    assert "Incoming re-review pending areas:" in handoff
    assert "INCOMING_REPLAY_REREVIEW_TRACE_ONLY_NOT_RESOLUTION_OR_APPROVAL" in handoff

    page.locator("#dw-clear-incoming-coverage").click()
    assert page.evaluate("() => incomingReviewCoverageSnapshot()") is None
    assert page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace)") == []
    state = page.evaluate("() => ({cursor:workspaceReplayDeltaQueue.cursorArea,acks:Object.values(workspaceReplayDeltaQueue.acknowledgements)})")
    assert state["cursor"] is None
    assert all(value is None for value in state["acks"])
    assert page.locator(".dw-rereview-queue").count() == 0



def test_r35_8_changed_candidate_replay_exposes_exact_field_level_basis_diff(page) -> None:
    page.evaluate("""() => { window.__fetchCount=0; window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')}; }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")
    page.locator('[data-dw-review-area="candidate"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()

    diff = page.evaluate("() => replayBasisDiffSnapshot(window.__workspace).find(x => x.area === 'candidate')")
    assert diff["status"] == "CHANGED"
    assert diff["changedFields"] == ["candidate_view", "candidate_mode"]
    fields = {item["key"]: item for item in diff["fields"]}
    assert fields["presentation"]["status"] == "SAME"
    assert fields["source_bundle"]["status"] == "SAME"
    assert fields["candidate_view"] == {
        "key": "candidate_view", "status": "VALUE_CHANGED",
        "incoming": "PMP420:FlowBlocked", "current": "PMP420:FailedOff",
    }
    assert fields["candidate_mode"] == {
        "key": "candidate_mode", "status": "VALUE_CHANGED",
        "incoming": "FROZEN_REPORT_CANDIDATE", "current": "TRANSIENT_ALTERNATIVE_BROWSE_ONLY",
    }
    assert page.locator('[data-dw-replay-basis-area="candidate"]').count() == 1
    page.locator('[data-dw-replay-basis-area="candidate"] summary').click()
    assert page.locator('[data-dw-replay-basis-area="candidate"] [data-dw-replay-basis-field="candidate_view"]').count() == 1
    assert page.evaluate("() => reviewCoverageReplayComparison(window.__workspace).find(x => x.area === 'candidate').status") == "CHANGED"
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_8_basis_diff_keeps_same_fields_visible_and_guides_rereview_by_changed_field_names(page) -> None:
    _render(page)
    page.locator('[data-dw-review-area="graph"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.locator('[data-dw-main="graph"]').click()
    page.locator('[data-dw-node="component:PMP420"]').click()

    diff = page.evaluate("() => replayBasisDiffSnapshot(window.__workspace).find(x => x.area === 'graph')")
    assert diff["changedFields"] == ["selected_node"]
    fields = {item["key"]: item for item in diff["fields"]}
    assert fields["graph_provenance"]["status"] == "SAME"
    assert fields["graph_nodes"]["status"] == "SAME"
    assert fields["selected_node"]["status"] == "VALUE_CHANGED"
    row = page.locator('[data-dw-rereview-area="graph"]')
    assert "变化字段：selected_node" in row.inner_text()
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Incoming changed basis fields: graph=[selected_node]" in handoff
    assert "INCOMING_REPLAY_BASIS_DIFF_DISPLAY_ONLY_NOT_CAUSAL_EXPLANATION_OR_RESOLUTION" in handoff


def test_r35_8_not_reproduced_or_not_provided_does_not_fabricate_field_diff(page) -> None:
    _render(page)
    page.locator('[data-dw-review-area="graph"]').click()
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.evaluate("""() => { window.__workspace.mechanism_graph={status:'UNAVAILABLE',nodes:[],edges:[]}; renderWorkspacePanels(); }""")

    comparison = page.evaluate("() => Object.fromEntries(reviewCoverageReplayComparison(window.__workspace).map(x => [x.area,x.status]))")
    assert comparison["graph"] == "NOT_REPRODUCED"
    assert comparison["identity"] == "NOT_PROVIDED"
    assert page.evaluate("() => replayBasisDiffSnapshot(window.__workspace)") == []
    assert page.locator('[data-dw-replay-basis-area="graph"]').count() == 0
    assert page.locator('[data-dw-replay-basis-area="identity"]').count() == 0
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Incoming changed basis fields: none" in handoff
    assert page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'graph').status") == "NOT_REPRODUCED"

def test_r35_9_incoming_replay_all_matched_has_explicit_convergence_summary(page) -> None:
    _render(page)
    page.evaluate("() => ['identity','candidate','evidence','graph','traces','agent'].forEach(id => setReviewAreaReviewed(id,true))")
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()

    state = page.evaluate("() => incomingReplayConvergenceState(window.__workspace)")
    assert state["contract"] == "BROWSER_INCOMING_REREVIEW_CONVERGENCE_V1"
    assert state["status"] == "ALL_MATCHED"
    assert state["matchedCount"] == 6
    assert state["deltaCount"] == 0
    assert state["revisitedCount"] == 0
    assert state["pendingCount"] == 0
    assert page.locator('[data-dw-rereview-convergence="ALL_MATCHED"]').count() == 1
    assert page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace)") == []


def test_r35_9_replay_deltas_all_revisited_does_not_reclassify_original_comparison(page) -> None:
    page.evaluate("""() => { window.__fetchCount=0; window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')}; }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")
    page.evaluate("() => ['identity','candidate','evidence','graph','traces','agent'].forEach(id => setReviewAreaReviewed(id,true))")
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()

    pending = page.evaluate("() => incomingReplayConvergenceState(window.__workspace)")
    assert pending["status"] == "REREVIEW_PENDING"
    assert pending["deltaCount"] == 1
    assert pending["pendingAreas"] == ["candidate"]
    page.locator('[data-dw-rereview-toggle-area="candidate"]').click()

    converged = page.evaluate("() => incomingReplayConvergenceState(window.__workspace)")
    assert converged["status"] == "DELTAS_ALL_REVISITED"
    assert converged["deltaCount"] == 1
    assert converged["revisitedCount"] == 1
    assert converged["pendingCount"] == 0
    assert page.evaluate("() => reviewCoverageReplayComparison(window.__workspace).find(x => x.area === 'candidate').status") == "CHANGED"
    assert page.locator('[data-dw-rereview-convergence="DELTAS_ALL_REVISITED"]').count() == 1
    handoff = page.locator("#dw-handoff-text").inner_text()
    assert "Incoming re-review convergence: DELTAS_ALL_REVISITED" in handoff
    assert "Incoming re-review convergence counts: matched=5, delta=1, revisited=1, pending=0" in handoff
    assert "INCOMING_REREVIEW_CONVERGENCE_SUMMARY_ONLY_NOT_DISCREPANCY_RESOLUTION_OR_APPROVAL" in handoff
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before


def test_r35_9_convergence_returns_to_pending_when_revisited_fingerprint_changes(page) -> None:
    _render(page)
    page.evaluate("() => ['identity','candidate','evidence','graph','traces','agent'].forEach(id => setReviewAreaReviewed(id,true))")
    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.evaluate("() => { resetWorkspaceReviewSession(); renderWorkspacePanels(); }")
    page.locator("#dw-incoming-coverage-manifest").fill(manifest)
    page.locator("#dw-load-incoming-coverage").click()
    page.locator('[data-dw-main="graph"]').click()
    page.locator('[data-dw-node="component:PMP420"]').click()
    assert page.evaluate("() => incomingReplayConvergenceState(window.__workspace).status") == "REREVIEW_PENDING"
    page.locator('[data-dw-rereview-toggle-area="graph"]').click()
    assert page.evaluate("() => incomingReplayConvergenceState(window.__workspace).status") == "DELTAS_ALL_REVISITED"

    first = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'graph').fingerprint")
    page.locator('[data-dw-node="component:FT420"]').click()
    second = page.evaluate("() => reviewReplayDeltaQueueSnapshot(window.__workspace).find(x => x.area === 'graph')")
    assert second["fingerprint"] != first
    assert second["revisited"] is False
    state = page.evaluate("() => incomingReplayConvergenceState(window.__workspace)")
    assert state["status"] == "REREVIEW_PENDING"
    assert state["pendingAreas"] == ["graph"]



def test_r35_10_stable_candidate_status_overview_keeps_local_and_incoming_state_independent(page) -> None:
    page.evaluate("""() => { window.__fetchCount=0; window.fetch=()=>{window.__fetchCount+=1;throw new Error('unexpected fetch')}; }""")
    _render(page)
    maintenance_before = page.evaluate("() => JSON.stringify(window.__maintenance)")
    storage_before = page.evaluate("() => JSON.stringify(window.localStorage._data)")

    initial = page.evaluate("() => reviewStatusOverviewState(window.__workspace)")
    assert initial["boundary"] == "R35_PRODUCT_EXPERIENCE_STABLE_CANDIDATE_READ_ONLY"
    assert initial["localReviewStatus"] == "REVIEW_INCOMPLETE"
    assert initial["incomingReplayLoaded"] is False
    assert initial["incomingReReviewStatus"] == "NO_INCOMING_REPLAY"
    assert page.locator('.dw-review-status-overview').count() == 1
    assert page.locator('.dw-review-status-overview').get_by_text('本页材料复核', exact=True).count() == 1
    assert page.locator('.dw-review-status-overview').get_by_text('接手重新查看', exact=True).count() == 1
    assert page.get_by_text('本页材料复核状态', exact=True).count() == 1

    page.evaluate("() => ['identity','candidate','evidence','graph','traces','agent'].forEach(id => setReviewAreaReviewed(id,true))")
    reviewed = page.evaluate("() => reviewStatusOverviewState(window.__workspace)")
    assert reviewed["localReviewStatus"] == "REVIEWED_NO_EXPLICIT_ATTENTION"
    assert reviewed["incomingReReviewStatus"] == "NO_INCOMING_REPLAY"

    manifest = page.evaluate("() => reviewCoverageReplayManifestBlock(window.__workspace)")
    page.locator('#dw-incoming-coverage-manifest').fill(manifest)
    page.locator('#dw-load-incoming-coverage').click()
    matched = page.evaluate("() => reviewStatusOverviewState(window.__workspace)")
    assert matched["localReviewStatus"] == "REVIEWED_NO_EXPLICIT_ATTENTION"
    assert matched["incomingReplayLoaded"] is True
    assert matched["incomingReReviewStatus"] == "ALL_MATCHED"
    assert page.get_by_text('接手重新查看状态', exact=True).count() == 1
    assert page.get_by_text('接手复核收口摘要', exact=True).count() == 0

    page.locator('[data-dw-candidate="PMP420:FailedOff"]').click()
    changed = page.evaluate("() => reviewStatusOverviewState(window.__workspace)")
    assert changed["localReviewStatus"] == "REVIEWED_WITH_ATTENTION"
    assert changed["incomingReReviewStatus"] == "REREVIEW_PENDING"
    assert changed["replayDeltaCount"] == 1
    assert changed["pendingReReviewCount"] == 1

    page.locator('[data-dw-rereview-toggle-area="candidate"]').click()
    revisited = page.evaluate("() => reviewStatusOverviewState(window.__workspace)")
    assert revisited["localReviewStatus"] == "REVIEWED_WITH_ATTENTION"
    assert revisited["incomingReReviewStatus"] == "DELTAS_ALL_REVISITED"
    assert revisited["pendingReReviewCount"] == 0
    assert page.evaluate("() => reviewCoverageReplayComparison(window.__workspace).find(x => x.area === 'candidate').status") == "CHANGED"

    handoff = page.locator('#dw-handoff-text').inner_text()
    assert "Local page review status: REVIEWED_WITH_ATTENTION" in handoff
    assert "Incoming re-review status: DELTAS_ALL_REVISITED" in handoff
    assert "Review status boundary: R35_PRODUCT_EXPERIENCE_STABLE_CANDIDATE_READ_ONLY" in handoff
    assert page.evaluate("() => window.__fetchCount") == 0
    assert page.evaluate("() => JSON.stringify(window.localStorage._data)") == storage_before
    assert page.evaluate("() => JSON.stringify(window.__maintenance)") == maintenance_before
