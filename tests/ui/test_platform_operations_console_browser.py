from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src/astrograph/resources"
HTML = (RESOURCES / "platform_ui.html").read_text(encoding="utf-8")
CSS = (RESOURCES / "platform_ui.css").read_text(encoding="utf-8")
MODULES = [
    "platform_core.js",
    "platform_operations.js",
    "platform_diagnostic_workspace.js",
    "platform_report.js",
    "platform_system.js",
    "platform_agent.js",
    "platform_dashboard.js",
    "platform_job.js",
    "platform_agent_live.js",
    "platform_knowledge.js",
    "platform_shell.js",
    "platform_graph_workbench.js",
    "platform_engine_settings.js",
    "platform_ui.js",
]


def _classic_test_bundle() -> str:
    chunks: list[str] = []
    for name in MODULES:
        source = (RESOURCES / name).read_text(encoding="utf-8")
        source = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
        if "\nexport {" in source:
            source = source.rsplit("\nexport {", 1)[0]
        chunks.append(source)
    return "\n".join(chunks)


JS = _classic_test_bundle()
OFFLINE_HTML = HTML.replace(
    '<link rel="stylesheet" href="/platform/assets/v0.21.1/platform_ui.css">', ""
).replace(
    '<script type="module" src="/platform/assets/v0.21.1/platform_ui.js"></script>', ""
)


def _chromium_executable() -> str | None:
    candidates = [
        os.environ.get("ASTROGRAPH_CHROMIUM_EXECUTABLE"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _launch(playwright):
    kwargs = {"headless": True, "args": ["--no-sandbox"]}
    executable = _chromium_executable()
    if executable:
        kwargs["executable_path"] = executable
    return playwright.chromium.launch(**kwargs)


def _prepare(page) -> None:
    page.evaluate(
        """Object.defineProperty(window, 'localStorage', {
          configurable: true,
          value: {
            _data: {}, getItem(k) { return this._data[k] ?? null; },
            setItem(k, v) { this._data[k] = String(v); },
            removeItem(k) { delete this._data[k]; }, clear() { this._data = {}; }
          }
        });"""
    )
    page.set_content(OFFLINE_HTML, wait_until="domcontentloaded")
    page.add_style_tag(content=CSS)
    page.add_script_tag(content=JS)
    page.evaluate(
        """() => {
          state.agentCapabilities={modes:[
            {id:'normal',display_name:'标准诊断',allowed_diagnose_tools:[]},
            {id:'thinking',display_name:'深入复核',allowed_diagnose_tools:[]},
            {id:'creative',display_name:'扩展排查',allowed_diagnose_tools:[]}
          ]};
          renderModeSelectors(); renderOperationsWizard();
        }"""
    )


def test_execution_diagnosis_and_release_are_rendered_as_three_states() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.evaluate(
            """() => enhanceOperationsResult(
              {job_id:'JOB-R33'},
              {diagnosis_disposition:'REVIEW_REQUIRED',release_status:'FORMAL_RELEASE_BLOCKED',requires_human_review:true},
              {execution_status:'SUCCEEDED'}
            )"""
        )
        text = page.locator('[data-ops-result-panel="conclusion"]').inner_text()
        assert "SUCCEEDED" in text
        assert "REVIEW_REQUIRED" in text
        assert "FORMAL_RELEASE_BLOCKED" in text
        browser.close()


def test_human_review_requirement_preserves_true_false_and_missing() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        states = page.evaluate(
            """() => [
              operatorStatusViewModel({}, {requires_human_review:true}).requiresHumanReview,
              operatorStatusViewModel({}, {requires_human_review:false}).requiresHumanReview,
              operatorStatusViewModel({}, {}).requiresHumanReview
            ]"""
        )
        assert states == [True, False, None]
        labels = []
        for maintenance in (
            {"requires_human_review": True},
            {"requires_human_review": False},
            {},
        ):
            page.evaluate(
                "([maintenance]) => enhanceOperationsResult({job_id:'JOB-R33-2'}, maintenance, {execution_status:'SUCCEEDED'})",
                [maintenance],
            )
            labels.append(page.locator('.ops-diagnosis-summary > div').nth(4).locator('strong').inner_text())
        assert labels == ["是", "否", "未提供"]
        browser.close()


def test_preflight_blocked_keeps_start_disabled() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.evaluate(
            """() => {
              state.files=[new File(['{}'],'input.json',{type:'application/json',lastModified:1})];
              state.preflight={status:'BLOCKED',can_start:false,artifact_kind:'PIPELINE_RUNTIME',blockers:[],warnings:[],capability_matrix:[]};
              state.preflightJobId='JOB-BLOCKED'; state.preflightSignature=fileSignature();
              renderFiles(); renderPreflight(); renderInputChecks();
            }"""
        )
        assert page.locator("#start-btn").is_disabled()
        assert "阻断" in page.locator("#preflight-result").inner_text()
        browser.close()


def test_preflight_pass_enables_start() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.evaluate(
            """() => {
              state.files=[new File(['{}'],'input.json',{type:'application/json',lastModified:1})];
              state.preflight={status:'PASS',can_start:true,artifact_kind:'PIPELINE_RUNTIME',blockers:[],warnings:[],capability_matrix:[]};
              state.preflightJobId='JOB-PASS'; state.preflightSignature=fileSignature();
              renderFiles(); renderPreflight(); renderInputChecks();
            }"""
        )
        assert page.locator("#start-btn").is_enabled()
        assert "服务器识别的工件合同" in page.locator("#ops-recognized-inputs").inner_text()
        browser.close()


def test_missing_evidence_bundle_is_explicit() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.evaluate("() => enhanceOperationsResult({job_id:'JOB-R33'},{evidence_summary:null},{execution_status:'SUCCEEDED'})")
        page.evaluate("() => document.querySelector('[data-ops-result-tab=\"evidence\"]').click()")
        assert "正式证据摘要不可用" in page.locator('[data-ops-result-panel="evidence"]').inner_text()
        browser.close()


def test_probe_request_body_contains_only_probe_id() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        payload = page.evaluate("() => probeRequestBody('COMMAND_RESPONSE_RESIDUAL')")
        assert payload == {"probe_id": "COMMAND_RESPONSE_RESIDUAL"}
        browser.close()


def test_probe_result_does_not_change_diagnosis_or_release_display() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        before, after = page.evaluate(
            """() => {
              const report={job_id:'JOB-R33',agent_execution:{status:'SUCCEEDED',investigation:{findings:[]}}};
              const maintenance={diagnosis_disposition:'REVIEW_REQUIRED',release_status:'FORMAL_RELEASE_BLOCKED'};
              const job={execution_status:'SUCCEEDED'};
              enhanceOperationsResult(report,maintenance,job);
              const read=()=>document.querySelector('[data-ops-result-panel="conclusion"]').innerText;
              const before=read();
              report.agent_execution.investigation.findings.push({probe_id:'COMMAND_RESPONSE_RESIDUAL',text:'probe complete',result_hash:'abc'});
              enhanceOperationsResult(report,maintenance,job);
              return [before,read()];
            }"""
        )
        assert before == after
        assert "REVIEW_REQUIRED" in after and "FORMAL_RELEASE_BLOCKED" in after
        browser.close()


def test_1366_by_768_at_160_percent_has_no_page_horizontal_overflow() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        page.locator("#font-range").evaluate("el => {el.value='160'; el.dispatchEvent(new Event('input',{bubbles:true}))}")
        page.locator('[data-view="new"]').click()
        page.wait_for_timeout(50)
        overflow = page.evaluate("document.documentElement.scrollWidth-document.documentElement.clientWidth")
        assert overflow <= 1
        browser.close()


def test_blocked_preflight_keeps_wizard_on_step_two() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        classes = page.evaluate(
            """() => {
              state.files=[new File(['{}'],'input.json',{type:'application/json',lastModified:1})];
              state.preflight={status:'BLOCKED',can_start:false,recognized_inputs:[]};
              state.preflightSignature='locked';
              renderOperationsWizard();
              return [...document.querySelectorAll('[data-ops-step]')].map(x=>x.className);
            }"""
        )
        assert "active" not in classes[0]
        assert "active" in classes[1] and "blocked" in classes[1] and "complete" not in classes[1]
        assert "active" not in classes[2] and "complete" not in classes[2]
        browser.close()


def test_recognized_input_false_values_preserve_missing_state() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        result = page.evaluate(
            """() => {
              const arrayHtml=preflightRecognizedInputsHtml({recognized_inputs:[{kind:'Command log',present:false}]});
              const mappingHtml=preflightRecognizedInputsHtml({recognized_inputs:{Telemetry:{label:'Telemetry',available:false}}});
              const host=document.createElement('div'); host.innerHTML=arrayHtml+mappingHtml;
              return {text:host.innerText,html:host.innerHTML};
            }"""
        )
        assert result["text"].count("缺失") == 2
        assert "[object Object]" not in result["html"]
        browser.close()


def test_agent_page_does_not_duplicate_investigation_findings() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        result = page.evaluate(
            """() => {
              state.report={agent_execution:{work_mode:'thinking',status:'SUCCEEDED',executed_actions:[],investigation:{findings:[{probe_id:'COMMAND_RESPONSE_RESIDUAL',text:'unique finding text'}]}}};
              renderAgentExecution();
              const host=document.querySelector('#agent-investigation');
              return {findings:host.querySelectorAll('details.finding').length,timelines:host.querySelectorAll('.ops-investigation').length};
            }"""
        )
        assert result == {"findings": 1, "timelines": 0}
        browser.close()


def test_creative_mode_is_only_in_advanced_agent_settings() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        primary = page.locator("#mode-grid").inner_text()
        advanced = page.locator("#agent-advanced-mode-segment").inner_text()
        assert "扩展排查（高级）" not in primary
        assert "扩展排查（高级）" in advanced
        assert "标准诊断" in primary and "增强诊断" in primary
        browser.close()


def test_direct_knowledge_navigation_loads_current_job_contracts() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        _prepare(page)
        result = page.evaluate(
            """async () => {
              state.currentJobId='JOB-DIRECT'; state.job=null; state.report=null; state.maintenance=null;
              const originalFetch=window.fetch;
              window.fetch=async url => {
                const body=String(url).endsWith('/maintenance')
                  ? {job_id:'JOB-DIRECT',component_id:'PMP420',fault_id:'FlowBlocked'}
                  : String(url).endsWith('/report')
                    ? {job_id:'JOB-DIRECT',analysis:{knowledge_enrichment:{document_ingestion:[],local_retrieval:{hits:[]}}}}
                    : {job_id:'JOB-DIRECT',status:'SUCCEEDED'};
                return new Response(JSON.stringify(body),{status:200,headers:{'content-type':'application/json'}});
              };
              await loadKnowledgeWorkspace();
              const text=document.querySelector('#knowledge-operations').innerText;
              window.fetch=originalFetch;
              return {text,job:state.job?.job_id,report:state.report?.job_id,maintenance:state.maintenance?.job_id};
            }"""
        )
        assert result["job"] == result["report"] == result["maintenance"] == "JOB-DIRECT"
        assert "PMP420" in result["text"] and "FlowBlocked" in result["text"]
        browser.close()


def test_graph_workbench_navigation_auth_and_candidate_provenance():
    with sync_playwright() as driver:
        browser = _launch(driver)
        page = browser.new_page(viewport={'width': 1366, 'height': 860})
        _prepare(page)
        page.evaluate("""() => {
          window.__expertCalls = [];
          window.__loggedIn = false;
          window.fetch = async (url, opts={}) => {
            window.__expertCalls.push({url:String(url),method:opts.method||'GET',csrf:opts.headers?.['X-Material-CSRF']||''});
            const payload = String(url).endsWith('/review-session') && opts.method==='POST'
              ? (window.__loggedIn=true, {authenticated:true,reviewer:'subject-matter-expert',csrf:'ephemeral-csrf'})
              : String(url).endsWith('/review-session')
              ? {configured:true,authenticated:window.__loggedIn,reviewer:'subject-matter-expert',csrf:window.__loggedIn?'ephemeral-csrf':null,publisher:false,graph_model_configured:true,release_signing_configured:false,neo4j_publication_enabled:false,session_ttl_seconds:1800}
              : String(url).endsWith('/graph/build')
              ? {status:'SUCCEEDED',staging_status:'PENDING_EXPERT_REVIEW',source_group_id:'JOB-123456789ABCDEF012345678',equipment_scope_requested:'SAT_ADCS_V2',proposal_count:1,chunks_processed:1,candidate_relations:[{source_group_id:'JOB-123456789ABCDEF012345678',source:{entity_type:'FAULT_MODE',name:'ADCS_RW_JAM'},target:{entity_type:'FAILURE_MECHANISM',name:'<img src=x onerror=alert(1)>'},relation_type:'ASSOCIATED_WITH',evidence_quote:'日志提示轴承异常',polarity:'AFFIRMATIVE',provenance:{source_role:'references',filename:'manual.md',source_sha256:'abcdddeeff',source_locators:['p.3']}}]}
              : {snapshot:0,graph_proposals:[],documents:[],cases:[]};
            return new Response(JSON.stringify(payload),{status:200,headers:{'content-type':'application/json'}});
          };
        }""")
        page.locator('[data-view="graph-review"]').click()
        page.locator('#graph-review-credential').fill('one-time-local-secret')
        page.locator('#graph-login-form button[type="submit"]').click()
        page.wait_for_function("document.querySelector('#graph-auth-status').textContent.includes('subject-matter-expert')")
        assert page.locator('#graph-review-credential').input_value() == ''
        assert page.locator('#graph-login-form').evaluate("el => el.classList.contains('hidden')")
        assert page.evaluate("Object.keys(localStorage._data).every(k=>!k.includes('material'))")
        page.locator('[data-view="graph-build"]').click()
        page.locator('#graph-equipment-scope').fill('SAT_ADCS_V2')
        page.locator('#graph-files').set_input_files({'name':'manual.md','mimeType':'text/markdown','buffer':b'example spacecraft graph manual'})
        page.locator('#graph-build-submit').click()
        page.wait_for_function("document.querySelector('#graph-build-result').textContent.includes('ADCS_RW_JAM')")
        result = page.locator('#graph-build-result')
        assert '日志提示轴承异常' in result.inner_text()
        assert result.locator('a[href*="/graph/sources/JOB-"]').count() == 1
        assert result.locator('img').count() == 0  # untrusted extracted names must be HTML-escaped
        calls = page.evaluate("window.__expertCalls.filter(c=>c.url.endsWith('/graph/build'))")
        assert len(calls) == 1 and calls[0]['csrf'] == 'ephemeral-csrf'
        browser.close()


def test_graph_release_requires_admin_readiness_and_responsive_console():
    with sync_playwright() as driver:
        browser = _launch(driver)
        page = browser.new_page(viewport={'width': 390, 'height': 844})
        _prepare(page)
        page.evaluate("""() => {
          session={configured:true,authenticated:true,reviewer:'expert',csrf:'test-csrf',publisher:false,graph_model_configured:true,neo4j_publication_enabled:false,release_signing_configured:false};
          authPanel();
          releaseSnapshot=3;
          releaseId='SAT_ADCS_V2-SNAPSHOT-00000003';
          readiness();
        }""")
        page.evaluate("nav('graph-release')")
        page.locator('#graph-publish-confirm').check()
        assert page.locator('#graph-export').is_disabled()
        assert page.locator('#graph-publish').is_disabled()
        assert '知识版本与发布' in page.locator('#view-graph-release h1').inner_text()
        for view in ('graph-build','graph-review','graph-release'):
            page.evaluate('v=>nav(v)',view)
            assert page.locator('#view-'+view).get_attribute('class').split().__contains__('active')
            overflow = page.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth')
            bad = page.evaluate('''() => [...document.querySelectorAll('*')].filter(e=>e.getBoundingClientRect().right > window.innerWidth+2 && e.getBoundingClientRect().width > 0 && getComputedStyle(e).display!=='none').map(e=>({tag:e.tagName,id:e.id,cls:e.className,rect:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).slice(0,16)''')
            assert overflow <= 1,(view, overflow, bad)
        browser.close()


def test_material_multihop_guard_is_visible_and_unknown_is_not_certified() -> None:
    """An ABSTAIN state must be seen before expanding graph details."""
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1280, "height": 780})
        _prepare(page)
        page.evaluate("""() => renderResultLogReview({
          artifact_kind:'RESULT_LOG_REVIEW', analysis:{
            result_log_review:{
              conclusion:'上游故障待核实',evidence:[],evidence_gaps:[],
              candidates:[{rank:1,fault_id:'UNKNOWN_TEST',fault_name:'未知模式',status:'UNKNOWN',
                cause_hypotheses:[{text:'仅为上游线索'}],handling_measures:[],
                verification_checks:[],diagnosis_source_ids:[],log_evidence_ids:[],
                reference_citation_ids:[]}]},
            material_reasoning:{answer_policy:{all_candidates_abstained:true},
              candidates:[{fault_id:'UNKNOWN_TEST',fault_name:'未知模式',
              multi_hop:{decision:'ABSTAIN_NO_APPROVED_PATH',
                explanation:'不能证明根因',knowledge_gap_codes:['MISSING_APPROVED_FAULT_RELATIONS'],
                approved_paths:[],public_reference_paths:[]}}]}
          }
        })""")
        page.evaluate("() => nav('result', false)")
        container = page.locator('#result-content')
        assert container.get_by_role('status').is_visible()
        assert 'ABSTAIN' in container.get_by_role('status').inner_text()
        assert '无法确诊根因' in container.get_by_role('status').inner_text()
        assert '未经独立证实' in container.inner_text()
        assert 'ABSTAIN_NO_APPROVED_PATH' in container.inner_text()
        assert container.locator('summary', has_text='2～3 跳证据约束推理').count() == 1
        browser.close()


def test_hierarchical_rag_stages_and_unverified_modality_are_visible_without_causal_claim() -> None:
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1280, "height": 780})
        _prepare(page)
        page.evaluate('''() => renderResultLogReview({
            artifact_kind:'RESULT_LOG_REVIEW',analysis:{
              result_log_review:{conclusion:'待人工核验',evidence:[],evidence_gaps:[],candidates:[{
                rank:1,fault_id:'ADCS_RW_FRICTION_INCREASE',fault_name:'飞轮摩擦异常',
                status:'UNKNOWN',cause_hypotheses:[],handling_measures:[],verification_checks:[],
                diagnosis_source_ids:[],log_evidence_ids:[],reference_citation_ids:[]}]},
              material_reasoning:{answer_policy:{all_candidates_abstained:true},
                candidates:[{fault_id:'ADCS_RW_FRICTION_INCREASE',fault_name:'飞轮摩擦异常',
                  reference_retrieval:{retrieval_mode:'HIERARCHICAL_HYBRID',status:'SUCCEEDED',
                    terminology:{matches:[{concept_id:'reaction_wheel',matched_alias:'反作用飞轮'}]},
                    query_plan:[{role:'CURATED_SYNONYM',query:'reaction wheel'}],
                    retrieval_trace:{document_index_count:8,chunk_index_count:200,
                      unsafe_representation_excluded_count:3,
                      modalities:{TEXT:190,IMAGE:3,TABLE:7},excluded_modalities:{IMAGE:3},
                      stages:[{stage:'DOCUMENT_SELECTION',selected:[{document_id:'doc:1',score:0.19}]},
                              {stage:'SECTION_SELECTION',selected:[{document_id:'doc:1',section_path:['ADCS']}]},
                              {stage:'FINE_CITATION_RERANK_AND_DIVERSIFY',returned:['chunk-1']}]},
                    hits:[{knowledge_id:'chunk-1',filename:'manual.md',source_locators:['line:44'],
                      modalities:['TABLE'],retrieval_channels:['BM25','CHAR_NGRAM'],
                      knowledge_validation_status:'UNREVIEWED_UPLOADED_REFERENCE',
                      table_header_context:'| component | limit |',
                      quote:'| reaction wheel | 3A |'}]},
                  multi_hop:{decision:'ABSTAIN_NO_APPROVED_PATH',approved_paths:[],
                             public_reference_paths:[],knowledge_gap_codes:[]}}
                ]}
            }
        })''')
        page.evaluate("() => nav('result',false)")
        container=page.locator('#result-content')
        assert container.get_by_role('status').is_visible()
        assert 'ABSTAIN' in container.get_by_role('status').inner_text()
        retrieval=container.locator('summary',has_text='文档 → 章节 → 片段')
        assert retrieval.count()==1
        retrieval.click()
        assert '排除未审/不安全模态片段 3' in container.inner_text()
        assert '表头（原文上下文）' in container.inner_text()
        container.locator('summary',has_text='查看检索计划、阶段和模态覆盖').click()
        assert 'CURATED_SYNONYM' in container.inner_text()
        assert 'SECTION_SELECTION' in container.inner_text()
        browser.close()
