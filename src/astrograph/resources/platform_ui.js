import { state } from './platform_core.js';
import { renderReport } from './platform_report.js';
import { loadCapabilities, capabilitiesHtml, loadLLMSettings, loadLLMProfiles, loadNeo4jSettings, loadMCPProviders, setupSystemHooks } from './platform_system.js';
import { setMode, loadAgentCapabilities, loadTools, setupAgentHooks } from './platform_agent.js';
import { loadDashboard, loadHistory, openHistory, setupDashboardHooks } from './platform_dashboard.js';
import { invalidatePreflight, renderInputChecks, renderFiles, loadJobDiagnostics, setupJobHooks } from './platform_job.js';
import { renderAgentExecution } from './platform_agent_live.js';
import { loadKnowledgeWorkspace } from './platform_knowledge.js';
import { initShell, nav, setupShellHooks } from './platform_shell.js';
import { setupEngineSettings, loadEngineSettings } from './platform_engine_settings.js';
import { setupGraphWorkbench, loadGraphBuild, loadGraphReview, loadGraphRelease } from './platform_graph_workbench.js';
import { renderOperationsWizard, setupOperationsHooks } from './platform_operations.js';
import { renderDiagnosticWorkspace } from './platform_diagnostic_workspace.js';

setupSystemHooks({loadAgentCapabilities:(...args)=>loadAgentCapabilities(...args),loadTools:(...args)=>loadTools(...args)});
setupAgentHooks({invalidatePreflight:(...args)=>invalidatePreflight(...args),renderInputChecks:(...args)=>renderInputChecks(...args),renderAgentExecution:(...args)=>renderAgentExecution(...args)});
setupDashboardHooks({nav:(...args)=>nav(...args)});
setupJobHooks({nav:(...args)=>nav(...args),loadDashboard:(...args)=>loadDashboard(...args)});
setupOperationsHooks({nav:(...args)=>nav(...args),setMode:(...args)=>{setMode(...args);renderOperationsWizard()},renderDiagnosticWorkspace});
setupShellHooks({
  loadDashboard,loadHistory,openHistory,loadJobDiagnostics,loadAgentCapabilities,loadTools,loadKnowledgeWorkspace,loadGraphBuild,loadGraphReview,loadGraphRelease,
  loadLLMSettings,loadLLMProfiles,loadNeo4jSettings,loadMCPProviders,loadEngineSettings,loadCapabilities,capabilitiesHtml,renderReport,setMode,
});
initShell();setupGraphWorkbench();setupEngineSettings();

(async()=>{
  renderFiles();renderInputChecks();renderOperationsWizard();
  await loadCapabilities();
  await Promise.all([loadAgentCapabilities(),loadLLMSettings(),loadLLMProfiles(),loadNeo4jSettings(),loadMCPProviders(),loadTools()]);
  renderOperationsWizard();
  if(state.currentJobId)await loadJobDiagnostics(state.currentJobId);
  const initial=location.hash.slice(1);
  nav(['dashboard','new','result','history','agent','knowledge','graph-build','graph-review','graph-release','bench','system','engine'].includes(initial)?initial:'dashboard',false);
})();
