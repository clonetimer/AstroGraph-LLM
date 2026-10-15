"""Extract the user's embedded scene assets; drop all region and time-window UI.

Mechanical extraction only: the original uploaded HTML remains unchanged.
Run from the workspace: .venv/Scripts/python source/scripts/extract_home_scene.py
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
original = (root / 'earth_sate.html').read_text(encoding='utf-8')
out = root / 'source/src/astrograph/resources'
script = original.split('<script type="module">', 1)[1].rsplit('</script>', 1)[0]
library, rest = script.split('const embeddedTextures = ', 1)
(out / 'home_three.js').write_text(library + '\nexport default THREE;\n', encoding='utf-8')
textures, rest = rest.split('\nfunction gmstRadians', 1)
scene = 'import THREE from "./home_three.js";\nconst embeddedTextures = ' + textures + '\nfunction gmstRadians' + rest
# Remove geometry creation as well as inputs: no invisible geographic overlays.
scene = scene.replace('let regionGroup = new THREE.Group();\nearthGroup.add(regionGroup);\nlet regionSignature = "";', '')
scene = re.sub(r'function regionSurface\(region, color\) \{.*?(?=function updateFromEvaluation)', '', scene, flags=re.S)
scene = scene.replace('  buildRegions(evaluation?.regions || evaluation?.area_results || []);', '')
scene = re.sub(r'const drawer = document.querySelector\("#orbit-region-drawer"\);.*?(?=const satelliteTag)', '', scene, flags=re.S)
scene = scene.replace('  document.querySelector("#orbit-satellite-name").textContent = orbit.satellite_name || "演示卫星01";', '')
scene = scene.replace('  document.querySelector("#orbit-scale").textContent = `仿真倍率 ${simulationScale.toFixed(0)}×`;', '')
scene = scene.replace('let cameraDistance = 3.45;', 'let cameraDistance = 4.3;').replace('cameraDistance = 3.45;', 'cameraDistance = 4.3;')
scene = scene.replace('distance=3.45', 'distance=4.3')
scene = scene.replace('let demoScale=240;', 'let demoScale=60;')
scene = re.sub(r'let regionCounter=2;.*?(?=const getSimEpoch)', '', scene, flags=re.S)
scene = re.sub(r'function getSampleTimeJudgement\(millis\) \{.*?(?=function computeEvaluation)', '', scene, flags=re.S)
scene = re.sub(r' const evaluatedRegions=.*?\n const matchedRegionNames=.*?\n', '', scene)
scene = scene.replace('  regions:evaluatedRegions,matched_region_names:matchedRegionNames,', '')
scene = scene.replace('  time_evaluation:getSampleTimeJudgement(simulatedEpoch),', '')
scene = scene.split('function renderRegionList', 1)[0]
scene += '''
let scenePaused = matchMedia('(prefers-reduced-motion: reduce)').matches;
function syncScene() {
  const result = computeEvaluation();
  window.dispatchEvent(new CustomEvent('orbit:evaluation', {detail:result}));
}
function setPaused(paused) {
  const now=Date.now();
  simulationStartEpoch=getSimEpoch(now); simulationStartReal=now;
  scenePaused=paused; demoScale=paused?0:60;
  document.querySelector('#scene-pause').textContent=paused?'继续运行':'暂停演示';
  document.querySelector('#scene-pause').setAttribute('aria-pressed',String(paused));
  syncScene();
}
document.querySelector('#scene-pause').addEventListener('click',()=>setPaused(!scenePaused));
setPaused(scenePaused);
setInterval(()=>{if(!document.hidden)syncScene()},500);
'''
scene = scene.replace('simulationScale = Number(orbit.time_scale) || 1;', 'simulationScale = Number(orbit.time_scale) || 0;')
scene = scene.replace('  satellite.rotation.y += 0.006;', '  if (simulationScale) satellite.rotation.y += 0.006;')
(out / 'home_scene.js').write_text(scene, encoding='utf-8')
print('Extracted original Three.js, textures, Earth, orbit and satellite; removed region rendering and controls.')
