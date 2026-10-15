#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from astrograph.governance.agent_tool_specification import build_agent_tool_governance_catalog

p=argparse.ArgumentParser()
p.add_argument('--output', required=True)
a=p.parse_args()
out=Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
catalog=build_agent_tool_governance_catalog()
out.write_text(json.dumps(catalog.model_dump(mode='json'), ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
print(json.dumps({'status':'PASS','output':str(out),'tool_count':len(catalog.tools),'catalog_sha256':catalog.catalog_sha256},ensure_ascii=False,sort_keys=True))
