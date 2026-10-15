#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,zipfile
from pathlib import Path
from astrograph.evaluation.nasa_adapt_adapter import system_catalog
from astrograph.simulation.adapt_component_model_registry import build_adapt_mechanism_model_registry

p=argparse.ArgumentParser();p.add_argument('--adapt-systems-zip',required=True);p.add_argument('--member',default='AdaptDXC/AdaptDXC.xml');p.add_argument('--output',required=True);a=p.parse_args()
with zipfile.ZipFile(a.adapt_systems_zip) as z: catalog=system_catalog(z.read(a.member))
registry=build_adapt_mechanism_model_registry(catalog)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(registry.model_dump(mode='json'),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','output':str(out),'registry_sha256':registry.registry_sha256,'coverage':registry.coverage},ensure_ascii=False,sort_keys=True))
