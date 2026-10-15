#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from astrograph.knowledge.generic_gap_review import build_gap_review_packet

ROOT=Path(__file__).resolve().parents[2]

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default=str(ROOT))
    p.add_argument('--gaps',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.project_root).resolve()
    gaps_path=Path(args.gaps); gaps_path=gaps_path if gaps_path.is_absolute() else root/gaps_path
    gaps=json.loads(gaps_path.read_text(encoding='utf-8'))
    packet=build_gap_review_packet(root,gaps)
    out=Path(args.output);out=out if out.is_absolute() else root/out;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(packet.model_dump(mode='json'),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'items':len(packet.items),'active_kg_modified':False},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
