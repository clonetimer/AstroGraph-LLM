"""Run the frozen R20 OPS-SAT experiment with data/results outside the source."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from astrograph.evaluation.ops_sat_external_validation import run
from astrograph.evaluation.external_dataset_contract import file_identity

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--official-modeling-notebook',type=Path,required=True)
    args=parser.parse_args()
    for path in (args.data_root,args.output,args.official_modeling_notebook):
        if path.resolve().is_relative_to(ROOT):parser.error('external data, provenance and results must stay outside source')
    provenance={'url':'https://raw.githubusercontent.com/kplabs-pl/OPS-SAT-AD/main/modeling_examples.ipynb',**file_identity(args.official_modeling_notebook)}
    if provenance['sha256']!='224f8a73561c25e421f1eb72b84a850118002eecf278642c690e8b4fce2d4faa':parser.error('official modeling notebook differs from frozen R20 semantics evidence')
    report=run(args.data_root,args.output,ROOT/'configs/evaluation/r20_ops_sat.json',provenance)
    print(json.dumps({'report_sha256':report['report_sha256'],'eligible_segments':report['eligible_segments'],'qualification':report['qualification']}))
    return 0

if __name__=='__main__':raise SystemExit(main())
