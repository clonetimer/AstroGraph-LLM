"""Run the complete test suite and keep evidence OUTSIDE the immutable source.

No fail-fast, ignored tests, auto-installation, model downloads, or network claims.
Existing pytest skip markers remain visible in the XML and -ra log.
"""
from __future__ import annotations
import argparse,ast,datetime,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from xml.etree import ElementTree
from importlib.metadata import version,PackageNotFoundError
from verify_source import verify


def digest(root:Path)->str:
    return hashlib.sha256((root/"SHA256SUMS.txt").read_bytes()).hexdigest()


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--threads",type=int,default=1)
    args=p.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    if args.threads<1:p.error("--threads must be positive")
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output=(args.output or root.parent/("validation-local-"+stamp)).resolve()
    if output==root or output.is_relative_to(root) or root.is_relative_to(output):
        p.error("--output must be outside and not an ancestor of the source root")
    output.mkdir(parents=True,exist_ok=False)
    # One unchanged historical regression is bound to the original design report.
    # Keep this opaque user-supplied fixture adjacent to source/, as the original
    # test requires; fail preflight rather than relying on an author's workspace.
    artifact=root/"governance/evidence/research/document_corpus/early_design_report_r001/design_drift_candidates.json"
    fixture=root.parent/"智能故障检测软件_设计报告.docx"
    expected=json.loads(artifact.read_text(encoding="utf-8"))["source_document_sha256"]
    observed=hashlib.sha256(fixture.read_bytes()).hexdigest() if fixture.is_file() else None
    material={"path":str(fixture),"expected_sha256":expected,"actual_sha256":observed,
              "status":"PASS" if observed==expected else "FAIL"}
    (output/"material_preflight.json").write_text(json.dumps(material,ensure_ascii=False,indent=2),encoding="utf-8")
    if material["status"]!="PASS":
        failure={"status":"FAILED_PREREQUISITE","pytest_executed":False,
                 "reason":"Keep the supplied design report adjacent to source/; checksum mismatch or missing file."}
        (output/"result.json").write_text(json.dumps(failure,indent=2),encoding="utf-8")
        print(json.dumps(failure));return 2
    initial=verify(root)
    (output/"source_before.json").write_text(json.dumps(initial,indent=2),encoding="utf-8")
    if initial["status"]!="PASS":
        print(json.dumps(initial));return 2
    syntax=[]
    for path in sorted((root/"src").rglob("*.py")):
        try:ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
        except (SyntaxError,UnicodeError) as exc:syntax.append(str(exc))
    if syntax:
        (output/"syntax_errors.json").write_text(json.dumps(syntax,indent=2));return 2
    packages={}
    for name in ["astrograph-core","pytest","numpy","pandas","pydantic","torch","scikit-learn","fastapi","httpx","jsonschema","PyYAML"]:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]="NOT_INSTALLED"
    environment={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
                 "packages":packages,"source_checksums_sha256":digest(root),"threads":args.threads,
                 "claim_boundary":"Software regression only; skip reasons and external-service status must be reported."}
    (output/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTEST_ADDOPTS="",PYTHONHASHSEED="0")
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:env[k]=str(args.threads)
    env["PYTHONPATH"]=str(root/"src")+os.pathsep+str(root)
    cmd=[sys.executable,"-m","pytest","-p","no:cacheprovider","-q","-ra","--durations=20","--junitxml="+str(output/"pytest.xml")]
    start=time.perf_counter();utc_start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("Complete pytest is running. Log:",output/"pytest.log",flush=True)
    with (output/"pytest.log").open("w",encoding="utf-8") as log:
        log.write("COMMAND: "+" ".join(cmd)+"\nCWD: "+str(root)+"\nUTC_START: "+utc_start+"\n");log.flush()
        proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    after=verify(root)
    (output/"source_after.json").write_text(json.dumps(after,indent=2),encoding="utf-8")
    counts={"passed":0,"failed":0,"errors":0,"skipped":0};skips=[]
    if (output/"pytest.xml").is_file():
        for tc in ElementTree.parse(output/"pytest.xml").findall(".//testcase"):
            if tc.find("failure") is not None:counts["failed"]+=1
            elif tc.find("error") is not None:counts["errors"]+=1
            elif tc.find("skipped") is not None:
                counts["skipped"]+=1;skips.append({"test":tc.attrib,"reason":tc.find("skipped").attrib.get("message","")})
            else:counts["passed"]+=1
    ok=proc.returncode==0 and after["status"]=="PASS" and digest(root)==environment["source_checksums_sha256"]
    result={"status":"PASS" if ok else "FAIL","pytest_exit_code":proc.returncode,"counts":counts,
            "duration_s":time.perf_counter()-start,"started_at":utc_start,
            "finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"command":cmd,
            "source_checksums_sha256":environment["source_checksums_sha256"],"skip_details":skips}
    (output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="skip_details"},ensure_ascii=False,indent=2))
    return 0 if ok else 1


if __name__=="__main__":raise SystemExit(main())
