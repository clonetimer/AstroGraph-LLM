"""Rebuild the repaired portable ZIP offline from a SHA-pinned V2.7 distribution.

No network, pip, user-data import or private credentials. Existing outputs are
never overwritten. This refresh preserves the Windows runtime/dependencies;
it does not claim those binaries were executed or newly downloaded.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
import zipfile

BASELINE_SHA256='d6f979721f80be2eff1521ad66cddea2fa12d18bd66067ed80dfabd8fd744fc2'
VERSION='2.7.1-preview.1'
ARCHIVE_NAME='AstroGraph-LLM-Win10-x64-V2.7.1-preview.zip'
STAMP=(2026,11,4,0,0,0)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def source_files(root):
    for directory in ('src','portable','docs','.github'):
        base=root/directory
        if not base.exists(): continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                yield path
    for name in ('README.md','.gitignore','SOURCE_ORIGIN.json'):
        if (root/name).is_file(): yield root/name


def refresh_distribution(repo_root: Path, baseline_zip: Path, output_dir: Path):
    root=Path(repo_root).resolve(); base=Path(baseline_zip).resolve(); output=Path(output_dir).resolve()
    if digest(base)!=BASELINE_SHA256: raise ValueError('baseline SHA-256 does not match verified V2.7 ZIP')
    if output.exists(): raise FileExistsError('output directory must not already exist')
    spec=importlib.util.spec_from_file_location('portable_builder',root/'portable/build.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    with tempfile.TemporaryDirectory(prefix='astrograph-refresh-') as scratch:
        scratch=Path(scratch)
        # The entire baseline is pinned before any extraction.
        builder._extract_archive(base,scratch)
        package=scratch/'AstroGraph-LLM-Win10-x64'
        old=json.loads((package/'MANIFEST.json').read_text(encoding='utf-8'))
        for item in old['files']:
            path=package/item['path']
            if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('baseline manifest mismatch: '+item['path'])
        for name in ('app','source','data'):
            shutil.rmtree(package/name,ignore_errors=True)
        for path in package.glob('RELEASE_NOTES*.md'): path.unlink()
        builder.stage_app(root,package/'app'); (package/'data').mkdir()
        for name in ('START.bat','CHECK.bat','README_zh-CN.md','verify_manifest.py'):
            if name.endswith('.bat'):
                text=(root/'portable'/name).read_text(encoding='ascii').replace('\r\n','\n')
                (package/name).write_bytes(text.replace('\n','\r\n').encode('ascii'))
            else:
                shutil.copy2(root/'portable'/name,package/name)
        (package/'README_zh-CN.txt').write_text((root/'portable/README_zh-CN.md').read_text(encoding='utf-8'),encoding='utf-8-sig')
        shutil.copy2(root/'docs/RELEASE_NOTES_V2.7.1.md',package/'RELEASE_NOTES_V2.7.1.md')
        receipts=[]
        for path in source_files(root):
            rel=path.relative_to(root); target=package/'source'/rel
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            receipts.append({'path':rel.as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)})
        source_hash=hashlib.sha256(json.dumps(receipts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        write_json(package/'provenance/refresh.json',{'version':VERSION,'baseline_zip_sha256':BASELINE_SHA256,
            'source_inventory_sha256':source_hash,'source_files':receipts,'dependencies_changed':False,
            'remote_source_commit':None,'native_windows_execution_verified':False})
        pe=[]
        for path in package.rglob('*'):
            if path.suffix.lower() in {'.exe','.dll','.pyd'}:
                builder._verify_amd64(path);pe.append(path.relative_to(package).as_posix())
        manifest={k:old[k] for k in ('schema_version','source_base_commit','target','python_version','runtime_archive_sha256','dependencies')}
        manifest.update(version=VERSION,source_commit=None,check_path_quoting_fix=True,built_at='2026-11-04T00:00:00Z',
            timestamp_policy='fixed epoch for reproducible builds; not execution timestamp',
            baseline_zip_sha256=BASELINE_SHA256,source_inventory_sha256=source_hash,
            readable_application_source_included=True,native_windows_execution_verified=False,
            pdf_dependency_upgrade_verified=False,pe_amd64_count=len(pe),
            files=[{'path':p.relative_to(package).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
                   for p in sorted(package.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'])
        write_json(package/'MANIFEST.json',manifest)
        output.mkdir(parents=True)
        destination=output/package.name;shutil.copytree(package,destination)
        archive=output/ARCHIVE_NAME
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for path in sorted(destination.rglob('*')):
                if path.is_file():
                    info=zipfile.ZipInfo(path.relative_to(output).as_posix(),STAMP)
                    info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                    z.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        with zipfile.ZipFile(archive) as z:
            if z.testzip(): raise ValueError('generated ZIP CRC failed')
        archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n',encoding='ascii')
        return destination,archive

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    package,archive=refresh_distribution(Path(__file__).resolve().parents[1],args.baseline,args.output_dir)
    print(json.dumps({'zip':str(archive),'sha256':digest(archive)},indent=2))
