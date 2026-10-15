"""Optional OFFLINE OCR + local vision candidates for scanned engineering manuals.

Machine outputs are always UNREVIEWED. They require independent page/figure
review before they may be used as trustworthy diagnosis/reference evidence.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener

from astrograph.retrieval.structured_document import DocumentElement, DocumentElementKind

MAX_PAGES = 40
MAX_PDF_BYTES = 80 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 300000


def _source_pdf(pdf: Path) -> tuple[str, int]:
    pdf = Path(pdf).resolve(strict=True)
    if not pdf.is_file() or pdf.suffix.lower() != '.pdf' or pdf.stat().st_size > MAX_PDF_BYTES:
        raise ValueError('only bounded local PDF input is permitted')
    if not shutil.which('pdfinfo'):
        raise RuntimeError('pdfinfo (Poppler) is unavailable')
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True,
                          check=True, timeout=20).stdout
    match = re.search(r'(?m)^Pages:\s+(\d+)\s*$', info)
    if not match:
        raise ValueError('pdfinfo page count missing')
    pages = int(match[1])
    if pages < 1 or pages > MAX_PAGES:
        raise ValueError('scanned PDF exceeds explicit 40-page OCR budget')
    return hashlib.sha256(pdf.read_bytes()).hexdigest(), pages


def render_page_png(pdf: Path, page: int, destination: Path, *, resolution: int = 160) -> Path:
    if not shutil.which('pdftoppm'):
        raise RuntimeError('pdftoppm (Poppler) is unavailable')
    if not 1 <= page <= MAX_PAGES:
        raise ValueError('page number outside bounded policy')
    subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-r',str(resolution),
                    '-singlefile','-png',str(pdf),str(destination.with_suffix(''))],
                   capture_output=True, timeout=45, check=True)
    if not destination.is_file() or destination.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError('page raster missing or oversized')
    return destination


def ocr_scanned_pdf(pdf: str | Path, *, languages: str = 'chi_sim+eng',
                    maximum_chars: int = MAX_TEXT_CHARS) -> dict[str, Any]:
    """Run local Tesseract per page with immutable page evidence coordinates."""
    pdf = Path(pdf).resolve()
    sha, pages = _source_pdf(pdf)
    if not re.fullmatch(r'[A-Za-z0-9_+]{3,50}', languages):
        raise ValueError('invalid OCR language selector')
    available = {p.strip() for p in subprocess.run(['tesseract','--list-langs'],
        capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()}
    if any(lang not in available for lang in languages.split('+')):
        raise RuntimeError('requested offline OCR language pack is missing')
    elements=[]
    warnings=[]
    with tempfile.TemporaryDirectory(prefix='astrograph-ocr-') as directory:
        for page in range(1, pages + 1):
            png = render_page_png(pdf, page, Path(directory)/f'page_{page:03d}.png')
            output = subprocess.run(['tesseract',str(png),'stdout','-l',languages,'--psm','3'],
                                    capture_output=True, timeout=45, check=True)
            text = output.stdout.decode('utf-8', errors='replace').strip()
            if not text:
                warnings.append(f'OCR_PAGE_EMPTY:{page}')
                continue
            if len(text) > 30000:
                raise ValueError('OCR page text exceeds safety budget')
            elements.append(DocumentElement(element_id=f'ocr:{page:04d}',
                kind=DocumentElementKind.TEXT,
                section_path=[f'Page {page}'],text_representation=text,
                source_locator=f'pdf#page:{page}:ocr:tesseract',
                representation_method='offline-tesseract-v1',representation_reviewed=False))
    if sum(len(e.text_representation) for e in elements) > maximum_chars:
        raise ValueError('OCR total extracted text exceeds safety budget')
    return {'status':'OCR_UNREVIEWED' if elements else 'OCR_NO_TEXT',
            'source_sha256':sha, 'parser':'TESSERACT_OFFLINE_LOCAL',
            'page_count':pages, 'recognized_pages':len(elements),
            'review_required':True, 'warnings':warnings, 'elements':elements,
            'network_access_allowed':False,
            'claim_boundary':'OCR_TEXT_UNVERIFIED_NEVER_AUTOMATIC_DIAGNOSIS_EVIDENCE'}


def docling_pdf_candidates(pdf: str | Path) -> dict[str, Any]:
    """Optional structure-aware Docling conversion, held for expert review.

    No remote service or model download permitted; deployment must preinstall
    model artifacts. Granular locators are preserved where Docling provides them.
    """
    source = Path(pdf).resolve()
    sha, pages = _source_pdf(source)
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as exc:
        raise RuntimeError('Docling optional dependency not installed') from exc
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.do_formula_enrichment = True
    options.enable_remote_services = False
    options.document_timeout = 180
    # Forbid implicit Hugging Face/network model downloads on a deployment that
    # has not explicitly preinstalled the Docling model artefacts.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    converted = converter.convert(str(source))
    doc = converted.document
    elements = []
    for idx, (item, _depth) in enumerate(doc.iterate_items(), start=1):
        raw_text = str(getattr(item,'text','') or '').strip()
        category = str(getattr(item,'label','')).lower()
        if 'table' in category:
            kind = DocumentElementKind.TABLE
            if hasattr(item,'export_to_markdown'):
                raw_text = item.export_to_markdown(doc) or raw_text
        elif 'formula' in category:
            kind = DocumentElementKind.FORMULA
        elif 'picture' in category or 'image' in category:
            kind = DocumentElementKind.IMAGE
            raw_text = raw_text or '[docling image asset requires independently reviewed vision description]'
        else:
            kind = DocumentElementKind.TEXT
        if not raw_text:
            continue
        if len(raw_text) > 30000:
            raw_text = raw_text[:30000]
        page_numbers = sorted({getattr(prov,'page_no',0) for prov in (getattr(item,'prov',None) or [])})
        locator = f'docling#pages:{",".join(map(str,page_numbers)) or "unknown"}:item:{idx}'
        elements.append(DocumentElement(element_id=f'docling:{idx:05d}',kind=kind,
            section_path=[f'Page {page_numbers[0]}' if page_numbers else 'Unlocated'],
            text_representation=raw_text,source_locator=locator,
            asset_uri=(f'{source.name}#{locator}' if kind == DocumentElementKind.IMAGE else None),
            representation_method='docling-offline-v1', representation_reviewed=False))
        if len(elements) > 5000:
            raise ValueError('Docling extracted elements exceed processing budget')
    return {'status':'STRUCTURED_PARSE_UNREVIEWED' if elements else 'NO_REPRESENTATIONS',
            'source_sha256':sha,'page_count':pages,'parser':'DOCLING_LOCAL',
            'review_required':True,'elements':elements,
            'network_access_allowed':False,
            'claim_boundary':'TABLE_FORMULA_OCR_AND_PICTURE_OUTPUT_NOT_YET_INDEPENDENTLY_REVIEWED'}


def describe_diagram_local(png: bytes, *, endpoint: str, model: str, timeout_s: float = 30) -> dict[str, Any]:
    """Optional OpenAI-compatible local VLM, yields a candidate ONLY.

    Rejects non-loopback URLs to prevent accidental spacecraft document exfiltration.
    No automatic citations, numeric readings or causal claims are allowed.
    """
    from ipaddress import ip_address
    url = urlsplit(endpoint)
    try:
        loopback = ip_address(url.hostname or '').is_loopback
    except ValueError:
        loopback = url.hostname == 'localhost'
    if url.scheme != 'http' or not loopback or url.username or url.password or url.query or url.fragment:
        raise ValueError('vision endpoint must be a loopback HTTP service')
    if not endpoint.endswith('/chat/completions') or not model or len(model) > 120:
        raise ValueError('requires exact local chat-completions endpoint and model')
    if not png or len(png) > MAX_IMAGE_BYTES or not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('invalid or oversized page PNG')
    payload = {'model':model,'temperature':0,'max_tokens':850,'messages':[
        {'role':'system','content':('Describe only visibly identifiable labels, objects, plot axes, and equations. '
            'Never invent numeric measurements, failure causes, fault states or control actions. '
            'Return JSON with keys summary (string) and observations (array of strings). '
            'If unreadable, say so.')},
        {'role':'user','content':[{'type':'text','text':'Produce a candidate visual transcription for expert validation.'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode('ascii')}}]}]}
    request=Request(endpoint, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    class RejectRedirects(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError('vision model endpoint redirects are prohibited')
    opener = build_opener(ProxyHandler({}), RejectRedirects())
    with opener.open(request,timeout=min(max(1,timeout_s),60)) as response:
        raw = response.read(100000)
    content=json.loads(raw)['choices'][0]['message']['content']
    if isinstance(content,str) and content.strip().startswith('```'):
        content=content.strip().strip('`').removeprefix('json').strip()
    value = json.loads(content)
    if not isinstance(value,dict) or set(value) != {'summary','observations'} or not isinstance(value['summary'],str):
        raise ValueError('malformed local vision model answer')
    observations=value['observations']
    if not isinstance(observations,list) or len(observations)>12 or any(not isinstance(v,str) or len(v)>250 for v in observations):
        raise ValueError('malformed model observations')
    return {'status':'VISION_CANDIDATE_REQUIRES_EXPERT_REVIEW',
        'image_sha256':hashlib.sha256(png).hexdigest(), 'summary':value['summary'][:600],
        'observations':observations,'reviewed':False,'eligible_for_causes':False,
        'endpoint_is_loopback':True,'claim_boundary':'UNVERIFIED_VISUAL_MODEL_OUTPUT'}


def main(argv: list[str] | None = None) -> int:
    # CLI can reuse configuration saved from the administrator workbench.
    # No implicit model downloads or remote visual inference are introduced.
    platform_root = os.environ.get('ASTROGRAPH_PLATFORM_ROOT','').strip()
    if platform_root:
        from astrograph.integrations.material_engine_settings import load_active_engine_environment
        load_active_engine_environment(Path(platform_root))
    parser=argparse.ArgumentParser(description='Offline scanned-PDF/Docling candidate exporter (never auto-approves)')
    parser.add_argument('pdf',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engine',choices=['tesseract','docling'],default=(os.environ.get('ASTROGRAPH_MATERIAL_OCR_ENGINE') or 'tesseract'))
    parser.add_argument('--languages',default=os.environ.get('ASTROGRAPH_MATERIAL_OCR_LANGUAGES','chi_sim+eng'))
    parser.add_argument('--vision-endpoint',default=os.environ.get('ASTROGRAPH_MATERIAL_VISION_ENDPOINT',''))
    parser.add_argument('--vision-model',default=os.environ.get('ASTROGRAPH_MATERIAL_VISION_MODEL',''))
    parser.add_argument('--vision-pages',type=int,default=int(os.environ.get('ASTROGRAPH_MATERIAL_VISION_PAGES','0')))
    args=parser.parse_args(argv)
    result=(ocr_scanned_pdf(args.pdf,languages=args.languages) if args.engine=='tesseract'
            else docling_pdf_candidates(args.pdf))
    if args.vision_endpoint or args.vision_model:
        if not args.vision_endpoint or not args.vision_model or not 0 < args.vision_pages <= 5:
            parser.error('vision requires local endpoint, model and 1..5 pages')
        result['visual_candidates']=[]
        with tempfile.TemporaryDirectory(prefix='astrograph-vlm-') as directory:
            for page in range(1,min(args.vision_pages,result['page_count'])+1):
                png=render_page_png(args.pdf,page,Path(directory)/f'p{page}.png').read_bytes()
                result['visual_candidates'].append({'page':page,**describe_diagram_local(
                    png,endpoint=args.vision_endpoint,model=args.vision_model)})
    output={**result,'elements':[item.model_dump(mode='json') for item in result['elements']]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if os.name == 'posix':
        try: os.chmod(args.output,0o600)
        except OSError: pass
    print(json.dumps({'status':result['status'],'output':str(args.output),
                      'elements':len(result['elements']),'approved':False},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
