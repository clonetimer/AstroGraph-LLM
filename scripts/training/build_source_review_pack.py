"""Generate source-bound review drafts using a configured local model, never approvals."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from astrograph.oag.contracts import LiteratureDocument,OntologyProposal,content_hash
from astrograph.training.small_model import RetrievalQuery
from astrograph.training.review_pack import ReviewDraft,write_review_pack
from astrograph.llm.openai_compatible import OpenAICompatibleConfig,OpenAICompatibleStructuredBackend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base-url',required=True);p.add_argument('--model',required=True)
    args=p.parse_args();source_rows=json.loads(args.sources.read_text());drafts=[]
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(base_url=args.base_url,model=args.model,api_key='local-no-auth',timeout_s=120,tokenizer_mode='disabled'))
    for row in source_rows:
        doc=LiteratureDocument.model_validate(row['document'])
        for task,model in [('ONTOLOGY_EXTRACTION',OntologyProposal),('RETRIEVAL_QUERY',RetrievalQuery)]:
            inputs={'documents':[doc.model_dump(mode='json')]} if task=='ONTOLOGY_EXTRACTION' else {'source_ids':[doc.document_id],'source_text':doc.text,'source_type':'KNOWLEDGE_NOT_TELEMETRY'}
            instructions=('Extract only directly stated relationships. Use exact source quotes containing subject and object. Do not infer any unmentioned relation. Return an empty triples list if none is explicitly supported.' if task=='ONTOLOGY_EXTRACTION' else 'Write a short literature search query about this source. Cite only the provided source ID. Do not treat the source as a measurement or assert a confirmed satellite fault.')
            try:
                answer=backend.generate_structured(agent_id='material-review-draft',output_model=model,payload={**inputs,'instructions':instructions})
                answer=answer.model_dump(mode='json') if hasattr(answer,'model_dump') else answer
            except Exception as exc:answer={'draft_error':type(exc).__name__,'needs_manual_completion':True}
            drafts.append(ReviewDraft(example_id='DRAFT-'+content_hash({'source':doc.document_id,'task':task})[:20],task=task,
                source_group=row['source_group'],source_artifact_sha256=row['source_artifact_sha256'],source_url=doc.source_url,
                split=row['split'],input=inputs,candidate_output=answer,candidate_author=args.model))
    write_review_pack(drafts,args.output)
    print(json.dumps({'status':'REVIEW_REQUIRED','draft_count':len(drafts),'output':str(args.output)}))


if __name__=='__main__':main()
