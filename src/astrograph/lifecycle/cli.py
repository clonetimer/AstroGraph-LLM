from __future__ import annotations
import argparse,json
from pathlib import Path
from .knowledge_feedback import *

def main(argv=None):
    p=argparse.ArgumentParser(description="Expert-governed AstroGraph document/feedback lifecycle")
    p.add_argument("--db",type=Path,required=True);p.add_argument("--policy",type=Path,required=True)
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("stage-document");a.add_argument("--proposal",type=Path,required=True);a.add_argument("--event-id",required=True)
    a=sub.add_parser("review-document");a.add_argument("--document-id",required=True);a.add_argument("--decision",choices=["APPROVE","REJECT"],required=True);a.add_argument("--reviewer",required=True);a.add_argument("--reason",required=True);a.add_argument("--event-id",required=True)
    a=sub.add_parser("stage-knowledge");a.add_argument("--proposal",type=Path,required=True);a.add_argument("--event-id",required=True)
    a=sub.add_parser("review-knowledge");a.add_argument("--proposal-id",required=True);a.add_argument("--decision",choices=["APPROVE_FOR_OFFLINE_EVALUATION","REJECT"],required=True);a.add_argument("--reviewer",required=True);a.add_argument("--reason",required=True);a.add_argument("--event-id",required=True)
    a=sub.add_parser("export-knowledge");a.add_argument("--proposal-id",required=True)
    a=sub.add_parser("submit-feedback");a.add_argument("--feedback",type=Path,required=True);a.add_argument("--event-id",required=True)
    a=sub.add_parser("review-candidate");a.add_argument("--candidate-id",required=True);a.add_argument("--decision",choices=["APPROVE_FOR_OFFLINE_EVALUATION","REJECT"],required=True);a.add_argument("--reviewer",required=True);a.add_argument("--reason",required=True);a.add_argument("--event-id",required=True)
    a=sub.add_parser("export-candidate");a.add_argument("--candidate-id",required=True)
    a=sub.add_parser("list-documents")
    args=p.parse_args(argv);policy=LifecyclePolicy.model_validate_json(args.policy.read_text(encoding="utf-8"));store=KnowledgeFeedbackStore(args.db,policy)
    if args.cmd=="stage-document": out=store.stage_document(DocumentProposal.model_validate_json(args.proposal.read_text(encoding="utf-8")),event_id=args.event_id)
    elif args.cmd=="review-document": out=store.review_document(document_id=args.document_id,decision=args.decision,reviewer=args.reviewer,reason=args.reason,event_id=args.event_id)
    elif args.cmd=="stage-knowledge": out=store.stage_knowledge_proposal(KnowledgeProposal.model_validate_json(args.proposal.read_text(encoding="utf-8")),event_id=args.event_id)
    elif args.cmd=="review-knowledge": out=store.review_knowledge_proposal(proposal_id=args.proposal_id,decision=args.decision,reviewer=args.reviewer,reason=args.reason,event_id=args.event_id)
    elif args.cmd=="export-knowledge": out=store.export_knowledge_proposal(args.proposal_id)
    elif args.cmd=="submit-feedback": out=store.submit_feedback(ExpertFeedback.model_validate_json(args.feedback.read_text(encoding="utf-8")),event_id=args.event_id)
    elif args.cmd=="review-candidate": out=store.review_candidate(candidate_id=args.candidate_id,decision=args.decision,reviewer=args.reviewer,reason=args.reason,event_id=args.event_id)
    elif args.cmd=="export-candidate": out=store.export_candidate(args.candidate_id)
    else: out={"documents":store.approved_documents()}
    print(json.dumps(out,ensure_ascii=False,indent=2));return 0

if __name__ == "__main__":  # pragma: no cover - console entrypoint convenience
    raise SystemExit(main())
