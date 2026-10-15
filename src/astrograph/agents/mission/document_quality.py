"""Stage-86 document provider readiness and quality-boundary receipt."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def document_provider_quality_receipt(settings: Any) -> dict[str, Any]:
    mineru_requested=bool(getattr(settings,"allow_mineru",False))
    mineru_exe=getattr(settings,"mineru_executable",None)
    mineru_resolved=None
    if mineru_exe:
        mineru_resolved=str(Path(mineru_exe)) if Path(mineru_exe).is_absolute() else shutil.which(mineru_exe)
    a2o_requested=bool(getattr(settings,"allow_a2o",False))
    a2o_root=getattr(settings,"a2o_source_root",None)
    a2o_available=bool(a2o_root and Path(a2o_root).is_dir())
    external=[]
    if mineru_requested and mineru_resolved and Path(mineru_resolved).is_file():
        mineru="CONFIGURED_NOT_DOMAIN_QUALITY_VALIDATED"
    elif mineru_requested:
        mineru="BLOCKED_EXTERNAL_PROVIDER"
        external.append("MINERU_EXECUTABLE_UNAVAILABLE")
    else:
        mineru="NOT_REQUESTED"
    if a2o_requested and a2o_available:
        a2o="CONFIGURED_NOT_DOMAIN_QUALITY_VALIDATED"
    elif a2o_requested:
        a2o="BLOCKED_EXTERNAL_PROVIDER"
        external.append("A2O_SOURCE_ROOT_UNAVAILABLE")
    else:
        a2o="NOT_REQUESTED"
    return {
        "schema_version":"astrograph-document-provider-quality-receipt.v1",
        "native_text":{"status":"AVAILABLE","formats":["txt","md","markdown"],"semantic_rewrite_performed":False},
        "mineru":{"status":mineru,"configured_executable":mineru_exe,"resolved_executable":mineru_resolved},
        "anything2ontology":{"status":a2o,"source_root":a2o_root},
        "external_blockers":external,
        "quality_claims":{
            "adapter_contract_validated":True,
            "real_complex_pdf_quality_validated":False,
            "real_domain_ontology_extraction_quality_validated":False,
        },
        "diagnostic_vote_allowed":False,
        "active_kg_auto_write":False,
        "claim_boundary":(
            "Native text parsing is available. External parser/extractor configuration is reported separately from "
            "domain-quality validation; adapter availability alone is never a MinerU/A2O quality claim."
        ),
    }


__all__=["document_provider_quality_receipt"]
