from __future__ import annotations

from fastapi import APIRouter

from astrograph.integrations import SatSimPipelineClosureRequest, SatSimPipelineClosureWorkflow


def build_compat_router() -> APIRouter:
    router = APIRouter(tags=["compatibility"])

    def invoke(request: SatSimPipelineClosureRequest) -> dict:
        return SatSimPipelineClosureWorkflow().invoke(request).model_dump(mode="json")

    router.add_api_route(
        "/v1/compat/sat-sim/pipeline-diagnose",
        invoke,
        methods=["POST"],
        summary="Sat-Sim compatibility pipeline",
    )
    router.add_api_route(
        "/v1/integrations/sat-sim/pipeline-diagnose",
        invoke,
        methods=["POST"],
        deprecated=True,
        include_in_schema=False,
    )
    return router
