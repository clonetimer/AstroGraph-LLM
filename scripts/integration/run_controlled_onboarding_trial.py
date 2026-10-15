from __future__ import annotations

import argparse
from pathlib import Path

from astrograph.agents.dataset_onboarding import (
    ControlledDatasetOnboardingWorkflow,
    SchemaMappingProposalAgent,
)
from astrograph.llm.openai_compatible import OpenAICompatibleStructuredBackend


def _units(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        field, separator, unit = value.partition("=")
        if not separator or not field or not unit:
            raise ValueError(f"invalid --unit value {value!r}; expected FIELD=UNIT")
        result[field] = unit
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only Profiler -> Mapping Proposal -> Contract Critic trial."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument(
        "--semantic-schema",
        type=Path,
        default=Path("configs/semantics/core_semantic_schema.json"),
    )
    parser.add_argument("--llm-config", type=Path, default=Path("configs/local_llm.vllm.json"))
    parser.add_argument("--unit", action="append", default=[], metavar="FIELD=UNIT")
    parser.add_argument("--component", action="append", default=[], metavar="COMPONENT_INSTANCE")
    parser.add_argument("--reference", action="append", default=[], metavar="SOURCE_REFERENCE")
    args = parser.parse_args()

    backend = OpenAICompatibleStructuredBackend.from_json(args.llm_config)
    workflow = ControlledDatasetOnboardingWorkflow(
        proposer=SchemaMappingProposalAgent(backend=backend)
    )
    result = workflow.run(
        dataset_id=args.dataset_id,
        telemetry_path=args.telemetry,
        semantic_schema_path=args.semantic_schema,
        declared_units=_units(args.unit),
        approved_component_instances=args.component,
        approved_source_references=args.reference,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.validation.status == "READY_FOR_HUMAN_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
