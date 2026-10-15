export const PRESETS = Object.freeze({
  diagnosis:{profile:'DIAGNOSIS',tools:['system.describe','artifact.verify','workflows.get','runtime.describe','runtime.reference_plan','runtime.diagnosis.execute']},
  shadow:{profile:'SHADOW',tools:['system.describe','artifact.verify','workflows.get','workflows.audit_capabilities','runtime.describe','runtime.reference_plan','agent.shadow.validate_plan']},
  development:{profile:'DEVELOPMENT',tools:['system.describe','artifact.verify','research.artifact.verify_json','research.route_a.selection_gate','workflows.get']},
  posthoc:{profile:'POSTHOC_AUDIT',tools:['system.describe','artifact.verify','research.artifact.verify_json','workflows.get']},
  evidence:{profile:'EVIDENCE_CURATION',tools:['system.describe','artifact.verify','evidence.card.propose','evidence.card.validate','optional_toolkits.catalog','workflows.get']},
  confirmatory:{profile:'CONFIRMATORY_SEALED',tools:['system.describe','artifact.verify','confirmatory.campaign.verify_preregistration','confirmatory.campaign.execute_once','workflows.get']},
  maintainer:{profile:'MAINTAINER',tools:['system.describe','profiles.describe','artifact.verify','dataset.intake_archive','workflows.list','workflows.get','workflows.validate','workflows.audit_capabilities','runtime.describe','runtime.reference_plan','runtime.diagnosis.execute','authoritative_p2.audit','optional_toolkits.catalog']}
});
