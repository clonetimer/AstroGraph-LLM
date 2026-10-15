"""SYNTHETIC orchestration test ONLY. No trained model, physical KG or accuracy claim."""
def distribution(ids,values):
    return {'status':'AVAILABLE','class_ids':ids,'values':values,'score_kind':'PROBABILITY'}

def synthetic_data(payload):
    req=payload['request']
    return {'model_release':req['model_release'],'input_sha256':req['observation']['payload_sha256'],
        'anomaly':distribution(['NORMAL','ANOMALY'],[.1,.9]),'component':distribution(['fixture_component'],[1.]),
        'fault':distribution(['FIXTURE_F1','FIXTURE_F2'],[.2,.8] if payload['derived_observations'] else [.8,.2])}

def synthetic_knowledge(payload):
    req=payload['request']
    def rank(source):
        return {'ranking_id':source,'episode_id':req['request_id'],'source':source,'producer_id':'SYNTHETIC_FIXTURE','producer_version':'1',
            'candidates':[{'fault_id':f,'rank':i+1,'score':.8-i*.6,'source':source} for i,f in enumerate(['FIXTURE_F2','FIXTURE_F1'])]}
    return {'input_sha256':req['observation']['payload_sha256'],'knowledge_sha256':req['knowledge_release']['sha256'],
            'forward':rank('KG_FORWARD'),'backward':rank('KG_BACKWARD')}

def synthetic_inspect(payload):
    req=payload['request'];obs=req['observation']
    return {'observations':[{'evidence_id':'fixture_metric','source_input_sha256':obs['payload_sha256'],
        'metric_id':'fixture.constant','unit':'1','value':.5,'available_at':req['cutoff_at'],'source_evidence_ids':[obs['artifact_id']]}]}
