"""ADAPT XML catalog and SCN separation. Unknown units and truth fail closed."""
from __future__ import annotations
import hashlib,math,re,xml.etree.ElementTree as ET
from .external_dataset_contract import digest,seal
NS={'d':'urn:org:dx-competition:system'}
XSI='{http://www.w3.org/2001/XMLSchema-instance}type'

def system_catalog(xml_bytes):
    """Parse the public DXC ADAPT system catalog without inventing bindings.

    R23 preserves the legacy v1 fields and additionally records instance-level
    command bindings plus nominal/fault mode separation.  Command bindings are
    taken only from explicit ``component/command`` records in the XML; they are
    not inferred from names, topology, or scenario traces.
    """
    root=ET.fromstring(xml_bytes)
    def text(e,name):return (e.findtext('d:'+name,default='',namespaces=NS) or '').strip()
    types={}
    mode_groups={}; nominal_mode_groups={}; all_mode_groups={}
    for group in root.findall('d:modeCatalog/d:modeGroup',NS):
        group_name=text(group,'name')
        records=[{'name':text(m,'name'),'kind':m.attrib.get(XSI,'')} for m in group.findall('d:mode',NS)]
        all_mode_groups[group_name]=records
        mode_groups[group_name]=[item['name'] for item in records if item['kind']=='faultMode']
        nominal_mode_groups[group_name]=[item['name'] for item in records if item['kind']!='faultMode']
    for t in root.findall('d:componentTypeCatalog/d:componentType',NS):
        name=text(t,'name');unit=text(t,'engUnits');kind=t.attrib.get(XSI,'')
        refs=[(ref.text or '').strip() for ref in t.findall('d:modesRef',NS)]
        types[name]={'type_id':name,'kind':kind,'unit':unit or None,'unit_status':'VERIFIED_FROM_XML' if unit else 'UNIT_UNVERIFIED',
          'modes_ref':text(t,'modesRef'),'modes_refs':refs,
          'fault_modes':sorted({mode for ref in refs for mode in mode_groups.get(ref,[])}),
          'nominal_modes':sorted({mode for ref in refs for mode in nominal_mode_groups.get(ref,[])}),
          'data_type':text(t,'sensorValue/d:dataType'),
          'commands':[text(c,'name') for c in t.findall('.//d:command',NS)]}
    components=[];connections=[];command_bindings=[]
    command_targets={}
    for system in root.findall('d:systems/d:system',NS):
        for c in system.findall('.//d:component',NS):
            cid=text(c,'name');type_id=text(c,'componentType')
            instance_commands=[]
            for command in c.findall('d:command',NS):
                command_id=text(command,'name')
                if not command_id:continue
                data_type=(command.findtext('d:commandValue/d:dataType',default='',namespaces=NS) or '').strip() or None
                binding={'command_id':command_id,'component_id':cid,'component_type':type_id,'data_type':data_type,'binding_source':'COMPONENT_INSTANCE_XML'}
                command_bindings.append(binding);instance_commands.append(command_id)
                command_targets.setdefault(command_id,set()).add(cid)
            if cid and type_id:
                inherited={k:v for k,v in types.get(type_id,{'kind':'UNKNOWN','unit_status':'UNIT_UNVERIFIED','unit':None,'fault_modes':[],'nominal_modes':[]}).items() if k!='type_id'}
                components.append({'component_id':cid,'component_type':type_id,**inherited,'command_ids':sorted(instance_commands)})
        for c in system.findall('.//d:connection',NS):connections.append({'c1':text(c,'c1'),'c2':text(c,'c2')})
    if not components:raise ValueError('CATALOG_EMPTY_OR_SCHEMA_UNADMITTED')
    command_bindings=sorted(command_bindings,key=lambda row:(row['command_id'],row['component_id']))
    ambiguous_command_ids=sorted(command_id for command_id,targets in command_targets.items() if len(targets)>1)
    return seal({'schema_version':'astrograph-adapt-system-catalog.v1','system_universe':'NASA_ADAPT_EPS','xml_sha256':hashlib.sha256(xml_bytes).hexdigest(),
      'component_types':types,'fault_mode_groups':mode_groups,'nominal_mode_groups':nominal_mode_groups,'mode_groups':all_mode_groups,
      'target_universe':sorted(c['component_id']+':'+m for c in components for m in c.get('fault_modes',[])),
      'components':components,'connections':connections,'topology_sha256':digest(connections),
      'command_bindings':command_bindings,'command_bindings_sha256':digest(command_bindings),'ambiguous_command_ids':ambiguous_command_ids})

def require_formal_unit(component):
    if component.get('unit_status')!='VERIFIED_FROM_XML':raise ValueError('UNIT_UNVERIFIED')

def parse_scenario(source: str):
    """Tick times retained without guessed conversion; no name enters online input."""
    online=[];injections=[];info_count=0
    for line in source.splitlines():
        line=line.strip()
        if not line or line.startswith('//'):continue
        m=re.fullmatch(r'sensors\s+@(\d+)\s*\{(.*)\}\s*;',line)
        if m:
            values={}
            for item in m[2].split(','):
                key,value=item.strip().split('=',1);key=key.strip();value=value.strip()
                if key in values:raise ValueError('DUPLICATE_SENSOR')
                number=(value=='true') if value in ('true','false') else float(value)
                if not math.isfinite(number):raise ValueError('NONFINITE_OBSERVATION')
                values[key]=number
            online.append({'kind':'sensors','time_tick':int(m[1]),'values':values});continue
        m=re.fullmatch(r'command\s+@(\d+)\s+(\w+)\s*=\s*(true|false)\s*;',line)
        if m:online.append({'kind':'command','time_tick':int(m[1]),'command_id':m[2],'value':m[3]=='true'});continue
        m=re.fullmatch(r'faultInjection\s+@(\d+)\s+isInjection\s*=\s*(true|false),\s*fault\s*=\s*\{(.*?)\},\s*parameters\s*=\s*\{(.*?)\}\s*;',line)
        if m:
            faults=[]
            for assignment in m[3].split(','):
                component,mode=assignment.strip().split('=',1);faults.append(component.strip()+':'+mode.strip())
            if not faults:raise ValueError('FAULT_TRUTH_EMPTY')
            injections.append({'time_tick':int(m[1]),'is_injection':m[2]=='true','fault_ids':sorted(faults),'parameters_raw':m[4]});continue
        if line.startswith('scenarioInfo @') and line.endswith(';'):info_count+=1;continue
        raise ValueError('SCENARIO_UNPARSED_RECORD:'+line[:80])
    # DXC09 supplied SCN starts with sensors; DXC10 additionally has scenarioInfo.
    if info_count>1 or not any(r['kind']=='sensors' for r in online):raise ValueError('SCENARIO_INCOMPLETE')
    sensor_times=[r['time_tick'] for r in online if r['kind']=='sensors']
    # Multiple ordered observations can legitimately carry the same tick. Retain
    # them verbatim; only reversal is invalid, never sort or discard bad records.
    if any(b<a for a,b in zip(sensor_times,sensor_times[1:])):raise ValueError('TIMESTAMP_REVERSED')
    truth={'schema_version':'astrograph-fault-set-truth.v1','true_fault_ids':sorted({f for r in injections if r['is_injection'] for f in r['fault_ids']}),
      'injection_events':injections,'injection_time':min((r['time_tick'] for r in injections if r['is_injection']),default=None),
      'time_unit':'UNVERIFIED_RAW_TICK','status':'PARSED_INJECTION_TRUTH' if injections else 'SCORING_BLOCKED_TRUTH_UNAVAILABLE',
      'semantics':'fault IDs injected at any time; event truth preserves removal and distinct times; absence is not asserted nominal'}
    return online,truth
