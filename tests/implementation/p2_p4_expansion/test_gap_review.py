import json
from pathlib import Path
from astrograph.knowledge.generic_gap_review import build_gap_review_packet

ROOT=Path(__file__).resolve().parents[3]

GAPS=[
 {'channel':'sensor.eps.battery_current_a','anomaly_id':'ANOM_BATTERY_CURRENT_DEVIATION'},
 {'channel':'sensor.adcs.rw.winding_temp_c_0','anomaly_id':'ANOM_RW0_WINDING_TEMP_DEVIATION'},
 {'channel':'adcs.rw.command_torque_nm_0','anomaly_id':'ANOM_RW0_COMMAND_TORQUE_DEVIATION'},
 {'channel':'comm.downlink_rate_bps','anomaly_id':'ANOM_DOWNLINK_RATE_DEVIATION'},
 {'channel':'counter.comm.packet_error_count','anomaly_id':'ANOM_PACKET_ERROR_COUNT_DEVIATION'},
]

def test_gap_review_does_not_auto_close_knowledge():
    packet=build_gap_review_packet(ROOT,GAPS)
    assert len(packet.items)==5
    exact=[x for x in packet.items if x.gap_kind=='EXACT_CONTEXT_ROLE_REVIEW']
    unsafe=[x for x in packet.items if x.gap_kind=='NO_SAFE_RELATION_CANDIDATE']
    assert {x.channel for x in exact}=={'sensor.eps.battery_current_a','adcs.rw.command_torque_nm_0'}
    assert len(unsafe)==3
    assert packet.active_kg_modified is False
    assert packet.automatic_promotion_allowed is False
    assert all(x.automatic_aliasing_allowed is False and x.active_kg_write_allowed is False for x in packet.items)

def test_non_exact_hints_are_not_relations():
    packet=build_gap_review_packet(ROOT,GAPS)
    downlink=next(x for x in packet.items if x.channel=='comm.downlink_rate_bps')
    assert downlink.gap_kind=='NO_SAFE_RELATION_CANDIDATE'
    assert any(h.channel=='comm.downlink_bps' for h in downlink.related_catalog_hints)
    assert all(h.automatic_relation_allowed is False for h in downlink.related_catalog_hints)
