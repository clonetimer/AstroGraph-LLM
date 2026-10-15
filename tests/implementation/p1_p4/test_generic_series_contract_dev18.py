from __future__ import annotations
from hashlib import sha256

import pandas as pd
import pytest

from astrograph.agents.mission.generic_series_contract import (
    build_generic_series_input_contract, materialize_model_window,
)


def _source_hash():
    return sha256(b"source").hexdigest()


def test_contract_is_deterministic_and_order_bound():
    frame=pd.DataFrame({'time_s':[0,1,2,3], 'a':[1.,2.,3.,4.], 'b':[5.,6.,7.,8.]})
    one=build_generic_series_input_contract(frame,timestamp_column='time_s',channel_order=['a','b'],channel_units=['V','A'],source_data_sha256=_source_hash())
    two=build_generic_series_input_contract(frame.copy(),timestamp_column='time_s',channel_order=['a','b'],channel_units=['V','A'],source_data_sha256=_source_hash())
    swapped=build_generic_series_input_contract(frame,timestamp_column='time_s',channel_order=['b','a'],channel_units=['A','V'],source_data_sha256=_source_hash())
    assert one==two
    assert one.quality_status=='VALID' and one.model_input_eligible is True
    assert one.canonical_input_sha256!=swapped.canonical_input_sha256


def test_contract_content_change_changes_hash():
    frame=pd.DataFrame({'time_s':[0,1,2], 'a':[1.,2.,3.]})
    one=build_generic_series_input_contract(frame,timestamp_column='time_s',channel_order=['a'],channel_units=['V'],source_data_sha256=_source_hash())
    changed=frame.copy();changed.loc[2,'a']=30
    two=build_generic_series_input_contract(changed,timestamp_column='time_s',channel_order=['a'],channel_units=['V'],source_data_sha256=_source_hash())
    assert one.canonical_input_sha256!=two.canonical_input_sha256


def test_missing_and_irregular_timestamp_fail_closed_for_model_but_are_audited():
    frame=pd.DataFrame({'time_s':[0,1,1,3], 'a':[1.,None,3.,4.]})
    contract=build_generic_series_input_contract(frame,timestamp_column='time_s',channel_order=['a'],channel_units=['V'],source_data_sha256=_source_hash())
    assert contract.quality_status=='DEGRADED'
    assert contract.model_input_eligible is False
    assert 'MISSING_VALUES:a' in contract.reason_codes
    assert 'TIMESTAMP_DUPLICATES' in contract.reason_codes
    with pytest.raises(ValueError,match='not eligible'):
        materialize_model_window(frame,contract=contract,window_size=2)


def test_window_is_exact_finite_tail_and_hash_bound():
    frame=pd.DataFrame({'time_s':range(6), 'a':[1.,2.,3.,4.,5.,6.], 'b':[6.,5.,4.,3.,2.,1.]})
    contract=build_generic_series_input_contract(frame,timestamp_column='time_s',channel_order=['a','b'],channel_units=['V','A'],source_data_sha256=_source_hash())
    window=materialize_model_window(frame,contract=contract,window_size=3)
    assert window.start_row==3 and window.end_row_exclusive==6
    assert window.values==((4.0,3.0),(5.0,2.0),(6.0,1.0))
    other=materialize_model_window(frame,contract=contract,window_size=3,end_row_exclusive=5)
    assert other.input_data_sha256!=window.input_data_sha256
