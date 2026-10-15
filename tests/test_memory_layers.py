"""D5: layered memory with human-gate promotion and pollution guard tests."""
from __future__ import annotations

import pytest

from astrograph.data_driven.memory_layers import MemoryEntry, MemoryLayer, MemoryStore


def _entry(entry_id: str = 'M-1', layer: MemoryLayer = MemoryLayer.WORKING) -> MemoryEntry:
    return MemoryEntry(entry_id=entry_id, layer=layer, content='phm unified adapter: single-split gain needs multi-fold validation')


def test_write_and_entries(tmp_path):
    store = MemoryStore(tmp_path)
    store.write(_entry())
    assert len(store.entries()) == 1


def test_promote_requires_human(tmp_path):
    store = MemoryStore(tmp_path)
    store.write(_entry())
    with pytest.raises(ValueError):
        store.promote('M-1', human_approved=False, target_layer=MemoryLayer.SEMANTIC)


def test_promote_with_human(tmp_path):
    store = MemoryStore(tmp_path)
    store.write(_entry())
    promoted = store.promote('M-1', human_approved=True, target_layer=MemoryLayer.SEMANTIC)
    assert promoted.layer == MemoryLayer.SEMANTIC


def test_demote_not_delete(tmp_path):
    store = MemoryStore(tmp_path)
    store.write(_entry())
    demoted = store.demote('M-1', reason='superseded by newer evidence')
    assert demoted.demoted is True
    assert 'DEMOTED' in demoted.content
    assert len(store.entries()) == 1  # never deleted


def test_negative_sample_recorded(tmp_path):
    store = MemoryStore(tmp_path)
    store.write(MemoryEntry(entry_id='M-NEG', content='threshold 0.015945 caused false positive', is_negative=True))
    audit = store.audit()
    assert audit['negative_samples'] == 1


def test_audit_layers(tmp_path):
    store = MemoryStore(tmp_path)
    store.write(_entry())
    store.write(MemoryEntry(entry_id='M-2', layer=MemoryLayer.EPISODIC, content='gyro bias supported after AND fix'))
    audit = store.audit()
    assert audit['layers']['WORKING'] == 1
    assert audit['layers']['EPISODIC'] == 1