from __future__ import annotations

from .base import DatasetSource
from .esa_adb import ESAADBDataSource
from .file_source import LocalFileDataSource
from .models import DataSourceType, DatasetSourceConfig
from .sat_agent_platform import SatAgentPlatformDataSource
from .simulation import SimulationDataSource


def build_data_source(config: DatasetSourceConfig) -> DatasetSource:
    if config.source_type == DataSourceType.SIMULATION:
        return SimulationDataSource(config)
    if config.source_type == DataSourceType.SAT_AGENT_PLATFORM:
        return SatAgentPlatformDataSource(config)
    if config.source_type == DataSourceType.ESA_ADB:
        return ESAADBDataSource(config)
    if config.source_type == DataSourceType.LOCAL_FILE:
        return LocalFileDataSource(config)
    raise ValueError(
        "memory data sources require explicit in-process frames and cannot be built from a path config"
    )
