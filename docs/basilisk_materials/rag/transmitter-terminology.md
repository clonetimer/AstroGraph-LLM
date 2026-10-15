# 固放与 RF 放大器术语

角色：SIMULATION_KNOWLEDGE_NOT_CURRENT_OBSERVATION。当前源码核对，非人工批准的诊断真值。

本地 transmitter runner 明确使用 RF Power Amplifier、TWT 和 SSPA。保留原文“固放”或使用有此来源支撑的 SSPA/固态功率放大器，不能扩写为 solid rocket motor。

来源：src/components/transmitter/runner.py:1-28；SHA256：fab3c02c70814abdbb86eeb8d39a9814158b8b30bf1d45350856a1531f6a2afd。

```python
"""RF Power Amplifier (TWT/SSPA) runner module.

This module provides runner functions for simulating TWT and SSPA amplifiers
following the standard runner pattern used in this project.

Usage:
    # Run TWT simulation
    result = run_twt_nominal_case()
    result = run_twt_power_sweep()

    # Run SSPA simulation
    result = run_sspa_nominal_case()
    result = run_sspa_power_sweep()

    # Run comparison
    result = run_amplifier_comparison()
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from .builder import (
    TransmitterConfig,
    TransmitterResult,
    compute_transmitter,
```
