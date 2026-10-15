# 发射机功率放大器故障

角色：SIMULATION_KNOWLEDGE_NOT_CURRENT_OBSERVATION。当前源码核对，非人工批准的诊断真值。

组件 L2 声明涉及 rf_power_w、pa_efficiency 和 heat_generation_w；整星 comm_data_transmitter_rate_loss 分支将故障简化为降码率。两者不是相同保真度，不可据降码率单独确认固放具体失效机理。

来源：src/components/transmitter/faults.py:70-100；SHA256：05790a8c5a0552eab8c202ab1f96ac3e94eb7a43512aff35a7817e28b65911ae。

```python

from ..fault_base import ParameterEffect, TimeWindowFault


class TransmitterL2FaultType(str, Enum):
    POWER_AMPLIFIER_FAULT = 'power_amplifier_fault'
    FREQUENCY_DRIFT_FAULT = 'frequency_drift_fault'
    SIGNAL_LOSS = 'signal_loss'


def _effects_for(kind: TransmitterL2FaultType) -> tuple[ParameterEffect, ...]:
    effects_by_type = {
        TransmitterL2FaultType.POWER_AMPLIFIER_FAULT: (
            ParameterEffect('rf_power_w', 'multiply', 0.35, 'PA fault reduces RF output and raises waste heat'),
            ParameterEffect('pa_efficiency', 'multiply', 0.55, 'PA fault reduces RF output and raises waste heat'),
            ParameterEffect('heat_generation_w', 'add', 8.0, 'PA fault reduces RF output and raises waste heat'),
        ),
        TransmitterL2FaultType.FREQUENCY_DRIFT_FAULT: (
            ParameterEffect('frequency_error_hz', 'add', 2500.0, 'oscillator fault shifts carrier frequency'),
        ),
        TransmitterL2FaultType.SIGNAL_LOSS: (
            ParameterEffect('availability', 'override', 0.0, 'RF chain loss removes transmit availability'),
            ParameterEffect('rf_power_w', 'override', 0.0, 'RF chain loss removes transmit availability'),
        ),
    }
    return effects_by_type[kind]


@dataclass
class TransmitterFault(TimeWindowFault):
    """Base L2-light fault for transmitter."""
```
