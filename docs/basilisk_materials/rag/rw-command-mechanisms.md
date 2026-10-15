# 反作用轮卡滞与力矩能力下降

角色：SIMULATION_KNOWLEDGE_NOT_CURRENT_OBSERVATION。当前源码核对，非人工批准的诊断真值。

本地指令管理器将卡滞实现为逆转速方向制动；低于容差后输出归零。力矩能力下降缩放指令与限幅。两者是不同分支，但末段稳态观测仍可能难以区分。不要把仿真参数直接当作诊断规则阈值。

来源：src/sat_sim/bsk_engine/project_native_modules.py:456-541；SHA256：2fa5e544e18d62cb9b08c214cf0e55cd669fd66bf00559bfa6d73f5ae2ef6597。

```python
class ReactionWheelCommandFaultManager(sysModel.SysModel):
    """Modify wheel torque commands before they reach the wheel dynamics."""

    EFFECTS = {
        "rw_jamming", "adcs_rw_jamming", "rw_motor_failure", "adcs_rw_motor_failure",
        "rw_friction_degradation", "adcs_rw_torque_authority_loss", "reaction_wheel_speed_limit", "adcs_reaction_wheel_speed_limit",
        "adcs_actuator_failure", "adcs_control_loop_failure",
    }

    def __init__(self, events: Sequence[BSKEventSpec], *, wheel_count: int = 3, max_torque_nm: float = 0.2) -> None:
        super().__init__()
        self.ModelTag = "project_rw_command_fault_manager"
        self.events = tuple(event for event in events if event.effect in self.EFFECTS)
        self.wheel_count = int(wheel_count)
        self.max_torque_nm = abs(float(max_torque_nm))
        self.commandInMsg = messaging.ArrayMotorTorqueMsgReader()
        self.speedInMsg = messaging.RWSpeedMsgReader()
        self.commandOutMsg = messaging.ArrayMotorTorqueMsg()
        self.dragOutMsg = messaging.ArrayMotorTorqueMsg()
        self.torqueLimitOutMsg = messaging.ArrayMotorTorqueMsg()
        self.speedLimitOutMsg = messaging.RWSpeedMsg()

    def Reset(self, current_sim_nanos: int) -> None:  # noqa: N802
        self.UpdateState(current_sim_nanos)

    def UpdateState(self, current_sim_nanos: int) -> None:  # noqa: N802
        command_payload = self.commandInMsg()
        speed_payload = self.speedInMsg()
        source = list(command_payload.motorTorque)
        output = list(source)
        drag = [0.0] * len(output)
        torque_limits = [self.max_torque_nm] * len(output)
        speed_limits = [0.0] * len(list(speed_payload.wheelSpeeds))
        speeds = list(speed_payload.wheelSpeeds)

        for event in _active(self.events, current_sim_nanos):
            index = _wheel_index(event, self.wheel_count)
            if event.effect in {"rw_motor_failure", "adcs_rw_motor_failure"}:
                scale = max(0.0, min(_event_number(event, ("torque_scale", "remaining_torque_ratio"), 0.0), 1.0))
                output[index] *= scale
                torque_limits[index] = self.max_torque_nm * scale
            elif event.effect == "adcs_actuator_failure":
                output[index] = 0.0; torque_limits[index] = 0.0
            elif event.effect == "adcs_control_loop_failure":
                scale = max(0.0, min(_event_number(event, ("command_scale",), 0.0), 1.0))
                for wheel_index in range(self.wheel_count):
                    output[wheel_index] *= scale
            elif event.effect == "adcs_rw_torque_authority_loss":
                scale = max(0.0, min(_event_number(event, ("torque_scale", "remaining_torque_ratio"), 0.5), 1.0))
                output[index] *= scale
                torque_limits[index] = self.max_torque_nm * scale
            elif event.effect == "rw_friction_degradation":
                coefficient = max(_event_number(event, ("drag_nms",), 0.01), 0.0)
                drag[index] = coefficient
                output[index] -= coefficient * float(speeds[index])
            elif event.effect in {"reaction_wheel_speed_limit", "adcs_reaction_wheel_speed_limit"}:
                limit = max(_event_number(event, ("max_speed_rad_s",), 50.0), 0.0)
                speed_limits[index] = limit
                speed = float(speeds[index])
                if abs(speed) >= limit:
                    gain = max(_event_number(event, ("overspeed_brake_gain_nms",), 0.02), 0.0)
                    brake = min(self.max_torque_nm, gain * max(0.0, abs(speed) - limit))
                    output[index] = -math.copysign(brake, speed)
            elif event.effect in {"rw_jamming", "adcs_rw_jamming"}:
                brake = min(max(_event_number(event, ("brake_torque_nm",), self.max_torque_nm), 0.0), self.max_torque_nm)
                tolerance = max(_event_number(event, ("lock_tolerance_rad_s",), 0.25), 0.0)
                speed = float(speeds[index])
                output[index] = -math.copysign(brake, speed) if abs(speed) > tolerance else 0.0
                torque_limits[index] = 0.0

        for index in range(min(self.wheel_count, len(output))):
            output[index] = max(-self.max_torque_nm, min(self.max_torque_nm, float(output[index])))

        out_payload = messaging.ArrayMotorTorqueMsgPayload()
        out_payload.motorTorque = output
        self.commandOutMsg.write(out_payload, current_sim_nanos, self.moduleID)
        drag_payload = messaging.ArrayMotorTorqueMsgPayload()
        drag_payload.motorTorque = drag
        self.dragOutMsg.write(drag_payload, current_sim_nanos, self.moduleID)
        torque_payload = messaging.ArrayMotorTorqueMsgPayload()
        torque_payload.motorTorque = torque_limits
        self.torqueLimitOutMsg.write(torque_payload, current_sim_nanos, self.moduleID)
        speed_payload_out = messaging.RWSpeedMsgPayload()
        speed_payload_out.wheelSpeeds = speed_limits
        self.speedLimitOutMsg.write(speed_payload_out, current_sim_nanos, self.moduleID)

```
