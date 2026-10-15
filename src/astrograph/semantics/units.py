from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DimensionVector(BaseModel):
    """SI base-dimension exponents in M, L, T, I, temperature, amount, luminous order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mass: int = 0
    length: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminous_intensity: int = 0


DIMENSIONLESS = DimensionVector()
TIME = DimensionVector(time=1)
INVERSE_TIME = DimensionVector(time=-1)
TEMPERATURE = DimensionVector(temperature=1)
ANGULAR_RATE = DimensionVector(time=-1)
POWER = DimensionVector(mass=1, length=2, time=-3)
ENERGY = DimensionVector(mass=1, length=2, time=-2)
THERMAL_CAPACITY = DimensionVector(mass=1, length=2, time=-2, temperature=-1)
THERMAL_CONDUCTANCE = DimensionVector(mass=1, length=2, time=-3, temperature=-1)
VOLTAGE = DimensionVector(mass=1, length=2, time=-3, current=-1)
ELECTRIC_CURRENT = DimensionVector(current=1)
LINEAR_SPEED = DimensionVector(length=1, time=-1)
LENGTH = DimensionVector(length=1)
MOMENTUM = DimensionVector(mass=1, length=2, time=-1)
MAGNETIC_FIELD = DimensionVector(mass=1, length=0, time=-2, current=-1)
MAGNETIC_DIPOLE_MOMENT = DimensionVector(length=2, current=1)
MASS = DimensionVector(mass=1)
MASS_FLOW_RATE = DimensionVector(mass=1, time=-1)
PRESSURE = DimensionVector(mass=1, length=-1, time=-2)
FORCE = DimensionVector(mass=1, length=1, time=-2)
LINEAR_MOMENTUM = DimensionVector(mass=1, length=1, time=-1)
MOMENT_OF_INERTIA = DimensionVector(mass=1, length=2)


class UnitDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1)
    dimension: DimensionVector
    scale_to_si: float = 1.0
    offset_to_si: float = 0.0

    @field_validator("scale_to_si")
    @classmethod
    def valid_scale(cls, value: float) -> float:
        if not math.isfinite(value) or value == 0.0:
            raise ValueError("unit scale_to_si must be finite and non-zero")
        return value

    @field_validator("offset_to_si")
    @classmethod
    def valid_offset(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("unit offset_to_si must be finite")
        return value


class UnitRegistry:
    def __init__(self, definitions: Iterable[UnitDefinition]) -> None:
        self._definitions: dict[str, UnitDefinition] = {}
        for definition in definitions:
            if definition.symbol in self._definitions:
                raise ValueError(f"duplicate unit symbol: {definition.symbol}")
            self._definitions[definition.symbol] = definition

    def get(self, symbol: str) -> UnitDefinition:
        try:
            return self._definitions[symbol]
        except KeyError as exc:
            raise ValueError(f"unknown unit: {symbol}") from exc

    def require_dimension(self, symbol: str, dimension: DimensionVector) -> None:
        actual = self.get(symbol).dimension
        if actual != dimension:
            raise ValueError(f"unit {symbol} has dimension {actual.model_dump()}, expected {dimension.model_dump()}")

    def convert(self, value: float, *, source: str, target: str) -> float:
        source_unit = self.get(source)
        target_unit = self.get(target)
        if source_unit.dimension != target_unit.dimension:
            raise ValueError(f"cannot convert incompatible units: {source} -> {target}")
        si_value = float(value) * source_unit.scale_to_si + source_unit.offset_to_si
        converted = (si_value - target_unit.offset_to_si) / target_unit.scale_to_si
        if not math.isfinite(converted):
            raise ValueError("unit conversion produced a non-finite value")
        return converted

    def convert_optional(
        self,
        values: Iterable[float | None],
        *,
        source: str,
        target: str,
    ) -> list[float | None]:
        return [None if value is None else self.convert(value, source=source, target=target) for value in values]


DEFAULT_UNIT_REGISTRY = UnitRegistry(
    [
        UnitDefinition(symbol="1", dimension=DIMENSIONLESS),
        UnitDefinition(symbol="%", dimension=DIMENSIONLESS, scale_to_si=0.01),
        UnitDefinition(symbol="s", dimension=TIME),
        UnitDefinition(symbol="ms", dimension=TIME, scale_to_si=0.001),
        UnitDefinition(symbol="1/s", dimension=INVERSE_TIME),
        # Information amount is dimensionless in the SI dimension vector, so a
        # bit rate is an inverse-time flow with an explicit semantic unit.
        UnitDefinition(symbol="bit/s", dimension=INVERSE_TIME),
        UnitDefinition(symbol="bit", dimension=DIMENSIONLESS),
        UnitDefinition(symbol="K", dimension=TEMPERATURE),
        UnitDefinition(symbol="degC", dimension=TEMPERATURE, offset_to_si=273.15),
        UnitDefinition(symbol="rad/s", dimension=ANGULAR_RATE),
        UnitDefinition(symbol="deg/s", dimension=ANGULAR_RATE, scale_to_si=math.pi / 180.0),
        UnitDefinition(symbol="rpm", dimension=ANGULAR_RATE, scale_to_si=2.0 * math.pi / 60.0),
        UnitDefinition(symbol="deg", dimension=DIMENSIONLESS, scale_to_si=math.pi / 180.0),
        UnitDefinition(symbol="rad", dimension=DIMENSIONLESS, scale_to_si=1.0),
        UnitDefinition(symbol="W", dimension=POWER),
        UnitDefinition(symbol="kW", dimension=POWER, scale_to_si=1000.0),
        UnitDefinition(symbol="J", dimension=ENERGY),
        UnitDefinition(symbol="Wh", dimension=ENERGY, scale_to_si=3600.0),
        UnitDefinition(symbol="J/K", dimension=THERMAL_CAPACITY),
        UnitDefinition(symbol="W/K", dimension=THERMAL_CONDUCTANCE),
        UnitDefinition(symbol="N*m", dimension=ENERGY),
        UnitDefinition(symbol="V", dimension=VOLTAGE),
        UnitDefinition(symbol="A", dimension=ELECTRIC_CURRENT),
        UnitDefinition(symbol="m/s", dimension=LINEAR_SPEED),
        UnitDefinition(symbol="m", dimension=LENGTH),
        UnitDefinition(symbol="km", dimension=LENGTH, scale_to_si=1000.0),
        UnitDefinition(symbol="dB", dimension=DIMENSIONLESS),
        UnitDefinition(symbol="dBi", dimension=DIMENSIONLESS),
        UnitDefinition(symbol="T", dimension=MAGNETIC_FIELD),
        UnitDefinition(symbol="nT", dimension=MAGNETIC_FIELD, scale_to_si=1e-9),
        UnitDefinition(symbol="A*m^2", dimension=MAGNETIC_DIPOLE_MOMENT),
        UnitDefinition(symbol="N*m*s", dimension=MOMENTUM),
        UnitDefinition(symbol="MRP", dimension=DIMENSIONLESS),
        UnitDefinition(symbol="kg", dimension=MASS),
        UnitDefinition(symbol="kg/s", dimension=MASS_FLOW_RATE),
        UnitDefinition(symbol="Pa", dimension=PRESSURE),
        UnitDefinition(symbol="kPa", dimension=PRESSURE, scale_to_si=1000.0),
        UnitDefinition(symbol="N", dimension=FORCE),
        UnitDefinition(symbol="N*s", dimension=LINEAR_MOMENTUM),
        UnitDefinition(symbol="kg*m^2", dimension=MOMENT_OF_INERTIA),
    ]
)


__all__ = [
    "ANGULAR_RATE",
    "DEFAULT_UNIT_REGISTRY",
    "DIMENSIONLESS",
    "DimensionVector",
    "ELECTRIC_CURRENT",
    "ENERGY",
    "FORCE",
    "INVERSE_TIME",
    "LENGTH",
    "LINEAR_MOMENTUM",
    "LINEAR_SPEED",
    "MAGNETIC_DIPOLE_MOMENT",
    "MAGNETIC_FIELD",
    "MASS",
    "MASS_FLOW_RATE",
    "MOMENT_OF_INERTIA",
    "MOMENTUM",
    "PRESSURE",
    "POWER",
    "TEMPERATURE",
    "THERMAL_CAPACITY",
    "THERMAL_CONDUCTANCE",
    "TIME",
    "UnitDefinition",
    "UnitRegistry",
    "VOLTAGE",
]
