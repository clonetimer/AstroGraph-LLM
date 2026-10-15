from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .contracts import ChannelMapCatalog


class PortDirection(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    OBSERVATION = "OBSERVATION"


class PortRole(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port_role_id: str = Field(min_length=1)
    direction: PortDirection
    quantity_kind_ids: list[str] = Field(min_length=1)
    required: bool = True
    description: str = ""

    @field_validator("quantity_kind_ids")
    @classmethod
    def unique_quantity_kinds(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("port quantity_kind_ids must be unique")
        return value


class ComponentArchetype(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archetype_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    family: str = Field(min_length=1)
    functional_role_ids: list[str] = Field(min_length=1)
    state_quantity_kind_ids: list[str] = Field(default_factory=list)
    port_roles: list[PortRole] = Field(min_length=1)
    description: str = ""

    @model_validator(mode="after")
    def unique_roles(self) -> "ComponentArchetype":
        role_ids = [item.port_role_id for item in self.port_roles]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("component archetype port_role_id values must be unique")
        if len(self.functional_role_ids) != len(set(self.functional_role_ids)):
            raise ValueError("functional_role_ids must be unique")
        if len(self.state_quantity_kind_ids) != len(set(self.state_quantity_kind_ids)):
            raise ValueError("state_quantity_kind_ids must be unique")
        return self

    def port_role(self, port_role_id: str) -> PortRole:
        for role in self.port_roles:
            if role.port_role_id == port_role_id:
                return role
        raise KeyError(f"unknown port role {port_role_id} for archetype {self.archetype_id}")


class ComponentArchetypeCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-component-archetype-catalog.v1"
    catalog_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    semantic_schema_id: str = Field(min_length=1)
    archetypes: list[ComponentArchetype] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_archetypes(self) -> "ComponentArchetypeCatalog":
        ids = [item.archetype_id for item in self.archetypes]
        if len(ids) != len(set(ids)):
            raise ValueError("component archetype IDs must be unique")
        return self

    def archetype(self, archetype_id: str) -> ComponentArchetype:
        for item in self.archetypes:
            if item.archetype_id == archetype_id:
                return item
        raise KeyError(f"unknown component archetype: {archetype_id}")

    def validate_quantity_kinds(self, known_quantity_kind_ids: set[str]) -> None:
        referenced: set[str] = set()
        for archetype in self.archetypes:
            referenced.update(archetype.state_quantity_kind_ids)
            for role in archetype.port_roles:
                referenced.update(role.quantity_kind_ids)
        unknown = sorted(referenced - known_quantity_kind_ids)
        if unknown:
            raise ValueError(f"archetype catalog references unknown quantity kinds: {unknown}")

    def validate_channel_maps(self, channel_maps: "ChannelMapCatalog") -> None:
        for channel_map in channel_maps.maps:
            if channel_map.semantic_schema_id != self.semantic_schema_id:
                raise ValueError(
                    f"channel map {channel_map.mapping_id} uses a different semantic schema"
                )
            for entry in channel_map.entries:
                archetype = self.archetype(entry.component_archetype_id)
                if entry.port_role_id is None:
                    raise ValueError(
                        f"channel {entry.semantic_channel_id} has no declared archetype port role"
                    )
                role = archetype.port_role(entry.port_role_id)
                if entry.quantity_kind_id not in set(role.quantity_kind_ids):
                    raise ValueError(
                        f"channel {entry.semantic_channel_id} quantity kind {entry.quantity_kind_id} "
                        f"is not allowed by port role {entry.port_role_id}"
                    )


class PortBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port_role_id: str = Field(min_length=1)
    semantic_channel_ids: list[str] = Field(min_length=1)

    @field_validator("semantic_channel_ids")
    @classmethod
    def unique_channels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("port binding semantic_channel_ids must be unique")
        return value


class ComponentInstance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_instance_id: str = Field(min_length=1)
    component_archetype_id: str = Field(min_length=1)
    parent_component_instance_id: str | None = None
    port_bindings: list[PortBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_port_bindings(self) -> "ComponentInstance":
        ids = [item.port_role_id for item in self.port_bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("component instance port_role_id bindings must be unique")
        return self


class AssetTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-asset-topology.v1"
    asset_instance_id: str = Field(min_length=1)
    asset_archetype_id: str = Field(min_length=1)
    components: list[ComponentInstance] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_component_tree(self) -> "AssetTopology":
        ids = [item.component_instance_id for item in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component instance IDs must be unique")
        known = set(ids)
        for component in self.components:
            parent = component.parent_component_instance_id
            if parent == component.component_instance_id:
                raise ValueError("component cannot be its own parent")
            if parent is not None and parent not in known:
                raise ValueError(f"unknown parent component instance: {parent}")
        parent_by_id = {
            item.component_instance_id: item.parent_component_instance_id for item in self.components
        }
        for component_id in known:
            visited: set[str] = set()
            current: str | None = component_id
            while current is not None:
                if current in visited:
                    raise ValueError("component hierarchy contains a cycle")
                visited.add(current)
                current = parent_by_id[current]
        return self

    def validate_against(
        self,
        catalog: ComponentArchetypeCatalog,
        *,
        known_semantic_channel_ids: set[str] | None = None,
    ) -> None:
        known_channels = known_semantic_channel_ids or set()
        validate_channels = known_semantic_channel_ids is not None
        for component in self.components:
            archetype = catalog.archetype(component.component_archetype_id)
            for binding in component.port_bindings:
                archetype.port_role(binding.port_role_id)
                if validate_channels:
                    unknown = sorted(set(binding.semantic_channel_ids) - known_channels)
                    if unknown:
                        raise ValueError(f"port binding references unknown semantic channels: {unknown}")


__all__ = [
    "AssetTopology",
    "ComponentArchetype",
    "ComponentArchetypeCatalog",
    "ComponentInstance",
    "PortBinding",
    "PortDirection",
    "PortRole",
]
