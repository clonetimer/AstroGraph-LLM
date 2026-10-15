from __future__ import annotations

from .contracts import ExtensionCapability, ExtensionDescriptor, ExtensionProvider


class ExtensionRegistry:
    """Runtime-neutral registry for optional providers.

    Registration is explicit and side-effect free. Core/runtime modules depend only on
    this interface; DOR/FEC/ISGN/M2N implementations stay in compatibility/integration
    packages and may be loaded by deployment-specific factories.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ExtensionProvider] = {}

    def register(self, provider: ExtensionProvider) -> None:
        descriptor = provider.descriptor
        if descriptor.extension_id in self._providers:
            raise ValueError(f"duplicate extension_id: {descriptor.extension_id}")
        self._providers[descriptor.extension_id] = provider

    def descriptors(self) -> list[ExtensionDescriptor]:
        return [self._providers[key].descriptor for key in sorted(self._providers)]

    def providers_for(self, capability: ExtensionCapability) -> list[ExtensionProvider]:
        return [
            self._providers[key]
            for key in sorted(self._providers)
            if capability in self._providers[key].descriptor.capabilities
        ]

    def get(self, extension_id: str) -> ExtensionProvider:
        try:
            return self._providers[extension_id]
        except KeyError as exc:
            raise KeyError(f"unknown AstroGraph extension: {extension_id}") from exc
