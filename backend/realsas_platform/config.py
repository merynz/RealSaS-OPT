from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class PlatformSettings:
    database_url: str
    temporal_target: str
    temporal_namespace: str
    artifact_root: str
    environment: str

    @classmethod
    def from_env(cls) -> "PlatformSettings":
        return cls(
            database_url=os.environ["REALSAS_PLATFORM_DATABASE_URL"],
            temporal_target=os.environ.get("REALSAS_TEMPORAL_TARGET", "127.0.0.1:7233"),
            temporal_namespace=os.environ.get("REALSAS_TEMPORAL_NAMESPACE", "default"),
            artifact_root=os.environ.get("REALSAS_ARTIFACT_ROOT", ".realsas-platform-artifacts"),
            environment=os.environ.get("REALSAS_PLATFORM_ENV", "development"),
        )
