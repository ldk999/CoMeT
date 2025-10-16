from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


def _ensure_list(values: Iterable[int]) -> List[int]:
    return list(values)


@dataclass
class RerouteOverlay:
    t_apply_ns: int
    scope: str
    from_core: int
    to_core: int
    mb_ids: List[int] = field(default_factory=list)
    reason: str = ""
    expected_gain: float = 0.0


@dataclass
class MigrationOverlay:
    t_apply_ns: int
    obj_type: str
    obj_id: str
    src: str
    dst: str
    size_bytes: int
    cost: float


@dataclass
class Overlays:
    reroutes: List[RerouteOverlay] = field(default_factory=list)
    migrations: List[MigrationOverlay] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.reroutes and not self.migrations


@dataclass
class WindowView:
    window_idx: int
    pending_mb_ids: List[int]


__all__ = [
    "RerouteOverlay",
    "MigrationOverlay",
    "Overlays",
    "WindowView",
]
