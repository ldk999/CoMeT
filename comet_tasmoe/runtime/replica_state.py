from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ReplicaRecord:
    cores: List[int]
    ts_created_ns: int
    last_migration_ns: int = 0


@dataclass
class ReplicaState:
    replicas: Dict[str, ReplicaRecord] = field(default_factory=dict)

    def record_replica(self, expert: str, cores: List[int], ts_ns: int) -> None:
        self.replicas[expert] = ReplicaRecord(cores=list(cores), ts_created_ns=ts_ns, last_migration_ns=ts_ns)

    def update_replica(self, expert: str, cores: List[int], ts_ns: int) -> None:
        if expert not in self.replicas:
            self.record_replica(expert, cores, ts_ns)
            return
        record = self.replicas[expert]
        record.cores = list(cores)
        record.last_migration_ns = ts_ns

    def cores_for(self, expert: str) -> List[int]:
        if expert not in self.replicas:
            return []
        return list(self.replicas[expert].cores)

    def last_move_ts(self, expert: str) -> int:
        if expert not in self.replicas:
            return 0
        return self.replicas[expert].last_migration_ns


__all__ = ["ReplicaState", "ReplicaRecord"]
