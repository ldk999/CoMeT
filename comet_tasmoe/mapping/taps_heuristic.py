from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Any

def _flatten_mesh(cores: Iterable[int]) -> List[int]:
    xs, ys = cores
    return [idx for idx in range(xs * ys)]


def _estimate_hotness(num_experts: int, zipf_s: float) -> List[float]:
    weights = [1.0 / ((idx + 1) ** zipf_s) for idx in range(num_experts)]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def build_placement(
    cfg_arch: Dict[str, object],
    cfg_workload: Dict[str, object],
    taps_params: Dict[str, object],
    H: Any | None = None,
) -> Dict[str, object]:
    mesh_cfg = cfg_arch.get("mesh", {})
    cores_dims = mesh_cfg.get("cores", [8, 8])
    core_ids = _flatten_mesh(cores_dims)

    model_name = str(cfg_workload.get("model", "mixtral-8x7b"))
    expert_count = sum(ch.isdigit() for ch in model_name)
    if expert_count == 0:
        expert_count = 8
    experts = [f"E{idx}" for idx in range(expert_count)]

    routing_cfg = cfg_workload.get("routing", {})
    zipf_s = float(routing_cfg.get("zipf_s", 1.0))
    hotness = _estimate_hotness(expert_count, zipf_s)

    placement: Dict[str, object] = {"experts": {}, "tensors": {}}
    capacity_cfg = taps_params.get("capacity", {}) if taps_params else {}
    core_capacity = float(capacity_cfg.get("core_flops_tps", 5e12))

    per_core_load: Dict[int, float] = defaultdict(float)

    for idx, expert in enumerate(experts):
        target_core = min(core_ids, key=lambda cid: per_core_load[cid])
        per_core_load[target_core] += hotness[idx] * core_capacity
        placement["experts"][expert] = {"cores": [target_core], "policy": "single"}

    replicate_k = int(taps_params.get("replicate_hot_k", 0)) if taps_params else 0
    if replicate_k > 0:
        hot_indices = sorted(range(len(hotness)), key=lambda idx: hotness[idx], reverse=True)[
            :replicate_k
        ]
        for idx in hot_indices:
            expert = experts[idx]
            primary_core = placement["experts"][expert]["cores"][0]
            secondary_core = min(
                (cid for cid in core_ids if cid != primary_core),
                key=lambda cid: per_core_load[cid],
                default=primary_core,
            )
            if secondary_core != primary_core:
                placement["experts"][expert]["cores"].append(secondary_core)
                placement["experts"][expert]["policy"] = "replicate"
                per_core_load[secondary_core] += hotness[idx] * core_capacity * 0.5

    dram_cfg = cfg_arch.get("dram", {})
    banks_dims = dram_cfg.get("banks", [32, 32, 8])
    total_banks = 1
    for dim in banks_dims:
        total_banks *= int(dim)
    tensors_cfg = cfg_workload.get("tensors", {})
    shard_bytes = int(tensors_cfg.get("weight_shard_mb", 1)) * 1024 * 1024

    for idx, expert in enumerate(experts):
        bank_id = idx % total_banks
        z = bank_id // (banks_dims[0] * banks_dims[1])
        rem = bank_id % (banks_dims[0] * banks_dims[1])
        y = rem // banks_dims[0]
        x = rem % banks_dims[0]
        placement["tensors"][f"W_{expert}_0"] = {
            "dram_bank": f"B({x},{y},{z})",
            "size_bytes": shard_bytes,
        }

    return placement


__all__ = ["build_placement"]
