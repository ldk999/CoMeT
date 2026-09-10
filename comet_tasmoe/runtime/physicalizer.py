from __future__ import annotations

import json
from typing import Dict, Iterable, List, Sequence, Tuple

from .overlay_types import Overlays
from .replica_state import ReplicaState


def _base_frequency_ghz(hw: Dict[str, object]) -> float:
    dvfs = hw.get("dvfs", {}).get("domains", [])
    if not dvfs:
        return 2.0
    steps = dvfs[0].get("steps_ghz", [2.0])
    return float(steps[0])


def physicalize_window(
    win_idx: int,
    logical_compute: Iterable[Dict[str, object]],
    logical_memory: Iterable[Dict[str, object]],
    placement: Dict[str, object],
    overlays: Overlays,
    state: ReplicaState,
    hw: Dict[str, object],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    freq_ghz = _base_frequency_ghz(hw)
    win_start_ns = int(win_idx * 1_000_000)

    reroute_map: Dict[int, int] = {}
    for overlay in overlays.reroutes:
        for mb_id in overlay.mb_ids:
            reroute_map[mb_id] = overlay.to_core

    expert_placement = placement.get("experts", {})
    tensor_placement = placement.get("tensors", {})

    compute_rows: List[Dict[str, object]] = []
    memory_rows: List[Dict[str, object]] = []

    for local_idx, event in enumerate(logical_compute):
        mb_id = int(event.get("mb_id", 0))
        experts = json.loads(event.get("experts_json", "[]"))
        flops = json.loads(event.get("flops_json", "[]"))
        target_core = None
        for expert in experts:
            mapping = expert_placement.get(expert)
            if mapping and mapping.get("cores"):
                target_core = int(mapping["cores"][0])
                state.update_replica(expert, list(mapping["cores"]), win_start_ns)
                break
        if target_core is None:
            target_core = 0
        if mb_id in reroute_map:
            target_core = reroute_map[mb_id]

        total_flops = sum(float(value) for value in flops)
        duration_cycles = int(total_flops / max(freq_ghz, 1e-3))
        compute_rows.append(
            {
                "evt_id": win_idx * 1000 + local_idx,
                "t_start_ns": win_start_ns + local_idx * 10_000,
                "core_id": target_core,
                "duration_cycles": duration_cycles,
                "flops": total_flops,
                "op": event.get("op_type", "moe"),
                "dvfs_domain": 0,
                "power_hint": total_flops / max(duration_cycles, 1) if duration_cycles else total_flops,
            }
        )

        for expert in experts:
            tensor_id = f"W_{expert}_0"
            tensor_info = tensor_placement.get(tensor_id)
            if tensor_info:
                memory_rows.append(
                    {
                        "evt_id": win_idx * 10_000 + local_idx,
                        "t_start_ns": win_start_ns + local_idx * 10_000,
                        "src_core": target_core,
                        "dst_bank": tensor_info.get("dram_bank", ""),
                        "bytes": tensor_info.get("size_bytes", 0),
                        "rw": "read",
                        "hops": abs(target_core - local_idx) % 4,
                        "noc_qos": 0,
                    }
                )

    for offset, mem_event in enumerate(logical_memory):
        tensor_id = str(mem_event.get("tensor_id", ""))
        tensor_info = tensor_placement.get(tensor_id, {})
        memory_rows.append(
            {
                "evt_id": win_idx * 10_000 + len(memory_rows) + offset,
                "t_start_ns": win_start_ns + len(memory_rows) * 5_000,
                "src_core": 0,
                "dst_bank": tensor_info.get("dram_bank", ""),
                "bytes": int(mem_event.get("bytes", 0)),
                "rw": mem_event.get("access", "read"),
                "hops": 0,
                "noc_qos": 0,
            }
        )

    return compute_rows, memory_rows


__all__ = ["physicalize_window"]
