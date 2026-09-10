from __future__ import annotations

from typing import Dict, Iterable, Sequence

ALPHA_CMP = 0.8
ACCESS_ENERGY_PJ = 35.0
LEAK_SLOPE = 0.02


def _sum_column(rows: Sequence[Dict[str, object]], key: str) -> float:
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return total


def compute_power_from_events(
    compute_rows: Sequence[Dict[str, object]],
    memory_rows: Sequence[Dict[str, object]],
    cfg_arch: Dict[str, object],
) -> Dict[str, float]:
    total_flops = _sum_column(compute_rows, "flops")
    window_ns = max(1.0, float(len(compute_rows) * 10_000))
    p_logic = ALPHA_CMP * total_flops / window_ns

    bytes_total = _sum_column(memory_rows, "bytes")
    p_mem = (bytes_total * ACCESS_ENERGY_PJ) / 1e3

    avg_temp = 60.0 + LEAK_SLOPE * p_logic
    leak = avg_temp * 0.01

    return {
        "logic_power_w": p_logic + leak,
        "mem_power_w": p_mem + leak / 2,
    }


__all__ = ["compute_power_from_events"]
