from __future__ import annotations

import logging
from typing import Dict, Sequence

import math

from .power_injector import compute_power_from_events
from .dvfs_refresh_models import dvfs_step, refresh_multiplier

LOGGER = logging.getLogger(__name__)


def submit_events_to_comet(
    compute_rows: Sequence[Dict[str, object]],
    memory_rows: Sequence[Dict[str, object]],
    cfg_arch: Dict[str, object],
) -> Dict[str, object]:
    power = compute_power_from_events(compute_rows, memory_rows, cfg_arch)
    freq_ghz = dvfs_step(70.0, "all")

    core_cycles: Dict[int, float] = {}
    total_cycles = 0.0
    start_times: list[int] = []
    for row in compute_rows:
        core_id = int(float(row.get("core_id", 0)))
        cycles = float(row.get("duration_cycles", row.get("flops", 0.0)))
        core_cycles[core_id] = core_cycles.get(core_id, 0.0) + cycles
        total_cycles += cycles
        try:
            start_times.append(int(row.get("t_start_ns", 0)))
        except (TypeError, ValueError):
            start_times.append(0)

    if not core_cycles:
        core_cycles[0] = 0.0

    util_core = {
        core: (cycles / total_cycles) if total_cycles > 0 else 0.0
        for core, cycles in core_cycles.items()
    }
    temps_core = [60.0 + 25.0 * util for util in util_core.values()]

    bank_totals: Dict[str, float] = {}
    for row in memory_rows:
        bank = str(row.get("dst_bank", "B(0,0,0)"))
        try:
            volume = float(row.get("bytes", 0.0))
        except (TypeError, ValueError):
            volume = 0.0
        bank_totals[bank] = bank_totals.get(bank, 0.0) + volume
    temps_bank = [55.0 + 15.0 * math.tanh(total / 1e9) for total in bank_totals.values()]

    noc_load = min(1.0, len(memory_rows) / 1000.0)
    refresh_rates = [refresh_multiplier(temp) for temp in temps_bank] or [1.0]

    bytes_sum = sum(bank_totals.values())
    temps_all = temps_core + temps_bank if temps_bank else temps_core
    avg_temp = sum(temps_all) / len(temps_all) if temps_all else 60.0

    metrics = {
        "T_peak_logic": max(temps_core) if temps_core else 60.0,
        "T_peak_dram": max(temps_bank) if temps_bank else 55.0,
        "T_avg": avg_temp,
        "ips": len(compute_rows) * freq_ghz,
        "lat_ms": len(compute_rows) / max(freq_ghz, 1e-3),
        "bw_util": bytes_sum / 1e9,
        "refresh_mhz": (sum(refresh_rates) / len(refresh_rates)) * 1e3,
        "noc_load": noc_load,
        "power_logic_w": power["logic_power_w"],
        "power_mem_w": power["mem_power_w"],
    }

    telemetry = {
        "temps_core": temps_core,
        "temps_bank": temps_bank,
        "util_core": list(util_core.values()),
        "noc_load": noc_load,
        "ts_ns": min(start_times) if start_times else 0,
    }

    LOGGER.debug("submit_events_to_comet metrics: %s", metrics)
    return {"metrics": metrics, "telemetry": telemetry}


__all__ = ["submit_events_to_comet"]
