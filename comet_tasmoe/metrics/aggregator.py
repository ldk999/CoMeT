from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional dependency shim
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised when pandas unavailable
    from comet_tasmoe.utils import pd_compat as pd


@dataclass
class MetricsAggregator:
    scenario: str
    workload: str
    outdir: Path
    window_metrics: List[Dict[str, float]] = field(default_factory=list)

    def collect(self, metrics: Dict[str, float]) -> None:
        self.window_metrics.append(metrics)

    def flush(self) -> pd.DataFrame:
        df = reduce_window_metrics(self.window_metrics)
        if df.empty:
            return df

        df.insert(0, "workload", self.workload)
        df.insert(0, "scenario", self.scenario)

        self.outdir.mkdir(parents=True, exist_ok=True)
        out_path = self.outdir / f"{self.workload}.csv"
        df.to_csv(out_path, index=False)
        return df


def reduce_window_metrics(win_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    columns = [
        "T_peak_logic",
        "T_peak_dram",
        "T_avg",
        "ips",
        "lat_ms",
        "bw_util",
        "refresh_mhz",
        "noc_load",
    ]

    if not win_metrics:
        return pd.DataFrame([{column: 0.0 for column in columns}])

    count = len(win_metrics)
    summary = {
        "T_peak_logic": max(float(m.get("T_peak_logic", 0.0)) for m in win_metrics),
        "T_peak_dram": max(float(m.get("T_peak_dram", 0.0)) for m in win_metrics),
        "T_avg": sum(float(m.get("T_avg", 0.0)) for m in win_metrics) / count,
        "ips": sum(float(m.get("ips", 0.0)) for m in win_metrics) / count,
        "lat_ms": sum(float(m.get("lat_ms", 0.0)) for m in win_metrics) / count,
        "bw_util": sum(float(m.get("bw_util", 0.0)) for m in win_metrics) / count,
        "refresh_mhz": sum(float(m.get("refresh_mhz", 0.0)) for m in win_metrics) / count,
        "noc_load": sum(float(m.get("noc_load", 0.0)) for m in win_metrics) / count,
    }
    return pd.DataFrame([summary], columns=columns)


__all__ = ["reduce_window_metrics", "MetricsAggregator"]
