from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # pragma: no cover - optional dependency shim
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised when pandas unavailable
    from comet_tasmoe.utils import pd_compat as pd

from comet_tasmoe.adapters.comet_sink import submit_events_to_comet
from comet_tasmoe.mapping.taps_heuristic import build_placement
from comet_tasmoe.metrics.aggregator import MetricsAggregator
from comet_tasmoe.runtime.overlay_types import MigrationOverlay, Overlays, RerouteOverlay, WindowView
from comet_tasmoe.runtime.physicalizer import physicalize_window
from comet_tasmoe.runtime.replica_state import ReplicaState
from comet_tasmoe.runtime.tacs_runtime import Telemetry, tacs_step
from comet_tasmoe.workloads.generate_traces import generate_logical_traces

LOGGER = logging.getLogger(__name__)


def _default_placement(cfg_arch: Dict[str, object], expert_count: int) -> Dict[str, object]:
    cores = cfg_arch.get("mesh", {}).get("cores", [8, 8])
    total_cores = cores[0] * cores[1]
    placement = {"experts": {}, "tensors": {}}
    for idx in range(expert_count):
        core_id = idx % total_cores
        placement["experts"][f"E{idx}"] = {"cores": [core_id], "policy": "single"}
        placement["tensors"][f"W_E{idx}_0"] = {"dram_bank": "B(0,0,0)", "size_bytes": 1_048_576}
    return placement


def _read_csv_dicts(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _chunk(data: Sequence[Dict[str, object]], size: int) -> List[List[Dict[str, object]]]:
    return [data[idx : idx + size] for idx in range(0, len(data), size)]


def _df_to_overlays(reroute_df: pd.DataFrame, migrate_df: pd.DataFrame) -> Overlays:
    overlays = Overlays()

    if not reroute_df.empty:
        for row in reroute_df.to_dict(orient="records"):
            overlays.reroutes.append(
                RerouteOverlay(
                    t_apply_ns=int(row.get("t_apply_ns", 0)),
                    scope=str(row.get("scope", "mb")),
                    from_core=int(row.get("from_core", 0)),
                    to_core=int(row.get("to_core", 0)),
                    mb_ids=json.loads(str(row.get("mb_ids_json", "[]"))),
                    reason=str(row.get("reason", "")),
                    expected_gain=float(row.get("expected_gain", 0.0)),
                )
            )

    if not migrate_df.empty:
        for row in migrate_df.to_dict(orient="records"):
            overlays.migrations.append(
                MigrationOverlay(
                    t_apply_ns=int(row.get("t_apply_ns", 0)),
                    obj_type=str(row.get("obj_type", "weight")),
                    obj_id=str(row.get("obj_id", "")),
                    src=str(row.get("src", "")),
                    dst=str(row.get("dst", "")),
                    size_bytes=int(row.get("size_bytes", 0)),
                    cost=float(row.get("cost", 0.0)),
                )
            )

    return overlays


def _load_yaml(path: str) -> Dict[str, object]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - optional dependency shim
        try:
            data = json.loads(text)
        except json.JSONDecodeError as json_exc:
            raise RuntimeError(
                "PyYAML is not installed and configuration is not valid JSON"
            ) from json_exc
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return data


def run_experiment(
    cfg_arch_path: str,
    cfg_workload_path: str,
    tas_params_path: str,
    scenario: str,
    outdir: str,
) -> str:
    cfg_arch = _load_yaml(cfg_arch_path)
    cfg_workload = _load_yaml(cfg_workload_path)
    tas_params = _load_yaml(tas_params_path)

    traces_dir = Path("comet_tasmoe/traces")
    traces_dir.mkdir(parents=True, exist_ok=True)
    logical_compute_path, logical_memory_path = generate_logical_traces(
        cfg_workload, str(traces_dir)
    )

    logical_compute = _read_csv_dicts(Path(logical_compute_path))
    logical_memory = _read_csv_dicts(Path(logical_memory_path))

    expert_set: set[str] = set()
    for row in logical_compute:
        expert_set.update(json.loads(row.get("experts_json", "[]")))
    expert_count = len(expert_set)
    if scenario in {"taps", "tas"}:
        placement = build_placement(cfg_arch, cfg_workload, tas_params.get("taps", {}))
    else:
        placement = _default_placement(cfg_arch, expert_count)

    state = ReplicaState()
    window_size = max(1, len(logical_compute) // 5)
    compute_windows = _chunk(logical_compute, window_size)
    memory_windows = _chunk(logical_memory, window_size)
    total_windows = max(len(compute_windows), len(memory_windows))

    aggregator = MetricsAggregator(scenario=scenario, workload=Path(cfg_workload_path).stem, outdir=Path(outdir) / scenario)
    telemetry_data = {
        "temps_core": [60.0] * 4,
        "temps_bank": [55.0] * 4,
        "util_core": [0.1] * 4,
        "noc_load": 0.1,
        "ts_ns": 0,
    }

    for win_idx in range(total_windows):
        compute_chunk = compute_windows[win_idx] if win_idx < len(compute_windows) else []
        memory_chunk = memory_windows[win_idx] if win_idx < len(memory_windows) else []
        overlays = Overlays()
        if scenario in {"tacs", "tas"}:
            tele = Telemetry(
                temps_core=list(telemetry_data["temps_core"]),
                temps_bank=list(telemetry_data["temps_bank"]),
                util_core=list(telemetry_data["util_core"]),
                noc_load=float(telemetry_data["noc_load"]),
                ts_ns=int(telemetry_data["ts_ns"]),
            )
            pending = WindowView(window_idx=win_idx, pending_mb_ids=[int(row["mb_id"]) for row in compute_chunk])
            reroute_df, migrate_df = tacs_step(
                tele,
                placement,
                state,
                tas_params.get("tacs", {}),
                pending,
            )
            overlays = _df_to_overlays(reroute_df, migrate_df)

        compute_df, memory_df = physicalize_window(
            win_idx,
            compute_chunk,
            memory_chunk,
            placement,
            overlays,
            state,
            cfg_arch,
        )

        result = submit_events_to_comet(compute_df, memory_df, cfg_arch)
        aggregator.collect(result["metrics"])
        telemetry_data = result["telemetry"]

    _ = aggregator.flush()
    output_path = aggregator.outdir / f"{Path(cfg_workload_path).stem}.csv"
    LOGGER.info("Experiment %s %s complete", scenario, cfg_workload_path)
    return str(output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a single TAS-MoE scenario")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--tas", required=True)
    parser.add_argument("--scenario", required=True, choices=["baseline", "taps", "tacs", "tas"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_experiment(args.arch, args.workload, args.tas, args.scenario, args.out)


if __name__ == "__main__":
    main()
