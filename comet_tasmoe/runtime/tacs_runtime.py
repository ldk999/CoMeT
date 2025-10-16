from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency shim
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised when pandas unavailable
    from comet_tasmoe.utils import pd_compat as pd

from .overlay_types import MigrationOverlay, RerouteOverlay, WindowView
from .replica_state import ReplicaState

LOGGER = logging.getLogger(__name__)


@dataclass
class Telemetry:
    temps_core: List[float]
    temps_bank: List[float]
    util_core: List[float]
    noc_load: float
    ts_ns: int


def _pressure_scores(tele: Telemetry, params: Dict[str, object]) -> List[float]:
    thresholds = params.get("thresholds", {})
    t_hot = float(thresholds.get("Thot", 80.0))
    t_emg = float(thresholds.get("Temg", 90.0))
    q_target = float(params.get("q_target", 0.7))

    scores: List[float] = []
    gamma = float(params.get("gamma", 1.0))
    for temp, util in zip(tele.temps_core, tele.util_core):
        temp_term = max(0.0, float(temp) - t_hot) / max(1e-6, t_emg - t_hot)
        util_term = max(0.0, float(util) - q_target) / max(1e-6, q_target)
        scores.append(temp_term + gamma * util_term)
    return scores


def _select_candidates(scores: List[float], k: int) -> List[int]:
    if not scores:
        return []
    indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    hot: List[int] = []
    for idx in indices:
        if scores[idx] <= 0:
            break
        hot.append(int(idx))
        if len(hot) >= k:
            break
    return hot


def _find_cool_cores(scores: List[float], temps: Iterable[float], limit: int) -> List[int]:
    indexed = sorted(enumerate(float(temp) for temp in temps), key=lambda item: item[1])
    result: List[int] = []
    for idx, _temp in indexed:
        if idx < len(scores) and scores[idx] <= 0:
            result.append(int(idx))
        if len(result) >= limit:
            break
    return result


def tacs_step(
    tele: Telemetry,
    placement: Dict[str, object],
    state: ReplicaState,
    params: Dict[str, object],
    pending_window: WindowView,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores = _pressure_scores(tele, params)
    max_parallel = int(params.get("max_parallel_migrations", 1))
    hot_cores = _select_candidates(scores, max_parallel)

    cool_cores = _find_cool_cores(scores, tele.temps_core, max_parallel * 2)
    LOGGER.debug("Hot cores: %s -> cool candidates: %s", hot_cores, cool_cores)

    reroutes: List[RerouteOverlay] = []
    migrations: List[MigrationOverlay] = []

    cost_cfg = params.get("cost", {})
    lambda_hops = float(cost_cfg.get("lambda_hops", 1.0))
    mu_noc = float(cost_cfg.get("mu_noc", 1.0))
    eta_temp = float(cost_cfg.get("eta_temp", 1.0))

    for src_core in hot_cores:
        if not pending_window.pending_mb_ids:
            break
        dst_core = next((core for core in cool_cores if core != src_core), src_core)
        if dst_core == src_core:
            continue
        mb_id = pending_window.pending_mb_ids[0]
        hops = abs(dst_core - src_core)
        cost = lambda_hops * hops + mu_noc * tele.noc_load + eta_temp * (
            scores[src_core] - scores[dst_core]
        )
        reroutes.append(
            RerouteOverlay(
                t_apply_ns=tele.ts_ns,
                scope="mb",
                from_core=src_core,
                to_core=dst_core,
                mb_ids=[mb_id],
                reason="hot_core",
                expected_gain=max(0.0, scores[src_core] - scores[dst_core]),
            )
        )
        tensor_id = f"W_E{mb_id % max(1, len(placement.get('experts', {})))}_0"
        src_bank = placement.get("tensors", {}).get(tensor_id, {}).get("dram_bank", "")
        migrations.append(
            MigrationOverlay(
                t_apply_ns=tele.ts_ns,
                obj_type="weight",
                obj_id=tensor_id,
                src=src_bank,
                dst=src_bank,
                size_bytes=int(
                    placement.get("tensors", {}).get(tensor_id, {}).get("size_bytes", 0)
                ),
                cost=float(cost),
            )
        )

    reroute_df = pd.DataFrame(
        [
            {
                "t_apply_ns": r.t_apply_ns,
                "scope": r.scope,
                "from_core": r.from_core,
                "to_core": r.to_core,
                "mb_ids_json": json.dumps(r.mb_ids),
                "reason": r.reason,
                "expected_gain": r.expected_gain,
            }
            for r in reroutes
        ],
        columns=[
            "t_apply_ns",
            "scope",
            "from_core",
            "to_core",
            "mb_ids_json",
            "reason",
            "expected_gain",
        ],
    )

    migrate_df = pd.DataFrame(
        [
            {
                "t_apply_ns": m.t_apply_ns,
                "obj_type": m.obj_type,
                "obj_id": m.obj_id,
                "src": m.src,
                "dst": m.dst,
                "size_bytes": m.size_bytes,
                "cost": m.cost,
            }
            for m in migrations
        ],
        columns=[
            "t_apply_ns",
            "obj_type",
            "obj_id",
            "src",
            "dst",
            "size_bytes",
            "cost",
        ],
    )

    return reroute_df, migrate_df


__all__ = ["Telemetry", "tacs_step"]
