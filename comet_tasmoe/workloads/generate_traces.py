from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple
import random


def _infer_expert_count(model_name: str) -> int:
    digits: list[str] = []
    for ch in model_name:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return 8
    return max(1, int(''.join(digits)))


def _zipf_weights(count: int, s: float) -> list[float]:
    return [1.0 / ((idx + 1) ** s) for idx in range(count)]


def generate_logical_traces(cfg_workload: dict, out_dir: str) -> Tuple[str, str]:
    """Generate deterministic logical traces for compute and memory layers."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    expert_count = _infer_expert_count(cfg_workload.get("model", ""))
    experts = [f"E{idx}" for idx in range(expert_count)]
    seed = int(cfg_workload.get("routing", {}).get("seed", 0))
    rng = random.Random(seed)

    topk = int(cfg_workload.get("routing", {}).get("topk", 2))
    zipf_s = float(cfg_workload.get("routing", {}).get("zipf_s", 1.0))
    weights = _zipf_weights(expert_count, zipf_s)

    seq_cfg = cfg_workload.get("sequence", {})
    tokens = int(seq_cfg.get("tokens", 1024))
    micro_batch = int(seq_cfg.get("micro_batch", 32))
    steps = int(seq_cfg.get("steps", 1))
    micro_batches = micro_batch * steps
    tokens_per_mb = max(1, tokens // max(1, micro_batch))

    ops_cfg = cfg_workload.get("operators", {})
    fc_flops = float(ops_cfg.get("fc", {}).get("flop_per_token", 0.0))
    attn_flops = float(ops_cfg.get("attn", {}).get("flop_per_token", 0.0))
    kv_bytes = int(ops_cfg.get("attn", {}).get("kv_bytes_per_token", 0))

    tensor_cfg = cfg_workload.get("tensors", {})
    weight_mb = int(tensor_cfg.get("weight_shard_mb", 1))
    weight_bytes = weight_mb * 1024 * 1024

    compute_path = out_path / "logical_compute.csv"
    memory_path = out_path / "logical_memory.csv"

    event_id = 0
    tensor_event_id = 0

    with compute_path.open("w", newline="", encoding="utf-8") as comp_f:
        comp_writer = csv.writer(comp_f)
        with memory_path.open("w", newline="", encoding="utf-8") as mem_f:
            mem_writer = csv.writer(mem_f)

            comp_writer.writerow([
                "event_id",
                "t_gen_ns",
                "mb_id",
                "token_id",
                "experts_json",
                "flops_json",
                "op_type",
            ])
            mem_writer.writerow([
                "event_id",
                "tensor_id",
                "access",
                "bytes",
                "locality_hint",
            ])

            for mb_id in range(micro_batches):
                experts_sel = rng.choices(experts, weights=weights, k=topk)
                flops_per_expert = [
                    (fc_flops + attn_flops) * tokens_per_mb / max(1, topk)
                    for _ in range(topk)
                ]
                t_gen = mb_id * 1000
                comp_writer.writerow(
                    [
                        event_id,
                        t_gen,
                        mb_id,
                        0,
                        json.dumps(experts_sel),
                        json.dumps(flops_per_expert),
                        "moe",
                    ]
                )

                for expert in experts_sel:
                    tensor_event_id += 1
                    mem_writer.writerow(
                        [
                            tensor_event_id,
                            f"W_{expert}_0",
                            "read",
                            weight_bytes,
                            "shard0",
                        ]
                    )

                kv_bytes_total = kv_bytes * tokens_per_mb
                if kv_bytes_total > 0:
                    tensor_event_id += 1
                    mem_writer.writerow(
                        [
                            tensor_event_id,
                            f"KV_{mb_id}",
                            "read",
                            kv_bytes_total,
                            "seq",
                        ]
                    )

                event_id += 1

    return (str(compute_path), str(memory_path))


__all__ = ["generate_logical_traces"]
