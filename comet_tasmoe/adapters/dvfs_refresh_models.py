from __future__ import annotations


def refresh_multiplier(temp_c: float) -> float:
    if temp_c < 40.0:
        return 1.0
    if temp_c < 55.0:
        return 2.0
    if temp_c < 70.0:
        return 4.0
    return 8.0


def dvfs_step(temp_c: float, domain: str) -> float:
    if temp_c < 75.0:
        return 2.0
    if temp_c < 82.0:
        return 1.8
    if temp_c < 88.0:
        return 1.6
    return 1.2


__all__ = ["refresh_multiplier", "dvfs_step"]
