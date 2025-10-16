"""Lightweight pandas compatibility layer for environments without pandas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence


class DataFrame:
    """A minimal in-house DataFrame implementation mimicking pandas APIs used here."""

    def __init__(
        self,
        data: Iterable[Mapping[str, Any]] | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        rows = [dict(row) for row in (data or [])]
        if columns is None:
            inferred: List[str] = []
            if rows:
                for key in rows[0].keys():
                    inferred.append(str(key))
            columns = inferred
        self._columns: List[str] = list(columns)
        self._rows: List[dict[str, Any]] = []
        for row in rows:
            normalized = {column: row.get(column) for column in self._columns}
            for column in self._columns:
                if column not in normalized:
                    normalized[column] = None
            self._rows.append(normalized)

    @property
    def empty(self) -> bool:
        return not self._rows

    def to_dict(self, orient: str = "records") -> List[dict[str, Any]]:
        if orient != "records":
            raise ValueError("Only 'records' orient is supported in the pandas compatibility layer")
        result: List[dict[str, Any]] = []
        for row in self._rows:
            result.append({column: row.get(column) for column in self._columns})
        return result

    def insert(self, loc: int, column: str, value: Any) -> None:
        loc = max(0, min(loc, len(self._columns)))
        if column in self._columns:
            self._columns.remove(column)
        self._columns.insert(loc, column)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
            if len(values) != len(self._rows):
                raise ValueError("Column length does not match DataFrame length")
        else:
            values = [value] * len(self._rows)
        for idx, row in enumerate(self._rows):
            row[column] = values[idx]

    def to_csv(self, path: str | Path, index: bool = False) -> None:  # noqa: ARG002 - parity with pandas
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(self._columns)
            for row in self._rows:
                writer.writerow([row.get(column, "") for column in self._columns])

    def __iter__(self):  # type: ignore[override]
        return iter(self._rows)


__all__ = ["DataFrame"]
