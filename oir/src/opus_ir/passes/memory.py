"""Deterministic LDS planning: explicit offsets are honored, the rest are packed in declaration order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..ir import Kernel


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class LdsAllocation:
    name: str
    offset_bytes: int
    size_bytes: int
    alias_of: Optional[str] = None

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes


@dataclass(frozen=True)
class LdsPlan:
    allocations: Tuple[LdsAllocation, ...]
    total_bytes: int
    overlaps: Tuple[Tuple[str, str], ...]

    def by_name(self) -> Dict[str, LdsAllocation]:
        return {item.name: item for item in self.allocations}


def plan_lds(kernel: Kernel) -> LdsPlan:
    """Plan every LDS region. ``alias_of`` regions reuse their backing region's offset."""
    owners = [buf for buf in kernel.lds if buf.alias_of is None]
    fixed = [buf for buf in owners if buf.offset_bytes is not None]
    placed: List[LdsAllocation] = [LdsAllocation(buf.name, buf.offset_bytes, buf.total_bytes) for buf in fixed]
    cursor = max((a.end_bytes for a in placed), default=0)
    for buf in owners:
        if buf.offset_bytes is None:
            offset = align_up(cursor, buf.alignment)
            placed.append(LdsAllocation(buf.name, offset, buf.total_bytes))
            cursor = offset + buf.total_bytes

    overlaps = []
    ordered = sorted(placed, key=lambda a: a.offset_bytes)
    for left, right in zip(ordered, ordered[1:]):
        if right.offset_bytes < left.end_bytes:
            overlaps.append((left.name, right.name))

    by_name = {a.name: a for a in placed}
    for buf in kernel.lds:
        if buf.alias_of is not None and buf.alias_of in by_name:
            base = by_name[buf.alias_of]
            placed.append(LdsAllocation(buf.name, base.offset_bytes, buf.total_bytes, buf.alias_of))

    total = max((a.end_bytes for a in placed), default=0)
    return LdsPlan(tuple(placed), total, tuple(overlaps))
