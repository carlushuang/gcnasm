"""Stable, structured diagnostics returned to humans and agents (same contract as cake-ir)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    path: str = "kernel"
    notes: Tuple[str, ...] = ()

    def render(self) -> str:
        head = "%s[%s] %s: %s" % (self.severity.value, self.code, self.path, self.message)
        if not self.notes:
            return head
        return head + "\n" + "\n".join("  note: " + note for note in self.notes)


@dataclass
class VerificationResult:
    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(item.severity is Severity.ERROR for item in self.diagnostics)

    @property
    def errors(self) -> List[Diagnostic]:
        return [item for item in self.diagnostics if item.severity is Severity.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [item for item in self.diagnostics if item.severity is Severity.WARNING]

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        path: str = "kernel",
        notes: Iterable[str] = (),
    ) -> None:
        self.diagnostics.append(Diagnostic(code, severity, message, path=path, notes=tuple(notes)))

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    def require_ok(self) -> None:
        if not self.ok:
            raise VerificationError(self)

    def render(self) -> str:
        if not self.diagnostics:
            return "verification succeeded"
        return "\n".join(item.render() for item in self.diagnostics)


class VerificationError(RuntimeError):
    def __init__(self, result: VerificationResult, message: Optional[str] = None) -> None:
        self.result = result
        super().__init__(message or result.render())
