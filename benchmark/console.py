"""Console helpers with Windows-safe text output."""
from __future__ import annotations

import sys
from typing import IO, Any

from rich.console import Console


class EncodingSafeTextIO:
    """Proxy text streams and replace characters unsupported by their encoding."""

    def __init__(self, wrapped: IO[str]):
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        try:
            return self._wrapped.write(text)
        except UnicodeEncodeError:
            encoding = self.encoding or "utf-8"
            safe_text = text.encode(encoding, errors="replace").decode(encoding)
            return self._wrapped.write(safe_text)

    def flush(self) -> None:
        self._wrapped.flush()

    def isatty(self) -> bool:
        return self._wrapped.isatty()

    def fileno(self) -> int:
        return self._wrapped.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self._wrapped, "encoding", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def make_console(file: IO[str] | None = None) -> Console:
    """Create a Rich console that will not crash on legacy encodings."""
    return Console(
        file=EncodingSafeTextIO(file or sys.stdout),
        emoji=False,
        safe_box=True,
    )
