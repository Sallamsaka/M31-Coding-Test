"""Package init. Its one job is to make stdout able to carry what we print.

The Windows console defaults to cp1252, which cannot encode `Δ`, `σ`, `π` or
`∩`. This project writes maths in its docstrings and its summary lines, so that
default has now crashed two runs:

  * §D: a `∩` in a print raised UnicodeEncodeError.
  * Today: `Δt` in `ablations.py`'s contrast labels crashed the time ablation in
    `report()` -- AFTER all seven arms had trained, discarding ~2 hours. The
    results survived only because each arm is written to a ledger as it
    completes.

Both times the response was to replace the character with ASCII. That treats the
symptom and loses the notation, and it has to be remembered forever by every
future edit. Reconfiguring the stream fixes the cause once.

`errors="backslashreplace"` rather than the default: if reconfiguration is ever
impossible, an odd-looking `\u0394` in a log is strictly better than a crash at
the end of a long run. Printing must never be able to fail.
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError, OSError):
        # Not a reconfigurable text stream (pytest capture, a pipe, a redirect
        # to a file object). Nothing to do, and nothing worth failing over.
        pass
