"""Printing must never be able to fail, whatever is in the string.

This bug class has bitten twice. §D records a `∩` in a print raising
UnicodeEncodeError; today `Δt` in `ablations.py`'s contrast labels crashed the
time ablation inside `report()` -- AFTER all seven arms had trained, discarding
roughly two hours. The arms survived only because each is written to a ledger as
it completes.

It is invisible in review: the character renders fine in an editor, the module
imports, the tests pass, and it raises only when the line is actually printed --
which for a summary is the last thing a long run does.

Both previous fixes replaced the character with ASCII. That treats the symptom,
loses the notation, and has to be remembered by every future edit. The cause is
that the Windows console is cp1252, so `src/__init__` reconfigures the streams to
UTF-8 once. These tests pin that, and they check the STREAM rather than auditing
source files -- an earlier version of this test audited the files and failed on
docstrings and a matplotlib axis label that were never going to reach stdout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD = "\u0394 \u03c3 \u03c0 \u2229 \u2192 \u03c9"      # delta sigma pi cap arrow omega


def _run(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, encoding="utf-8", cwd=str(ROOT))


def test_importing_the_package_makes_these_characters_printable():
    r = _run(f"import src; print({HARD!r})")
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip() == HARD


def test_the_probe_is_not_vacuous():
    """Without the package import the same print MUST fail, or this suite is
    asserting something the platform already guaranteed."""
    r = _run(f"print({HARD!r})")
    assert r.returncode != 0 and "UnicodeEncodeError" in r.stderr, (
        "cp1252 console assumed, but the bare print succeeded -- if the default "
        "encoding has changed, this guard is no longer testing anything")


def test_a_module_that_prints_maths_runs_under_the_console_encoding():
    """End to end: the module whose summary line actually crashed."""
    r = _run("import src.ablations as m; print(m.ARMS.keys())")
    assert r.returncode == 0, r.stderr[-400:]
