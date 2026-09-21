"""The single source of truth for every time operation in this project.

No other module may call ``pd.to_datetime`` directly -- ``tests/test_time.py``
greps for it. Every timestamp in the pipeline passes through :func:`to_ts`.

The two conventions everything else depends on
----------------------------------------------
::

    FEATURE_WINDOW  =  ts <  anchor                     right-OPEN at the anchor
    LABEL_WINDOW    =  anchor <= first_dx <= anchor+5y   BOTH edges closed

Both were settled by measurement rather than taste, and both are load-bearing:

* **Feature edge exclusive.** Across all 358 provided test patients, zero events
  fall at or after their given anchor, while 11 patients have events in the
  preceding 24 hours at hours 2 through 21. Had the organisers cut at an
  intra-day timestamp, several of those would have retained an early-morning
  event on the anchor date. They did not, so the cut is midnight.
* **Label right edge inclusive.** 119 first-diagnoses land exactly on
  ``anchor + 5y`` and zero land beyond it. This is structural, not luck:
  ``anchor + 5y`` is by construction the day of the last encounter, and
  conditions carry a date with no time, so a diagnosis recorded at that final
  visit lands precisely on the boundary. An exclusive edge silently discards
  2% of all positives.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "FEATURE_WINDOW",
    "LABEL_WINDOW",
    "LABEL_RIGHT_CLOSED",
    "to_ts",
    "anchor_from_encounters",
    "label_window_end",
    "is_pre_anchor",
]

FEATURE_WINDOW = "ts < anchor"
LABEL_WINDOW = "anchor <= first_dx <= anchor + 5y"
LABEL_RIGHT_CLOSED = True


def to_ts(s: pd.Series) -> pd.Series:
    """Parse any Synthea time column to tz-naive ``datetime64[ns]``.

    ``format="ISO8601"`` handles both shapes present in the data -- the
    date-only columns (``conditions``, ``careplans``, ``allergies``) and the
    ``...Z`` stamps everywhere else -- in one pass, and *raises* on anything
    else rather than silently falling back to object dtype, which is the quiet
    failure mode of a bare ``to_datetime`` on mixed formats.

    ``utc=True`` then ``tz_localize(None)`` puts every column on one tz-naive
    number line. The alternative -- staying tz-aware -- risks an object-dtype
    comparison that returns elementwise ``False`` without raising, which is the
    single worst failure available here.
    """
    return pd.to_datetime(s, format="ISO8601", utc=True).dt.tz_localize(None)


def anchor_from_encounters(enc: pd.DataFrame, years: int = 5) -> pd.Series:
    """``anchor = normalize(max over rows of (STOP else START)) - years``.

    Three details, each of which would be a silent bug if wrong:

    1. The ``STOP``/``START`` fallback is **per row**, then the max is taken --
       matching the ReadMe's "the latest STOP, or START if STOP is blank".
       ``train_val`` has zero blank STOPs so both readings agree there, but the
       test split has five.
    2. ``DateOffset(years=...)`` is calendar arithmetic and clamps 2016-02-29 to
       2011-02-28. ``Timedelta(days=1826)`` would shift every anchor by one or
       two days depending on how many leap years the span crosses, which flips
       the 119 boundary diagnoses.
    3. ``.normalize()`` floors to midnight -- see the module docstring.

    Parameters
    ----------
    enc
        Encounters with ``PATIENT`` plus already-parsed ``START``/``STOP``.

    Returns
    -------
    Series indexed by ``PATIENT``.
    """
    for col in ("PATIENT", "START", "STOP"):
        if col not in enc.columns:
            raise KeyError(f"anchor_from_encounters needs column {col!r}")
    last_per_row = enc["STOP"].where(enc["STOP"].notna(), enc["START"])
    last = last_per_row.groupby(enc["PATIENT"]).max()
    return (last - pd.DateOffset(years=years)).dt.normalize()


def label_window_end(anchor: pd.Series, years: int = 5) -> pd.Series:
    """``anchor + years``, calendar-correct. The window is closed at this value."""
    return anchor + pd.DateOffset(years=years)


def is_pre_anchor(ts: pd.Series, anchor: pd.Series) -> pd.Series:
    """``ts < anchor`` -- the one place the feature-window comparison is written.

    Stamped onto the unified event frame exactly once so no downstream module
    re-implements it and drifts.
    """
    return ts < anchor
