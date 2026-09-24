"""Patient timelines -> token sequences for the transformer.

Where the feature matrix throws order away and asks "how many times did X
happen", this keeps the order and asks the model to read the story.

Design decisions, each measured or derived
------------------------------------------
**One position per event, with the lab value fused into the token.**
``OBS_8480-6_Q7`` is one token meaning "systolic BP was measured and it was in
the 7th decile". The alternative -- ``OBS_8480-6`` then ``Q7`` as two positions
-- carries identical information and makes every sequence 37% longer. Measured
on this data: median 259 tokens fused vs 354 split, and 19% of patients
truncated at block 512 instead of 33%.

Fusion is applied only where there is volume to support it. It multiplies the
vocabulary -- measured, the numeric observation vocabulary goes 123 -> 974, a
**7.9x** increase -- so each embedding row sees roughly a tenth of the data.
What rescues it is the skew: 56 of the 113 numeric codes end up with fewer
than 50 events per fused token, but those 56 carry only **0.76% of numeric
events**. So fuse where the volume is and fall back to a shared ``Q0..Q9``
token, emitted as a separate position, where it is not.

**No gap tokens.** Inserting a token between events to mark elapsed time costs
19.5% of sequence length here, and attention is O(T^2), so that is 1.43x on the
attention term. The same information reaches the model for ~192 parameters and
zero length through the Δt attention bias. What this file emits instead is a
parallel ``dt`` array -- days before the anchor, per position -- which the model
turns into a continuous encoding.

**No rank positions.** Measured: knowing an event's index tells you only 13.6%
of what you would want to know about when it happened
(``I(rank; Δt) = 0.68`` of ``H(Δt) = 4.99`` bits). "The most recent event before
the cutoff" happened anywhere from 7 to 224 days ago. So position indices are
not emitted at all; the model's positional signal is a function of ``dt``.

**Within-timestamp order is arbitrary and we say so.** 97.7% of events share a
timestamp with at least one other. Sorting by ``(kind, code)`` is for
reproducibility only -- the model is made invariant to it by masking on
timestamp rather than on index, so shuffling within a group cannot change a
prediction beyond float32 reassociation noise (measured 3.0e-07).
``tests/test_model.py`` asserts that.

**Order and time come from ``ts_seq``, not ``ts``.** Conditions, careplans and
allergies carry a date with no time, so at midnight they sort *before* the
encounter that produced them -- measured, that was 81.4% of conditions, 100%
of them misplaced. ``ts_seq`` snaps those onto their same-day encounter.
Labels deliberately keep reading raw ``ts``; see ``cohort._seq_timestamp``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import load_cohort, load_events, load_obs_text
from .examples import ExampleConfig, load_examples

__all__ = ["SeqConfig", "Vocab", "SeqPack", "build_vocab", "build_sequences",
           "PAD", "BOS", "ANCHOR", "MASK"]

PAD, BOS, ANCHOR, MASK = "[PAD]", "[BOS]", "[ANCHOR]", "[MASK]"


@dataclass(frozen=True)
class SeqConfig:
    block_size: int = 512            # 91.3% of patients complete, 67.8% of all events
    min_patients_per_code: int = 5
    n_value_bins: int = 10
    fuse_min_events_per_bin: int = 50   # below this, fall back to a shared Q token
    vocab_post_anchor: bool = False
    """Count codes over training patients' WHOLE record, not only pre-anchor.

    Needed only by full-timeline pretraining (§E56.5): a code that first appears
    after the anchor is a next-event TARGET there, and would otherwise map to
    [UNK_kind]. Lab level edges stay fitted on pre-anchor values, which is what
    every model input sees. Off (default) reproduces the old vocabulary.
    """
    text_answers: bool = False
    """Fuse a text-valued observation's answer into its token.

    ``OBS_72166-2=Former smoker`` instead of ``OBS_72166-2`` for every (code,
    answer) pair seen in >= min_patients_per_code training patients; rarer
    answers keep the plain code. Off (default) reproduces the old vocabulary.
    """
    adaptive_bins: bool = False
    """Fuse EVERY numeric lab, with as many levels as its volume supports.

    Off (default): a lab gets 10 fused tokens only if it has >= 50 events per
    level; rarer labs emit the plain code plus a SHARED `Q0..Q9` token as a
    second position. At one layer nothing binds that shared level to its lab
    (§E56.6), so the value is effectively orphaned.
    On: each lab gets clamp(n_events // 50, 2, 10) levels, all fused, so every
    numeric event is one position and its level is never separated from it.
    """
    kinds: tuple[str, ...] = ("ENC", "COND", "MED", "PROC", "OBS", "IMM",
                              "CP", "ALG", "DEV", "IMG")


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    edges: dict[str, list[float]]     # code -> decile cut points, train-fitted
    fused: set[str]                   # codes whose value is fused into the token
    meta: dict = field(default_factory=dict)
    # REASONCODE -> id. 0 = no reason, 1 = unseen reason. Train-fitted like the
    # rest of the vocabulary; only read when use_reason_embedding is on.
    reasons: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.itos)

    def get(self, tok: str, kind: str) -> int:
        return self.stoi.get(tok, self.stoi[f"[UNK_{kind}]"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "itos": self.itos, "edges": self.edges,
            "fused": sorted(self.fused), "meta": self.meta}, indent=2))


@dataclass
class SeqPack:
    tokens: np.ndarray      # (n_examples, block_size) int32, right-padded with 0
    dt: np.ndarray          # (n_examples, block_size) float32, days before cutoff
    lengths: np.ndarray     # (n_examples,) int32
    eid: np.ndarray
    pid: np.ndarray         # NOT unique once augmented -- group by this for CV
    split: np.ndarray
    is_real: np.ndarray
    # Age at this example's cutoff, in days. One number per example: the model
    # derives age at every event as `age_days - dt`, so no sequence changes.
    age_days: np.ndarray | None = None
    # Reason id per position, parallel to `tokens` (0 = none / header / anchor).
    reasons: np.ndarray | None = None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SeqPack(tokens={self.tokens.shape}, "
                f"median_len={int(np.median(self.lengths))})")


def _decile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    """Interior cut points, de-duplicated.

    ``np.unique`` is load-bearing: several codes are effectively binary, and
    naive quantiles then produce repeated edges, which makes ``searchsorted``
    assign every value to one bucket.
    """
    q = np.nanquantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return [float(x) for x in np.unique(q)]


def build_vocab(root: Path | str = ".", cfg: SeqConfig | None = None,
                fit_pids: set[int] | None = None) -> Vocab:
    """Fitted on TRAIN patients, pre-anchor events only.

    Both restrictions are asserted rather than assumed. Fitting the vocabulary
    or the decile edges on val/test is the quietest leak available here, and
    EHRSHOT's reference implementation makes exactly that mistake.

    ``fit_pids`` narrows the fit set *within* train. This exists because the
    train-only restriction above is necessary but not sufficient: a caller that
    then carves an inner dev split out of train -- which every search config
    does -- has already let those dev patients shape the vocabulary they are
    about to be scored against. A code clears
    ``min_patients_per_code`` partly on their evidence, and a decile edge moves
    on their values. That is bug D17, and it is in the eight ledger entries
    that predate this parameter.

    The leak is membership-only, so it is small; it is also invisible to the
    shuffled-label test, because nothing here reads ``y``. Passing the inner
    training pids is what closes it. ``None`` reproduces the historical
    behaviour exactly, so every existing caller is unchanged.
    """
    cfg = cfg or SeqConfig()
    root = Path(root)
    cohort, events = load_cohort(root), load_events(root)

    allowed = set(cohort.loc[cohort.split == "train", "pid"])
    if fit_pids is None:
        # Default excludes the locked test set -- see cv.trainable_pids. The
        # vocabulary decides which codes clear `min_patients_per_code` and where
        # the decile edges fall, so building it on the locked patients is the
        # same class of leak as D17, one level up.
        from ..cv import trainable_pids
        train_pids = trainable_pids(str(root))
    else:
        train_pids = {int(x) for x in fit_pids}
        assert train_pids, "empty fit_pids"
        assert train_pids <= allowed, (
            f"fit_pids contains {len(train_pids - allowed)} non-train patients; "
            "the vocabulary may never see val or test")
    ev = events[events.is_pre_anchor & events.pid.isin(train_pids)
                & events.kind.isin(cfg.kinds)]
    assert len(ev) > 0 and ev.is_pre_anchor.all()

    _code_ev = ev
    if cfg.vocab_post_anchor:
        # Training patients only (train_pids), any time: see SeqConfig.
        _code_ev = events[events.pid.isin(train_pids) & events.kind.isin(cfg.kinds)]
    per_code = _code_ev.groupby("token", observed=True).pid.nunique()
    keep = sorted(per_code[per_code >= cfg.min_patients_per_code].index)

    # Which numeric codes have the volume to support ten separate tokens?
    num = ev[ev.is_numeric & ev.value.notna()]
    counts = num.groupby("token", observed=True).size()
    edges, fused = {}, set()
    for tok, n in counts.items():
        if cfg.adaptive_bins:
            _nb = int(min(cfg.n_value_bins, max(2, n // cfg.fuse_min_events_per_bin)))
            e = _decile_edges(num.loc[num.token == tok, "value"].to_numpy(), _nb)
            if len(e) < 1:
                continue                               # constant: no level at all
            edges[tok] = e
            fused.add(tok)
            continue
        e = _decile_edges(num.loc[num.token == tok, "value"].to_numpy(), cfg.n_value_bins)
        if len(e) < 2:
            continue                                   # near-constant: no value token
        edges[tok] = e
        if n / (len(e) + 1) >= cfg.fuse_min_events_per_bin:
            fused.add(tok)

    itos = [PAD, BOS, ANCHOR] + [f"[UNK_{k}]" for k in cfg.kinds]
    itos += [f"SEX_{v}" for v in ("M", "F")]
    for col, pfx in (("race", "RACE"), ("ethnicity", "ETH"), ("marital", "MARITAL")):
        itos += [f"{pfx}_{v}" for v in sorted(cohort[col].fillna("NA").unique())]
    itos += [f"Q{i}" for i in range(cfg.n_value_bins)]   # shared, for unfused codes
    for tok in keep:
        if tok in fused:
            itos += [f"{tok}_Q{i}" for i in range(len(edges[tok]) + 1)]
        else:
            itos.append(tok)

    # Text answers (flag): new rows for (code, answer) pairs, before [MASK].
    text_pairs = []
    if cfg.text_answers:
        ot = load_obs_text(root)
        ot = ot[ot.pid.isin(train_pids)].assign(ts=lambda d: d.ts.astype("datetime64[ns]"))
        # Pre-anchor and kind-filtered by construction: join onto `ev`.
        j = ev[["pid", "ts", "token"]].astype({"token": str}).assign(
            pid=lambda d: d.pid.astype("int64"),
            ts=lambda d: d.ts.astype("datetime64[ns]")).merge(ot, on=["pid", "ts", "token"])
        pp = (j.token + "=" + j.answer).groupby(j.pid).unique().explode()
        npat = pp.value_counts()
        text_pairs = sorted(npat[npat >= cfg.min_patients_per_code].index)
        itos += text_pairs

    # Appended LAST on purpose: inserting it near [PAD] would shift every
    # code's id and silently invalidate any cached tokenisation.
    itos.append(MASK)

    stoi = {t: i for i, t in enumerate(itos)}
    assert stoi[PAD] == 0, "PAD must be 0 so padding_idx=0 and masks are != 0"

    # Reasons: same fit set and the same >= min_patients rule as codes.
    _r = ev[ev.reason.notna()]
    _rp = _r.groupby(_r.reason.astype(str), observed=True).pid.nunique()
    reasons = {"[NO_REASON]": 0, "[UNK_REASON]": 1}
    for _code in sorted(_rp[_rp >= cfg.min_patients_per_code].index):
        reasons[_code] = len(reasons)

    return Vocab(stoi, itos, edges, fused, reasons=reasons, meta={
        "split": "train", "n_patients": len(train_pids),
        # Records whether the fit set was narrowed below the full train split,
        # so a vocabulary built for an inner-CV fold is distinguishable from
        # the default one rather than silently interchangeable with it.
        "fit_pids_given": fit_pids is not None,
        "n_codes_kept": len(keep), "n_fused_codes": len(fused),
        "min_patients_per_code": cfg.min_patients_per_code,
        "adaptive_bins": cfg.adaptive_bins,
        "vocab_post_anchor": cfg.vocab_post_anchor,
        "text_answers": cfg.text_answers, "n_text_pairs": len(text_pairs),
    })


def build_sequences(root: Path | str = ".", vocab: Vocab | None = None,
                    cfg: SeqConfig | None = None,
                    ex_cfg: ExampleConfig | None = None,
                    full_pids: set | None = None) -> SeqPack:
    """One sequence per **example**, truncated at that example's own cutoff.

    Built by slicing rather than re-filtering. Each patient's events are laid
    out once, sorted by ``ts_seq``; an example is then the prefix ending at
    ``searchsorted(cutoff)``, of which the most recent ``budget`` tokens are
    kept. A patient with ten cutoffs costs ten slices of one shared array, not
    ten passes over the event frame.
    """
    cfg = cfg or SeqConfig()
    root = Path(root)
    vocab = vocab or build_vocab(root, cfg)
    cohort, events = load_cohort(root), load_events(root)
    ex = load_examples(root, ex_cfg)

    if full_pids is not None:
        # FULL-TIMELINE pack, for pretraining only: one row per given patient,
        # cut one day after their last event, so every event -- before and after
        # the anchor -- is in view. The caller must pass TRAINING patients only;
        # train() asserts it. Never used as a model input for prediction.
        fp = sorted(int(p) for p in full_pids)
        ev_f = events[events.pid.isin(fp)]
        last = ev_f.groupby("pid", observed=True).ts_seq.max().reindex(fp)
        ex = pd.DataFrame({"eid": np.arange(len(fp)), "pid": fp,
                           "cutoff": (last + pd.Timedelta(days=1)).to_numpy(),
                           "split": "train", "is_real": True})
        pre = events[events.pid.isin(fp) & events.kind.isin(cfg.kinds)].copy()
    else:
        pre = events[events.is_pre_anchor & events.kind.isin(cfg.kinds)].copy()

    # Resolve each event to a token id, vectorised per group.
    tok = pre.token.astype(str).to_numpy()
    kind = pre.kind.astype(str).to_numpy()
    val = pre.value.to_numpy()
    isnum = pre.is_numeric.to_numpy()

    ids = np.empty(len(pre), np.int32)
    extra_q = np.full(len(pre), -1, np.int32)      # separate Q token, or -1
    for i in range(len(pre)):
        t = tok[i]
        if isnum[i] and t in vocab.edges and np.isfinite(val[i]):
            b = int(np.searchsorted(vocab.edges[t], val[i], side="right"))
            if t in vocab.fused:
                ids[i] = vocab.get(f"{t}_Q{b}", kind[i])
                continue
            ids[i] = vocab.get(t, kind[i])
            extra_q[i] = vocab.stoi[f"Q{min(b, cfg.n_value_bins - 1)}"]
        else:
            ids[i] = vocab.get(t, kind[i])

    if vocab.meta.get("text_answers"):
        ot = load_obs_text(root)
        # Both sides at ns: a parquet round trip can change datetime resolution,
        # and a resolution mismatch would match nothing (the assert below).
        ans = ot.assign(ts=ot.ts.astype("datetime64[ns]")).set_index(
            ["pid", "ts", "token"]).answer
        key = pd.MultiIndex.from_arrays([pre.pid.astype("int64").to_numpy(),
                                         pre.ts.astype("datetime64[ns]").to_numpy(),
                                         pre.token.astype(str).to_numpy()])
        a = ans.reindex(key).to_numpy()
        n_hit = 0
        for i in np.flatnonzero(pd.notna(a)):
            t2 = vocab.stoi.get(f"{tok[i]}={a[i]}")
            if t2 is not None:
                ids[i] = t2
                n_hit += 1
        # A silent zero would make B5 a copy of B0 that "shows nothing".
        assert n_hit > 0, "text_answers on but no answer token was emitted"
        print(f"  text answers fused into {n_hit:,} events", flush=True)
    if vocab.meta.get("adaptive_bins"):
        # The point of the flag: no numeric event may be split into two positions.
        assert int((extra_q >= 0).sum()) == 0, "adaptive_bins emitted a shared Q token"
    pre["tid"], pre["qid"] = ids, extra_q
    _rs = pre.reason.astype(object)
    _known = _rs.map(vocab.reasons)
    pre["rid"] = np.where(_rs.isna(), 0,
                          np.where(_known.isna(), 1, _known.fillna(0))).astype(np.int32)
    pre = pre.sort_values(["pid", "ts_seq", "ko", "token"])

    # Flatten the optional value token into its own position, at the SAME
    # instant, so a slice by timestamp can never separate a code from its value.
    tid_v, qid_v = pre.tid.to_numpy(), pre.qid.to_numpy()
    rid_v = pre.rid.to_numpy()
    ts_v, pid_v = pre.ts_seq.to_numpy(), pre.pid.to_numpy()
    has_q = qid_v >= 0
    order = np.argsort(np.concatenate([np.arange(len(pre)) * 2,
                                       np.flatnonzero(has_q) * 2 + 1]), kind="stable")
    flat_tok = np.concatenate([tid_v, qid_v[has_q]])[order]
    # A shared Q value token carries no reason of its own.
    flat_rid = np.concatenate([rid_v, np.zeros(int(has_q.sum()), np.int32)])[order]
    flat_ts = np.concatenate([ts_v, ts_v[has_q]])[order]
    flat_pid = np.concatenate([pid_v, pid_v[has_q]])[order]

    # Per-patient contiguous blocks, so an example is a slice.
    uniq = np.unique(flat_pid)
    starts = np.searchsorted(flat_pid, uniq, side="left")
    ends = np.searchsorted(flat_pid, uniq, side="right")
    span = {int(p): (int(a), int(b)) for p, a, b in zip(uniq, starts, ends)}

    n, L = len(ex), cfg.block_size
    T = np.zeros((n, L), np.int32)
    D = np.zeros((n, L), np.float32)
    lens = np.zeros(n, np.int32)
    age = np.zeros(n, np.float32)
    RS = np.zeros((n, L), np.int32)

    demo_cols = [("gender", "SEX"), ("race", "RACE"),
                 ("ethnicity", "ETH"), ("marital", "MARITAL")]
    pat = cohort.set_index("pid")
    anchor_tok = vocab.stoi[ANCHOR]

    for row in ex.itertuples():
        pid, cut = int(row.pid), np.datetime64(row.cutoff)
        d = pat.loc[pid]
        head = [vocab.stoi[BOS]]
        for col, pfx in demo_cols:
            v = d[col]
            head.append(vocab.stoi.get(f"{pfx}_{'NA' if pd.isna(v) else v}",
                                       vocab.stoi[BOS]))

        a, b = span.get(pid, (0, 0))
        stop = a + int(np.searchsorted(flat_ts[a:b], cut, side="left"))
        budget = L - len(head) - 1
        lo = max(a, stop - budget)             # keep the MOST RECENT events

        body_t = flat_tok[lo:stop]
        body_d = ((cut - flat_ts[lo:stop]) / np.timedelta64(1, "s") / 86400.0)

        k = len(head) + len(body_t) + 1
        age[row.eid] = (cut - np.datetime64(d["birthdate"])) / np.timedelta64(1, "D")
        T[row.eid, :len(head)] = head
        T[row.eid, len(head):k - 1] = body_t
        RS[row.eid, len(head):k - 1] = flat_rid[lo:stop]
        T[row.eid, k - 1] = anchor_tok
        first_dt = float(body_d[0]) if len(body_d) else 0.0
        D[row.eid, :len(head)] = first_dt
        D[row.eid, len(head):k - 1] = body_d
        D[row.eid, k - 1] = 0.0
        lens[row.eid] = k

    return SeqPack(tokens=T, dt=D, lengths=lens,
                   eid=ex.eid.to_numpy(), pid=ex.pid.to_numpy(),
                   split=ex.split.to_numpy().astype("U5"),
                   is_real=ex.is_real.to_numpy(), age_days=age, reasons=RS)
