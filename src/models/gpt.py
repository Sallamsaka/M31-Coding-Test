"""A small transformer for patient timelines.

Sized after Delphi-2M (Nature 2025): ~2M parameters, on disease timelines, with
the positional-embedding line deleted and age sin/cos in its place. Our
architecture is the same shape for the same reasons.

Three departures from a textbook decoder, each measured
-------------------------------------------------------
**No positional lookup table.** Position indices are replaced by a continuous
function of time-before-anchor. Measured on this data, an event's index tells
you only 13.6% of what you would want to know about when it happened
(I(rank; Δt) = 0.68 bits of H(Δt) = 4.99). "The most recent event" happened
anywhere from 7 to 224 days ago.

**Concatenate the time encoding, never add it.** Expanding the attention logit
for ``x = w + π`` gives ``wMw + wMπ + πMw + πMπ`` -- two cross terms comparing a
content vector to a time vector through one shared projection. CEHR-BERT ran
the controlled test: summing time/age was *worse than injecting no time at all*
in 7 of 8 task-metric cells, and with triple the variance.

**Mask on timestamp, not on index.** A strict causal mask is triangular
*within* a group of simultaneous events, which imposes an order that does not
exist -- and 97.7% of events here share a timestamp with at least one other,
once date-only diagnoses are snapped onto the encounter that produced them.
Masking on ``dt`` instead makes the mask block-constant within each timestamp,
so permuting simultaneous events cannot change any output. The invariance is
architectural rather than learned, which at 2,791 patients is the difference
between getting it free and paying data for it.

Measured, a permutation moves the logits by at most **3.0e-07** -- not zero,
because reordering changes the summation order inside attention and float32
addition is not associative. That is `eps * sqrt(T)` accumulation, not a
broken invariance, and the test asserts the tolerance rather than equality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GPTConfig", "TimeEncoding", "PatientTransformer"]


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192
    d_time: int = 64             # width of the continuous time encoding
    n_dt_buckets: int = 10       # log-spaced buckets for the pairwise bias
    # Was 32, copied from T5 including the count. T5's 32 comes from a
    # 10^11-token regime; here it means n_head x 32 free scalars fitted on
    # 2,791 patients. Measured occupancy over 150 sampled patients: eight
    # buckets spanning 0.4-23 days hold ~2% of all attention pairs between
    # them, while ~70% of the mass sits in the 140-20,000 day range. The
    # resolution was in the wrong place, not merely excessive.
    max_dt_days: float = 40000.0 # top bucket edge; measured max Δt is 37,284
    n_outputs: int = 40          # the 40 scored conditions
    # Was 41, documented as '40 + 5-year death'. It was never true:
    # train_finetune builds every model with n_outputs=y.shape[1], and the
    # label array has 40 columns, so the 41st head was never constructed
    # and death has never been predicted. Adding it is a live idea (it
    # would make A6's 'predicting record-end' claim testable) but it needs
    # a 41st label column, which does not exist yet.
    attn_dropout: float = 0.1    # DropKey-style; measured 5x larger gain at small n
    resid_dropout: float = 0.1
    block_size: int = 512

    fusion: str = "none"
    """``none`` | ``readout`` | ``token`` -- how engineered features enter.

    Late (``readout``) fusion concatenates a projection of the feature vector
    to the pooled sequence summary: a jointly-trained ensemble, where the two
    views never interact. The ``token`` form prepends the projection as an
    extra position at ``dt = 0``, so attention can mix feature evidence with
    sequence evidence at every layer. Measured here, the sequence model and
    the feature model disagree about which patients are at risk (Spearman
    0.436 against 0.740 for two feature models), so how they are combined is
    a real question rather than a formality.
    """
    fusion_dim: int = 64
    n_features: int = 0

    modality_dropout: float = 0.0
    """Probability of zeroing the WHOLE feature vector for an example, in training.

    Not the same thing as `resid_dropout` inside `feat_proj`, which drops
    individual units and leaves the tabular branch informative. This drops the
    branch entirely for a random subset of examples per step, so the sequence
    branch has to be able to carry the prediction alone.

    Two reasons it earns a switch here. §E16 named "auxiliary sequence-only
    loss, modality dropout" as the fixes if the collapse check had failed, and
    implemented neither. And §E25.2 measured the model as OVERFITTING rather
    than under-trained, which makes a regulariser the right class of
    intervention -- this one happening to also attack the fact that
    `Linear(3296, 64)` is 211k parameters against a ~123k-parameter trunk.

    0.0 reproduces the previous behaviour exactly, so it is inert until set.
    """

    use_time_encoding: bool = True
    """Ablation switch: include the per-token Time2Vec encoding at all.

    False zeroes the time half of the token/time concatenation, leaving the
    token embedding and the linear mix intact so the shapes and the parameter
    count are unchanged. What remains is a model that knows WHICH codes occurred
    and, through the visibility mask, their order -- but not how far apart they
    are.
    """

    use_dt_bias: bool = True
    """Ablation switch: include the pairwise Δt attention bias.

    False replaces the learned per-(head, bucket) scalars with zeros. The
    visibility mask is untouched, so causality is preserved and only the
    *graded* notion of temporal distance is removed.
    """

    causal: bool = True
    """Causal (each position sees only the past) or bidirectional.

    Not an independent knob. The mask decides which pretraining objective can
    run at all -- next-token prediction is trivially solved under a
    bidirectional mask, and masked-token prediction is pointless under a
    causal one -- and it also decides the readout. See `readout`.
    """

    readout: str = "last"
    """``"last"`` reads the final ``[ANCHOR]`` position, ``"mean"`` pools.

    Determined by `causal`, not chosen separately. Under a causal mask the
    final position is the ONLY one that has seen the whole history, so
    mean-pooling would dilute it with vectors built from prefixes. Under a
    bidirectional mask every position has seen everything, ``[ANCHOR]`` holds
    no privileged view and carries no content of its own, so pooling T
    equally-informed vectors is free variance reduction.

    CORE-BEHRT found CLS-style readout worst -- on a bidirectional model,
    where that is the expected result. Read as evidence against ``[ANCHOR]``
    in general it would be a misreading.
    """


class TimeEncoding(nn.Module):
    """Time2Vec over ``log(1 + days_before_anchor)``.

    Two details that matter and are easy to get wrong:

    *Log-compress first.* Δt spans 0 to ~29,200 days -- 4.5 orders of
    magnitude. A sinusoidal ladder with the usual base 10^4 covers only 4, so
    the coarse end saturates. ``u = log(1+Δt)`` collapses the range to [0, 10.3]
    and matches the clinical intuition that 1→2 days is comparable in
    significance to 1→2 years.

    *Keep a non-periodic term.* ``φ[0] = ω₀u + α₀`` is a straight line, so the
    encoding stays monotone in time. A purely periodic basis wraps, and two very
    different Δt can collide. This is what Time2Vec adds over fixed sinusoids
    and it is why we use it rather than Delphi-2M's fixed ladder.

    Cost: ``2 * d_time`` learnable numbers -- 128 at d_time=64.
    """

    def __init__(self, d_time: int):
        super().__init__()
        assert d_time >= 2
        # Spread initial frequencies over the log-day range so some components
        # resolve days and others resolve decades.
        # Band chosen so every component is periodic over the OBSERVED range of
        # u = log(1+dt), which spans [0, 10.28].
        #
        # Floor 0.6: one full cycle needs omega = 2*pi/10.28 = 0.611. The old
        # floor of 0.05 gave a period of 125.7 in u-units, so 38 of 63
        # components completed less than one cycle across the entire dataset --
        # quasi-linear at init, and therefore redundant with the explicit
        # linear term on the next line. 60% of the basis was doing nothing the
        # linear term was not already doing.
        #
        # Ceiling 8.0: the log transform sets a resolution limit. At the recent
        # end u(2)-u(1) = 0.405, so the finest resolvable frequency is about
        # pi/0.405 = 7.8; past that, adjacent days alias. The old ceiling of 3.0
        # left the top of the expressible band unused.
        #
        # Tancik et al. (2020) is the established mechanism: performance vs the
        # frequency scale is a broad plateau with sharp cliffs at both ends, and
        # the scale matters far more than the number of features -- so this band
        # is a hyperparameter to sweep, not a constant to trust.
        w = torch.exp(torch.linspace(math.log(0.6), math.log(8.0), d_time - 1))
        self.w = nn.Parameter(w)
        self.a = nn.Parameter(torch.rand(d_time - 1) * 2 * math.pi)
        self.w0 = nn.Parameter(torch.tensor(0.3))
        self.a0 = nn.Parameter(torch.tensor(-1.0))

    def forward(self, dt_days: torch.Tensor) -> torch.Tensor:
        u = torch.log1p(dt_days.clamp(min=0.0)).unsqueeze(-1)   # (B,T,1)
        periodic = torch.sin(u * self.w + self.a)               # (B,T,d-1)
        linear = self.w0 * u + self.a0                          # (B,T,1)
        return torch.cat([linear, periodic], dim=-1)


class Block(nn.Module):
    """Pre-LayerNorm block with a pairwise Δt bias added to the attention logits."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.resid_dropout),
        )
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))   # (B,nh,T,T)
        att = att + bias                                          # Δt bias + mask
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        x = x + self.resid_drop(self.proj(y))
        return x + self.mlp(self.ln2(x))


class PatientTransformer(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd, padding_idx=0)
        self.time = TimeEncoding(cfg.d_time)
        # Concatenate-then-project. See module docstring for why not summing.
        self.mix = nn.Linear(cfg.n_embd + cfg.d_time, cfg.n_embd, bias=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.n_outputs)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight          # tied

        # 32 log-spaced Δt buckets per head. Bucket 0 is EXACTLY Δt == 0, kept
        # separate so "same instant" is distinguishable from "a minute apart":
        # the smallest non-zero gap measured here is 0.00064 days -- 55
        # seconds -- and 7.6% of consecutive gaps fall inside bucket 1 alone.
        #
        # The top edge must exceed the largest Δt in the data (measured
        # 37,284 days) or the longest histories all collapse into one bucket.
        self.feat_proj = None
        if cfg.fusion != "none" and cfg.n_features > 0:
            self.feat_proj = nn.Sequential(
                nn.Linear(cfg.n_features, cfg.fusion_dim),
                nn.GELU(),
                nn.Dropout(cfg.resid_dropout),
            )
            if cfg.fusion == "readout":
                self.head = nn.Linear(cfg.n_embd + cfg.fusion_dim, cfg.n_outputs)
            else:                                    # token: lift to model width
                self.feat_to_tok = nn.Linear(cfg.fusion_dim, cfg.n_embd)

        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_head, cfg.n_dt_buckets))
        self.register_buffer(
            "bucket_edges",
            torch.expm1(torch.linspace(0.0, math.log1p(cfg.max_dt_days),
                                       cfg.n_dt_buckets - 1)),
            persistent=False)

        self.apply(self._init)
        for n, p in self.named_parameters():           # nanoGPT scaled residual init
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _attn_bias(self, dt: torch.Tensor, pad: torch.Tensor,
                   strict: bool, causal: bool | None = None) -> torch.Tensor:
        """Pairwise Δt bias combined with the visibility mask.

        ``dt`` counts *down* to the anchor, so a larger value is further in the
        past. Position ``i`` may attend to ``j`` when ``dt[j] >= dt[i]`` -- j is
        at the same instant or earlier.

        ``strict=True`` (``dt[j] > dt[i]``) hides everything at the current
        instant, which is what causal pretraining needs: with 97.7%
        simultaneity, "predict the next token" would otherwise mostly mean
        "predict which of 20 simultaneous lab results I happened to list
        next", i.e. predicting our own tie-break. Delphi-2M masks the same way.

        ``causal=False`` drops the direction constraint entirely -- every real
        position sees every other. The Δt bias is untouched, so the model
        still knows how far apart two events are; it simply no longer has to
        pretend it cannot look forward. Used by the bidirectional arms, where
        the pretraining objective is masked-token rather than next-token.

        Every form is constant within a timestamp block, so permuting
        simultaneous events cannot change any output.
        """
        B, T = dt.shape
        gap = (dt.unsqueeze(2) - dt.unsqueeze(1)).abs()               # (B,T,T)
        idx = torch.bucketize(gap, self.bucket_edges)                 # 0 iff gap==0
        if self.cfg.use_dt_bias:
            bias = self.dt_bias[:, idx.clamp(max=self.cfg.n_dt_buckets - 1)]
            bias = bias.permute(1, 0, 2, 3)                           # (B,nh,T,T)
        else:
            # Ablated: no graded temporal distance. The visibility mask below is
            # untouched, so this removes "how far apart" without removing
            # "which came first".
            bias = torch.zeros(B, self.cfg.n_head, T, T, device=dt.device)

        causal = self.cfg.causal if causal is None else causal
        if not causal:
            # Bidirectional: every real position sees every other. The Δt bias
            # still applies, so timing information is unchanged -- only the
            # direction constraint is dropped.
            visible = torch.ones(B, T, T, dtype=torch.bool, device=dt.device)
        elif strict:
            visible = dt.unsqueeze(1) > dt.unsqueeze(2)
        else:
            visible = dt.unsqueeze(1) >= dt.unsqueeze(2)
        visible = visible & pad.unsqueeze(1) & pad.unsqueeze(2)
        visible = visible | torch.eye(T, dtype=torch.bool, device=dt.device)
        return bias.masked_fill(~visible.unsqueeze(1), float("-inf"))

    def encode(self, tokens: torch.Tensor, dt: torch.Tensor,
               strict: bool = False, causal: bool | None = None,
               inject: tuple[int, torch.Tensor] | None = None) -> torch.Tensor:
        pad = tokens != 0
        t = self.time(dt)
        if not self.cfg.use_time_encoding:
            t = torch.zeros_like(t)     # keep shapes and parameter count fixed
        x = self.mix(torch.cat([self.tok(tokens), t], dim=-1))
        if inject is not None:                # replace a position's embedding
            pos, vec = inject
            x = torch.cat([vec, x[:, pos + 1:]], dim=1) if pos == 0 else x
        bias = self._attn_bias(dt, pad, strict, causal)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor, dt: torch.Tensor,
                lengths: torch.Tensor,
                features: torch.Tensor | None = None) -> torch.Tensor:
        """Logits per example, from whichever readout the mask implies."""
        # Applied once, HERE, rather than inside each fusion branch: both
        # `readout` and `token` consume `features` downstream, so dropping it at
        # the entry point guarantees the two modes get identical treatment. A
        # per-branch implementation would be two chances to make them differ.
        #
        # Training only -- `self.training` is what makes evaluation
        # deterministic, and the mask is per-EXAMPLE (shape (B, 1)) not per-unit,
        # which is the whole distinction from `resid_dropout`.
        #
        # Deliberately NOT rescaled by 1/(1-p). Standard inverted dropout keeps
        # the expected input to the next layer constant, but here the point is
        # that the sequence branch must cope with the feature branch being
        # genuinely absent -- which is exactly the condition at `features=0` in
        # the collapse check. Rescaling would train it on a reweighted vector it
        # never sees at evaluation.
        if (self.training and self.cfg.modality_dropout > 0
                and features is not None):
            keep = (torch.rand(features.shape[0], 1, device=features.device)
                    >= self.cfg.modality_dropout).to(features.dtype)
            features = features * keep
        if self.cfg.fusion == "token" and features is not None:
            # The feature token must sit at the OLDEST instant, not dt = 0.
            #
            # dt counts down to the anchor, so dt = 0 is the most recent
            # moment, and under the causal rule (attend to j iff dt[j] >=
            # dt[i]) nothing can look forward to it. Placed there the token
            # was invisible to every other position and to the readout:
            # measured, changing the feature vector moved the output by
            # exactly 0.0000. Anchoring it at each row's maximum dt makes
            # dt[token] >= dt[i] true for every i, so it behaves as a prefix
            # that the whole sequence can attend to -- which is what a
            # summary of the entire history should be.
            f = self.feat_to_tok(self.feat_proj(features)).unsqueeze(1)
            extra_tok = torch.full((len(tokens), 1), 1, dtype=tokens.dtype,
                                   device=tokens.device)   # non-pad placeholder
            oldest = dt.max(dim=1, keepdim=True).values
            tokens = torch.cat([extra_tok, tokens], dim=1)
            dt = torch.cat([oldest, dt], dim=1)
            lengths = lengths + 1
            h = self.encode(tokens, dt, strict=False, inject=(0, f))
        else:
            h = self.encode(tokens, dt, strict=False)

        if self.cfg.readout == "mean":
            m = (tokens != 0).unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = h[torch.arange(len(h), device=h.device), lengths - 1]

        if self.cfg.fusion == "readout" and features is not None:
            pooled = torch.cat([pooled, self.feat_proj(features)], dim=-1)
        return self.head(pooled)

    def lm_forward(self, tokens: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Next-token logits for pretraining, under the strictly-earlier mask.

        Forced causal regardless of ``cfg.causal``: with a bidirectional mask
        the target token is in the input and the objective is trivial. The
        bidirectional arms pretrain with :meth:`mlm_forward` instead.
        """
        return self.lm_head(self.encode(tokens, dt, strict=True, causal=True))

    def mlm_forward(self, tokens: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Masked-token logits, the bidirectional pretraining objective.

        Caller replaces a sample of positions with ``[MASK]`` and scores only
        those. Forced bidirectional: reconstructing a hidden token from one
        side only is next-token prediction with fewer signals per pass.
        """
        return self.lm_head(self.encode(tokens, dt, strict=False, causal=False))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
