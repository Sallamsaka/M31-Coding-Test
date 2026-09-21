"""How many patients each label would need, against how many it has.

§T calls this the highest-value item in the triage, and the reason is that it
converts a disappointing null into a predicted one. §C2 reported that all the
added feature blocks sit "inside the noise floor" and read that as a failure to
find signal. The Riley criteria say it is what a sample this size is *expected*
to produce: with 29 positives, the rarest label cannot support a non-trivial
predictor set at all, whatever the features are.

**The criterion.** Riley, Snell, Ensor, Burke, Harrell, Moons & Collins (Stat
Med 2019; BMJ 2020) require the expected uniform shrinkage factor S to be at
least 0.9 -- i.e. a model whose coefficients need shrinking by more than 10% is
overfitted by construction. For a binary outcome,

    S  =  1  +  P / (n * ln(1 - R2_cs / S))

so, solving for the number of parameters a given n supports,

    P  =  n * (S - 1) * ln(1 - R2_cs / S)

Both terms are negative at S = 0.9, so P is positive.

**R2_cs from the C-statistic.** Riley et al. (Part II, Stat Med 2021) obtain the
anticipated Cox-Snell R2 by simulation, because no closed form exists. Same
approach here, under the standard assumption that the linear predictor is
normal: if LP ~ N(mu, sigma^2) then

    C = Phi(sigma / sqrt(2))      =>   sigma = sqrt(2) * Phi^-1(C)

and mu is solved so that E[expit(LP)] equals the outcome prevalence. R2_cs then
follows from the expected log-likelihoods of the fitted and null models. Stated
explicitly because it is an assumption, not a measurement: a linear predictor
that is skewed or bimodal gives a different R2 at the same C.

**What this is not.** It is a statement about *this* estimand at *this*
prevalence, and it says nothing about which features are good. It bounds how
many parameters the data can support before shrinkage exceeds 10%; we feed 3,320.

Run: ``python -m src.riley``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

S_TARGET = 0.9          # Riley's criterion (i): <=10% expected shrinkage
N_SIM = 200_000


def r2_cs_from_c(c_stat: float, prevalence: float, rng=None) -> float:
    """Anticipated Cox-Snell R^2 for a model with this C and this prevalence.

    Normal-linear-predictor assumption; see the module docstring.
    """
    if not (0.5 < c_stat < 1.0) or not (0.0 < prevalence < 1.0):
        return float("nan")
    rng = rng or np.random.default_rng(0)
    sigma = np.sqrt(2.0) * norm.ppf(c_stat)
    z = rng.standard_normal(N_SIM)

    # Centre the linear predictor so the implied event rate matches prevalence.
    def gap(mu):
        return expit(mu + sigma * z).mean() - prevalence
    lo, hi = -50.0, 50.0
    if gap(lo) * gap(hi) > 0:
        return float("nan")
    mu = brentq(gap, lo, hi, xtol=1e-10)

    p = expit(mu + sigma * z)
    # Expected log-likelihood per observation, model vs null. The outcome is
    # Bernoulli(p), so E[loglik] = p*log(p) + (1-p)*log(1-p) under the model.
    ll_model = np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
    q = prevalence
    ll_null = q * np.log(q) + (1 - q) * np.log(1 - q)
    # R2_cs = 1 - exp(-2 * (LL_model - LL_null) / n), per-observation form.
    return float(1.0 - np.exp(-2.0 * (ll_model - ll_null)))


def max_parameters(n: int, r2_cs: float, s: float = S_TARGET) -> float:
    """Parameters supportable at <=10% expected shrinkage."""
    if not np.isfinite(r2_cs) or r2_cs <= 0 or r2_cs >= s:
        return float("nan")
    return n * (s - 1.0) * np.log(1.0 - r2_cs / s)


def shrinkage(n: int, n_params: int, r2_cs: float, s0: float = S_TARGET) -> float:
    """Expected shrinkage S implied by (n, P, R2). Inverse of `max_parameters`."""
    if not np.isfinite(r2_cs) or r2_cs <= 0:
        return float("nan")

    def f(s):
        if s <= r2_cs:
            return 1e6
        return 1.0 + n_params / (n * np.log(1.0 - r2_cs / s)) - s
    try:
        return float(brentq(f, max(r2_cs + 1e-9, 1e-6), 0.999999, xtol=1e-12))
    except ValueError:
        return float("nan")


def required_n(n_params: int, r2_cs: float, s: float = S_TARGET) -> float:
    """Patients needed to support `n_params` at <=10% shrinkage."""
    if not np.isfinite(r2_cs) or r2_cs <= 0 or r2_cs >= s:
        return float("nan")
    return n_params / ((s - 1.0) * np.log(1.0 - r2_cs / s))


def main(root: str = ".", n_train: int = 2433, n_params: int = 3320) -> None:
    """`n_train` defaults to the CV pool; `n_params` to the LR design matrix."""
    t = pd.read_csv(f"{root}/outputs/per_code_val.csv")
    rng = np.random.default_rng(0)

    rows = []
    for r in t.itertuples():
        c = float(r.auroc_at_risk)
        prev = float(r.prevalence)
        # Scale the at-risk denominator from the 365-patient val set up to the
        # training pool; prevalence is the transferable quantity, not the count.
        n_at_risk_train = n_train * (float(r.n_at_risk) / t.n_at_risk.max()) \
            if t.n_at_risk.max() > 0 else n_train
        r2 = r2_cs_from_c(c, prev, rng)
        rows.append({
            "code": r.code, "description": r.description[:38],
            "prevalence": prev, "c_stat": c,
            "events_train_est": int(round(prev * n_at_risk_train)),
            "r2_cs": r2,
            "max_params": max_parameters(int(n_at_risk_train), r2),
            "required_n_for_3320": required_n(n_params, r2),
        })
    df = pd.DataFrame(rows).sort_values("prevalence")
    df.to_csv(f"{root}/outputs/riley_table.csv", index=False)

    ok = df.dropna(subset=["max_params"])
    print(f"Riley criterion (i): expected shrinkage <= 10%, S = {S_TARGET}")
    print(f"assumed development set n = {n_train:,} (CV pool), "
          f"candidate parameters fed = {n_params:,}\n")
    print(f"{'prev':>7}{'C':>7}{'events':>8}{'R2_cs':>8}{'max params':>12}"
          f"{'need n for 3,320':>18}")
    for r in ok.head(8).itertuples():
        print(f"{r.prevalence:>7.3f}{r.c_stat:>7.3f}{r.events_train_est:>8d}"
              f"{r.r2_cs:>8.4f}{r.max_params:>12.1f}{r.required_n_for_3320:>18,.0f}")
    print("   ... (full table in outputs/riley_table.csv)")
    for r in ok.tail(3).itertuples():
        print(f"{r.prevalence:>7.3f}{r.c_stat:>7.3f}{r.events_train_est:>8d}"
              f"{r.r2_cs:>8.4f}{r.max_params:>12.1f}{r.required_n_for_3320:>18,.0f}")

    med = ok.max_params.median()
    print(f"\n  median supportable parameters across {len(ok)} scorable labels: "
          f"{med:.1f}")
    print(f"  we feed {n_params:,} -- a factor of {n_params/med:,.0f} over the median.")
    starved = int((ok.max_params < 10).sum())
    print(f"  labels supporting fewer than 10 parameters: {starved}/{len(ok)}")
    print("\n  This does not say the features are bad. It says the sample cannot")
    print("  support this many of them, which is what C2's null was measuring.")


if __name__ == "__main__":
    main()
