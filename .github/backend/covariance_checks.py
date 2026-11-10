"""Turn a covariance matrix into standard errors, and say so when it cannot.

WHY THIS IS A MODULE AND NOT A ONE-LINER. Six places in this repository
computed standard errors as

    std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))

and that clamp is right for a negative of round-off size and catastrophically
wrong for anything larger. A covariance diagonal is a variance: it cannot be
negative, so a negative one means the matrix is not positive semi-definite and
the standard error does not exist. Reporting it as 0.0 reports PERFECT
PRECISION from a failed computation, which is the worst available answer for a
quantity whose whole job is to say how much to trust a number.

That is not hypothetical. In geometry_uncertainty, measured on water with the
relative sigma floor off, the covariance reached a condition number of 1.2e32
with a minimum eigenvalue of -2.0e13, and the old clamp reported sigma =
0.000000. It also made the behaviour platform dependent: on another LAPACK the
round-off landed positive and produced a plausible 0.0119 instead, so a test
pinned to either value passed on one machine and failed on another with neither
number meaning anything.

The remaining call sites are less exposed -- most build a Gram matrix
(J^T W J + reg), which is positive semi-definite in exact arithmetic, so a
negative diagonal there needs ill-conditioning rather than arriving by
construction. No trigger has been demonstrated for those. They are routed
through here anyway, because the cost of being wrong is a confidently quoted
zero and the cost of this check is one comparison per parameter.

THE GOOD PATTERN ALREADY IN THE REPO. torsion_uncertainty builds its
covariance from an eigendecomposition, masks eigenvalues at or below a rank
threshold, and warns that the result is a pseudo-inverse in the deficient
directions. Its clamp really is round-off only. Where a caller can do that,
it is better than this: it fixes the matrix instead of reporting on it.
"""

from __future__ import annotations

import numpy as np

#: How negative a variance may be, relative to the largest variance on the
#: diagonal, before it is treated as a failure rather than round-off. Round-off
#: on a well-conditioned covariance lands near 1e-16 of that scale; a collapsed
#: one lands near -1. The threshold is far below the round-off it must tolerate
#: and far above the failure it must catch.
NEGATIVE_VARIANCE_REL_TOL = 1e-9


def std_err_from_covariance(cov, labels=None, context: str = "",
                            rel_tol: float = NEGATIVE_VARIANCE_REL_TOL,
                            warn=None):
    """Standard errors from a covariance diagonal, with failures made visible.

    Parameters
    ----------
    cov : (n, n) covariance matrix, or an (n,) diagonal already extracted.
    labels : optional names for the parameters, used only in the message.
    context : what produced this covariance, for the message.
    rel_tol : see NEGATIVE_VARIANCE_REL_TOL.
    warn : optional callable taking one string. Defaults to printing. Pass a
        logger or a collector to capture instead.

    Returns
    -------
    (std_err, info). ``std_err`` is (n,) with **nan** wherever the variance was
    negative beyond tolerance, and the ordinary square root elsewhere; nan
    rather than 0.0 so a failure cannot be mistaken for precision, and
    per-element rather than wholesale so one bad parameter does not discard the
    rest. ``info`` carries 'n_negative', 'worst_rel', 'clamped_indices' and
    'ok'.
    """
    arr = np.asarray(cov, dtype=float)
    diag = np.diag(arr) if arr.ndim == 2 else arr.ravel()
    diag = np.asarray(diag, dtype=float)

    scale = float(np.max(np.abs(diag))) if diag.size else 0.0
    floor = -float(rel_tol) * max(scale, np.finfo(float).tiny)
    bad = diag < floor

    std = np.sqrt(np.maximum(diag, 0.0))
    std = np.where(bad, np.nan, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        worst = float(np.min(diag) / scale) if scale > 0 else 0.0
    info = {
        "n_negative": int(np.sum(bad)),
        "worst_rel": worst,
        "clamped_indices": np.flatnonzero(bad).tolist(),
        "ok": not bool(np.any(bad)),
    }

    if info["n_negative"]:
        names = ""
        if labels is not None:
            try:
                names = ", ".join(str(labels[i]) for i in info["clamped_indices"])
                names = f" ({names})"
            except Exception:  # noqa: BLE001 - a label problem must not matter here
                names = ""
        where = f" in {context}" if context else ""
        message = (
            f"covariance{where} is not positive semi-definite: "
            f"{info['n_negative']} of {diag.size} variances are negative"
            f"{names}, the worst at {worst:.2e} of the largest variance. "
            f"Those standard errors are reported as nan, not 0, because they "
            f"do not exist -- a negative variance means the matrix could not "
            f"be inverted meaningfully, not that the parameter is perfectly "
            f"determined."
        )
        if warn is None:
            print(f"  [covariance] WARNING: {message}", flush=True)
        else:
            warn(message)

    return std, info
