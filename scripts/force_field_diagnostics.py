"""Is this correction's force field trustworthy? Two cheap checks.

WHY THIS EXISTS. On 2026-11-10 the benchmark replaced HF/6-31G corrections
with MP2/cc-pVTZ ones -- unambiguously the better level of theory by any
general measure -- and ozone's hybrid result went from 1.70 mA to 7.86 mA, 4.6
times worse. The cause was not subtle: MP2 puts ozone's antisymmetric stretch
at 2212 cm^-1 against a measured 1103, exactly double, and alpha carries
1/omega, so the correction inherited it. The RMS correction error went from 117
MHz at HF to 794 MHz at MP2, with the SIGN wrong on B and C.

Nothing in the engine noticed. It computed a force field, computed alpha from
it, applied it, fitted a structure and reported a number to two decimal places.
"Better level of theory" is not a scalar, and until now there was no way for a
result to carry that caveat.

TWO SIGNALS, because they fail for different reasons.

1. MULTIREFERENCE CHARACTER, from one level. Ozone's ground state has
   substantial biradical character: its O-O-O pi system is not well described
   by a single determinant. MP2 is perturbation theory ON TOP of that single
   determinant, so it amplifies the error instead of fixing it. The symptom is
   a large double-excitation amplitude -- MP2 is being asked to do
   non-perturbative work. Measured across the whole reference set, max|t2|:

       ozone               0.1920 (6-31g)  0.1794 (cc-pVDZ)  0.1697 (cc-pVTZ)
       vinyl fluoride      0.0969          0.0876            0.0797
       formyl fluoride                     0.0799
       fluoroacetylene                     0.0689
       isocyanic acid                      0.0661
       acetyl fluoride                     0.0647
       water               0.0478          0.0471            0.0448
       chlorofluoromethane                 0.0299
       fluoroethane                        0.0209

   Ozone is more than double everything else, and the ordering holds across
   three bases, so the threshold sits in an empty gap rather than being tuned.

   Two other candidates were measured and rejected. The HOMO-LUMO gap does
   flag ozone (12.4 eV, the smallest in the set) but only 1.2x below vinyl
   fluoride's 15.2 eV, too narrow to threshold. |E_corr|/|E_SCF| does not
   discriminate at all: every molecule in the set lands at 0.0026 +/- 0.0002.

2. CROSS-LEVEL DISAGREEMENT, when two levels are available. Two force fields
   that disagree badly cannot both be right, and the engine often has both --
   every base/corr benchmark pair computes two. Largest relative disagreement
   over modes:

       ozone      HF [730, 851, 1267] vs MP2 [748, 1171, 2212]   74.6%
       water      HF [1737, 3989, 4145] vs MP2 [1650, 3871, 3992]  5.3%

   This catches a bad force field that the multireference check misses, since
   a level can be wrong for reasons other than multireference character.

WHERE THIS LIVES, AND WHAT THAT MEANS. This is benchmark tooling, next to the
benchmark that uses it, not part of the shipped engine. The consequence is
worth stating plainly rather than leaving implicit: a caller who uses
``build_correction_table_from_hessian`` directly gets NO force-field warning.
Only runs that go through accuracy_upgrades_benchmark do.

That was a deliberate choice over wiring it into the correction path. The
multireference probe costs an MP2 calculation -- seconds for a four-atom
molecule and not seconds for a large one -- so putting it in the engine's hot
path would have meant an opt-in flag there anyway, and the engine does not
otherwise reach for electronic structure it was not asked for. If a library
user ever needs the warning, the functions are importable as they stand; the
move to make then is to call them from the correction path, not to rewrite
them.

WHAT THESE DO NOT DO. Neither says which level is RIGHT. A flag means the
correction is not to be trusted at face value, not that the other level is
better. For ozone the cheaper HF/6-31G happens to score better, but its own nu3
is 15% high and its alpha errors partly cancel -- luck, not quality. Deciding
the right level needs a reference, which is what the r_e structures are for.
"""

from __future__ import annotations

import numpy as np

#: max|t2| above which a force field is reported as multireference-suspect.
#: 0.12 sits between the largest well-behaved value measured (vinyl fluoride,
#: 0.097 at 6-31g) and the smallest ozone value (0.170 at cc-pVTZ). See the
#: module docstring for the full table.
MULTIREFERENCE_T2_WARN = 0.12

#: Largest relative frequency disagreement between two levels, above which
#: neither force field is trusted. 0.25 sits between water's 5.3% and ozone's
#: 74.6%.
FREQUENCY_DISAGREEMENT_WARN = 0.25

#: Basis for the multireference probe. Small deliberately: the diagnostic is
#: about the WAVEFUNCTION's character, which the table above shows is
#: basis-insensitive, so paying for a large basis buys nothing.
MULTIREFERENCE_PROBE_BASIS = "cc-pvdz"


def multireference_diagnostic(elements, coords_ang, basis: str = MULTIREFERENCE_PROBE_BASIS,
                              charge: int = 0, spin: int = 0,
                              t2_warn: float = MULTIREFERENCE_T2_WARN) -> dict:
    """Does this molecule need more than one determinant?

    Runs RHF then MP2 at ``basis`` and reports the largest double-excitation
    amplitude, which is the cheapest standard indicator of multireference
    character. A flagged molecule is one where single-reference correlation
    methods -- MP2 above all, but CCSD(T) too, to a lesser degree -- should not
    be trusted to give a force field good enough for a vibration-rotation
    correction.

    Costs one MP2 calculation, seconds for a small molecule and not free for a
    large one, which is why callers opt in.

    Returns a dict with 'max_t2', 'homo_lumo_ev', 'e_corr_fraction',
    'flagged', 'basis' and 'reason'. On failure it returns the same shape with
    'max_t2' None and the exception text in 'reason', because a diagnostic that
    raises would take down the run it was meant to qualify.
    """
    out = {"max_t2": None, "homo_lumo_ev": None, "e_corr_fraction": None,
           "flagged": False, "basis": str(basis), "reason": ""}
    try:
        from pyscf import gto, mp, scf
    except ImportError as exc:  # pragma: no cover - pyscf is a hard dep in practice
        out["reason"] = f"pyscf unavailable: {exc}"
        return out

    coords = np.asarray(coords_ang, dtype=float)
    try:
        mol = gto.M(atom=[(str(e), tuple(map(float, xyz)))
                          for e, xyz in zip(elements, coords)],
                    basis=str(basis), charge=int(charge), spin=int(spin),
                    unit="Angstrom", verbose=0)
        mf = scf.RHF(mol)
        mf.max_cycle = max(int(getattr(mf, "max_cycle", 50)), 200)
        mf.kernel()
        if not mf.converged:
            out["reason"] = "SCF did not converge, so no diagnostic"
            return out
        n_occ = mol.nelectron // 2
        if 0 < n_occ < len(mf.mo_energy):
            out["homo_lumo_ev"] = float(
                (mf.mo_energy[n_occ] - mf.mo_energy[n_occ - 1]) * 27.211386245988)
        pt = mp.MP2(mf)
        pt.kernel()
        t2 = getattr(pt, "t2", None)
        if t2 is None:
            out["reason"] = "MP2 returned no amplitudes"
            return out
        out["max_t2"] = float(np.abs(np.asarray(t2)).max())
        if mf.e_tot:
            out["e_corr_fraction"] = float(abs(pt.e_corr) / abs(mf.e_tot))
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill the run
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    if out["max_t2"] > float(t2_warn):
        out["flagged"] = True
        out["reason"] = (
            f"max|t2| = {out['max_t2']:.4f} exceeds {float(t2_warn):.2f}: this "
            f"molecule has substantial multireference character, so a "
            f"single-reference force field -- MP2 especially -- may be badly "
            f"wrong. Measured precedent: ozone sits at 0.18 and its MP2 "
            f"antisymmetric stretch is 101% high, which made its corrected "
            f"result 4.6x worse than a cheaper level."
        )
    else:
        out["reason"] = (f"max|t2| = {out['max_t2']:.4f}, within the "
                         f"single-reference range for this set")
    return out


def frequency_disagreement(omega_a, omega_b, label_a: str = "A",
                           label_b: str = "B",
                           warn: float = FREQUENCY_DISAGREEMENT_WARN) -> dict:
    """How badly do two levels' harmonic frequencies disagree?

    Compared mode by mode after sorting, relative to the SMALLER of each pair
    so the measure is symmetric in the two levels and a doubling reads as 100%
    rather than 50%. Modes are matched by frequency order, which is adequate
    for the gross disagreement this is looking for and would not be for a
    careful mode correspondence.

    Returns 'max_rel', 'worst_index', 'worst_pair', 'flagged', 'reason'.
    """
    a = np.sort(np.asarray(omega_a, dtype=float))
    b = np.sort(np.asarray(omega_b, dtype=float))
    out = {"max_rel": None, "worst_index": None, "worst_pair": None,
           "flagged": False, "reason": ""}
    if a.size != b.size or a.size == 0:
        out["reason"] = (f"cannot compare {a.size} modes against {b.size}; "
                         f"the two levels are not describing the same molecule")
        return out

    denom = np.minimum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0.0, np.abs(a - b) / denom, 0.0)
    k = int(np.argmax(rel))
    out["max_rel"] = float(rel[k])
    out["worst_index"] = k
    out["worst_pair"] = (float(a[k]), float(b[k]))

    if out["max_rel"] > float(warn):
        out["flagged"] = True
        out["reason"] = (
            f"{label_a} and {label_b} disagree by {100 * out['max_rel']:.0f}% "
            f"on mode {k} ({a[k]:.0f} vs {b[k]:.0f} cm^-1), above the "
            f"{100 * float(warn):.0f}% threshold. They cannot both be right, "
            f"and alpha carries 1/omega, so neither correction should be "
            f"trusted until the disagreement is resolved."
        )
    else:
        out["reason"] = (f"{label_a} and {label_b} agree to "
                         f"{100 * out['max_rel']:.0f}% on every mode")
    return out
