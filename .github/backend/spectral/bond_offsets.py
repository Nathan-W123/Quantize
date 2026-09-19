"""Bond-class offsets: telling the prior which way the theory is wrong.

The quantum prior in this engine is Gaussian in displacement from the level of
theory's own geometry, which asserts that the theory is unbiased -- that its
error is as likely to be short as long. That assertion is false, and measurably
so. At RHF/6-31G across the benchmark set:

    class   n     mean      range          all one sign
    C-F     6   +27.8   +24.6 .. +30.7          yes
    C-H    11   -11.1   -23.1 ..  -1.9          yes
    C-O     3    +7.1    +1.3 .. +13.3          yes
    C-C     4   -11.8   -18.8 ..  +4.1           no

Six C-F bonds in five different molecules, every one too long, with a 2 mA
spread about a 28 mA bias. A zero-mean prior cannot remove that: the fit can
only satisfy the rotational constants by pushing some other bond the wrong way
to compensate, which is exactly the mixed-sign per-bond pattern measured on
vinyl fluoride (C-F +30.7, C=C -18.8) and acetyl fluoride (C-F +28.8,
C-C -17.8).

So move the prior's centre. Each bond class gets an offset, the theory geometry
is corrected by it, and the corrected geometry becomes both the starting point
and the centre of the Gaussian. Nothing about the optimiser changes; it is
handed a better-centred prior. This is the same idea as the offset corrections
used in semi-experimental equilibrium work, applied to the prior rather than to
the final structure.

Honesty about calibration
-------------------------
An offset table fitted on the molecules it is then scored on is circular, and
the circularity is large here -- C-F would be corrected by its own error.
:func:`leave_one_out_offsets` is therefore the only entry point the benchmark
uses: each molecule is corrected using offsets measured on the *other*
molecules alone. A class with no other member gets no correction, which is why
chlorofluoromethane's C-Cl (+78 mA, the largest single error in the set) is
left uncorrected under leave-one-out. That is the honest answer for a class
seen once.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares


def bond_class(elem_i: str, elem_j: str) -> str:
    """Order-independent element-pair label, e.g. ``"C-F"``.

    Bond order is deliberately ignored. C=C and C-C share a class because the
    reference set has too few of either to calibrate separately, and the
    dominant error at a small basis is per-element-pair rather than per-order.
    Splitting them would be the right move on a larger calibration set.
    """
    a, b = sorted((str(elem_i), str(elem_j)))
    return f"{a}-{b}"


def bond_classes_of(mol) -> dict[str, str]:
    """Map each named bond of a reference molecule to its class.

    ``mol.bonds`` maps a name to a list of symmetry-equivalent index pairs; the
    class comes from the first pair, since equivalent bonds share elements.
    """
    out: dict[str, str] = {}
    for name, pairs in mol.bonds.items():
        i, j = int(pairs[0][0]), int(pairs[0][1])
        out[name] = bond_class(mol.elems[i], mol.elems[j])
    return out


def measure_offsets(mol, theory_coords) -> dict[str, list[float]]:
    """Signed theory-minus-reference bond errors, in Angstrom, keyed by class.

    Positive means the level of theory makes that bond too long.
    """
    ref = mol.internal_coordinates(np.asarray(mol.geometry, dtype=float))
    got = mol.internal_coordinates(np.asarray(theory_coords, dtype=float))
    classes = bond_classes_of(mol)
    out: dict[str, list[float]] = defaultdict(list)
    for name, cls in classes.items():
        out[cls].append(float(got[name] - ref[name]))
    return dict(out)


def leave_one_out_offsets(per_molecule: dict[str, dict[str, list[float]]],
                          exclude_key: str) -> dict[str, float]:
    """Mean offset per class, computed without the molecule being scored.

    ``per_molecule`` maps a molecule key to the output of :func:`measure_offsets`.
    Classes whose only members belong to the excluded molecule are absent from
    the result, so callers correct nothing for them.
    """
    pooled: dict[str, list[float]] = defaultdict(list)
    for key, by_class in per_molecule.items():
        if key == exclude_key:
            continue
        for cls, errs in by_class.items():
            pooled[cls].extend(errs)
    return {cls: float(np.mean(errs)) for cls, errs in pooled.items() if errs}


def offset_corrected_geometry(mol, theory_coords, offsets_ang,
                              angle_weight: float = 0.02):
    """Theory geometry with each bond class shifted by ``-offset``.

    Bonds are driven to ``theory - offset``, angles held at their theory
    values, and the six rigid-body directions pinned by a weak tether to the
    input geometry so the returned frame is comparable to it. ``angle_weight``
    is in Angstrom per degree: angles are held, not corrected, because the
    reference set gives no usable per-class angle bias (the angle errors are
    small and change sign within a class).

    Returns Cartesians in Angstrom, shaped like ``theory_coords``.
    """
    x0 = np.asarray(theory_coords, dtype=float)
    theory_int = mol.internal_coordinates(x0)
    classes = bond_classes_of(mol)
    bond_names = list(mol.bonds.keys())
    angle_names = list(mol.angles.keys())

    target_bond = {n: theory_int[n] - float(offsets_ang.get(classes[n], 0.0))
                   for n in bond_names}

    def residuals(flat):
        coords = flat.reshape(x0.shape)
        got = mol.internal_coordinates(coords)
        res = [got[n] - target_bond[n] for n in bond_names]
        res += [angle_weight * (got[n] - theory_int[n]) for n in angle_names]
        # Rigid-body tether. Weak enough not to fight the internal targets,
        # strong enough to remove the six flat directions.
        res += list(1e-4 * (coords - x0).ravel())
        return np.asarray(res, dtype=float)

    sol = least_squares(residuals, x0.ravel(), method="trf",
                        xtol=1e-14, ftol=1e-14, max_nfev=4000)
    return sol.x.reshape(x0.shape)


def residual_sigma_after_offsets(per_molecule: dict[str, dict[str, list[float]]],
                                 exclude_key: str,
                                 floor_ang: float = 0.002) -> float:
    """Prior width left once the class offsets are removed, in Angstrom.

    sigma_x states how far the level of theory is trusted, and the engine
    calibrates alpha_q from it. Correcting a systematic bias without also
    narrowing sigma_x is incoherent -- and measurably harmful: on vinyl
    fluoride the offsets moved the prior from 19.0 to 5.7 mA while sigma_x
    stayed at 20 mA, so the optimiser went on treating a good prior as a loose
    one and let the spectroscopic residual pull the structure back out to
    17.2 mA, worse than the prior it started from.

    What remains after the offsets is the *within-class* spread, so that is
    what this returns: the pooled RMS of each bond's error about its own class
    mean, over the calibration molecules only. Classes with a single member
    contribute no spread and are skipped, since one point cannot estimate a
    width. ``floor_ang`` keeps a small or lucky calibration set from claiming
    an implausibly tight prior.
    """
    pooled: dict[str, list[float]] = defaultdict(list)
    for key, by_class in per_molecule.items():
        if key == exclude_key:
            continue
        for cls, errs in by_class.items():
            pooled[cls].extend(errs)
    resid: list[float] = []
    for errs in pooled.values():
        if len(errs) < 2:
            continue
        arr = np.asarray(errs, dtype=float)
        resid.extend(arr - arr.mean())
    if not resid:
        return float(floor_ang)
    return float(max(np.sqrt(np.mean(np.square(resid))), floor_ang))


def uncorrected_class_bias(per_molecule: dict[str, dict[str, list[float]]],
                           exclude_key: str) -> float:
    """Typical size of a bond-class bias, for a class nothing calibrated.

    A class seen only in the molecule being scored gets no offset, and the
    honest statement about its error is not "small" -- it is "the size a bond
    class bias usually has". That is measurable: the RMS of the per-class mean
    offsets over the calibration molecules.
    """
    pooled: dict[str, list[float]] = defaultdict(list)
    for key, by_class in per_molecule.items():
        if key == exclude_key:
            continue
        for cls, errs in by_class.items():
            pooled[cls].extend(errs)
    means = [float(np.mean(v)) for v in pooled.values() if v]
    if not means:
        return 0.0
    return float(np.sqrt(np.mean(np.square(means))))


def prior_sigma_for_molecule(mol, per_molecule: dict[str, dict[str, list[float]]],
                             exclude_key: str, floor_ang: float = 0.002) -> float:
    """Prior width for *this* molecule, bond by bond, in Angstrom.

    :func:`residual_sigma_after_offsets` returns one number for the whole
    calibration set: the spread left once class offsets are removed. That is
    right only if every class in the molecule actually got an offset. Where one
    did not, the residual error is the full uncorrected bias, and quoting the
    corrected spread instead tells the optimiser the prior is far better than
    it is.

    The cost of getting this wrong is not subtle. Chlorofluoromethane's C-Cl is
    +78 mA and the only member of its class, so leave-one-out declines to
    correct it and the "corrected" prior is still 45 mA out -- while sigma_x
    claimed 5-6. Centring the prior there and asserting that precision moved
    the hybrid from 13.16 mA to 19.71. The same mechanism, milder, cost formyl
    fluoride 5.24 -> 6.45.

    So sigma_x becomes the RMS over the molecule's own bonds of what is
    actually left for each: the within-class spread where a class was
    calibrated, and :func:`uncorrected_class_bias` where it was not.
    """
    corrected = leave_one_out_offsets(per_molecule, exclude_key)
    spread = residual_sigma_after_offsets(per_molecule, exclude_key,
                                          floor_ang=floor_ang)
    unknown = uncorrected_class_bias(per_molecule, exclude_key)
    classes = bond_classes_of(mol)
    per_bond = [spread if classes[name] in corrected else unknown
                for name in mol.bonds]
    if not per_bond:
        return float(max(spread, floor_ang))
    return float(max(np.sqrt(np.mean(np.square(per_bond))), floor_ang))
