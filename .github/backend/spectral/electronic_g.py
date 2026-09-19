"""Rotational g-tensor, so the electronic correction stops being zero.

``resolve_corrections`` has always known how to apply the electronic
correction to a rotational constant,

    delta_elec = -(m_e / m_p) * g_alpha * B_obs

but nothing in this repository could produce ``g_alpha``. Absent a supplied
g-tensor the code correctly refuses to guess and reports zero with a sigma wide
enough to cover |g| <= 0.7, which is honest but leaves a known, computable
term on the table. STRFIT added the electronic correction to its r_e^SE
evaluation in July 2025; the field includes it.

This module computes it, from the same SCF the geometry came from. It needs
``pip install pyscf-properties`` alongside PySCF; like the quantum backends
themselves that is an optional extra, and the tests skip without it.

Frame
-----
The g-tensor is returned by the property code in the input Cartesian frame.
A rotational constant belongs to a principal axis, so the tensor is rotated
into the principal axis frame with the engine's own ``inertia_paf`` and then
labelled A/B/C by increasing moment of inertia -- the same ordering
``rotational_constants_mhz`` uses. Doing this by hand from the input frame is
the obvious way to silently mislabel two components.

Why a patch is needed
---------------------
``pyscf-properties`` 0.1.0 is the only distribution of PySCF's rotational
g-tensor and it predates a change in ``lib.krylov``: its CPHF response closure
reshapes the trial vectors to ``(3, nmo, nocc)``, hardcoding the three
components of the perturbation, while krylov now hands it a variable number of
trial vectors. Every call dies with a reshape error. :func:`_patch_cphf_vind`
replaces that one closure with a component-count-agnostic version; the physics
is untouched.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from backend.spectral.centrifugal_distortion import inertia_paf

#: Set once, the first time a g-tensor is requested.
_PATCHED = False


def _patch_cphf_vind() -> None:
    """Make pyscf-properties' CPHF response closure accept any vector count."""
    global _PATCHED
    if _PATCHED:
        return
    from pyscf import lib
    from pyscf.prop.nmr import rhf as nmr_rhf

    def gen_vind(mf, mo_coeff, mo_occ):
        vresp = mf.gen_response(singlet=True, hermi=2)
        orbo = mo_coeff[:, mo_occ > 0]
        nocc = orbo.shape[1]
        _, nmo = mo_coeff.shape

        def vind(mo1):
            mo1 = np.asarray(mo1).reshape(-1, nmo, nocc)
            dm1 = [reduce(np.dot, (mo_coeff, x * 2, orbo.T.conj())) for x in mo1]
            dm1 = np.asarray([d1 - d1.conj().T for d1 in dm1])
            v1mo = lib.einsum("xpq,pi,qj->xij", vresp(dm1),
                              mo_coeff.conj(), orbo)
            return v1mo.reshape(mo1.shape[0], -1)

        return vind

    nmr_rhf.gen_vind = gen_vind
    _PATCHED = True


def rotational_g_tensor(elems, coords_ang, masses_amu, method="hf",
                        basis="6-31g", charge=0, multiplicity=1):
    """Diagonal rotational g values in the principal axis frame.

    Returns ``{"A": g_aa, "B": g_bb, "C": g_cc}``, the dict shape
    ``resolve_corrections`` expects for its ``g_tensor`` argument. A is the
    axis of smallest moment of inertia, matching the A/B/C ordering used
    everywhere else in the engine.

    Costs one extra SCF plus a CPHF solve -- cheap next to the Hessians the
    correction table already needs.
    """
    _patch_cphf_vind()
    from pyscf import dft, gto, scf
    from pyscf.prop import rotational_gtensor

    coords = np.asarray(coords_ang, dtype=float)
    mol = gto.M(atom=[(e, tuple(xyz)) for e, xyz in zip(elems, coords)],
                basis=basis, charge=int(charge), spin=int(multiplicity) - 1,
                unit="Angstrom", verbose=0)
    if str(method).lower() in ("hf", "rhf", "scf"):
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = str(method)
    mf.max_cycle = max(int(getattr(mf, "max_cycle", 50)), 200)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for the g-tensor "
                           f"({method}/{basis})")

    g_input_frame = np.asarray(rotational_gtensor.RHF(mf).kernel(), dtype=float)
    g_input_frame = 0.5 * (g_input_frame + g_input_frame.T)

    # Rotate into the principal axis frame. inertia_paf returns eigenvalues in
    # increasing order with V's columns the corresponding axes, so column 0 is
    # the A axis.
    _, v_paf, _ = inertia_paf(coords, np.asarray(masses_amu, dtype=float))
    g_paf = v_paf.T @ g_input_frame @ v_paf
    return {k: float(g_paf[i, i]) for i, k in enumerate(("A", "B", "C"))}
