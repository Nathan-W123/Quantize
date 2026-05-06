"""
Optional matplotlib-based plots for torsion/LAM results.

All functions are safe to import when matplotlib is unavailable — they raise
ImportError only when actually called.  Use ``try_import=False`` to suppress
the guard and require matplotlib to be present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for torsion plots. Install it with: pip install matplotlib"
        ) from exc
    import matplotlib.pyplot as plt
    import numpy as np
    return plt, np


def plot_torsion_potential(
    spec,
    *,
    n_points: int = 200,
    n_wavefunctions: int = 0,
    output_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the torsion potential V(α) with optional wavefunction overlays.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec
    n_points : number of α grid points (0 to 2π)
    n_wavefunctions : number of eigenfunctions to overlay (0 = potential only)
    output_path : save figure to this path if given
    title : optional figure title
    show : call plt.show() if True

    Returns
    -------
    Path of saved figure, or None.
    """
    plt, np = _require_matplotlib()
    from backend.torsion_hamiltonian import solve_ram_lite_levels

    alpha = np.linspace(0, 2 * np.pi, n_points)
    pot = spec.potential
    V = np.full_like(alpha, pot.v0)
    for k, vc in pot.vcos.items():
        V += vc * np.cos(k * alpha)
    for k, vs in pot.vsin.items():
        V += vs * np.sin(k * alpha)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.degrees(alpha), V, "k-", lw=1.5, label="V(α)")
    ax.set_xlabel("Torsion angle α (degrees)")
    ax.set_ylabel("Potential (cm⁻¹)")

    if n_wavefunctions > 0:
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=n_wavefunctions)
        energies = out["energies_cm-1"]
        vecs = out.get("eigenvectors")
        if vecs is not None:
            m_vals = out.get("m_vals", [])
            for i in range(min(n_wavefunctions, len(energies))):
                e0 = float(energies[i])
                if vecs.ndim == 2 and vecs.shape[1] > i:
                    psi = vecs[:, i].real
                    # Project onto real-space grid via Fourier sum
                    psi_alpha = np.zeros_like(alpha)
                    for j, m in enumerate(m_vals):
                        psi_alpha += psi[j] * np.cos(m * alpha)
                    scale = 0.3 * (V.max() - V.min()) / (np.abs(psi_alpha).max() + 1e-12)
                    ax.plot(np.degrees(alpha), e0 + scale * psi_alpha,
                            lw=0.8, label=f"vt={i} ({e0:.1f} cm⁻¹)")

    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path: Optional[Path] = None
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_torsion_wavefunctions(
    spec,
    *,
    n_levels: int = 4,
    J: int = 0,
    K: int = 0,
    n_points: int = 200,
    output_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot torsion probability densities |ψ_vt(α)|² on top of the potential.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec
    n_levels : number of torsional levels to plot
    J, K : rotational quantum numbers for solve_ram_lite_levels
    n_points : α grid resolution
    output_path : save figure here if given
    title : figure title
    show : call plt.show()

    Returns
    -------
    Path of saved figure, or None.
    """
    plt, np = _require_matplotlib()
    from backend.torsion_hamiltonian import solve_ram_lite_levels

    alpha = np.linspace(0, 2 * np.pi, n_points)
    pot = spec.potential
    V = np.full_like(alpha, pot.v0)
    for k, vc in pot.vcos.items():
        V += vc * np.cos(k * alpha)
    for k, vs in pot.vsin.items():
        V += vs * np.sin(k * alpha)

    out = solve_ram_lite_levels(spec, J=J, K=K, n_levels=n_levels)
    energies = out["energies_cm-1"]
    vecs = out.get("eigenvectors")
    m_vals = list(out.get("m_vals", []))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.degrees(alpha), V, "k-", lw=1.5, label="V(α)", zorder=5)
    ax.axhline(0, color="gray", lw=0.4, ls="--")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i in range(min(n_levels, len(energies))):
        e0 = float(energies[i])
        color = colors[i % len(colors)]
        ax.axhline(e0, color=color, lw=0.5, ls=":")
        if vecs is not None and vecs.ndim == 2 and vecs.shape[1] > i and m_vals:
            psi = vecs[:, i].real
            prob = np.zeros_like(alpha)
            for j, m in enumerate(m_vals):
                for jj, mm in enumerate(m_vals):
                    prob += psi[j] * psi[jj] * np.cos((m - mm) * alpha)
            prob = np.abs(prob)
            scale = 0.25 * (V.max() - V.min()) / (prob.max() + 1e-12)
            ax.fill_between(np.degrees(alpha), e0, e0 + scale * prob,
                            alpha=0.35, color=color, label=f"vt={i} ({e0:.1f} cm⁻¹)")

    ax.set_xlabel("Torsion angle α (degrees)")
    ax.set_ylabel("Energy (cm⁻¹)")
    ax.legend(fontsize=8, loc="upper right")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path: Optional[Path] = None
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_tunneling_splitting_table(
    rows: Sequence[dict],
    *,
    output_path: Optional[Path | str] = None,
    title: Optional[str] = "A/E Tunneling Splittings",
    show: bool = False,
) -> Optional[Path]:
    """Bar chart of A/E tunneling splittings vs. torsional level vt.

    Parameters
    ----------
    rows : list of dicts from predict_tunneling_splitting
        Each dict must contain 'vt' and 'splitting_cm-1' (or 'splitting_MHz').
    output_path : save figure here if given
    title : figure title
    show : call plt.show()

    Returns
    -------
    Path of saved figure, or None.
    """
    plt, np = _require_matplotlib()

    if not rows:
        return None

    vt_vals = [int(r["vt"]) for r in rows]
    split_cm1 = [float(r.get("splitting_cm-1", r.get("splitting_MHz", 0.0) / 29979.2458)) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["steelblue" if s >= 0 else "tomato" for s in split_cm1]
    ax.bar(vt_vals, split_cm1, color=colors, edgecolor="k", linewidth=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Torsional level vt")
    ax.set_ylabel("A–E splitting (cm⁻¹)")
    ax.set_xticks(vt_vals)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path: Optional[Path] = None
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path
