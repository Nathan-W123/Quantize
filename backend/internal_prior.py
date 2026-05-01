import numpy as np

from backend.quantum import wilson_B


def _internal_values(coords, labels):
    coords = np.asarray(coords, dtype=float)
    vals = []
    for lbl in labels:
        parts = lbl.split()
        kind = parts[0].lower()
        idx = [int(x) - 1 for x in parts[1].split("-")]
        if kind == "bond" and len(idx) == 2:
            i, j = idx
            vals.append(float(np.linalg.norm(coords[i] - coords[j])))
        elif kind == "angle" and len(idx) == 3:
            i, j, k = idx
            v1 = coords[i] - coords[j]
            v2 = coords[k] - coords[j]
            c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
            vals.append(float(np.degrees(np.arccos(c))))
        elif kind == "dihedral" and len(idx) == 4:
            i, j, k, l = idx
            b0 = coords[i] - coords[j]
            b1 = coords[k] - coords[j]
            b2 = coords[l] - coords[k]
            b1n = b1 / max(np.linalg.norm(b1), 1e-12)
            v = b0 - np.dot(b0, b1n) * b1n
            w = b2 - np.dot(b2, b1n) * b1n
            x = np.dot(v, w)
            y = np.dot(np.cross(b1n, v), w)
            vals.append(float(np.degrees(np.arctan2(y, x))))
        else:
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


class InternalPriorEngine:
    """
    Internal-coordinate priors for bonds/angles/dihedrals.
    Provides stacked prior Jacobian/residual compatible with spectral stacking.
    """

    def __init__(
        self,
        coords,
        elems,
        use_dihedrals=False,
        prior_targets=None,
        prior_sigmas=None,
        auto_from_initial=True,
        sigma_bond=0.05,
        sigma_angle_deg=3.0,
        sigma_dihedral_deg=15.0,
    ):
        self.elems = list(elems)
        self.use_dihedrals = bool(use_dihedrals)
        self.B0, self.labels = wilson_B(np.asarray(coords, dtype=float), self.elems)
        kinds = [lbl.split()[0].lower() for lbl in self.labels]
        keep = []
        for i, k in enumerate(kinds):
            if k in ("bond", "angle"):
                keep.append(i)
            elif k == "dihedral" and self.use_dihedrals:
                keep.append(i)
        self.keep_idx = np.asarray(keep, dtype=int)
        self.labels = [self.labels[i] for i in self.keep_idx]
        self.B0 = self.B0[self.keep_idx] if self.B0.size else np.zeros((0, 3 * len(self.elems)))

        if prior_targets is None and auto_from_initial:
            prior_targets = _internal_values(np.asarray(coords, dtype=float), self.labels)
        if prior_targets is None:
            prior_targets = np.zeros(len(self.labels), dtype=float)
        self.prior_targets = np.asarray(prior_targets, dtype=float)
        if self.prior_targets.size != len(self.labels):
            raise ValueError("prior_targets length must match selected internal coordinates.")

        if prior_sigmas is None:
            sig = []
            for lbl in self.labels:
                k = lbl.split()[0].lower()
                if k == "bond":
                    sig.append(float(sigma_bond))
                elif k == "angle":
                    sig.append(float(sigma_angle_deg))
                else:
                    sig.append(float(sigma_dihedral_deg))
            prior_sigmas = sig
        self.prior_sigmas = np.maximum(np.asarray(prior_sigmas, dtype=float), 1e-12)
        if self.prior_sigmas.size != len(self.labels):
            raise ValueError("prior_sigmas length must match selected internal coordinates.")

    def stacked(self, coords):
        """
        Returns weighted Jacobian and residual in prior space:
        Jp = B/sigma, rp = (target - q(coords))/sigma
        """
        if len(self.labels) == 0:
            n = np.asarray(coords, dtype=float).size
            return np.zeros((0, n), dtype=float), np.zeros(0, dtype=float)
        B, labels_all = wilson_B(np.asarray(coords, dtype=float), self.elems)
        label_to_idx = {lbl: i for i, lbl in enumerate(labels_all)}
        rows = []
        for i, lbl in enumerate(self.labels):
            j = label_to_idx.get(lbl, None)
            if j is not None:
                rows.append(B[j])
            else:
                # Connectivity can change transiently in noisy optimization regions.
                # Fall back to the initial Jacobian row so priors remain well-defined.
                rows.append(self.B0[i])
        Bk = np.asarray(rows, dtype=float)
        q = _internal_values(np.asarray(coords, dtype=float), self.labels)
        r = self.prior_targets - q
        Jw = Bk / self.prior_sigmas[:, None]
        rw = r / self.prior_sigmas
        return Jw, rw

    def diagnostics(self, coords):
        if len(self.labels) == 0:
            return {"prior_wrms": 0.0, "n_priors": 0}
        _, rw = self.stacked(coords)
        return {"prior_wrms": float(np.sqrt(np.mean(rw ** 2))), "n_priors": int(len(self.labels))}

    def diagnostics_for_conformers(self, conformer_coords, conformer_weights=None):
        """
        Optional conformer-indexed prior diagnostics for mixture models.
        """
        coords_list = list(conformer_coords or [])
        if not coords_list:
            return {"prior_wrms_by_conformer": [], "prior_wrms_mixture": 0.0}
        if conformer_weights is None:
            weights = np.ones(len(coords_list), dtype=float)
        else:
            weights = np.asarray(conformer_weights, dtype=float)
        weights = weights / max(np.sum(weights), 1e-12)
        wrms_vals = []
        for c in coords_list:
            d = self.diagnostics(c)
            wrms_vals.append(float(d["prior_wrms"]))
        wrms_vals = np.asarray(wrms_vals, dtype=float)
        mix = float(np.sum(weights * wrms_vals))
        return {
            "prior_wrms_by_conformer": wrms_vals.tolist(),
            "prior_wrms_mixture": mix,
        }
