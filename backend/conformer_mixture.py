from __future__ import annotations

import numpy as np

_KCAL_PER_MOL_PER_K = 0.00198720425864083
_HARTREE_TO_KCAL_PER_MOL = 627.509474
_CM1_TO_KCAL_PER_MOL = 0.002859144


def _energy_to_kcal_mol(value: float, unit: str) -> float:
    u = str(unit or "kcal/mol").strip().lower()
    x = float(value)
    if u in {"kcal/mol", "kcal", "kcal_mol"}:
        return x
    if u in {"cm-1", "cm1", "cm_1"}:
        return x * _CM1_TO_KCAL_PER_MOL
    if u in {"hartree", "ha"}:
        return x * _HARTREE_TO_KCAL_PER_MOL
    if u in {"dimensionless", "arb", "arbitrary"}:
        return x
    raise ValueError(f"Unsupported conformer energy unit: {unit!r}")


class ConformerMixture:
    """
    Conformer-mixture model with support for absolute coordinates or offsets.

    Conformers are represented as offsets from a reference geometry so the
    optimization can still be expressed in a single Cartesian coordinate vector.
    """

    def __init__(
        self,
        reference_coords,
        conformer_defs=None,
        weight_mode="fixed",
        temperature_k=298.15,
        energy_unit="kcal/mol",
    ):
        self.reference_coords = np.asarray(reference_coords, dtype=float)
        self.weight_mode = str(weight_mode).strip().lower()
        self.temperature_k = float(temperature_k)
        self.energy_unit = str(energy_unit).strip().lower()
        defs = conformer_defs or []
        self.conformers: list[dict] = []

        if not defs:
            self.conformers.append(
                {
                    "name": "base",
                    "offset": np.zeros_like(self.reference_coords),
                    "weight": 1.0,
                    "energy": 0.0,
                    "energy_kcal_mol": 0.0,
                    "energy_unit": self.energy_unit,
                    "source": "reference",
                    "metadata": {"source": "reference"},
                }
            )
        else:
            for i, c in enumerate(defs):
                name = str(c.get("name", f"conf_{i + 1}"))
                if "offset" in c and c["offset"] is not None:
                    off = np.asarray(c["offset"], dtype=float)
                elif "coords" in c and c["coords"] is not None:
                    off = np.asarray(c["coords"], dtype=float) - self.reference_coords
                else:
                    off = np.zeros_like(self.reference_coords)
                unit = str(c.get("energy_unit", self.energy_unit))
                energy = float(c.get("energy", 0.0))
                self.conformers.append(
                    {
                        "name": name,
                        "offset": off,
                        "weight": float(c.get("weight", 1.0)),
                        "energy": energy,
                        "energy_kcal_mol": _energy_to_kcal_mol(energy, unit),
                        "energy_unit": unit,
                        "source": str(c.get("source", "explicit")),
                        "metadata": dict(c.get("metadata") or {}),
                    }
                )
        self._normalize_fixed_weights()

    def _normalize_fixed_weights(self):
        if self.weight_mode == "uniform":
            w = np.ones(len(self.conformers), dtype=float)
        else:
            w = np.array([max(c["weight"], 0.0) for c in self.conformers], dtype=float)
        s = float(np.sum(w))
        if s <= 0.0:
            w = np.ones(len(self.conformers), dtype=float) / max(1, len(self.conformers))
        else:
            w = w / s
        for c, wi in zip(self.conformers, w):
            c["weight"] = float(wi)

    def _boltzmann_weights(self):
        e = np.array([float(c["energy_kcal_mol"]) for c in self.conformers], dtype=float)
        e = e - np.min(e)
        beta = 1.0 / max(_KCAL_PER_MOL_PER_K * max(self.temperature_k, 1e-6), 1e-12)
        x = np.exp(-beta * e)
        x = x / max(np.sum(x), 1e-12)
        return x

    def weights(self):
        if self.weight_mode == "boltzmann":
            return self._boltzmann_weights()
        if self.weight_mode == "uniform":
            n = len(self.conformers)
            return np.ones(n, dtype=float) / max(n, 1)
        return np.array([float(c["weight"]) for c in self.conformers], dtype=float)

    def conformer_coords(self, base_coords=None):
        base = self.reference_coords if base_coords is None else np.asarray(base_coords, dtype=float)
        return [base + np.asarray(c["offset"], dtype=float) for c in self.conformers]

    def diagnostics(self):
        w = self.weights()
        return {
            "names": [c["name"] for c in self.conformers],
            "weights": [float(v) for v in w],
            "energies": [float(c["energy"]) for c in self.conformers],
            "energies_kcal_mol": [float(c["energy_kcal_mol"]) for c in self.conformers],
            "energy_units": [str(c["energy_unit"]) for c in self.conformers],
            "sources": [str(c.get("source", "explicit")) for c in self.conformers],
            "weight_mode": self.weight_mode,
            "temperature_k": float(self.temperature_k),
            "n_conformers": len(self.conformers),
            "metadata": [dict(c.get("metadata") or {}) for c in self.conformers],
        }
