import numpy as np


class ConformerMixture:
    """
    Lightweight conformer-mixture model.

    Conformers are represented as offsets from a reference geometry so the
    optimization can still be expressed in a single Cartesian coordinate vector.
    """

    def __init__(self, reference_coords, conformer_defs=None, weight_mode="fixed", temperature_k=298.15):
        self.reference_coords = np.asarray(reference_coords, dtype=float)
        self.weight_mode = str(weight_mode).strip().lower()
        self.temperature_k = float(temperature_k)
        defs = conformer_defs or []
        self.conformers = []
        if not defs:
            self.conformers.append(
                {
                    "name": "base",
                    "offset": np.zeros_like(self.reference_coords),
                    "weight": 1.0,
                    "energy": 0.0,
                }
            )
        else:
            for i, c in enumerate(defs):
                name = str(c.get("name", f"conf_{i+1}"))
                if "offset" in c and c["offset"] is not None:
                    off = np.asarray(c["offset"], dtype=float)
                elif "coords" in c and c["coords"] is not None:
                    off = np.asarray(c["coords"], dtype=float) - self.reference_coords
                else:
                    off = np.zeros_like(self.reference_coords)
                self.conformers.append(
                    {
                        "name": name,
                        "offset": off,
                        "weight": float(c.get("weight", 1.0)),
                        "energy": float(c.get("energy", 0.0)),
                    }
                )
        self._normalize_fixed_weights()

    def _normalize_fixed_weights(self):
        w = np.array([max(c["weight"], 0.0) for c in self.conformers], dtype=float)
        s = float(np.sum(w))
        if s <= 0.0:
            w = np.ones(len(self.conformers), dtype=float) / max(1, len(self.conformers))
        else:
            w = w / s
        for c, wi in zip(self.conformers, w):
            c["weight"] = float(wi)

    def _boltzmann_weights(self):
        # Energies expected in relative kcal/mol-like arbitrary units.
        # Keep scale robust by using softmax-like normalization.
        e = np.array([float(c["energy"]) for c in self.conformers], dtype=float)
        e = e - np.min(e)
        beta = 1.0 / max(self.temperature_k, 1e-6)
        x = np.exp(-beta * e)
        x = x / max(np.sum(x), 1e-12)
        return x

    def weights(self):
        if self.weight_mode == "boltzmann":
            return self._boltzmann_weights()
        return np.array([float(c["weight"]) for c in self.conformers], dtype=float)

    def conformer_coords(self, base_coords):
        base = np.asarray(base_coords, dtype=float)
        return [base + np.asarray(c["offset"], dtype=float) for c in self.conformers]

    def diagnostics(self):
        w = self.weights()
        return {
            "names": [c["name"] for c in self.conformers],
            "weights": [float(v) for v in w],
        }
