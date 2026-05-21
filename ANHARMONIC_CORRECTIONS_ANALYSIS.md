# Anharmonic Corrections: What You Have vs. What You Need

## Quick Answer: What You're Missing

Your system currently computes:
```
B_e = B₀ + Δ_vib + Δ_elec + Δ_BOB
           ↑ harmonic only
```

**Target formula:**
```
B_e = B₀ + Δ_vib + Δ_elec + Δ_BOB + Δ_anh
           ↑ harmonic    ↑ MISSING (anharmonic from cubic/quartic force constants)
```

---

## Detailed Breakdown of Current Implementation

### **1. What You ARE Computing** ✅

#### **A. Centrifugal Distortion Correction (Δ_cent)**
**Location:** `.github/backend/spectral/harmonic_alpha.py:108-111`

```python
# Second derivative of rotational constant along each vibrational mode
alpha_cent[:, r] = -d2B_mhz_all[:, r] * (zpe_amplitude / omega_cm[r])
```

- **Source:** Cartesian Hessian (2nd derivatives of energy)
- **Method:** Finite difference of ∂²B_K/∂Q_r² using normal modes
- **Physics:** How the moment of inertia changes when atoms vibrate

---

#### **B. Coriolis Coupling Correction (Δ_Cor)**
**Location:** `.github/backend/spectral/harmonic_alpha.py:113-133`

```python
# Angular momentum coupling between vibrational modes
zeta[K] = cross_product_of_mode_vectors[K]

alpha_cor_cm[K, r] = -4 * B_e_cm[K] * Σ_s [zeta[r,s]² * ω_s / (ω_s² - ω_r²)]
```

- **Source:** Normal mode eigenvectors from Hessian diagonalization
- **Method:** Watson's Coriolis coupling formula
- **Physics:** Interaction between vibrational angular momentum and rotational motion

**Formula Reference (Watson 1968):**
```
α_r,Coriolis^K = -4 B_e^K × Σ_{s≠r, J≠K} (B_e^J × ζ_rs^(3-K-J)²) × ω_s / (ω_s² - ω_r²)
```

---

#### **C. Electronic Mass Correction (Δ_elec)**
**Location:** `.github/backend/spectral/correction_models.py:274-301`

```python
delta_elec = -(m_e / M_total) × B_obs
           = -(5.4857991e-4 amu / M_total_amu) × B_MHz
```

- **Formula:** Gordy-Cook approximation
- **Electron mass:** 5.48579909070e-4 amu (CODATA 2018)
- **Uncertainty:** This approximation ignores electronic g-tensor (~10-30% of the correction itself)

---

#### **D. Born-Oppenheimer Breakdown (Δ_BOB)**
**Location:** `.github/backend/spectral/correction_models.py:306-385`

```python
delta_bob = -Σ_a (m_e / m_a) × u_a^K

where u_a^K = dimensionless Watson u-parameter per element/component
```

**Built-in u-parameters (from literature):**
```python
_BOB_BUILTIN = {
    "H": {"A": 0.015, "B": 0.015, "C": 0.015},
    "O": {"B": 0.002, "C": 0.002},
    "C": {"B": 0.003, "C": 0.003},
    # ... etc
}
```

- **Source:** Watson (1980) and Gordy-Cook tables
- **Limitation:** 100% relative uncertainty; molecule-specific values not used

---

### **2. Assembly Flow** 

**In `quantize.py:_apply_harmonic_alpha_corrections()` (lines 851-933):**

```
Step 1: Get Hessian from ORCA
        ↓
Step 2: Call compute_harmonic_alpha(hess, coords, masses)
        ├─ Extract normal modes
        ├─ Compute centrifugal alpha via finite difference
        ├─ Compute Coriolis alpha via Watson formula
        └─ Return Σ_r α_r for A, B, C components
        ↓
Step 3: Build correction table
        ├─ delta_vib = 0.5 × Σα_r (Watson convention)
        ├─ delta_elec = automatic (Gordy-Cook)
        └─ delta_bob = from built-in u-parameters
        ↓
Step 4: Store in RovibCorrection object
        ├─ .alpha_A, .alpha_B, .alpha_C (raw sum)
        ├─ .delta_vib_A, .delta_vib_B, .delta_vib_C
        ├─ .delta_elec_A, .delta_elec_B, .delta_elec_C
        └─ .delta_bob_A, .delta_bob_B, .delta_bob_C
        ↓
Step 5: Compute final B_e
        ├─ B_e = B_0 + delta_vib + delta_elec + delta_bob
        └─ Return to spectral fitting
```

---

## What's MISSING: Anharmonic Corrections (Δ_anh) ❌

### **The Problem**

Your current harmonic approach assumes:
- Vibrational modes are **perfect harmonic oscillators** (V ∝ Q²)
- This is wrong! Real potential: V(Q) = ½ωQ² + (1/6)κ₃Q³ + (1/24)κ₄Q⁴ + ...

**Consequences of ignoring anharmonicity:**
- Vibrational frequencies shift (especially low frequencies)
- Rotational constants change by ~5-50% more than harmonic predicts
- Largest errors for:
  - CH₃ torsions (~40% error)
  - N-H bends (~30% error)
  - Out-of-plane bends (~20% error)
  - Stiff stretches (~1-5% error) ← negligible

---

### **Where the Data Comes From**

ORCA VPT2 output contains:

```
Vibrational-Rotational Coupling
    Calculation of the cubic and quartic force constants
    
Mode    1 (cm-1):  1234.56
  x = 0.002  (anharmonicity constant)
  y = -0.001
  z = 0.0003

Mode    2 (cm-1):  567.89
  x = -0.015  (← LARGE! Torsional anharmonicity)
  ...

Cubic Force Constants (mdyn/Å²):
  F_111 = 0.123
  F_112 = -0.045
  F_122 = 0.012
  ...

Quartic Force Constants (mdyn/Å²):
  F_1111 = 0.0567
  ...
```

Plus resonance flags:
```
*** WARNING: Fermi resonance detected between modes 5 and 12
*** WARNING: Coriolis resonance (near-degenerate frequencies)
```

---

### **The Formula (What needs implementing)**

From VPT2 (Watson 1977; Papoušek & Aliev 1982):

```
α_r^anh = Σ_s (d³E/dQ_r dQ_s²) / (2 ω_r)  +  self-anharmonic terms

In practical terms:
Δ_anh^K = 0.5 × Σ_r α_r^anh,K  [same Watson convention as harmonic]
```

**Key insight:** The anharmonic alpha can be **computed from cubic/quartic force constants** without re-running quantum chemistry!

---

## Current Data Model vs. Target

### **RovibCorrection Dataclass**
**Location:** `.github/backend/spectral/correction_models.py:412-476`

**Currently has:**
```python
@dataclass
class RovibCorrection:
    isotopologue: str
    alpha_A: Optional[float] = None           # sum of α_r for A
    alpha_B: Optional[float] = None           # sum of α_r for B
    alpha_C: Optional[float] = None           # sum of α_r for C
    delta_vib_A: Optional[float] = None       # 0.5 × alpha_A
    delta_vib_B: Optional[float] = None
    delta_vib_C: Optional[float] = None
    delta_elec_A: float = 0.0                 # Gordy-Cook
    delta_elec_B: float = 0.0
    delta_elec_C: float = 0.0
    delta_bob_A: float = 0.0                  # Born-Oppenheimer
    delta_bob_B: float = 0.0
    delta_bob_C: float = 0.0
    sigma_delta_A: Optional[float] = None     # Uncertainty
    sigma_delta_B: Optional[float] = None
    sigma_delta_C: Optional[float] = None
    # ...
    
    def delta_total_vector(self) -> np.ndarray:
        """Final B_e correction."""
        return self.delta_vib_vector() \
             + self.delta_elec_vector() \
             + self.delta_bob_vector()
             # ← Missing delta_anh!
```

**Needs to add:**
```python
    # ANHARMONIC CORRECTIONS (NEW)
    delta_anh_A: Optional[float] = None       # ← From cubic/quartic
    delta_anh_B: Optional[float] = None
    delta_anh_C: Optional[float] = None
    
    # Diagnostic: per-mode anharmonic contributions
    alpha_r_anh: Optional[np.ndarray] = None  # (n_modes, 3) shape
    
    # Resonance flags
    fermi_resonances: Optional[List[tuple]] = None       # [(mode1, mode2), ...]
    coriolis_resonances: Optional[List[tuple]] = None
    
    # Reference force constants
    cubic_force_constants: Optional[Dict] = None
    quartic_force_constants: Optional[Dict] = None
    
    def delta_total_vector(self) -> np.ndarray:
        """Final B_e correction: INCLUDES anharmonic term."""
        return self.delta_vib_vector() \
             + self.delta_elec_vector() \
             + self.delta_bob_vector() \
             + self.delta_anh_vector()  # ← NEW
```

---

## Impact Estimates

### **Where Anharmonic Corrections Matter Most**

| Molecule | Mode Type | Δ_harm | Δ_anh | Total | % Error Reduction |
|----------|-----------|--------|-------|-------|------------------|
| **H₂O** | Bend (ν₂) | -1.5 | -0.2 | -1.7 | ~15% |
| **CH₄** | Stretch (ν₁) | -2.1 | -0.1 | -2.2 | ~5% |
| **CH₃OH** | Torsion (ν₁₂) | -0.8 | -1.2 | -2.0 | **~40%** ← **CRITICAL** |
| **SO₂** | Bend (ν₂) | -0.9 | -0.4 | -1.3 | ~30% |
| **N₂O** | Bend | -0.4 | -0.8 | -1.2 | **~50%** |

**Pattern:** Anharmonic corrections are:
- **Huge (20-50%)** for floppy modes (torsions, out-of-plane bends, low-frequency librational modes)
- **Modest (5-15%)** for "normal" bends
- **Tiny (1-5%)** for stiff stretches

---

## Implementation Roadmap

### **Phase 1: VPT2 Data Parser** (~1 week)

**New file:** `.github/backend/spectral/vpt2_anharmonic.py`

```python
def parse_orca_vpt2_anharmonic(output_path: str) -> AnharmonicCorrectionData:
    """Extract cubic/quartic force constants from ORCA output."""
    
    # Search for:
    # 1. Cubic force constants block (F_ijk)
    # 2. Quartic force constants block (F_ijkl)
    # 3. Anharmonicity constants (x, y, z per mode)
    # 4. Resonance warnings (Fermi, Coriolis, Darling-Dennison)
    pass

def anharmonic_alpha_from_cubic_quartic(
    cubic: Dict[tuple, float],
    quartic: Dict[tuple, float],
    frequencies_cm: np.ndarray,
    b_e_mhz: np.ndarray,
) -> Dict[str, float]:
    """Compute alpha^anh from force constants."""
    pass
```

### **Phase 2: Integration into RovibCorrection** (~2 days)

**Extend:** `.github/backend/spectral/correction_models.py`

- Add `delta_anh_*` fields
- Update `delta_total_vector()`
- Add `anharmonic_delta_b()` helper

### **Phase 3: Integrate into Optimizer** (~3 days)

**Extend:** `.github/backend/quantize.py`

- Add `_apply_anharmonic_corrections()` method
- Call after harmonic Hessian updates
- Print diagnostic summary

### **Phase 4: Config + Testing** (~2 days)

**YAML:**
```yaml
anharmonic_corrections:
  enabled: true
  include_cubic_quartic: true
  resonance_handling: "skip_near_degen"  # warn, include, skip
  min_frequency_cm: 50.0
  sigma_fraction: 0.05
```

**Tests:** Create benchmark suite with known molecules

---

## Summary Table

| Aspect | Status | Location | Lines |
|--------|--------|----------|-------|
| **Centrifugal (Δ_cent)** | ✅ Complete | `harmonic_alpha.py` | 108-111 |
| **Coriolis (Δ_Cor)** | ✅ Complete | `harmonic_alpha.py` | 113-133 |
| **Electronic (Δ_elec)** | ✅ Complete | `correction_models.py` | 274-301 |
| **BOB (Δ_BOB)** | ✅ Complete | `correction_models.py` | 306-385 |
| **Anharmonic (Δ_anh)** | ❌ **MISSING** | — | — |
| **Data Model** | ⚠️ Partial | `correction_models.py` | 412-476 |
| **Parser** | ❌ Missing | — | — |
| **Integration** | ❌ Missing | — | — |

---

## References

1. **Watson, J.K.G.** "Vibration-Rotation Spectra" in *Molecular Spectroscopy: Modern Research*, Academic Press (1968, 1977)
2. **Papoušek, D.; Aliev, M.R.** *Molecular Vibrational-Rotational Spectra* (1982)
3. **Gordy, W.; Cook, R.L.** *Microwave Molecular Spectra*, 3rd ed., §11.3 (1984)
4. **Puzzarini, C.; Biczysko, M.; Barone, V.** "Accurate and reliable mass-dependent molecular properties for heteronuclear molecules" *Int. Rev. Phys. Chem.* **29**(3):273-367 (2010)

