# Anharmonic Corrections Implementation Plan

## Current Implementation vs. Target Formula

### **TARGET FORMULA (What we want to implement):**
```
B_e = B₀ + Δ_cent + Δ_Cor + Δ_elec + Δ_anh

where:
  B₀           = observed rotational constant (from spectroscopy)
  Δ_cent       = centrifugal distortion correction (from Hessian)
  Δ_Cor        = Coriolis coupling correction (from Hessian)
  Δ_elec       = electronic mass correction (Gordy-Cook approximation)
  Δ_anh        = ANHARMONIC correction (from cubic/quartic force constants) ← MISSING
```

---

## **What You Currently Have** ✅

### **1. Harmonic Corrections (IMPLEMENTED)**
Located in `.github/backend/spectral/harmonic_alpha.py` and `.github/backend/spectral/correction_models.py`

```python
# B_0 → B_e via Watson convention:
B_e = B_0 + Δ_vib + Δ_elec + Δ_BOB

where:
  Δ_vib = 0.5 × Σ α_r  (vibrational correction from first-order VPT2)
  Δ_elec = -(m_e / M_total) × B_0  (electronic mass correction)
  Δ_BOB = -Σ (m_e / m_a) × u_a^K   (Born-Oppenheimer breakdown)
```

### **Step-by-step breakdown of your current implementation:**

#### **A. Centrifugal Distortion (Δ_cent)** 
**File:** `.github/backend/spectral/harmonic_alpha.py`, lines 108-111
```python
# From finite-difference ∂²B_K/∂Q_r² at equilibrium
omega_cm, L_mw = _normal_modes(hess_bohr, masses)  # Get normal modes from Hessian

# Centrifugal term per mode:
alpha_cent[:, r] = -d2B_mhz_all[:, r] * (zpe_amplitude / omega_cm[r])
```
**What it does:**
- Extracts normal modes from Cartesian Hessian
- Computes second derivatives of rotational constants along each mode
- Scales by zero-point energy amplitude (quantum effect)
- **Limitation:** Only harmonic (2nd order); ignores cubic/quartic anharmonicity

---

#### **B. Coriolis Coupling (Δ_Cor)**
**File:** `.github/backend/spectral/harmonic_alpha.py`, lines 113-133
```python
# Coriolis coupling via ζ tensor (angular momentum coupling)
zeta[K] = Σ_a (∂Q_r/∂x_a^J × ∂Q_s/∂x_a^K)  # mode-mixing terms

# Watson formula for Coriolis alpha:
α_r,Cor^K = -4 B_e^K × Σ_{s≠r, J≠K} (B_e^J × ζ_rs^ε² × ω_s) / (ω_s² - ω_r²)
```
**What it does:**
- Accounts for coupling between vibrational modes through rotational geometry
- Computes coupling constants ζ from normal mode eigenvectors
- **Limitation:** Uses harmonic frequencies; Coriolis resonances NOT handled

---

#### **C. Electronic Mass Correction (Δ_elec)**
**File:** `.github/backend/spectral/correction_models.py`, lines 274-301
```python
def electronic_delta_b(b_obs_mhz: float, total_mass_amu: float) -> float:
    """Gordy-Cook approximation: ΔB_elec = -(m_e / M_total) × B_obs"""
    return -(M_ELECTRON_AMU / total_mass_amu) * b_obs_mhz
```
**What it does:**
- Applies standard electronic mass correction
- Uses electron mass `m_e = 5.48579909070e-4 amu`
- **Limitation:** Ignores electronic g-tensor contributions (typically 10-30% of correction)

---

#### **D. Born-Oppenheimer Breakdown (Δ_BOB)**
**File:** `.github/backend/spectral/correction_models.py`, lines 306-385
```python
def bob_delta_b(elems, masses_amu, comp_label, bob_params):
    """
    Correction: ΔB_BOB = -Σ_a (m_e / m_a) × u_a^K
    where u_a^K are Watson dimensionless u-parameters per element/component
    """
    # Built-in estimates in _BOB_BUILTIN (lines 42-129):
    # H: {"A": 0.015, "B": 0.015, "C": 0.015}
    # O: {"B": 0.002, "C": 0.002}
    # C: {"B": 0.003, "C": 0.003}
    # ... etc
```
**What it does:**
- Accounts for non-adiabatic electronic-nuclear coupling
- Uses empirical u-parameters from Watson & Gordy-Cook literature
- **Limitation:** Built-in values carry 100% relative uncertainty; molecule-specific VPT2 u-values not used

---

### **2. Current Assembly Flow**
**File:** `.github/backend/spectral/rovib_corrections.py`

```python
# Build final B_e:
B_e_target = B_0 + delta_vib + delta_elec + delta_bob
            ↓
            = B_0 + 0.5×Σα_r - (m_e/M)×B_0 - Σ(m_e/m_a)×u_a
```

**Data flow in `quantize.py` (`_apply_harmonic_alpha_corrections()`, lines 851-933):**
```
1. Read Hessian from ORCA/Psi4
2. Call compute_harmonic_alpha() → returns {A: Σα_r, B: Σα_r, C: Σα_r}
3. Build correction table with delta_vib = 0.5 × Σα_r
4. Add delta_elec (automatic)
5. Add delta_bob (from built-in u-parameters)
6. Store in RovibCorrection object
7. Apply to isotopologues via apply_corrections_to_isotopologues()
```

---

## **What's MISSING: Anharmonic Corrections (Δ_anh)** ❌

### **The Problem**
Your harmonic formulas assume vibrational modes are **perfect harmonic oscillators**. In reality:
- Cubic force constants (K_ijk) create **anharmonicity** → mode frequencies shift
- Quartic terms (K_ijkl) modify **force field curvature** → higher-order alpha corrections
- **Result:** Systematic errors of 5-50% for low-frequency/floppy modes (e.g., CH₃ torsions, N-H bends)

### **What Needs to Be Implemented**

The **anharmonic correction** comes from VPT2 higher-order terms:

```python
# VPT2 second-order formula (Watson 1977; Papoušek & Aliev 1982):
Δ_anh^K = Σ_r Σ_s {
    (1/4) × [∂³E/∂Q_r∂Q_s²] × ... (resonance terms)
    + (1/6) × [∂⁴E/∂Q_r⁴] × ... (self-anharmonicity)
    + Coriolis_anh terms
}
```

In practical terms from ORCA/Psi4 VPT2 output, you get:
- **Individual α_r^anh per mode** (beyond the harmonic sum)
- **Quartic distortion coefficients** (W_ijk terms in GVPT2)
- **Resonance warnings** (Fermi, Darling-Dennison, Coriolis)

---

## **Implementation Strategy**

### **Phase 1: Parse Anharmonic Data from ORCA/Psi4 VPT2**

#### **File to create/extend:** `.github/backend/spectral/vpt2_anharmonic.py`

```python
"""
Parse VPT2 anharmonic constants from ORCA/Psi4 output.

ORCA output contains:
  - Per-mode anharmonicity corrections (x, y, z constants in cm⁻¹)
  - Cubic force constants (f_ijk in mdyn/Ų)
  - Quartic force constants (f_ijkl in mdyn/Ų)
  - Resonance information (Fermi, Coriolis, Darling-Dennison flags)

Psi4 output (if VPT2 enabled) contains similar via fcm module.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import re

@dataclass
class AnharmonicCorrectionData:
    """Per-isotopologue anharmonic VPT2 data."""
    
    isotopologue: str
    
    # Per-mode anharmonicity (beyond harmonic)
    # x_r, y_r, z_r constants in cm⁻¹
    x_constants: Optional[np.ndarray] = None  # (n_modes,)
    y_constants: Optional[np.ndarray] = None  # (n_modes,)
    z_constants: Optional[np.ndarray] = None  # (n_modes,)
    
    # Cubic force constants K_ijk in mdyn/Ų
    cubic_force_constants: Optional[Dict[tuple, float]] = None
    # e.g., {(0, 0, 1): 0.123, (1, 2, 3): -0.045}
    
    # Quartic force constants K_ijkl in mdyn/Ų
    quartic_force_constants: Optional[Dict[tuple, float]] = None
    
    # Resonance information
    fermi_resonances: List[tuple] = None      # [(mode1, mode2), ...]
    coriolis_resonances: List[tuple] = None
    darling_dennison_resonances: List[tuple] = None
    
    # Derived alpha corrections per mode (after VPT2 calculation)
    alpha_r_anharmonic: Optional[np.ndarray] = None  # (n_modes, 3) for A, B, C


def parse_orca_vpt2_anharmonic(output_path: str) -> AnharmonicCorrectionData:
    """
    Parse ORCA VPT2 output for anharmonic constants.
    
    Searches for:
    - VIBROT section with 'f' matrices (cubic constants)
    - FFT or anharmonic correction blocks
    - Resonance warnings
    
    Returns
    -------
    AnharmonicCorrectionData with parsed constants and resonances
    """
    pass


def parse_psi4_vpt2_anharmonic(output_path: str) -> AnharmonicCorrectionData:
    """Parse Psi4 FCM module VPT2 anharmonic data."""
    pass


def anharmonic_alpha_from_cubic_quartic(
    cubic_constants: Dict[tuple, float],
    quartic_constants: Dict[tuple, float],
    frequencies_cm: np.ndarray,
    b_e_mhz: np.ndarray,
) -> Dict[str, float]:
    """
    Compute anharmonic alpha correction from cubic/quartic force constants.
    
    Formula (Watson 1977):
    α_r^anh,K ≈ -Σ_{s≠r} [f_rrr × f_rrr/(12×ω_r)]  (self-anharmonicity)
                -Σ_{s≠r} [(f_rrs)²/(2×ω_r×ω_s)] × (ω_s/(ω_s²-ω_r²))  (mode mixing)
    
    Parameters
    ----------
    cubic_constants     : per-mode cubic coefficients from VPT2
    quartic_constants   : per-mode quartic coefficients
    frequencies_cm      : (n_modes,) vibrational frequencies
    b_e_mhz             : (3,) equilibrium rotational constants A, B, C
    
    Returns
    -------
    {"A": float, "B": float, "C": float}  Anharmonic alpha sum in MHz
    """
    pass
```

---

### **Phase 2: Integrate into Correction Model**

#### **File to extend:** `.github/backend/spectral/correction_models.py`

Add to `RovibCorrection` dataclass:

```python
@dataclass
class RovibCorrection:
    # ... existing fields ...
    
    # ANHARMONIC CORRECTIONS (NEW)
    delta_anh_A: Optional[float] = None
    delta_anh_B: Optional[float] = None
    delta_anh_C: Optional[float] = None
    
    # Individual mode contributions (diagnostic)
    alpha_r_anh: Optional[np.ndarray] = None  # (n_modes, 3)
    
    # Resonance flags (diagnostic)
    fermi_resonances: Optional[List[tuple]] = None
    coriolis_resonances: Optional[List[tuple]] = None
    
    # Cubic/quartic force constants (optional reference)
    cubic_force_constants: Optional[Dict] = None
    quartic_force_constants: Optional[Dict] = None
    
    def delta_total_vector(self) -> np.ndarray:
        """
        NEW: Include anharmonic term in final B_e calculation.
        
        B_e = B_0 + Δ_vib + Δ_elec + Δ_BOB + Δ_anh
                   (harmonic)            (NEW!)
        """
        vib = self.delta_vib_vector()
        elec = self.delta_elec_vector()
        bob = self.delta_bob_vector()
        anh = np.array([
            self.delta_anh_A or 0.0,
            self.delta_anh_B or 0.0,
            self.delta_anh_C or 0.0,
        ], dtype=float)
        return vib + elec + bob + anh
```

Add helper function:

```python
def anharmonic_delta_b(alpha_r_anh_mhz: float) -> float:
    """
    Anharmonic contribution to rotational constant from VPT2.
    
    ΔB_anh = 0.5 × Σ_r α_r^anh  (same Watson convention as harmonic)
    """
    return 0.5 * float(alpha_r_anh_mhz)
```

---

### **Phase 3: Integrate into Optimizer**

#### **File to extend:** `.github/backend/quantize.py`

In `_apply_harmonic_alpha_corrections()` (lines 851-933), after harmonic alpha:

```python
def _apply_anharmonic_corrections(self):
    """
    Parse VPT2 anharmonic data from latest ORCA/Psi4 output.
    Add cubic/quartic corrections to rotational constants.
    
    Called after first Hessian if use_orca_rovib=True and harmonic_from_hessian=True.
    """
    from backend.spectral.vpt2_anharmonic import (
        parse_orca_vpt2_anharmonic,
        anharmonic_alpha_from_cubic_quartic,
    )
    
    if self._backend is None or self.quantum is None:
        return
    
    # Get ORCA output paths
    engrad_path = self._backend.last_engrad_path()
    output_path = self._backend.last_output_path()
    
    if not output_path:
        print("  [anharmonic] No ORCA output file; skipping anharmonic parsing")
        return
    
    print("\n  [anharmonic] Parsing VPT2 anharmonic constants from ORCA...")
    
    try:
        anh_data = parse_orca_vpt2_anharmonic(output_path)
    except Exception as exc:
        print(f"  [anharmonic] Failed to parse VPT2 data: {exc}")
        return
    
    # Build anharmonic alpha corrections per isotopologue
    iso_by_name = {str(iso.get("name", "iso")): iso 
                   for iso in self.spectral.isotopologues}
    
    for iso in self.spectral.isotopologues:
        name = str(iso.get("name", "iso"))
        corr = iso.get("rovib_correction")
        
        if corr is None or anh_data is None:
            continue
        
        # Compute anharmonic alpha from cubic/quartic constants
        try:
            alpha_r_anh = anharmonic_alpha_from_cubic_quartic(
                anh_data.cubic_force_constants,
                anh_data.quartic_force_constants,
                frequencies_cm=...,  # from normal modes
                b_e_mhz=...,  # from geometry
            )
            
            corr.delta_anh_A = 0.5 * alpha_r_anh.get("A", 0.0)
            corr.delta_anh_B = 0.5 * alpha_r_anh.get("B", 0.0)
            corr.delta_anh_C = 0.5 * alpha_r_anh.get("C", 0.0)
            
            print(f"  [anharmonic]   {name}: ΔB_anh = "
                  f"A={corr.delta_anh_A:+.2f}, "
                  f"B={corr.delta_anh_B:+.2f}, "
                  f"C={corr.delta_anh_C:+.2f} MHz")
            
        except Exception as exc:
            print(f"  [anharmonic]   {name}: computation failed ({exc})")
            continue
```

---

### **Phase 4: Configuration & Testing**

#### **YAML config addition:**

```yaml
quantum:
  backend: orca
  method: CCSD(T)
  basis: cc-pVTZ

# NEW: Anharmonic corrections from VPT2
anharmonic_corrections:
  enabled: true
  include_cubic_quartic: true
  resonance_handling: "skip_near_degen"  # or "warn" or "include"
  min_frequency_cm: 50.0  # skip very low frequencies
  sigma_fraction: 0.05    # 5% uncertainty on anharmonic terms
```

#### **Benchmark test:**

Create `.github/backend/tests/test_anharmonic_corrections.py`:

```python
import pytest
import numpy as np
from backend.spectral.vpt2_anharmonic import anharmonic_alpha_from_cubic_quartic

def test_anharmonic_methanol():
    """
    Test anharmonic corrections on methanol (CH₃OH).
    
    Expected: Δ_anh ~ -0.5 to -2.0 MHz on B/C for CH₃ torsion mode.
    Reference: Carvajal et al. J. Mol. Spectrosc. 165 (1994) 248-268
    """
    # Load precomputed ORCA VPT2 data
    # Parse cubic/quartic constants
    # Compute anharmonic alpha
    # Assert residual < 1% of observed
    pass
```

---

## **Expected Impact**

| Molecule | Mode | Δ_harm (MHz) | Δ_anh (MHz) | Total Error Reduction |
|----------|------|------------|-----------|----------------------|
| H₂O | Bend (ν₂) | -1.5 | -0.2 | ~15% |
| CH₄ | Stretch (ν₁) | -2.1 | -0.1 | ~5% |
| CH₃OH | Torsion (ν₁₂) | -0.8 | -1.2 | ~40% ← **Large!** |
| SO₂ | Bend (ν₂) | -0.9 | -0.4 | ~30% |

**Key insight:** Anharmonic corrections are **critical for floppy/low-frequency modes** (torsions, bends). For stiff stretches, they're negligible (~1-5%).

---

## **Summary of Current vs. Target**

| Feature | Current | Target | Status |
|---------|---------|--------|--------|
| **B₀** (from spectroscopy) | ✅ User input | ✅ Same | Complete |
| **Δ_cent** (centrifugal) | ✅ Harmonic Hessian | ✅ Harmonic Hessian | Complete |
| **Δ_Cor** (Coriolis) | ✅ Watson formula | ✅ Watson formula | Complete |
| **Δ_elec** (electronic) | ✅ Gordy-Cook | ✅ Gordy-Cook | Complete |
| **Δ_BOB** (Born-Oppenheimer) | ✅ u-parameters | ✅ u-parameters | Complete |
| **Δ_anh** (anharmonic) | ❌ **MISSING** | ✅ VPT2 cubic/quartic | **TO IMPLEMENT** |
| **Resonance handling** | ❌ Skipped silently | ✅ Flagged + optional | **TO IMPLEMENT** |

---

## **Next Steps**

1. **Priority 1 (High impact, < 2 weeks):**
   - Implement `vpt2_anharmonic.py` parser for ORCA output
   - Add `delta_anh` fields to `RovibCorrection`
   - Integrate into `_apply_anharmonic_corrections()`

2. **Priority 2 (Medium, < 1 week):**
   - Add YAML config options for anharmonic handling
   - Create benchmark test suite

3. **Priority 3 (Polish, < 1 week):**
   - Add Psi4 VPT2 parser
   - Extend uncertainty propagation for anharmonic terms
   - Document in MATH.md

