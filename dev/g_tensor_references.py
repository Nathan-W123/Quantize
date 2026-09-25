"""Measured rotational g-tensor values, for validating the electronic correction.

The engine's electronic correction is

    delta_elec = -(m_e / m_p) * g_alpha * B_obs

and until now exactly one number in this repository tested the g_alpha behind
it: g_perp ~ -0.028 for OCS, quoted in a code comment without a citation. One
datum cannot distinguish a right implementation from a wrong one that happens
to land near a small negative number, and it certainly cannot test the sign,
the anisotropy, the axis labelling or the mass dependence.

This is 59 measured components across 20 molecules and isotopologues, spanning
four orders of magnitude (|g| from 0.019 to 5.26) and both signs, with three
isotopologue pairs. Every value is quoted as printed in its source alongside the
parsed float, so a transcription error is visible rather than silent -- the
failure mode this repository has already been bitten by once, in the
centrifugal-distortion coefficient table.

Conventions that matter
-----------------------
A g tensor is a molecular property, not a fitted Hamiltonian parameter, so there
is no A/S reduction of it and no I^r/III^r ambiguity. What does matter:

1. UNITS. Every value here is dimensionless in NUCLEAR MAGNETON units, so the
   rotational magnetic moment is mu_g = g_gg * mu_N * J_g. Bohr-magneton values
   would differ by a factor of 1836.
2. SIGN. Several sources print a magnitude and argue the sign in the text; those
   are recorded with the sign the authors assign, and the printed form is kept
   so the distinction survives.
3. AXES. a/b/c are the principal inertial axes with I_a <= I_b <= I_c, which
   maps directly onto this engine's A/B/C ordering. Linear and symmetric-top
   sources report g_perp and g_par instead, recorded as "perp" and "par".
4. VIBRATIONAL STATE. All v=0, so these are ground-state effective values. A
   CPHF calculation at an equilibrium geometry gives the equilibrium g, and the
   two differ by a vibrational correction that is not included here. That sets
   the floor on how well any calculation can be expected to agree.
5. J DEPENDENCE. Formaldehyde's values are averaged over three rotational
   states; the source notes a small J dependence.

Provenance
----------
Retrieved through the Crossref REST API, which returns deposited abstracts
containing the numeric tables for older AIP and De Gruyter papers that are
otherwise paywalled. Confidence is "high" where the number was read in an
accessible source and "medium" where it came from a secondary compilation.

NOT_VERIFIED records what was looked for and could not be read. It is not a
to-do list to be filled in from memory: H2O's parent g values in particular are
frequently misquoted, and only D2O could be verified here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GReference:
    """One measured g-tensor component, with everything needed to compare it."""

    molecule: str
    #: "A", "B", "C" for an asymmetric top; "perp"/"par" for linear and
    #: symmetric tops.
    component: str
    #: Signed dimensionless value in nuclear magneton units.
    value: float
    sigma: float
    #: Exactly as printed in the source, including any sign caveat.
    printed: str
    source: str
    convention: str
    confidence: str
    #: The source's own label for the quantity.
    quantity: str


#: Every verified component. Generated from the retrieval record rather than
#: typed, so the printed and parsed forms cannot disagree.
G_REFERENCES = [
    GReference(
        molecule='Carbonyl sulfide, OCS',
        component='perp',
        value=-0.028711,
        sigma=4e-05,
        printed='− 0.028711 ± 0.00004',
        source="W. H. Flygare, W. Huettner, R. L. Shoemaker, P. D. Foster, 'Magnetic Susceptibility Anisotropy, Molecular Quadrupole Moment, and the Sign of the Electric Dipole Moment in OCS', J. Chem. Phys. 50, 1714-1719 (1969). DOI 10.1063/1.1671264. Value read verbatim in the publisher-deposited abstract.",
        convention='Linear molecule: only g_perp exists; g along the internuclear axis is zero, so in an A/B/C dict B = C = g_perp and A is undefined/zero. Reduction (A vs S) and representation (Ir/IIIr) are NOT applicable to a g tensor — they only apply to centrifugal distortion constants. Sign: negative, determined together with the -OCS+ dipole polarity. Ground vibrational state (v=0) effective value from J = 0 -> 1 transitions in fields up to 30 000 G. THIS IS THE SOURCE FOR THE ~-0.028 NUMBER ALREADY IN THE REPO.',
        confidence='high',
        quantity='g_perp (= g_bb = g_cc)',
    ),
    GReference(
        molecule='Carbonyl sulfide, OCS',
        component='perp',
        value=-0.028127,
        sigma=4e-05,
        printed='− 0.028127 ± 0.00004',
        source='W. H. Flygare, W. Huettner, R. L. Shoemaker, P. D. Foster, J. Chem. Phys. 50, 1714-1719 (1969). DOI 10.1063/1.1671264.',
        convention="Same as the 32S species. The 32S/34S pair is a direct test of the mass dependence of the g tensor (g scales with 1/I), which is exactly the isotopologue handling in the repo's g_tensors_for_isotopologues. Ratio 0.028711/0.028127 = 1.0208.",
        confidence='high',
        quantity='g_perp (= g_bb = g_cc)',
    ),
    GReference(
        molecule='Sulfur dioxide, SO2',
        component='A',
        value=-0.6037,
        sigma=0.0005,
        printed='− (0.6037 ± 0.0005)',
        source="J. M. Pochan, R. G. Stone, W. H. Flygare, 'Molecular g Values, Magnetic Susceptibilities, Molecular Quadrupole Moments, and Second Moments of the Electronic Charge Distribution in OF2, O3, and SO2', J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790. Table read verbatim in the publisher-deposited abstract.",
        convention="Principal inertial axes, I_a <= I_b <= I_c. Stated verbatim: 'The b axis bisects the interatomic angle and the a axis is also in the molecular plane' — so b is the C2 axis and c is perpendicular to the molecular plane. Experiment gives absolute values of g_aa, g_bb, g_cc plus signs/magnitudes of the susceptibility anisotropies; 'Arguments are presented to show that all of the molecular g values are negative'. v=0 effective values, high-resolution microwave Zeeman.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Sulfur dioxide, SO2',
        component='B',
        value=-0.1161,
        sigma=0.0002,
        printed='− (0.1161 ± 0.0002)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention="b = C2 axis (bisects the O-S-O angle). Signs all negative per the authors' argument. v=0.",
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Sulfur dioxide, SO2',
        component='C',
        value=-0.0882,
        sigma=0.0004,
        printed='− (0.0882 ± 0.0004)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention="c = out-of-plane axis. Signs all negative per the authors' argument. v=0.",
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Ozone, O3',
        component='A',
        value=-2.968,
        sigma=0.035,
        printed='− (2.968 ± 0.035)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention='Principal inertial axes; b bisects the O-O-O angle (C2 axis), a also in plane, c out of plane. All g values assigned negative by the authors. v=0. NOTE: independently measured as 2.98933(8) by Mack & Muenter (next entries) but with the OPPOSITE PRINTED SIGN — see caveats.',
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Ozone, O3',
        component='B',
        value=-0.228,
        sigma=0.007,
        printed='− (0.228 ± 0.007)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention='b = C2 axis. All negative. v=0. Independent check: -0.22919(3) from Mack & Muenter.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Ozone, O3',
        component='C',
        value=-0.081,
        sigma=0.006,
        printed='− (0.081 ± 0.006)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention='c = out-of-plane axis. All negative. v=0. Independent check: -0.07623 from Mack & Muenter.',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Ozone, O3',
        component='A',
        value=2.98933,
        sigma=8e-05,
        printed='2.98933(8)  [printed WITHOUT a minus sign in the source abstract]',
        source="K. M. Mack, J. S. Muenter, 'Stark and Zeeman properties of ozone from molecular beam spectroscopy', J. Chem. Phys. 66, 5278-5283 (1977). DOI 10.1063/1.433909. Read verbatim in the publisher-deposited abstract.",
        convention="Molecular beam electric resonance, Stark and Stark-Zeeman transitions within single J states. The abstract calls these 'rotational magnetic moments, ga, gb, gc'. The printed sign of g_aa is POSITIVE here while g_bb and g_cc are printed negative, and while Pochan et al. give g_aa negative — so either a different sign convention or a typo. Use the magnitude and take the sign from Pochan et al. / from formaldehyde-like reasoning (g_aa < 0). Precision here is ~400x better than Pochan et al., so this is the number to benchmark against once the sign is settled.",
        confidence='high',
        quantity='g_aa (independent measurement)',
    ),
    GReference(
        molecule='Ozone, O3',
        component='B',
        value=-0.22919,
        sigma=3.0000000000000004e-05,
        printed='−0.22919(3)',
        source='K. M. Mack, J. S. Muenter, J. Chem. Phys. 66, 5278-5283 (1977). DOI 10.1063/1.433909.',
        convention="Molecular beam electric resonance. Same axis convention (b = C2). Agrees with Pochan's -0.228(7) within its error bar.",
        confidence='high',
        quantity='g_bb (independent measurement)',
    ),
    GReference(
        molecule='Ozone, O3',
        component='C',
        value=-0.07623,
        sigma=3.0,
        printed="−0.07623(b)  [uncertainty digit is literally printed as '(b)' — corrupted]",
        source='K. M. Mack, J. S. Muenter, J. Chem. Phys. 66, 5278-5283 (1977). DOI 10.1063/1.433909.',
        convention="Molecular beam electric resonance. c = out-of-plane. Value agrees with Pochan's -0.081(6).",
        confidence='high',
        quantity='g_cc (independent measurement)',
    ),
    GReference(
        molecule='Oxygen difluoride, OF2',
        component='A',
        value=-0.213,
        sigma=0.005,
        printed='− (0.213 ± 0.005)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention='Principal inertial axes; b bisects the F-O-F angle, a also in plane, c out of plane. All g values assigned negative. v=0. A cheap 3-atom FLUORINATED test case.',
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Oxygen difluoride, OF2',
        component='B',
        value=-0.058,
        sigma=0.002,
        printed='− (0.058 ± 0.002)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention='b bisects the F-O-F angle. All negative. v=0. NOTE |g_bb| < |g_cc| here, an ordering inversion relative to SO2/O3 — a good test that you are not accidentally sorting g components by magnitude instead of by axis.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Oxygen difluoride, OF2',
        component='C',
        value=-0.068,
        sigma=0.002,
        printed='− (0.068 ± 0.002)',
        source='J. M. Pochan, R. G. Stone, W. H. Flygare, J. Chem. Phys. 51, 4278-4286 (1969). DOI 10.1063/1.1671790.',
        convention='c = out-of-plane. All negative. v=0.',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Formaldehyde, H2CO',
        component='A',
        value=-2.899,
        sigma=0.002,
        printed='− 2.899 ± 0.002',
        source="W. Huettner, Mei-Kuo Lo, W. H. Flygare, 'Molecular g-Value Tensor, the Molecular Susceptibility Tensor, and the Sign of the Electric Dipole Moment in Formaldehyde', J. Chem. Phys. 48, 1206-1220 (1968). DOI 10.1063/1.1668783. Read verbatim in the publisher-deposited abstract.",
        convention="Principal inertial axis system. a is the C=O (dipole) axis, c is perpendicular to the molecular plane. IMPORTANT: 'A small J dependence was observed in the molecular g values where the absolute values increased with rotational excitation. The g values averaged over the three rotational states studied' — so this is a J-averaged v=0 effective value, not a pure v=0, J=0 extrapolation. High-field rotational Zeeman effect. Signs uniquely determined; the isotope dependence of the g values also fixed the +C-O- dipole polarity.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Formaldehyde, H2CO',
        component='B',
        value=-0.2256,
        sigma=0.0008,
        printed='− 0.2256 ± 0.0008',
        source='W. Huettner, Mei-Kuo Lo, W. H. Flygare, J. Chem. Phys. 48, 1206-1220 (1968). DOI 10.1063/1.1668783.',
        convention='a = C=O axis, c out-of-plane. J-averaged over three rotational states, v=0.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Formaldehyde, H2CO',
        component='C',
        value=-0.1004,
        sigma=0.0007,
        printed='− 0.1004 ± 0.0007',
        source='W. Huettner, Mei-Kuo Lo, W. H. Flygare, J. Chem. Phys. 48, 1206-1220 (1968). DOI 10.1063/1.1668783.',
        convention='c = out-of-plane axis. J-averaged over three rotational states, v=0.',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Formaldehyde-d2, D2CO',
        component='A',
        value=-1.445,
        sigma=0.002,
        printed='− 1.445 ± 0.002',
        source='W. Huettner, Mei-Kuo Lo, W. H. Flygare, J. Chem. Phys. 48, 1206-1220 (1968). DOI 10.1063/1.1668783.',
        convention="Same axis convention and same J-averaging caveat as H2CO. The H2CO/D2CO pair (g_aa ratio 2.899/1.445 = 2.006) is an excellent isotopologue test — note the repo's own code comment says HF/6-31g gives water g_bb +0.682 for H2-16O vs +0.341 for D2-16O, the same near-factor-of-2 behaviour.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Formaldehyde-d2, D2CO',
        component='B',
        value=-0.1917,
        sigma=0.0005,
        printed='− 0.1917 ± 0.0005',
        source='W. Huettner, Mei-Kuo Lo, W. H. Flygare, J. Chem. Phys. 48, 1206-1220 (1968). DOI 10.1063/1.1668783.',
        convention='a = C=O axis, c out-of-plane. J-averaged, v=0.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Formaldehyde-d2, D2CO',
        component='C',
        value=-0.0788,
        sigma=0.0004,
        printed='− 0.0788 ± 0.0004',
        source='W. Huettner, Mei-Kuo Lo, W. H. Flygare, J. Chem. Phys. 48, 1206-1220 (1968). DOI 10.1063/1.1668783.',
        convention='c = out-of-plane axis. J-averaged, v=0.',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Formaldehyde, H2CO (earlier independent measurement)',
        component='A',
        value=-2.935,
        sigma=0.03,
        printed='−2.935 ± 0.03',
        source="W. H. Flygare, 'Molecular Magnetic Moments and Susceptibility in Formaldehyde', J. Chem. Phys. 42, 1563-1568 (1965). DOI 10.1063/1.1696162. Read verbatim in the publisher-deposited abstract.",
        convention="Earlier, lower-precision measurement by the same group: 'The magnitudes and signs of the diagonal elements of the g tensor were determined.' Also reports g_bb = −0.17 ± 0.05, g_cc = −0.11 ± 0.05 and |g_bb − g_cc| = 0.08 ± 0.03. Consistent with the 1968 values within its larger error bars; useful only as an independent sanity check, not as the primary reference.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Water-d2, D2O',
        component='A',
        value=0.3253,
        sigma=0.0001,
        printed='Gaa = 0.32530(10)',
        source="J. Verhoeven, A. Dymanus, 'Magnetic Properties and Molecular Quadrupole Tensor of the Water Molecule by Beam-Maser Zeeman Spectroscopy', J. Chem. Phys. 52, 3222-3233 (1970). DOI 10.1063/1.1673462. Read verbatim in the publisher-deposited abstract.",
        convention="CAUTION ON NOTATION: the source writes 'the elements of the magnetic moment tensor Ggg', not g_gg. In beam-maser work G_gg is the rotational magnetic moment per unit angular momentum in nuclear magnetons, i.e. numerically the rotational g value. Cross-check that this is the same quantity the code computes: the repo's own comment states HF/6-31g gives D2-16O g_bb = +0.341, against the measured G_bb = 0.36009 — same sign, same magnitude, ~5% apart, which is the expected HF error. Principal inertial axes; POSITIVE g values, unlike almost everything else in this set, which makes D2O a good sign test. Measured on the 3(-2)-2(2) and 4(3)-5(1) transitions of D2O with a beam-maser Zeeman spectrometer. v=0.",
        confidence='high',
        quantity="G_aa (the source's notation for the rotational g / magnetic moment tensor element)",
    ),
    GReference(
        molecule='Water-d2, D2O',
        component='B',
        value=0.36009,
        sigma=0.00022,
        printed='Gbb = 0.36009(22)',
        source='J. Verhoeven, A. Dymanus, J. Chem. Phys. 52, 3222-3233 (1970). DOI 10.1063/1.1673462.',
        convention='See the G-vs-g notation caution on the g_aa row. Positive. b is the C2 axis for water in the usual assignment, but the source abstract does not spell the axis directions out — confirm against the paper before mapping to A/B/C.',
        confidence='high',
        quantity='G_bb',
    ),
    GReference(
        molecule='Water-d2, D2O',
        component='C',
        value=0.32513,
        sigma=0.00015000000000000001,
        printed='Gcc = 0.32513(15)',
        source='J. Verhoeven, A. Dymanus, J. Chem. Phys. 52, 3222-3233 (1970). DOI 10.1063/1.1673462.',
        convention='See the G-vs-g notation caution. Positive. Note G_aa and G_cc are nearly equal (0.32530 vs 0.32513) while G_bb is distinctly larger — a near-degeneracy that will expose any axis-assignment bug.',
        confidence='high',
        quantity='G_cc',
    ),
    GReference(
        molecule='Methyl fluoride, CH3F',
        component='perp',
        value=-0.062,
        sigma=0.002,
        printed='g_perp = −0.062 ± 0.002',
        source="C. L. Norris, E. F. Pearson, W. H. Flygare, 'Molecular Zeeman effect in methyl fluoride', J. Chem. Phys. 60, 1758-1760 (1974). DOI 10.1063/1.1681272. Read verbatim in the publisher-deposited abstract.",
        convention='Prolate symmetric top: g_par is along the C3 axis (= the a axis, smallest moment of inertia), g_perp is the doubly degenerate perpendicular component (b = c). v=0, high-field rotational Zeeman. A cheap FLUORINATED test case with OPPOSITE-SIGN parallel and perpendicular components — very good for catching a sign or axis bug.',
        confidence='high',
        quantity='g_perp (= g_bb = g_cc)',
    ),
    GReference(
        molecule='Methyl fluoride, CH3F',
        component='par',
        value=0.265,
        sigma=0.008,
        printed='g_par = 0.265 ± 0.008',
        source='C. L. Norris, E. F. Pearson, W. H. Flygare, J. Chem. Phys. 60, 1758-1760 (1974). DOI 10.1063/1.1681272.',
        convention='Along the C3 = a axis. POSITIVE, in contrast to g_perp = −0.062. v=0.',
        confidence='high',
        quantity='g_par (= g_aa, along C3)',
    ),
    GReference(
        molecule='Nitrous oxide, N2O',
        component='perp',
        value=-0.07606,
        sigma=0.0001,
        printed='g_perp = 0.07606 ± 0.0001  (reported as an ABSOLUTE VALUE; the authors argue conclusively that the sign is NEGATIVE, so use −0.07606)',
        source="W. H. Flygare, R. L. Shoemaker, W. Huettner, 'Magnetic-Susceptibility Anisotropy, Molecular g Value, and Molecular Quadrupole Moment of 15N15N16O', J. Chem. Phys. 50, 2414-2416 (1969). DOI 10.1063/1.1671397. Read verbatim in the publisher-deposited abstract.",
        convention="Linear molecule: only g_perp. Note this is the DOUBLY 15N-SUBSTITUTED species, not the normal 14N14N16O — you must feed the code 15N masses. The abstract states the measured quantity is the absolute value and that 'Arguments are given which show conclusively that the molecular g value is negative'. v=0, high-field rotational Zeeman.",
        confidence='high',
        quantity='g_perp',
    ),
    GReference(
        molecule='Thioformaldehyde, H2CS',
        component='A',
        value=-5.2602,
        sigma=0.0068,
        printed='−5.2602 ± 0.0068',
        source="S. L. Rock, W. H. Flygare, 'Molecular Rotational Zeeman Effect in Thioformaldehyde', J. Chem. Phys. 56, 4723-4728 (1972). DOI 10.1063/1.1676945. Read verbatim in the publisher-deposited abstract.",
        convention="Stated verbatim: 'c is the out-of-plane axis and the a axis is the dipole axis'. 'The molecular g values with uniquely determined signs' — so no sign ambiguity here, unusually. 'gaa represents the largest molecular g value yet measured.' v=0, high-field rotational Zeeman. This is the extreme-magnitude end of the validation set and the hardest single number in it for an approximate method.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Thioformaldehyde, H2CS',
        component='B',
        value=-0.1337,
        sigma=0.0004,
        printed='−0.1337 ± 0.0004',
        source='S. L. Rock, W. H. Flygare, J. Chem. Phys. 56, 4723-4728 (1972). DOI 10.1063/1.1676945.',
        convention='a = dipole axis, c = out-of-plane. Signs uniquely determined. v=0.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Thioformaldehyde, H2CS',
        component='C',
        value=-0.0239,
        sigma=0.0004,
        printed='−0.0239 ± 0.0004',
        source='S. L. Rock, W. H. Flygare, J. Chem. Phys. 56, 4723-4728 (1972). DOI 10.1063/1.1676945.',
        convention='c = out-of-plane axis. Signs uniquely determined. v=0. Note the 220:1 dynamic range within a single molecule (g_aa/g_cc), which is a strong test of the CPHF implementation across the whole tensor.',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Methanimine, CH2=NH',
        component='A',
        value=-1.27099,
        sigma=0.00022,
        printed='−1.27099(22)',
        source="H. Krause, D. H. Sutter, 'The Molecular Zeeman Effect of Imines. I. Methanimine, its Molecular g-Tensor, ...', Z. Naturforsch. A 44, 1063-1078 (1989). DOI 10.1515/zna-1989-1106. Read verbatim in the publisher-deposited abstract.",
        convention="'The observed vibronic ground state expectation values of the molecular g-values' — explicitly v=0 vibronic expectation values, referred to the molecular principal inertia axes. Rotational Zeeman effect; the molecule was made from ethylenediamine by flash pyrolysis. Very high precision (6 significant figures). Cheap 5-atom molecule.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Methanimine, CH2=NH',
        component='B',
        value=-0.18975,
        sigma=7.000000000000001e-05,
        printed='−0.18975(7)',
        source='H. Krause, D. H. Sutter, Z. Naturforsch. A 44, 1063-1078 (1989). DOI 10.1515/zna-1989-1106.',
        convention='v=0 vibronic ground-state expectation value, principal inertia axes.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Methanimine, CH2=NH',
        component='C',
        value=-0.0344,
        sigma=8e-05,
        printed='−0.03440(8)',
        source='H. Krause, D. H. Sutter, Z. Naturforsch. A 44, 1063-1078 (1989). DOI 10.1515/zna-1989-1106.',
        convention='v=0 vibronic ground-state expectation value, principal inertia axes. c is the out-of-plane axis for this planar molecule (not stated explicitly in the abstract — confirm in the paper).',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Ethylene oxide (oxirane), C2H4O',
        component='A',
        value=-0.0946,
        sigma=0.0003,
        printed='− (0.0946 ± 0.0003)',
        source="D. H. Sutter, W. Huettner, W. H. Flygare, 'Molecular g Values, Magnetic Susceptibility Anisotropies, and Molecular Quadrupole Moments in Ethylene Oxide', J. Chem. Phys. 50, 2869-2874 (1969). DOI 10.1063/1.1671477. Read verbatim in the publisher-deposited abstract.",
        convention="Stated verbatim: 'The b axis bisects the COC angle and the a axis is also in the COC plane' — so c is perpendicular to the COC plane. IMPORTANT: 'Only the relative signs of the g values are experimentally determined. Arguments are presented which favor one of the two sets of signs' — the absolute signs are an inference, not a measurement. v=0, first- and second-order molecular Zeeman effect. MIXED-SIGN TENSOR (one negative, two positive) — one of the best sign tests in the set.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Ethylene oxide (oxirane), C2H4O',
        component='B',
        value=0.0189,
        sigma=0.0004,
        printed='+ (0.0189 ± 0.0004)',
        source='D. H. Sutter, W. Huettner, W. H. Flygare, J. Chem. Phys. 50, 2869-2874 (1969). DOI 10.1063/1.1671477.',
        convention='b bisects the COC angle. Only relative signs measured; absolute signs argued. v=0.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Ethylene oxide (oxirane), C2H4O',
        component='C',
        value=0.0318,
        sigma=0.0006,
        printed='+(0.0318 ± 0.0006',
        source='D. H. Sutter, W. Huettner, W. H. Flygare, J. Chem. Phys. 50, 2869-2874 (1969). DOI 10.1063/1.1671477.',
        convention="c perpendicular to the COC plane. Only relative signs measured; absolute signs argued. v=0. NOTE: the deposited abstract is missing the closing parenthesis on this value ('+(0.0318 ± 0.0006'); the digits are unambiguous but confirm in the paper.",
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Cyclopropene, C3H4',
        component='A',
        value=-0.0897,
        sigma=0.0009,
        printed='−0.0897 ± 0.0009',
        source="R. C. Benson, W. H. Flygare, 'Molecular g Values, Magnetic Susceptibilities, Molecular Quadrupole Moments, and Sign of the Electric Dipole Moment in Cyclopropene', J. Chem. Phys. 51, 3087-3096 (1969). DOI 10.1063/1.1672460. Read verbatim in the publisher-deposited abstract.",
        convention="Stated verbatim: 'The a axis is perpendicular to the double bond, and the a and b axes are in the molecular plane' — so c is out of plane. 'Although only the relative signs of the g values are determined experimentally, the above signs are conclusively assigned on the basis of the molecular quadrupole moments.' v=0, high-field rotational Zeeman. Mixed-sign tensor.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Cyclopropene, C3H4',
        component='B',
        value=-0.14915,
        sigma=0.00016,
        printed='−0.14915 ± 0.00016',
        source='R. C. Benson, W. H. Flygare, J. Chem. Phys. 51, 3087-3096 (1969). DOI 10.1063/1.1672460.',
        convention='a and b in the molecular plane, c out of plane. Relative signs measured, absolute signs assigned from quadrupole moments. v=0. Note |g_bb| > |g_aa| — another axis-ordering test.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Cyclopropene, C3H4',
        component='C',
        value=0.05363,
        sigma=0.00077,
        printed='0.05363 ± 0.00077',
        source='R. C. Benson, W. H. Flygare, J. Chem. Phys. 51, 3087-3096 (1969). DOI 10.1063/1.1672460.',
        convention='c = out-of-plane axis; POSITIVE while g_aa and g_bb are negative. Relative signs measured, absolute signs assigned. v=0. (The 1,2-dideuterocyclopropene values in the same abstract are g_aa = −0.0689 ± 0.0010, g_bb = −0.13718 ± 0.00015, g_cc = 0.4538 ± 0.00018 — but that last one is almost certainly a typo for 0.04538, so I have NOT listed the d2 species as an entry. See caveats.)',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Cyclopropane, c-C3H6',
        component='perp',
        value=0.02675,
        sigma=0.00023,
        printed='g_perp = 0.02675(23)',
        source="O. Boettcher, V. Meyer, D. H. Sutter, 'On the Validity of Additivity Rules for the Molecular Magnetizability Tensor and the Molecular g-Tensor in van der Waals Complexes. A Rotational Zeeman Effect Study of 1,1-Dideutero-Cyclopropane', Z. Naturforsch. A 49, 585-588 (1994). DOI 10.1515/zna-1994-4-510. Read verbatim in the publisher-deposited abstract.",
        convention='Oblate symmetric top of the parent: g_par is along the C3 axis perpendicular to the ring plane, g_perp lies in the ring plane. BOTH COMPONENTS POSITIVE — the only all-positive polyatomic tensor I could verify, so it is a valuable counterweight to the mostly-negative rest of the set. Determined by microwave Fourier transform study of the rotational Zeeman effect of the 1,1-d2 isotopomer. v=0.',
        confidence='high',
        quantity='g_perp',
    ),
    GReference(
        molecule='Cyclopropane, c-C3H6',
        component='par',
        value=0.06998,
        sigma=0.00023,
        printed='g_par = 0.06998(23)',
        source='O. Boettcher, V. Meyer, D. H. Sutter, Z. Naturforsch. A 49, 585-588 (1994). DOI 10.1515/zna-1994-4-510.',
        convention='Along the C3 axis (perpendicular to the ring). Positive. v=0. CAUTION: the tensor is quoted in the symmetric-top parallel/perpendicular basis of the PARENT molecule, while the measurement was on the d2 species whose principal axes are rotated/relabelled — check how the paper maps these to the d2 inertial axes before comparing to an a/b/c calculation on C3H4D2.',
        confidence='high',
        quantity='g_par',
    ),
    GReference(
        molecule='Fluorobenzene, C6H5F',
        component='A',
        value=-0.067,
        sigma=0.0008,
        printed='− (0.0670 ± 0.0008)',
        source="W. Huettner, W. H. Flygare, 'Molecular g Values, Magnetic Susceptibility Anisotropies, and Molecular Quadrupole Moments in Fluorobenzene', J. Chem. Phys. 50, 2863-2868 (1969). DOI 10.1063/1.1671476. Read verbatim in the publisher-deposited abstract.",
        convention="Stated verbatim: 'The a axis passes through the CF bond and the b axis is also in the molecular plane' — c is perpendicular to the ring. 'Only the relative signs of the g values are obtained experimentally. However, by combining our magnetic susceptibility anisotropies with the average magnetic susceptibility we can show that the signs must be as assigned above.' v=0, first- and second-order molecular Zeeman effect. FLUORINATED, mixed-sign tensor. NOTE: this measurement was superseded by a reinvestigation with ~5x better resolution (Stolze, Stolze, Huebner, Sutter, Z. Naturforsch. A 37, 1165-1175 (1982), DOI 10.1515/zna-1982-1008) whose abstract does not print the numbers — if you want the best fluorobenzene values, go to that paper.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Fluorobenzene, C6H5F',
        component='B',
        value=-0.0397,
        sigma=0.0015,
        printed='− (0.0397 ± 0.0015)',
        source='W. Huettner, W. H. Flygare, J. Chem. Phys. 50, 2863-2868 (1969). DOI 10.1063/1.1671476.',
        convention='b in the molecular plane, c perpendicular to the ring. Relative signs measured; absolute signs fixed using the bulk susceptibility. v=0. Superseded by Z. Naturforsch. A 37, 1165 (1982).',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Fluorobenzene, C6H5F',
        component='C',
        value=0.0266,
        sigma=0.0017,
        printed='+ (0.0266 ± 0.0017)',
        source='W. Huettner, W. H. Flygare, J. Chem. Phys. 50, 2863-2868 (1969). DOI 10.1063/1.1671476.',
        convention='c perpendicular to the ring; POSITIVE while the in-plane components are negative. Relative signs measured; absolute signs fixed using the bulk susceptibility. v=0. Superseded by Z. Naturforsch. A 37, 1165 (1982).',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Benzene, C6H6',
        component='A',
        value=-0.068,
        sigma=0.025,
        printed='− (0.068 ± 0.025)',
        source="R. L. Shoemaker, W. H. Flygare, 'Molecular Quadrupole Moment, Molecular Magnetic Susceptibilities, and Molecular g Values in Benzene', J. Chem. Phys. 51, 2988-2991 (1969). DOI 10.1063/1.1672447. Read verbatim in the publisher-deposited abstract.",
        convention="Oblate symmetric top: a and b in the ring plane (degenerate), c perpendicular. These were NOT measured directly on benzene — 'The molecular quadrupole moment in benzene is derived from the values in fluorobenzene', i.e. the benzene g values come from an indirect route. Treat as a weak constraint only; the 37% uncertainty makes this useless as a tight test but fine as a sign/order-of-magnitude check.",
        confidence='high',
        quantity='g_aa = g_bb (in-plane, degenerate)',
    ),
    GReference(
        molecule='Benzene, C6H6',
        component='C',
        value=0.053,
        sigma=0.017,
        printed='+ (0.053 ± 0.017)',
        source='R. L. Shoemaker, W. H. Flygare, J. Chem. Phys. 51, 2988-2991 (1969). DOI 10.1063/1.1672447.',
        convention='c perpendicular to the ring; positive while the in-plane components are negative. Indirectly derived from fluorobenzene, not measured directly. Weak constraint.',
        confidence='high',
        quantity='g_cc (perpendicular to the ring)',
    ),
    GReference(
        molecule='Acetonitrile (methyl cyanide), CH3CN',
        component='perp',
        value=-0.0338,
        sigma=0.0008,
        printed='g_perp = − 0.0338 ± 0.0008',
        source="J. M. Pochan, R. L. Shoemaker, R. G. Stone, W. H. Flygare, 'Molecular g Values, Magnetic Susceptibility Anisotropies, Diamagnetic and Paramagnetic Susceptibilities, Second Moment of the Charge Distribution, and Molecular Quadrupole Moments of H3CCN and H3CNC', J. Chem. Phys. 52, 2478-2484 (1970). DOI 10.1063/1.1673331. Read verbatim in the publisher-deposited abstract.",
        convention="Prolate symmetric top. Stated verbatim: 'c = b is perpendicular to the C3 or a axis'. IMPORTANT: g_par was NOT measured — 'Assuming g_par = 0.310 from other similar systems' — so only g_perp is an experimental number here. The 14N quadrupole coupling prevented fully uncoupled spectra even at 25 000-30 000 G, which is why this is less precise than the CH3F case. v=0. The sign of the dipole is +H3CCN-.",
        confidence='high',
        quantity='g_perp',
    ),
    GReference(
        molecule='Acetonitrile-d3, CD3CN',
        component='perp',
        value=-0.0315,
        sigma=0.0008,
        printed='g_perp = − 0.0315 ± 0.0008',
        source='J. M. Pochan, R. L. Shoemaker, R. G. Stone, W. H. Flygare, J. Chem. Phys. 52, 2478-2484 (1970). DOI 10.1063/1.1673331.',
        convention='As for the d0 species; only g_perp measured. The d0/d3 pair is another isotopologue mass-dependence test (0.0338 vs 0.0315). v=0.',
        confidence='high',
        quantity='g_perp',
    ),
    GReference(
        molecule='Thioxoborane, HBS',
        component='perp',
        value=-0.0414,
        sigma=0.0002,
        printed='g_perp = −0.0414 ± 0.0002',
        source="E. F. Pearson, C. L. Norris, W. H. Flygare, 'Molecular Zeeman effect, electric dipole moment, and boron nuclear hyperfine coupling constants in HBS', J. Chem. Phys. 60, 1761-1764 (1974). DOI 10.1063/1.1681273. Read verbatim in the publisher-deposited abstract.",
        convention='Linear triatomic: only g_perp. Measured from the J = 0 -> 1 transition at high magnetic field. The H/D pair was used to fix the dipole sign (+HBS-). v=0. Cheap 3-atom molecule with a second-row atom.',
        confidence='high',
        quantity='g_perp',
    ),
    GReference(
        molecule='Thioxoborane-d, DBS',
        component='perp',
        value=-0.0356,
        sigma=0.0002,
        printed='g_perp = −0.0356 ± 0.0002',
        source='E. F. Pearson, C. L. Norris, W. H. Flygare, J. Chem. Phys. 60, 1761-1764 (1974). DOI 10.1063/1.1681273.',
        convention='Linear triatomic, only g_perp. H/D isotopologue pair with the entry above. v=0.',
        confidence='high',
        quantity='g_perp',
    ),
    GReference(
        molecule='Sulfur tetrafluoride, SF4',
        component='A',
        value=-0.0555,
        sigma=0.0008,
        printed='− 0.0555 ± 0.0008',
        source="R. G. Stone, H. L. Tigelaar, W. H. Flygare, 'Molecular Quadrupole Moments of SF4', J. Chem. Phys. 53, 3947-3950 (1970). DOI 10.1063/1.1673864. Read verbatim in the publisher-deposited abstract.",
        convention="Stated verbatim: 'The a axis contains the near-linear F-S-F group in SF4, and the c axis bisects the SF2 angle for the other two fluorine atoms. The c axis is the electric dipole axis.' Linear and quadratic Zeeman effects at ~21 000 G. v=0. FLUORINATED, and a near-isotropic tensor (all three components within 13% of each other) — a good test of whether small anisotropies survive the calculation.",
        confidence='high',
        quantity='g_aa',
    ),
    GReference(
        molecule='Sulfur tetrafluoride, SF4',
        component='B',
        value=-0.051,
        sigma=0.0021,
        printed='− 0.0510 ± 0.0021',
        source='R. G. Stone, H. L. Tigelaar, W. H. Flygare, J. Chem. Phys. 53, 3947-3950 (1970). DOI 10.1063/1.1673864.',
        convention='See the g_aa row for the axis definitions. v=0.',
        confidence='high',
        quantity='g_bb',
    ),
    GReference(
        molecule='Sulfur tetrafluoride, SF4',
        component='C',
        value=-0.0492,
        sigma=0.0023,
        printed='− 0.0492 ± 0.0023',
        source='R. G. Stone, H. L. Tigelaar, W. H. Flygare, J. Chem. Phys. 53, 3947-3950 (1970). DOI 10.1063/1.1673864.',
        convention='c bisects the SF2 angle and is the dipole axis. v=0. Note g_bb and g_cc overlap within their error bars, so do not write a test that depends on their ordering.',
        confidence='high',
        quantity='g_cc',
    ),
    GReference(
        molecule='Carbon monoxide, CO',
        component='perp',
        value=-0.2689,
        sigma=0.0001,
        printed='−0.26890 ± 0.00010',
        source="Primary: I. Ozier, P.-n. Yi, A. Khosla, N. F. Ramsey, 'Sign and Magnitude of the Rotational Moment of 12C16O', J. Chem. Phys. 46, 1530 (1967). Read via the secondary compilation: arXiv:2505.02511, 'A Computational Study of the Vibrational and Rotational g-Factors of the Diatomic Molecules LiH, LiF, CO, CS, SiO and SiS' (Section on active spaces / Fig. 4 caption and the experimental-values list).",
        convention="Diatomic: single g_J = g_perp; the component along the internuclear axis is zero. This is the standard high-precision CO reference value and the sign is determined (negative). I read it in a 2025 compilation (secondary source), NOT in Ozier's original paper, hence medium confidence. Other measurements of the same quantity, as printed in that compilation: (0.26910 ± 0.0005) Rosenblum, Nethercot Jr, Townes, Phys. Rev. 109, 400 (1958); (0.267 ± 0.003) Burrus, J. Chem. Phys. 30, 976-983 (1959); 0.262 ± 0.026 Wang & Keiderling, J. Chem. Phys. 98, 903-911 (1993). The older ones are printed WITHOUT a minus sign because those experiments determined only the magnitude — do not read that as a positive g.",
        confidence='medium',
        quantity='g_perp (= g_J)',
    ),
    GReference(
        molecule='Carbon monosulfide, CS',
        component='perp',
        value=-0.2702,
        sigma=0.0004,
        printed='−0.2702 ± 0.0004',
        source="Primary: J. McGurk, H. Tigelaar, S. Rock, C. Norris, W. Flygare, 'Detection, assignment of the microwave spectrum and the molecular Stark and Zeeman effects in CSe, and the Zeeman effect and sign of the dipole moment in CS', J. Chem. Phys. 58, 1420-1424 (1973). Read via arXiv:2505.02511.",
        convention="Diatomic, single g_J. Sign determined negative. Also printed in the same compilation: (0.269 ± 0.005) from Bates, Gallagher, Derr, J. Appl. Phys. 39, 3218-3221 (1968) — whose own abstract prints it as a magnitude only, '± (0.269) ±0.005)'. Read in a secondary compilation, not the original.",
        confidence='medium',
        quantity='g_perp (= g_J)',
    ),
    GReference(
        molecule='Silicon monoxide, SiO',
        component='perp',
        value=-0.15359,
        sigma=0.00012,
        printed='−0.15359 ± 0.00012',
        source="Primary: R. E. Davis, J. Muenter, 'Magnetic properties of silicon and germanium monoxide', J. Chem. Phys. 61, 2940-2945 (1974). Read via arXiv:2505.02511.",
        convention='Diatomic, single g_J, sign negative. Read in a secondary compilation, not the original paper.',
        confidence='medium',
        quantity='g_perp (= g_J)',
    ),
    GReference(
        molecule='Silicon monosulfide, SiS',
        component='perp',
        value=-0.09097,
        sigma=6.5e-05,
        printed='−0.09097 ± 0.000065',
        source="Primary: R. Honerjaeger, R. Tischer, 'gJ-Faktor der Molekeln GeO und SiS und Anisotropie ihrer Magnetisierbarkeit / Molecular gJ-Factor and Magnetic Susceptibility Anisotropy of GeO and SiS', Z. Naturforsch. A 28, 1374-1375 (1973). Read via arXiv:2505.02511.",
        convention='Diatomic, single g_J, sign negative. Read in a secondary compilation, not the original paper.',
        confidence='medium',
        quantity='g_perp (= g_J)',
    ),
    GReference(
        molecule='Lithium hydride, LiH',
        component='perp',
        value=-0.65842,
        sigma=0.00017,
        printed='−0.65842 ± 0.00017',
        source="Primary: R. R. Freeman, A. R. Jacobson, D. W. Johnson, N. F. Ramsey, 'The molecular Zeeman and hyperfine spectra of LiH and LiD by molecular beam high resolution electric resonance', J. Chem. Phys. 63, 2597-2602 (1975). Read via arXiv:2505.02511.",
        convention='Diatomic, single g_J, sign negative. Molecular beam high-resolution electric resonance. Also printed in the compilation: −(0.654 ± 0.007) from Lawrence, Anderson, Ramsey, Phys. Rev. 130, 1865 (1963). The largest-magnitude diatomic g in this set; a 2-electron-pair system, so extremely cheap to compute and a clean CPHF test. Read in a secondary compilation, not the original.',
        confidence='medium',
        quantity='g_perp (= g_J)',
    ),
    GReference(
        molecule='Lithium fluoride, LiF',
        component='perp',
        value=0.07367,
        sigma=0.0005,
        printed='0.07367 ± 0.00050',
        source="Primary: F. Mehran, R. Brooks, N. Ramsey, 'Rotational magnetic moments of alkali-halide molecules', Phys. Rev. 141, 93 (1966). Read via arXiv:2505.02511.",
        convention="Diatomic, single g_J. PRINTED POSITIVE — LiF is the one positive diatomic g in this group, which makes it valuable as a sign test, BUT be careful: these alkali-halide molecular-beam experiments generally determined magnitudes, so the printed positive sign may be a magnitude rather than a determined sign. Verify against Mehran et al. before writing a signed assertion. Also printed in the compilation: (0.0642 ± 0.0004) from Russell, Phys. Rev. 111, 1558 (1958) — a ~13% disagreement with Mehran's value, so the two are not mutually consistent. Read in a secondary compilation, not the original.",
        confidence='medium',
        quantity='g_perp (= g_J)',
    ),
]


#: What was looked for and could not be read. See the module docstring.
NOT_VERIFIED = [   'WATER, H2O (the parent species). I could only verify D2O. The H2O and HDO '
    'magnetic moments ARE in the same paper (Verhoeven & Dymanus, J. Chem. Phys. 52, '
    '3222 (1970)) — the abstract says the 6(-5)-5(-1) transition of H2O was measured '
    'and that H2O/HDO/D2O moments were compared — but the deposited abstract prints '
    'only the D2O tensor. The H2O numbers are in the paywalled full text and I did not '
    'read them. Do NOT let anyone fill this in from memory; the H2O rotational g '
    'values are frequently misquoted.',
    'AMMONIA, NH3. Sources exist and I identified them precisely, but neither '
    'publisher deposited an abstract and both are paywalled: S. G. Kukolich & W. H. '
    "Flygare, 'Molecular g values, magnetic susceptibilities, molecular quadrupole "
    "moment, and spin-rotation interaction in 15NH3', Mol. Phys. 17, 127-133 (1969), "
    "DOI 10.1080/00268976900100871; S. G. Kukolich, 'Magnetic susceptibility "
    "anisotropy and molecular quadrupole moment in 14NH3', Chem. Phys. Lett. 5, "
    '401-404 (1970), DOI 10.1016/0009-2614(70)80047-1; and S. G. Kukolich, Chem. Phys. '
    "Lett. 12, 216 (1971), DOI 10.1016/0009-2614(71)80656-5. Crossref returns 'no "
    "abstract' for all three. Also Kukolich & Flygare, Chem. Phys. Lett. 7, 43-46 "
    '(1970) for PH3/PH2D/PHD2.',
    'HYDROGEN FLUORIDE, HF (and HCl). Source identified: F. H. de Leeuw & A. Dymanus, '
    "'Magnetic properties and molecular quadrupole moment of HF and HCl by "
    "molecular-beam electric-resonance spectroscopy', J. Mol. Spectrosc. 48, 427-445 "
    '(1973), DOI 10.1016/0022-2852(73)90107-0. Elsevier deposited no abstract and '
    "Semantic Scholar reports the abstract field as 'elided by the publisher'. I did "
    'not read the numbers.',
    "H2S, HDS, D2S, PH3, PD3, DBr, DI. All measured in C. A. Burrus, 'Zeeman Effect in "
    'the 1- to 3-Millimeter Wave Region: Molecular g Factors of Several Light '
    "Molecules', J. Chem. Phys. 30, 976-983 (1959), DOI 10.1063/1.1730139. I read that "
    'abstract: it names the molecules (and states that g tensor components along the '
    'principal axes were obtained where applicable) but prints no numbers. Note this '
    'paper is also the source of one of the CO values.',
    'CH2F2, OCF2 (carbonyl fluoride), cis-CHF=CHF, CH2=CF2. These are in R. P. '
    'Blickensderfer, J. H. S. Wang, W. H. Flygare, J. Chem. Phys. 51, 3196-3205 '
    '(1969), DOI 10.1063/1.1672495. The deposited abstract describes the table and the '
    'axis convention (x and y in the heavy-atom plane, x bisecting the FCF angle) but '
    'the Crossref record truncates immediately before the numeric table. This would be '
    'the best source of additional cheap fluorinated asymmetric tops — worth one '
    'library lookup. Same situation for CHFO (formyl fluoride), CH2=CHF, CF2=CHF, '
    'CH3CH2F, CH3CHF2 in Rock, Hancock & Flygare, J. Chem. Phys. 54, 3450-3463 (1971), '
    'DOI 10.1063/1.1675364, whose abstract prints no numbers at all. Also: Lo & '
    "Flygare, 'The molecular g-values and magnetic susceptibility in methylene "
    "fluoride', J. Mol. Spectrosc. 25, 365-373 (1968), no abstract deposited.",
    'ETHENE / ETHYLENE (g_aa = -0.3954(9), g_bb = -0.1159(3), g_cc = 0.0453(2)) and '
    'ALLENE-d2, H2CCCD2 (g_aa = -0.30055(54), g_bb = -0.01248(2)). These numbers '
    'appeared in web-search result summaries but I could NOT open the underlying pages '
    '(tandfonline and ScienceDirect both return 403), so I have not actually read them '
    'and I am deliberately NOT promoting them to entries. If you want them the sources '
    "are: 'The molecular electric quadrupole tensor of ethene from the rotational "
    "Zeeman effect of CH2=CD2', Mol. Phys. 83(3) (1994), DOI "
    '10.1080/00268979400101431; and V. Meyer, D. H. Sutter, B. Vogelsanger, J. Mol. '
    'Spectrosc. 148, 436-446 (1991), DOI 10.1016/0022-2852(91)90399-u.',
    'OZONE values from Meerts, Stolte & Dymanus, Chem. Phys. 19, 467-472 (1977), DOI '
    '10.1016/0301-0104(77)85017-9 (a search summary reported g_aa = -2.9877(9), g_bb = '
    '-0.2295(3), g_cc = -0.0760(3)). Same situation: seen only in a search-engine '
    'summary, page not readable, so not entered. The two ozone measurements I DID read '
    '(Pochan et al. 1969 and Mack & Muenter 1977) already cover ozone well.',
    'RotGT-2023 (Y. V. Vishnevskiy, J. Chem. Phys. 159, 164307 (2023), DOI '
    '10.1063/5.0173313). THIS IS THE THING YOU ACTUALLY WANT: 278 statistically '
    'validated experimental rotational g tensor components for 129 molecules and '
    'isotopologues, WITH computed vibrational corrections to equilibrium values, plus '
    'CCSD(T) and PBE0 benchmarks. I could not get it: OpenAlex reports oa_status '
    "'closed', no repository copy; the ChemRxiv preprint PDF exists at chemrxiv.org "
    '(item 64df7d4600bbebf0e66591f6) but Cloudflare returns 403 to both curl and '
    "WebFetch; the author's group site lists the paper with no data link. If you or a "
    'collaborator has institutional access, one download of this paper and its '
    'supplementary information replaces everything above and gives you '
    'equilibrium-referenced values, which is strictly better for benchmarking an '
    'equilibrium-geometry CPHF calculation.',
    'Landolt-Boernstein, Group II Vol. 14 (1982), chapter 2.9.3 by W. Huettner, in '
    "Demaison, Dubrulle, Huettner & Tiemann, 'Molecular Constants Mostly from "
    "Microwave, Molecular Beam, and Sub-Doppler Laser Spectroscopy'. This is the "
    'canonical hand compilation of molecular g values and is what the modern '
    'semi-experimental-structure literature cites for experimental g constants (I '
    'found it cited exactly that way, as ref. 51, in an open-access 2024 Molecules '
    'paper on propane). Not open access, not readable here.',
    'Spherical-top g values (CH4, SiH4) from Ozier and co-workers — I did not search '
    'for these specifically, but they exist and would be good cheap additions. '
    "RotGT-2023's abstract mentions using SiH4 for a structure refinement, so SiH4 g "
    'data are in that set.']


#: The retrieval's own statement of the conventions and their limits.
RETRIEVAL_CAVEATS = "REDUCTION AND REPRESENTATION DO NOT APPLY HERE. Your brief asked me to record the reduction (A vs S) and representation (Ir/IIIr) for every value. Those concepts belong to centrifugal-distortion analysis, not to rotational g tensors: a g tensor is a molecular property, not a fitted Hamiltonian parameter, so there is no A/S reduction of it and no Ir/IIIr ambiguity. I have instead recorded the five conventions that DO matter for comparing a g tensor correctly, and every entry's convention field states them: (1) units — all values here are dimensionless in NUCLEAR MAGNETON units, i.e. the rotational magnetic moment is mu_g = g_gg * mu_N * J_g; if any of these had been in Bohr magnetons they would differ by 1836; (2) sign convention; (3) axis labelling (a/b/c = principal inertial axes with I_a <= I_b <= I_c, which maps directly onto the repo's A/B/C ordering, plus each paper's statement of which physical direction is which); (4) vibrational state; (5) any J-averaging.\n\nVIBRATIONAL STATE IS THE BIGGEST SYSTEMATIC. Every value in this set is a GROUND VIBRATIONAL STATE (v=0) effective value, not an equilibrium g_e. A CPHF calculation at an equilibrium geometry computes g_e. RotGT-2023's own abstract puts the residual error of ae-CCSD(T) against VIBRATIONALLY CORRECTED experimental values at 1.09% mean / 2.07% std — meaning the vibrational correction itself is of that order or larger. So: do not set test tolerances tighter than a few percent against these numbers, and do not interpret a 2% deviation as an implementation bug. For HF/6-31g (the repo's default in rotational_g_tensor) expect considerably worse than that; the literature notes HF systematically underestimates rotational g factors. Two entries carry an extra wrinkle: the formaldehyde values are additionally averaged over three rotational states because a real J dependence was observed, and the cyclopropane tensor was measured on the 1,1-d2 isotopomer but reported in the parent's symmetric-top basis.\n\nSIGNS ARE OFTEN AN INFERENCE, NOT A MEASUREMENT. In most of the Flygare-group high-field microwave work the experiment yields only the MAGNITUDES of g_aa, g_bb, g_cc plus their RELATIVE signs; the absolute signs are then assigned from molecular-quadrupole-moment or second-moment arguments. Abstracts say so explicitly for SO2/O3/OF2 ('Arguments are presented to show that all of the molecular g values are negative'), oxirane ('Only the relative signs ... are experimentally determined'), cyclopropene, fluorobenzene, and the older diatomic work. H2CS and H2CO are the happy exceptions with uniquely determined signs. Practical consequence: if your code reproduces magnitudes but flips a sign, check whether the paper actually measured that sign before filing a bug.\n\nFOUR CONFIRMED TEXT CORRUPTIONS IN THE SOURCE ABSTRACTS — AND TWO VALUES I REFUSED TO ENTER BECAUSE OF THEM. The deposited abstracts are OCR'd or hand-keyed and I caught real errors: (a) Mack & Muenter's ozone g_cc uncertainty is literally printed as '(b)'; (b) their g_aa is printed positive where two other sources give it negative; (c) the oxirane g_cc is missing its closing parenthesis; (d) the Z. Naturforsch. A titles come through mangled ('The Molecular 0-Tensor' for 'g-Tensor', 'Q bb = 4M (7)'). Two values were bad enough that I left them OUT of the entry list rather than risk corrupting your tests: KETENE (Huettner, Foster & Flygare, J. Chem. Phys. 50, 1710 (1969), DOI 10.1063/1.1671263) is printed as g_aa = -(0.4182 +/- 0.0009), g_bb = -(0.356 +/- 0.0013), g_cc = -(0.0238 +/- 0.0006) — but for ketene I_b is approximately I_c, so g_bb cannot plausibly be 15x g_cc; -0.356 is almost certainly a typo for -0.0356, and I will not guess which. CYCLOPROPENE-1,2-d2 in the same abstract as the parent is printed g_cc = 0.4538 +/- 0.00018, which is a transparent typo for 0.04538 (the parent is 0.05363) and whose stated uncertainty is 2500x smaller than the value. Both are perfectly good validation points once someone reads the printed table. More generally: I verified these numbers against publisher-deposited abstracts, which is one step short of the typeset tables. For any number you are about to freeze into a test, read the table.\n\nTHE OCS NUMBER YOUR REPO ALREADY USES IS CORRECT, AND NOW HAS A CITATION. dev/tests/test_rovib_corrections.py:456 asserts electronic_delta_b(..., g_value=-0.028) > 0.0 'as for OCS'. That is Flygare, Huettner, Shoemaker & Foster, J. Chem. Phys. 50, 1714 (1969), g_perp(16O12C32S) = -0.028711(4). You can tighten that test to the real value and add the 34S isotopologue (-0.028127) as a mass-dependence check, which exercises the nucprop mass-override path in rotational_g_tensor that your code comment goes to some trouble to explain.\n\nLINEAR MOLECULES NEED SPECIAL HANDLING IN YOUR A/B/C DICT. For OCS, CO, N2O, HBS, CS, SiO, SiS, LiH, LiF the component along the molecular axis is identically zero and only g_perp exists. rotational_g_tensor returns {A, B, C} from a principal-axis diagonalisation; for a linear molecule I_a -> 0, so the A entry is a 0/0 artifact and B = C = g_perp is what you compare. Make sure the test harness does not compare the A component for these.\n\nVALIDATE THE PYSCF SIGN CONVENTION FIRST, ON ONE KNOWN CASE. Before trusting any of this as a benchmark, confirm that pyscf.prop.rotational_gtensor's sign convention matches the spectroscopic one used by all these papers. CO (-0.26890) and OCS (-0.028711) are the right molecules for that check because both signs are experimentally settled and both are single numbers. If the sign comes out inverted, everything below will look wrong for a trivial reason. The repo's existing comment that HF/6-31g gives water g_bb = +0.682 for H2-16O is reassuring on this point, since the measured D2O G_bb is +0.36009 and the code gives +0.341 for D2O — right sign, right magnitude, ~5% low, which is the expected HF behaviour.\n\nON THE CENTRIFUGAL-DISTORTION PART OF YOUR REQUEST. You also asked for DJ, DJK, DK, deltaJ, deltaK, PhiJ with A/S reduction and Ir/IIIr representation recorded. I found NO distortion constants in this search — I went after the rotational g tensor as the task text directed, and the two literatures barely overlap. If you are about to turn on a centrifugal-distortion correction and need experimental distortion constants for validation, that is a separate search and a much easier one: the CDMS and JPL catalogues are open and publish the fitted constants together with the reduction and representation for every species, which is exactly the provenance you need. One relevant data point I did see in passing: an open-access paper on 2,2-difluoropropane reported that the centrifugal-distortion correction to the rotational constants was 1 kHz for A, 25 kHz for B and -23 kHz for C and was neglected as negligible, whereas the g-tensor electronic correction for the same molecule was -0.97 MHz for A. For that molecule the electronic correction is ~40x the distortion correction, which is worth knowing before you spend effort on the smaller of the two."


def by_molecule(name: str):
    """Every verified component for one molecule, newest source first in the list.

    A list rather than a dict, because several molecules here have two
    independent measurements of the same component -- ozone, formaldehyde and
    OCS -- and those duplicates are the most useful rows in the table. Ozone's
    two sources disagree on the printed sign of g_aa while agreeing on its
    magnitude to 0.7%, which is a test a calculation can settle.
    """
    return [r for r in G_REFERENCES if r.molecule == name]


def primary_tensor(name: str):
    """``{component: GReference}`` taking the first measurement of each.

    Use :func:`by_molecule` when the duplicates matter.
    """
    out = {}
    for r in G_REFERENCES:
        if r.molecule == name and r.component not in out:
            out[r.component] = r
    return out


def molecules():
    """Molecule names present, in first-appearance order."""
    seen, out = set(), []
    for r in G_REFERENCES:
        if r.molecule not in seen:
            seen.add(r.molecule)
            out.append(r.molecule)
    return out
