# Current task
Task: T-09
Status: building
Request: find chlorofluoromethane isotopologues (13C, D, ...) with published
ground-state constants; literature first, no engine change (D-40,
auto-picked at Q-08).

## Brief
- Outcome: a table of CH2ClF species with published ground-state rotational
  constants (value, uncertainty, reduction, source, whether readable), or a
  stated "not found" with what was searched.
- Why: T-08 -- two Cl isotopologues leave a valley the C-Cl / C-F offset
  misses sit in; 13C and D substitution move different atoms. The present
  constants (NBS Monograph 70, 1950s; 41810.1 / 5715.7 / 5194.6 MHz) may
  also be superseded by modern 35Cl / 37Cl fits.
- Accept: ground-state A0, B0, C0 (or Watson parameters convertible to
  them) from a published fit, reduction stated, read from the paper or a
  reliable compilation; unambiguous isotopic composition.
- Gate: adding species or replacing constants changes a benchmark input
  and every CH2ClF row; that is a separate question, not part of this
  search.

## Graph
- N1 · explore · NQCC CH2FCl page and its references (Vogt 2014's isotopologues) · after: none · done · handoff: no constants; primary source Blanco et al. JMS 174, 397 (1995)
- N2 · explore · literature search for 13C / D / CHD / CD2 rotational spectra · after: none · done · handoff: 13C only (Blanco 1995, paywalled); no D; modern 35Cl constants = NBS to <= 1.1 MHz
- N2b · explore · synthetic 13C test: does a third species close the valley? · after: N2 · running · handoff: no at both levels (RHF 2.91 -> 2.93/2.98, B3LYP 6.00 -> 5.86/5.97); verify running
- N3 · gate · Q-09: add which species / replace constants? · after: N1, N2 · pending · handoff: -

## Progress
- NQCC CH2FCl page (Bailey, last modified 2014-07-03): no rotational
  constants at all; 35Cl / 37Cl quadrupole tables and the r_o (Blanco et al.
  [3]) vs r_e^SE (Vogt [4]) structure table only. Refs: Muller JACS 75, 860
  (1953); Nandi & Chatterji SA 31A, 603 (1975); Blanco, Lesarri, Lopez,
  Alonso & Guarnieri, JMS 174, 397-416 (1995); Vogt, Demaison & Rudolph, Mol.
  Phys. 112, 2873 (2014). Not stated which isotopologues entered r_e^SE.
- Blanco et al. 1995 (ScienceDirect abstract via search, pii
  S0022285285700112 -- its "85" is the journal's code, not 1985): ground
  states of 12CH2 35ClF, 12CH2 37ClF and 13CH2 35ClF, 8-250 GHz; r_o C-Cl
  1.7621(2), C-F 1.370(1), C-H 1.09(1), ClCF 110.1(5) -- identical to
  NQCC's r_o [3]. No deuterated species named. Constants not yet read.
- Searches (3, incl. one extended) found no CHDFCl / CD2FCl rotational
  study; closest are CH2DF, CH2DCl, CHF2Cl / CDF2Cl (other molecules).
- NQCC ClFMethanes.pdf (Bailey 2008, extracted locally, dl_clfm/): CH2FCl
  structures (ropt, ~re, ro) and nqcc only; no rotational constants.
- ScienceDirect (Blanco 1995) and ResearchGate (Vogt 2014): 403 to
  automated access; not retried. Bologna PDF (unibo): the CH2ClF dimer
  (Favero et al. PCCP 2019), not the monomer.
- Pietropolli Charmet et al., J. Chem. Phys. 139, 164302 (2013), open at
  ricerca.sns.it (extracted locally, dl_sns/): Table II prints Blanco 1995's
  ground-state constants for CH2 35ClF, Watson A / Ir: A0 41811.2198(98),
  B0 5715.97941(42), C0 5194.89167(14) MHz (benchmark's NBS values 41810.1,
  5715.7, 5194.6: within 1.1 / 0.28 / 0.29 MHz, against fit sigmas 121 /
  6.3 / 5.6 -- replacing them changes nothing). 37Cl and 13C values not
  printed. Their own r_e^SE (CCSD(T)/cc-pCVTZ corrections, "different
  isotopic species (from Ref. 21)" = Blanco's three): C-H 1.083984(17),
  C-F 1.359385(15), C-Cl 1.764123(13), HCCl 107.9446, HCH 112.5632, FCCl
  110.0180 -- Vogt's within 0.7 mA (C-H) and 0.04 deg.
- Conclusion of the search: published species are 12CH2 35ClF, 12CH2 37ClF,
  13CH2 35ClF (Blanco 1995, paywalled); no deuterated study found (5
  searches). JPL/CDMS: no CH2ClF entry found.
- Synthetic test (t09/synth.py; 13C species built from Vogt r_e minus the
  level's own correction, never written to the reference), RHF: two
  species 2.91; + exact 13C 2.93; + 13C with the 35Cl species' relative
  correction error 2.98. The 13C species adds almost nothing (C near the
  centre of mass; sigmas set by the correction's model error). B3LYP:
  two species 6.00 (C-Cl +7.35, C-F -7.20); + exact 13C 5.86 (+7.00,
  -7.22); + 13C with 35Cl-like error 5.97 (+7.26, -7.21). Does not close
  the valley at either level. Independent check of the construction
  running.

## Observations

## Pending question

## Handoff
