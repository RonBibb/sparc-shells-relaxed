# sparc-shells-relaxed

**Paper 3 — Relaxed-Cap Shell Catalog with Constrained-Run Validation,
Hierarchical Υ, and DGC-Tier Reporting**

Working title:
*"Localized Residual Structure in SPARC Rotation Curves II:
Relaxed-Cap Catalog, Constrained Fits, Hierarchical Mass-to-Light
Marginalization, and Multi-Body Decomposition Under DGC Scaling"*

Lead author: Ron Bibb (independent researcher, ORCID: 0009-0004-1153-2464)
Status: pre-writing — empirical work and methodology validation substantially complete

---

## Purpose

This repository contains the analysis code, derived data, and figure sources
for Paper 3 in the SPARC shells series. It extends the Paper 1 methodology
(Bibb 2026, submitted to PASP as PASP-102415) by:

1. **Relaxing the σ/r cap from 0.4 to 0.8** to address shell-width pegging
   observed in the strict-cap fits
2. **Increasing the maximum shell count from 2 to 5**
3. **Adding hierarchical-Υ marginalization** to address the Paper 1 referee
   critique on NGC 5055 BIC collapse under joint posterior treatment
4. **Producing a constrained-run companion catalog** (DGC + Hill + 1% M_central
   limits applied during fitting) for interpretation-strength comparison
5. **Reporting implied-IMBH populations under DGC scaling** as bookkeeping
   (one possible interpretation, presented with explicit caveats; the catalog
   does not claim DGC is the correct mass scaling at sub-halo scales)
6. **Categorizing shells by DGC compatibility** (Tier A: n_BH ≤ 2 acceptable;
   Tier B/C: DGC-overbudget under the 1% M_central limit)
7. **Cross-tracer validation work** (Walton+ 2022 ULX positions; THINGS HI
   moment maps; NGC 5055 Chandra cross-match) used during methodology
   validation, scripts shared with the companion Paper 4 repo

Paper 4 (`sparc-shells-imbh`) uses this catalog as input for its
cross-tracer test.

---

## Repository structure

```
sparc-shells-relaxed/
├── README.md
├── .gitignore
├── source/                         — paper manuscript source
│   ├── methods/                    — methodology fragments
│   └── sections/                   — section drafts
├── scripts/                        — analysis code
│   ├── run_relaxed_caps_fits.py                — n≤2 relaxed-cap producer
│   ├── run_relaxed_caps_fits_with_BH_decomp.py — n≤5 + DGC multi-BH bookkeeping
│   ├── run_DGC_Hill_5shell_fits.py             — constrained-during-fit run
│   ├── multi_imbh_analysis.py                  — shell → IMBH population analysis
│   ├── fetch_walton_catalog.py                 — Vizier fetcher for Walton+ 2022
│   ├── walton_ulx_crossmatch.py                — ULX↔shell match (shared w/ Paper 4)
│   └── ngc5055_chandra_crossmatch.py           — NGC 5055 Chandra cross-match
├── data/
│   ├── raw/
│   │   └── sparc_sample123.csv                 — SPARC master catalog (Lelli+ 2016)
│   ├── processed/                              — derived catalogs (committed)
│   │   ├── sparc_relaxed_caps_with_BH_decomp.csv   — primary relaxed catalog (123 gx, n≤5)
│   │   ├── sparc_T2-T9_relaxed_caps_fits.csv       — earlier 102-galaxy n≤2 run
│   │   ├── sparc_T2-T9_y_T_fits.csv                — Paper 1 strict-cap catalog
│   │   ├── sparc_T2-T9_DGC_Hill_5shell_fits.csv    — constrained-during-fit run
│   │   ├── cap_comparison_summary.csv              — strict↔relaxed per-galaxy class
│   │   ├── run_relaxed_caps_fits.log
│   │   └── run_DGC_Hill_5shell_fits.log
│   └── external/                               — third-party data
│       ├── walton_2022_ulx.csv                 — Walton+ 2022 (committed, redistributable)
│       └── NGC_*_THINGS.FITS                   — THINGS moment maps (NOT committed)
└── figures/
    ├── paper/                                  — final figures for publication
    └── exploration/                            — diagnostic figures
```

---

## Key derived data products

| File | Description |
|---|---|
| `data/processed/sparc_relaxed_caps_with_BH_decomp.csv` | **Primary catalog.** 123 galaxies (T=0–11), relaxed-cap fits with up to 5 shells, plus DGC multi-BH decomposition columns. |
| `data/processed/sparc_T2-T9_relaxed_caps_fits.csv` | Earlier relaxed-cap run, 102 galaxies (T=2–9), n≤2 cap. Used for strict↔relaxed comparison. |
| `data/processed/sparc_T2-T9_y_T_fits.csv` | Paper 1 strict-cap catalog (T-dependent Υ_disk, hierarchical-Υ-ready). |
| `data/processed/sparc_T2-T9_DGC_Hill_5shell_fits.csv` | Constrained-during-fit run (DGC + Hill + 1% M_central, n≤5). For interpretation-strength comparison. |
| `data/processed/cap_comparison_summary.csv` | Per-galaxy classification under cap relaxation: **91 stable_adequate, 6 rescued, 5 still_failing** of 102 baseline. |

---

## Methodology summary

**Fitter:** Burkert backbone (ρ₀, a) + N Gaussian shells (M_i, r_i, σ_i)
fit to SPARC rotation curves jointly with the baryonic V² contribution at
canonical Υ_disk=0.5, Υ_bulge=0.7. BIC selects best N per galaxy.

**Caps (relaxed run):** σ/r ∈ [0.01, 0.80], N ∈ {0..5}.
**Caps (Paper 1 strict):** σ/r ∈ [0.01, 0.40], N ∈ {0..2}.

**Multi-BH decomposition (post-hoc bookkeeping):**
- DGC scaling: log10(M_BH) = -37.785 + 4.35 · log10(M_shell)
- For each shell, try n_BH ∈ {1..5} to find simplest decomposition where
  M_BH_each ≤ 1% × M_central (Kormendy & Ho 2013 from V_flat)
- Flag shells where no decomposition in this range succeeds

**Decomposition outcome (87 shells in 123-galaxy run):**
- 29 shells (33%) Tier A: 1–2 IMBHs under DGC
- 0 shells in 3–4 IMBH range (DGC slope 4.35 forbids middle ground)
- 58 shells (67%) Tier B/C: fail at n_BH=5 (DGC-overbudget; multiple
  interpretations possible, deferred)

**Cap relaxation classification (102 baseline galaxies, strict vs relaxed):**
- 91 **stable_adequate** (best_n and χ²_r unchanged within tolerance)
- 6 **rescued** (substantial χ²_r improvement, often with shell promotion)
- 5 **still_failing** (high χ²_r in both regimes; framework limit cases)

---

## External data — download instructions

The repository does **not** redistribute large third-party data files.
To reproduce the cross-tracer figures or HI analyses, you'll need:

**THINGS HI moment maps** (Walter et al. 2008) for NGC 2403, 2903, 3521,
5055, 6946:
- Available at: https://www2.mpia-hd.mpg.de/THINGS/Data.html
- Download `NA_MOM0`, `NA_MOM2`, `RO_MOM0`, `RO_MOM2` for each galaxy
- Place under `data/external/`
- Acknowledgment: "This work made use of THINGS, 'The HI Nearby Galaxy
  Survey' (Walter et al. 2008)"

**SPARC rotation curves** (Lelli, McGaugh & Schombert 2016):
- Available at: http://astroweb.cwru.edu/SPARC/
- Rotmod files (`*_rotmod.dat`) referenced by scripts as `Rotmod_LTG/`
- The repo includes the SPARC master catalog at `data/raw/sparc_sample123.csv`

**Walton+ 2022 ULX catalog**: included at `data/external/walton_2022_ulx.csv`
(redistribution permitted; original source is J/MNRAS/509/1587 on Vizier or
the project page at https://dwalton354.wixsite.com/djwalton/ulxcat).

---

## Status of analyses

### Completed
- [x] Relaxed-cap fitter, 123 galaxies, σ/r ≤ 0.8, n ≤ 5
- [x] Strict↔relaxed cap comparison with per-galaxy classification (cap_comparison_summary.csv)
- [x] DGC + Hill + M_central constrained-during-fit run for comparison
- [x] DGC multi-BH decomposition with Tier A/B/C categorization
- [x] σ/r distribution analysis showing relaxed cap is not noise-fitting
  (1/87 shells peg at σ/r = 0.8; relaxed median 0.22 < strict median 0.28)
- [x] χ²_r per-galaxy delta vs strict cap
- [x] Universal-failure reanalysis (6/10 materially improved under relaxed cap)
- [x] Multi-IMBH population analysis (multi_imbh_analysis.py)
- [x] NGC 5055 Chandra cross-match (ngc5055_chandra_crossmatch.py)
- [x] Walton+ 2022 ULX cross-tracer match (shared with Paper 4 repo)
- [x] Hierarchical Υ marginalization data and code (from Paper 1 v7.1.1)

### In progress / pending
- [ ] **Synthetic null-curve injection test** (false-positive rate at σ/r ≤ 0.8 vs 0.4) — next session
- [ ] Hierarchical Υ three diagnostics (convergence, posterior scatter, NGC 5055)
- [ ] Universal-failure taxonomy: focused session on the 5 still_failing galaxies
- [ ] DGC scope review (calibration sample, slope uncertainty at sub-halo scales)
- [ ] Manuscript drafting
- [ ] Zenodo v0.1.0 deposit for priority-stamping

---

## Connection to companion repos

- `sparc-halo-shells` (Paper I, https://github.com/RonBibb/sparc-halo-shells):
  methodology foundation, strict cap, v7.1.1 catalog, hierarchical Υ implementation,
  v7.2.0 archived at Zenodo (DOI: 10.5281/zenodo.20263016)
- `sparc-halo-shell-reality` (Paper II): population statistics, shell-reality
  validation (polishing phase, pending Paper I outcome)
- `sparc-shells-imbh` (Paper IV): cross-tracer association testing
  (ULX positions, HI density and dispersion perturbations) using this
  paper's catalog as input. Cross-tracer scripts in this repo's scripts/
  are shared with that work.

---

## Citation policy

This work is conducted as an independent research effort. If using outputs:
- Cite Paper 1 (Bibb 2026, PASP submitted as PASP-102415) for methodology foundation
- Cite this work once submitted (placeholder: Bibb 2026 Paper III, in prep.)
- Cite Lelli, McGaugh & Schombert (2016) for the SPARC sample
- Cite Walter et al. (2008) if using THINGS data
- Cite Walton et al. (2022) if using ULX cross-match
- Cite Davis, Graham & Combes (2019) for the BH–halo scaling used in
  the multi-BH decomposition (note: the present work explicitly does not
  claim DGC applies at sub-halo scales)

---

## Repository hygiene

- scripts/ runnable from repo root; data paths assume data/raw/ and data/external/
- data/raw/ is read-only conceptually — do not modify SPARC inputs
- All catalog outputs go to data/processed/
- THINGS FITS and other large third-party data are **not** committed to git;
  see "External data — download instructions" above
- Exploratory figures live in figures/exploration/; figures that will
  appear in the manuscript move to figures/paper/ once finalized
- Commit catalog regeneration with the script version that produced it
- Tag releases (v0.1.0, v0.2.0, …) at meaningful checkpoints

---

*Last updated: 2026-05-23*
