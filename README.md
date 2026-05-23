# sparc-shells-relaxed

**Paper 3 — Relaxed-Cap Shell Catalog with Hierarchical Υ and DGC-Tier Reporting**

Working title:
*"Localized Residual Structure in SPARC Rotation Curves II:
Relaxed-Cap Catalog, Hierarchical Mass-to-Light Marginalization,
and Multi-Body Decomposition Under DGC Scaling"*

Lead author: Ron Bibb (independent researcher, ORCID: 0009-0004-1153-2464)
Status: pre-writing — empirical work substantially complete, methodology validation in progress

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
4. **Reporting implied-IMBH populations under DGC scaling** as bookkeeping
   (one possible interpretation, presented with explicit caveats; the catalog
   does not claim DGC is the correct mass scaling at sub-halo scales)
5. **Categorizing shells by DGC compatibility** (Tier A: n_BH ≤ 2 acceptable;
   Tier B/C: DGC-overbudget under the 1% M_central limit)

Paper 4 (`sparc-shells-imbh`) uses this catalog as input for cross-tracer
testing.

---

## Repository structure

```
sparc-shells-relaxed/
├── README.md                       — this file
├── source/                         — paper manuscript source
│   ├── methods/                    — methodology fragments
│   └── sections/                   — section drafts
├── scripts/                        — analysis code
│   ├── run_relaxed_caps_fits.py
│   └── (other scripts as added)
├── data/
│   ├── raw/                        — SPARC inputs (links/symlinks)
│   ├── processed/                  — derived catalogs (CSV)
│   └── external/                   — Walton+ 2022, THINGS, etc.
└── figures/
    ├── paper/                      — final figures for publication
    └── exploration/                — diagnostic figures, not for paper
```

---

## Key derived data products

| File | Description |
|---|---|
| `data/processed/sparc_relaxed_caps_with_BH_decomp.csv` | Primary catalog: 123 galaxies (T=0-11), relaxed-cap fits with up to 5 shells, plus DGC multi-BH decomposition columns |
| `data/processed/sparc_T2-T9_relaxed_caps_fits.csv` | Earlier 102-galaxy run (T=2-9), n≤2 cap, used for strict↔relaxed comparison |
| `data/external/walton_2022_ulx.csv` | Walton+ 2022 ULX catalog (1843 candidates), used for cross-tracer reference |

---

## Methodology summary

**Fitter:** Burkert backbone (ρ₀, a) + N Gaussian shells (M_i, r_i, σ_i)
fit to SPARC rotation curves jointly with the baryonic V² contribution at
canonical Υ_disk=0.5, Υ_bulge=0.7. BIC selects best N ∈ {0..5} per galaxy.

**Shell width prior:** σ/r ∈ [0.01, 0.80] (relaxed from Paper 1's 0.40)

**Multi-BH decomposition (post-hoc bookkeeping):**
- DGC scaling: log10(M_BH) = -37.785 + 4.35 · log10(M_shell)
- For each shell, try n_BH ∈ {1..5} to find simplest decomposition where
  M_BH_each ≤ 1% × M_central (Kormendy & Ho 2013 from V_flat)
- Flag shells where no decomposition in this range succeeds

**Decomposition outcome (87 shells in 123-galaxy run):**
- 29 shells (33%) Tier A: 1-2 IMBHs under DGC
- 0 shells in 3-4 IMBH range (DGC slope 4.35 forbids middle ground)
- 58 shells (67%) Tier B/C: fail at n_BH=5 (DGC-overbudget; multiple
  interpretations possible, deferred)

---

## Status of analyses

### Completed
- [x] Relaxed-cap fitter, 123 galaxies
- [x] Strict-cap comparison (94/102 unchanged, 6 promoted, sample expansion)
- [x] DGC multi-BH decomposition with Tier A/B/C categorization
- [x] σ/r distribution analysis showing relaxed cap is not noise-fitting
- [x] χ²_r per-galaxy delta vs strict cap
- [x] Universal-failure reanalysis (rescue rate under relaxed cap)
- [x] Hierarchical Υ marginalization data and code (Paper 1 v7.1.1)

### In progress / pending
- [ ] Synthetic null-curve injection test (false-positive rate at σ/r ≤ 0.8)
- [ ] Hierarchical Υ three diagnostics (convergence, posterior scatter, NGC 5055)
- [ ] Universal-failure taxonomy: focused session on the 7 persistent failures
- [ ] Three-constraint sensitivity comparison (unconstrained/relaxed/strict)
- [ ] DGC scope review (calibration sample, slope uncertainty at sub-halo scales)
- [ ] Manuscript drafting

---

## Connection to companion repos

- `sparc-halo-shells` (Paper 1, RonBibb/sparc-halo-shells): methodology
  foundation, strict cap, v7.1.1 catalog, hierarchical Υ implementation
- `sparc-shells-imbh` (Paper 4): cross-tracer association testing
  (ULX positions, HI density and dispersion perturbations) using this
  paper's catalog as input

---

## Citation policy

This work is conducted as an independent research effort. If using outputs:
- Cite Paper 1 (Bibb 2026, PASP submitted) for methodology foundation
- Cite this work once submitted (placeholder: Bibb 2026 Paper III, in prep.)
- Cite Lelli, McGaugh & Schombert (2016) for the SPARC sample
- Cite Davis, Graham & Combes (2019) for the BH-halo scaling used in
  the multi-BH decomposition (note: the present work explicitly does not
  claim DGC applies at sub-halo scales)

---

## Repository hygiene notes

- Keep `scripts/` runnable from repo root
- Treat `data/raw/` as read-only; do not modify SPARC inputs
- All catalog outputs go to `data/processed/` with date-stamped versions
  if regenerated
- Exploratory figures live in `figures/exploration/`; figures that will
  appear in the manuscript get moved to `figures/paper/` once finalized
- Commit catalog regeneration with the script version that produced it
- Tag releases (v0.1, v0.2, …) at meaningful checkpoints

---

*Last updated: 2026-05-23*
