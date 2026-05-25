# SPARC III: Localized HI Disturbances at Shell Radii in Nearby Galaxies

**Authors:** Ron Bibb  
**ORCID:** 0009-0004-1153-2464  
**Status:** In preparation  
**Companion to:** PASP-102415 (Paper I, under review)

---

## Abstract

We present a multi-survey HI and stellar mass analysis of rotation curve shell
features identified in the SPARC galaxy catalog. Using azimuthally-resolved
moment-0 maps from THINGS (Walter et al. 2008), HALOGAS DR1 (Heald et al. 2020),
and unWISE W1 stellar mass images (Lang et al. 2016) for 8 shell-bearing and
3 control galaxies, we demonstrate that a subset of shells show statistically
significant localized disturbances rather than azimuthally uniform rings. The
strongest signal is NGC5055 shell 2 (r=14.87 kpc), where three independent HI
datasets and WISE W1 stellar mass images all show peak emission within 15° of
the known ultra-luminous X-ray source NGC5055 X-1. A natural ISM variability
baseline of ≤22% is established from control galaxies with no detected shells.
We interpret localized shell disturbances as gravitational deposits from past
minor merger events, with the spatial coherence maintained by compact object
remnants of sufficient mass.

---

## Key Findings

1. **NGC5055 shell 2 (r=14.87 kpc) — five-tracer convergence:**
   - THINGS NA HI:   peak φ=172°, ratio=13.77× (vs. ≤22% ISM baseline)
   - HALOGAS HR HI:  peak φ=185°, ratio=11.35×
   - HALOGAS LR HI:  peak φ=175°, ratio=17.99× (deepest dataset)
   - WISE W1 stars:  peak φ=355°, ratio=8.87× (secondary deposit)
   - NGC5055 X-1 (ULX): φ=184°, Δr=0.01 kpc from shell center
   - All independent tracers within 15° of X-1 position

2. **Control baseline (NGC2976, NGC7331 THINGS; NGC4559 HALOGAS):**
   - Natural ISM variability ≤22% at equivalent shell radii
   - Shell signals range from 7× to 18× locally — well above baseline

3. **Multi-tracer convergence:**
   - NGC2403 shell 2 (r=4.0 kpc): HI 96th pctile + W1 86th pctile
   - NGC2903 shell 1 (r=7.0 kpc): HI 85th pctile + W1 81st pctile
   - NGC6946 shell 1 (r=4.2 kpc): HI 86th pctile + W1 86th pctile

4. **HI/W1 tracer divergence in NGC5055:**
   - HI peaks near X-1 (gas-rich deposit, φ≈175-185°)
   - W1 peaks at φ≈355° (stellar deposit, opposite side of orbit)
   - Consistent with gas and stars separating along tidal arc during disruption

---

## Data

**Input surveys:**
- THINGS: 5 shell-bearing galaxies (NGC2403, NGC2903, NGC3521, NGC5055, NGC6946)
  + 2 controls (NGC2976, NGC7331)
- HALOGAS DR1 (Heald et al. 2020, Zenodo 10.5281/zenodo.3715549):
  5 shell-bearing (NGC0891, NGC1003, NGC2403, NGC5055, NGC5585)
  + 1 control (NGC4559)
- unWISE NEO7 W1 cutouts: 63 SPARC galaxies (shell-bearing + controls)
- Walton et al. (2022) ULX catalog: cross-matched against all shell positions

**SPARC rotation curve catalog:**
- Lelli, McGaugh & Schombert (2016), AJ 152, 157
- Shell decomposition from Paper I (PASP-102415)

---

## Methods

- Deprojection using published PA, inclination, and distance for each galaxy
- Azimuthal MOM0 profiles in 36 bins (10° each) at each shell radius
- Shell annulus: r_sh ± 0.25×r_sh; off-shell reference: r_sh + [1×, 3×] dsh
- S/N metric: median in-shell / median off-shell + percentile rank in host galaxy
- Control baseline: max natural ratio from n=0 SPARC galaxies

---

## Repositories

- Code: github.com/RonBibb/sparc-shells-relaxed
- Paper I: github.com/RonBibb/sparc-halo-shells
- Zenodo concept DOI (Paper I): 10.5281/zenodo.20072882

---

*Draft prepared: 2026-05-25*
