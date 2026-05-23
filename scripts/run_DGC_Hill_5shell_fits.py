"""
run_relaxed_caps_fits_v2.py — Hill-sphere-constrained, DGC-scaled, up-to-5-shell fits

Major changes from v1 (relaxed_caps_fits.py):

  (1) DGC SCALING (replaces M_BH = M_sh/4000):
      M_BH = K * M_sh^4.35 (super-linear; calibrated at MW)
      log10(K) = -37.785 (so MW anchor M_BH=4e6, M_halo=1.6e10 reproduces 4000x ratio)
      Based on Davis, Graham & Combes 2019: M_BH proportional to M_DM^4.35
      
      Note: this is super-linear. Doubling M_sh increases M_BH by 2^4.35 = 20x.
      At MW anchor: ratio M_halo/M_BH = 4000.
      At larger M_sh: ratio drops fast (e.g., M_sh=4e10 gives M_halo/M_BH ~ 150).

  (2) HILL SPHERE CONSTRAINT (per-shell):
      Implied M_BH (from DGC) must not exceed the Hill-sphere limit at the
      shell's (r, sigma) configuration:
          M_BH_max(r, sigma) = k_Hill * M_galaxy(<r) * (sigma/r)^3 / 3
      where k_Hill is configurable. Default k_Hill=1 (strict Hill sphere).
      Set k_Hill=inf to disable. Implemented as HARD REJECTION during fit
      multi-restart (configurations violating Hill are dropped).

  (3) SHELL COUNT EXTENDED 2 -> 5:
      n_shells in {0, 1, 2, 3, 4, 5}, BIC compared across all.

  (4) LARGE-SHELL PREFERENCE:
      When BIC values are close (delta_BIC < BIC_TIE_THRESHOLD = 6.0),
      prefer the configuration with FEWER shells (which forces larger shells
      to absorb the same total). Configurable.

  (5) CENTRAL BH ESTIMATE from backbone:
      M_central estimated from V_flat via M-sigma (Kormendy & Ho 2013):
          log(M_BH/Msun) = 8.5 + 4.4 * log(sigma / 200 km/s)
      where sigma ~ V_flat / sqrt(2)
      Reported in output for diagnostic / interpretation.

OUTPUT: sparc_T2-T9_DGC_Hill_5shell_fits.csv
        Same schema as v1 plus:
        - {n}_M_BH_implied (DGC-derived implied IMBH mass for each shell)
        - {n}_Hill_ratio   (M_BH_implied / M_BH_Hill_max; > 1 means violates)
        - M_central_estimate
        - fw_n3/n4/n5 columns matching n1/n2 schema
"""

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DATA = '/Users/ronbibb/Library/CloudStorage/OneDrive-Personal(2)/Documents/Academic/Rotmod_LTG'
_LOCAL_DATA = os.path.join(os.getcwd(), 'Rotmod_LTG')
DATA_DIR = _PACKAGE_DATA if os.path.isdir(_PACKAGE_DATA) else _LOCAL_DATA
SAMPLE_CSV = '/Users/ronbibb/Library/CloudStorage/OneDrive-Personal(2)/Documents/Academic/paper2_package/data/sparc_sample123.csv'
OUTPUT_CSV = '../data/sparc_T2-T9_DGC_Hill_5shell_fits.csv'
LOG_FILE   = '../data/run_DGC_Hill_5shell_fits.log'

# Physics constants
G = 4.302e-6   # kpc (km/s)^2 / M_sun
SIGMA_V_FLOOR = 1.0
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7

T_MIN = 2
T_MAX = 9

# Cap parameters (still relaxed)
SHELL_R_MAX_GRID = [12.0, 30.0, 80.0]
SHELL_WIDTH_MAX_FRAC = 0.8
SHELL_M_MAX = 1e12

# NEW: DGC scaling
# DGC 2019: M_BH ∝ M_DM^4.35 (super-linear). For shell mass M_sh treated as
# localized halo component: M_BH = K * M_sh^4.35.
# Calibrated at MW: (M_BH=4e6, M_halo=1.6e10), giving K = 4e6 / (1.6e10)^4.35
DGC_SLOPE = 4.35        # M_BH ∝ M_sh^DGC_SLOPE
# Compute K in log space for numerical stability
# log10(K) = log10(4e6) - 4.35 * log10(1.6e10) = 6.602 - 4.35 * 10.204 = -37.785
DGC_LOG10_K = 6.602 - DGC_SLOPE * 10.204   # = -37.785

# NEW: Hill sphere constraint
ENABLE_HILL_CONSTRAINT = True
HILL_K_FACTOR = 1.0     # 1.0 = strict Hill sphere; >1 relaxes
                        # set to np.inf to disable

# NEW: M_BH vs M_central constraint
# An off-nuclear BH should not exceed some fraction of the central SMBH.
#   1e-3: conservative IMBH (clearly subordinate to central)
#   1e-2: relaxed IMBH (high-mass IMBH / low-mass off-nuclear AGN)
#   1.0:  permissive (BH only constrained by Hill sphere)
#   inf:  disabled
ENABLE_MCENTRAL_CONSTRAINT = True
MCENTRAL_MAX_FRACTION = 1e-2   # max M_BH_implied / M_central

# NEW: Large-shell preference
ENABLE_LARGE_SHELL_PREFERENCE = True
BIC_TIE_THRESHOLD = 6.0  # if delta_BIC < this, prefer fewer shells

# NEW: max shells
MAX_SHELLS = 5

# NEW: M-sigma for central BH
KORMENDY_HO_A = 8.5     # log(M_BH/Msun) = A + B * log(sigma/200)
KORMENDY_HO_B = 4.4
SIGMA_FROM_VFLAT_FACTOR = 1.0 / np.sqrt(2)  # sigma ~ V_flat / sqrt(2)

TIME_BUDGET_PER_FIT = 60


# ============================================================
# Physics (unchanged from v1)
# ============================================================
def baryon_squared(vg, vd, vb):
    return (vg * np.abs(vg)
            + UPSILON_DISK * vd * np.abs(vd)
            + UPSILON_BULGE * vb * np.abs(vb))

def burkert_v2(r, rho_0, a):
    a = max(a, 1e-6)
    x = r / a
    M_enc = np.pi * rho_0 * a**3 * (
        np.log(1 + x**2) + 2 * np.log(1 + x) - 2 * np.arctan(x))
    return np.maximum(G * M_enc / np.maximum(r, 1e-6), 0)

def burkert_M_enclosed(r, rho_0, a):
    """Burkert enclosed mass at radius r (M_sun). Used for Hill sphere check."""
    a = max(a, 1e-6)
    x = r / a
    return np.pi * rho_0 * a**3 * (
        np.log(1 + x**2) + 2 * np.log(1 + x) - 2 * np.arctan(x))

def baryon_M_enclosed_at_r(r_target, r_arr, V2_bar_arr):
    """Estimate baryonic enclosed mass at r_target via V^2 * r / G."""
    # Interpolate V^2_bar to r_target, then M_bar = V^2 * r / G
    V2_at_r = np.interp(r_target, r_arr, V2_bar_arr)
    return V2_at_r * r_target / G

def nfw_v2(r, rho_s, r_s):
    r_s = max(r_s, 1e-6)
    x = r / r_s
    M_enc = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))
    return np.maximum(G * M_enc / np.maximum(r, 1e-6), 0)

def shell_v2(r, M_sh, r_sh, sigma_sh):
    sigma = max(sigma_sh, 1e-6)
    M_enc = 0.5 * M_sh * (1 + erf((r - r_sh) / (np.sqrt(2) * sigma)))
    return np.maximum(G * M_enc / np.maximum(r, 1e-6), 0)


# ============================================================
# DGC scaling and Hill sphere helpers
# ============================================================
def shell_to_implied_BH(M_sh):
    """Convert shell mass to implied BH mass via DGC scaling.
    M_BH = K * M_sh^4.35 (DGC super-linear; calibrated at MW anchor)
    Computed in log space for numerical stability.
    """
    M_sh = max(M_sh, 1e-3)
    log_MBH = DGC_LOG10_K + DGC_SLOPE * np.log10(M_sh)
    return 10 ** log_MBH

def hill_BH_max(r_sh, sigma_sh, M_gal_at_r, k_factor=HILL_K_FACTOR):
    """Maximum BH mass at radius r_sh with extent sigma_sh,
    given enclosed galaxy mass M_gal_at_r.

    Strict Hill sphere: r_Hill = sigma_sh requires
        M_BH = M_gal * (sigma/r)^3 / 3
    Multiplied by k_factor to allow relaxation.
    """
    if k_factor == np.inf:
        return np.inf
    return k_factor * M_gal_at_r * (sigma_sh / r_sh) ** 3 / 3.0

def shell_violates_constraint(M_sh, r_sh, sigma_sh, M_gal_func, M_central):
    """True if implied M_BH (DGC) violates EITHER:
       - Hill sphere bound at (r_sh, sigma_sh), OR
       - M_BH_implied > MCENTRAL_MAX_FRACTION * M_central
    """
    M_BH_implied = shell_to_implied_BH(M_sh)

    # Hill sphere check
    if ENABLE_HILL_CONSTRAINT and HILL_K_FACTOR != np.inf:
        M_BH_Hill = hill_BH_max(r_sh, sigma_sh, M_gal_func(r_sh))
        if M_BH_implied > M_BH_Hill:
            return True

    # Central SMBH ratio check
    if ENABLE_MCENTRAL_CONSTRAINT and MCENTRAL_MAX_FRACTION != np.inf:
        if M_BH_implied > MCENTRAL_MAX_FRACTION * M_central:
            return True

    return False


# Backward-compatible alias used elsewhere in script
def shell_violates_hill(M_sh, r_sh, sigma_sh, M_gal_func, M_central=None):
    """Legacy name; combined Hill + M_central check.
    M_central must be provided for the second constraint to apply.
    """
    if M_central is None:
        M_central = np.inf  # disables the M_central check if not given
    return shell_violates_constraint(M_sh, r_sh, sigma_sh, M_gal_func, M_central)

def central_BH_estimate(V_flat):
    """Estimate central SMBH mass from V_flat via M-sigma (Kormendy & Ho 2013).
    sigma ~ V_flat / sqrt(2)
    log10(M_BH/Msun) = 8.5 + 4.4 * log10(sigma / 200 km/s)
    """
    sigma = V_flat * SIGMA_FROM_VFLAT_FACTOR
    log_M = KORMENDY_HO_A + KORMENDY_HO_B * np.log10(sigma / 200.0)
    return 10 ** log_M


# ============================================================
# Models  (n_shells = 0,1,2,3,4,5)
# ============================================================
def model_burkert(V2_bar):
    def vt(r, rho_0, a):
        return np.sqrt(np.maximum(V2_bar + burkert_v2(r, rho_0, a), 0))
    return vt

def model_nfw(V2_bar):
    def vt(r, rho_s, r_s):
        return np.sqrt(np.maximum(V2_bar + nfw_v2(r, rho_s, r_s), 0))
    return vt

def make_fw_model(V2_bar, n_shells):
    """Returns vt(r, rho_0, a, [M_i, r_i, sf_i] for i=1..n_shells)"""
    def vt(r, *params):
        rho_0, a = params[0], params[1]
        v2 = V2_bar + burkert_v2(r, rho_0, a)
        for i in range(n_shells):
            base = 2 + 3 * i
            M_i, r_i, sf_i = params[base], params[base+1], params[base+2]
            sigma_i = sf_i * r_i
            v2 = v2 + shell_v2(r, M_i, r_i, sigma_i)
        return np.sqrt(np.maximum(v2, 0))
    return vt


# ============================================================
# Fitters
# ============================================================
def fit_burkert(r, vobs, sig, V2_bar, time_budget=TIME_BUDGET_PER_FIT):
    vt = model_burkert(V2_bar)
    starts = [[3e7, 8.0], [1e7, 15.0], [1e8, 4.0], [3e6, 30.0], [3e8, 2.0],
              [1e6, 50.0], [5e7, 6.0], [1e9, 1.5]]
    bounds = ([1e3, 0.1], [1e10, 200.0])
    best_chi2 = np.inf; best_p = None
    t0 = time.time()
    for p0 in starts:
        try:
            p, _ = curve_fit(vt, r, vobs, p0=p0, bounds=bounds, sigma=sig, maxfev=15000)
            chi2 = float(np.sum(((vobs - vt(r, *p)) / sig)**2))
            if chi2 < best_chi2:
                best_chi2 = chi2; best_p = p
        except Exception:
            continue
        if time.time() - t0 > time_budget:
            break
    return best_p, best_chi2

def fit_nfw(r, vobs, sig, V2_bar, time_budget=TIME_BUDGET_PER_FIT):
    vt = model_nfw(V2_bar)
    starts = [[1e7, 15.0], [5e6, 25.0], [3e7, 8.0], [2e6, 50.0],
              [5e7, 5.0], [1e6, 100.0], [3e7, 12.0], [1e8, 4.0]]
    bounds = ([1e3, 0.1], [1e10, 500.0])
    best_chi2 = np.inf; best_p = None
    t0 = time.time()
    for p0 in starts:
        try:
            p, _ = curve_fit(vt, r, vobs, p0=p0, bounds=bounds, sigma=sig, maxfev=15000)
            chi2 = float(np.sum(((vobs - vt(r, *p)) / sig)**2))
            if chi2 < best_chi2:
                best_chi2 = chi2; best_p = p
        except Exception:
            continue
        if time.time() - t0 > time_budget:
            break
    return best_p, best_chi2


def fit_fw_n_shells(r, vobs, sig, V2_bar, n_shells,
                     M_gal_func=None, M_central=None,
                     time_budget=TIME_BUDGET_PER_FIT):
    """Fit framework model with n_shells in {0..5}.
    If M_gal_func and M_central provided AND constraints enabled, reject shell
    configurations that violate Hill sphere OR M_central ratio bounds.
    """
    if n_shells == 0:
        return fit_burkert(r, vobs, sig, V2_bar, time_budget)

    vt = make_fw_model(V2_bar, n_shells)

    # Generate multi-restart initial conditions
    starts = []
    for r_max in SHELL_R_MAX_GRID:
        for rho0 in [1e7, 3e7, 1e8]:
            for a_init in [5.0, 10.0, 20.0]:
                # Evenly distribute shells across the radial range
                shell_radii = [(i + 1) / (n_shells + 1) * r_max for i in range(n_shells)]
                M_init = min(3e9 * (r_max / 12.0), 5e11)
                p0 = [rho0, a_init]
                for r_i in shell_radii:
                    p0.extend([M_init, r_i, 0.25])
                starts.append((p0, r_max))

    best_chi2 = np.inf
    best_p_physical = None
    t0 = time.time()

    # Bounds
    lb_base = [1e3, 0.1]
    ub_base_template = [1e10, 200.0]

    for p0, r_max in starts:
        lb = list(lb_base)
        ub = list(ub_base_template)
        for i in range(n_shells):
            lb.extend([1e6, 0.2, 0.01])
            ub.extend([SHELL_M_MAX, r_max, SHELL_WIDTH_MAX_FRAC])

        try:
            p, _ = curve_fit(vt, r, vobs, p0=p0, bounds=(lb, ub),
                             sigma=sig, maxfev=20000)

            # Combined Hill + M_central check on fitted shells (HARD REJECTION)
            if (ENABLE_HILL_CONSTRAINT or ENABLE_MCENTRAL_CONSTRAINT) and M_gal_func is not None:
                violates = False
                for i in range(n_shells):
                    base = 2 + 3 * i
                    M_i, r_i, sf_i = p[base], p[base+1], p[base+2]
                    sigma_i = sf_i * r_i
                    if shell_violates_constraint(M_i, r_i, sigma_i, M_gal_func, M_central):
                        violates = True
                        break
                if violates:
                    continue  # reject this fit; try next restart

            chi2 = float(np.sum(((vobs - vt(r, *p)) / sig)**2))
            if chi2 < best_chi2:
                best_chi2 = chi2
                # Convert sigma_frac to sigma in result
                rho_0_fit, a_fit = p[0], p[1]
                physical = [rho_0_fit, a_fit]
                for i in range(n_shells):
                    base = 2 + 3 * i
                    M_i, r_i, sf_i = p[base], p[base+1], p[base+2]
                    physical.extend([M_i, r_i, sf_i * r_i])
                best_p_physical = np.array(physical)

        except Exception:
            continue

        if time.time() - t0 > time_budget:
            break

    return best_p_physical, best_chi2


# ============================================================
# BIC, model selection with large-shell preference
# ============================================================
def bic_of(chi2, n_pts_used, k):
    return chi2 + k * np.log(n_pts_used)

def chi2_red_of(chi2, n_pts_used, k):
    return chi2 / max(n_pts_used - k, 1)

def select_best_n_shells(bics):
    """Select best n_shells with large-shell preference rule.
    If ENABLE_LARGE_SHELL_PREFERENCE: when delta_BIC < threshold,
    prefer the lower n_shells (forces larger shells).
    """
    bics = np.asarray(bics)
    if not ENABLE_LARGE_SHELL_PREFERENCE:
        return int(np.argmin(bics))

    n_min = int(np.argmin(bics))
    bic_min = bics[n_min]
    # Among configurations within BIC_TIE_THRESHOLD of minimum,
    # prefer the one with FEWEST shells
    candidates = np.where(bics - bic_min < BIC_TIE_THRESHOLD)[0]
    return int(candidates.min())


def fit_one_galaxy(galaxy, T, V_flat, logM_star, logM_halo, log_fh):
    rotmod_path = os.path.join(DATA_DIR, f'{galaxy}_rotmod.dat')
    if not os.path.exists(rotmod_path):
        msg = f"  [SKIP] {galaxy}: rotmod file not found"
        print(msg); log_fh.write(msg + '\n')
        return None

    d = np.loadtxt(rotmod_path, comments='#')
    r_all, vobs_all, evobs_all = d[:, 0], d[:, 1], d[:, 2]
    vgas, vdisk, vbul = d[:, 3], d[:, 4], d[:, 5]
    Vbar2_all = baryon_squared(vgas, vdisk, vbul)

    n_total = len(r_all)
    mask = vobs_all**2 > Vbar2_all
    n_used = int(mask.sum())
    n_excluded = n_total - n_used

    if n_used < 5:
        msg = f"  [SKIP] {galaxy}: only {n_used} usable points"
        print(msg); log_fh.write(msg + '\n')
        return None

    r = r_all[mask]; vobs = vobs_all[mask]; evobs = evobs_all[mask]
    Vbar2 = Vbar2_all[mask]
    sig = np.maximum(evobs, SIGMA_V_FLOOR)

    print(f"\n  {galaxy} (T={T}, V_flat={V_flat:.0f}, n_pts={n_used}/{n_total})")
    log_fh.write(f"\n{galaxy} (T={T}, V_flat={V_flat:.0f}, n_pts={n_used}/{n_total})\n")

    # Central BH estimate from V_flat
    M_central = central_BH_estimate(V_flat)
    log_fh.write(f"  M_central estimate: {M_central:.2e} M_sun (from V_flat={V_flat:.1f})\n")

    # 1) Fit Burkert first - we use it for M_gal_func in Hill constraint
    t0 = time.time()
    p_burk, chi2_burk = fit_burkert(r, vobs, sig, Vbar2)
    if p_burk is None:
        return None
    chi2r_burk = chi2_red_of(chi2_burk, n_used, 2)
    bic_burk = bic_of(chi2_burk, n_used, 2)
    print(f"    Burkert: chi^2={chi2_burk:.2f}, chi^2_r={chi2r_burk:.2f} ({time.time()-t0:.1f}s)")

    rho_0_burk, a_burk = p_burk

    # M_galaxy(<r) function: baryons + Burkert halo
    def M_gal_func(r_target):
        M_burk = burkert_M_enclosed(r_target, rho_0_burk, a_burk)
        M_bar = baryon_M_enclosed_at_r(r_target, r, Vbar2)
        return M_burk + M_bar

    # NFW
    t0 = time.time()
    p_nfw, chi2_nfw = fit_nfw(r, vobs, sig, Vbar2)
    if p_nfw is None:
        return None
    chi2r_nfw = chi2_red_of(chi2_nfw, n_used, 2)
    bic_nfw = bic_of(chi2_nfw, n_used, 2)
    print(f"    NFW:     chi^2={chi2_nfw:.2f}, chi^2_r={chi2r_nfw:.2f} ({time.time()-t0:.1f}s)")

    # FW fits for n_shells = 0..MAX_SHELLS
    fw_chi2 = [chi2_burk]
    fw_chi2r = [chi2r_burk]
    fw_bic = [bic_burk]
    fw_params = [p_burk]

    for n_sh in range(1, MAX_SHELLS + 1):
        t0 = time.time()
        # k = 2 backbone params + 3 per shell
        k = 2 + 3 * n_sh
        p_fw, chi2_fw = fit_fw_n_shells(r, vobs, sig, Vbar2, n_sh,
                                          M_gal_func=M_gal_func,
                                          M_central=M_central)
        if p_fw is None:
            print(f"    FW {n_sh}sh: NO FIT (Hill rejected all restarts)")
            log_fh.write(f"  FW{n_sh}: NO FIT (Hill rejected)\n")
            fw_chi2.append(np.inf)
            fw_chi2r.append(np.inf)
            fw_bic.append(np.inf)
            fw_params.append(None)
            continue
        chi2r_fw = chi2_red_of(chi2_fw, n_used, k)
        bic_fw = bic_of(chi2_fw, n_used, k)
        print(f"    FW {n_sh}sh:  chi^2={chi2_fw:.2f}, chi^2_r={chi2r_fw:.2f}, BIC={bic_fw:.2f} ({time.time()-t0:.1f}s)")
        fw_chi2.append(chi2_fw)
        fw_chi2r.append(chi2r_fw)
        fw_bic.append(bic_fw)
        fw_params.append(p_fw)

    # Select best n_shells with large-shell preference
    fw_best_n_shells = select_best_n_shells(fw_bic)
    fw_best_chi2 = fw_chi2[fw_best_n_shells]
    fw_best_chi2r = fw_chi2r[fw_best_n_shells]
    fw_best_bic = fw_bic[fw_best_n_shells]
    fw_best_params = fw_params[fw_best_n_shells]

    # Naive BIC-min (without preference rule) for comparison
    fw_naive_n = int(np.argmin(fw_bic))
    naive_preferred_flag = (fw_naive_n != fw_best_n_shells)

    print(f"    BIC-best (naive): n_shells={fw_naive_n}, BIC={fw_bic[fw_naive_n]:.2f}")
    print(f"    BIC-best (large-shell pref): n_shells={fw_best_n_shells}, BIC={fw_best_bic:.2f}")
    log_fh.write(f"  Burk: BIC={bic_burk:.2f}\n")
    log_fh.write(f"  NFW:  BIC={bic_nfw:.2f}\n")
    for n_sh in range(MAX_SHELLS + 1):
        log_fh.write(f"  FW{n_sh}: BIC={fw_bic[n_sh]:.2f}\n")
    log_fh.write(f"  Best n_shells: {fw_best_n_shells} (naive: {fw_naive_n})\n")

    # Build result row
    result = {
        'Galaxy': galaxy, 'T': T, 'V_flat': V_flat,
        'logM_star': logM_star, 'logM_halo': logM_halo,
        'n_pts_total': n_total, 'n_pts_used': n_used, 'n_excluded': n_excluded,
        'is_clean': bool(n_excluded == 0),
        'M_central_estimate': M_central,
        'burk_rho0': p_burk[0], 'burk_a_kpc': p_burk[1],
        'burk_chi2': chi2_burk, 'burk_chi2_red': chi2r_burk, 'burk_bic': bic_burk,
        'nfw_rho_s': p_nfw[0], 'nfw_r_s_kpc': p_nfw[1],
        'nfw_chi2': chi2_nfw, 'nfw_chi2_red': chi2r_nfw, 'nfw_bic': bic_nfw,
        'fw_best_n_shells': fw_best_n_shells,
        'fw_naive_best_n_shells': fw_naive_n,
        'large_shell_pref_used': naive_preferred_flag,
        'fw_best_chi2': fw_best_chi2,
        'fw_best_chi2_red': fw_best_chi2r,
        'fw_best_bic': fw_best_bic,
    }

    # Per-n_shells results (extract shells, implied M_BH, Hill check)
    for n_sh in range(MAX_SHELLS + 1):
        result[f'fw_n{n_sh}_chi2']     = fw_chi2[n_sh]
        result[f'fw_n{n_sh}_chi2_red'] = fw_chi2r[n_sh]
        result[f'fw_n{n_sh}_bic']      = fw_bic[n_sh]
        if n_sh == 0 or fw_params[n_sh] is None:
            for i in range(1, MAX_SHELLS + 1):
                result[f'fw_n{n_sh}_M_sh{i}']      = np.nan
                result[f'fw_n{n_sh}_r_sh{i}_kpc']  = np.nan
                result[f'fw_n{n_sh}_sigma_sh{i}_kpc'] = np.nan
                result[f'fw_n{n_sh}_M_BH_impl{i}'] = np.nan
                result[f'fw_n{n_sh}_Hill_ratio{i}'] = np.nan
                result[f'fw_n{n_sh}_central_ratio{i}'] = np.nan
            continue
        p_fw = fw_params[n_sh]
        for i in range(1, MAX_SHELLS + 1):
            if i > n_sh:
                result[f'fw_n{n_sh}_M_sh{i}']      = np.nan
                result[f'fw_n{n_sh}_r_sh{i}_kpc']  = np.nan
                result[f'fw_n{n_sh}_sigma_sh{i}_kpc'] = np.nan
                result[f'fw_n{n_sh}_M_BH_impl{i}'] = np.nan
                result[f'fw_n{n_sh}_Hill_ratio{i}'] = np.nan
                result[f'fw_n{n_sh}_central_ratio{i}'] = np.nan
                continue
            base = 2 + 3 * (i - 1)
            M_i = p_fw[base]
            r_i = p_fw[base+1]
            sigma_i = p_fw[base+2]   # already converted to physical sigma
            M_BH_impl = shell_to_implied_BH(M_i)
            M_BH_Hill = hill_BH_max(r_i, sigma_i, M_gal_func(r_i))
            hill_ratio = M_BH_impl / M_BH_Hill if M_BH_Hill > 0 and np.isfinite(M_BH_Hill) else np.inf
            central_ratio = M_BH_impl / M_central if M_central > 0 else np.inf
            result[f'fw_n{n_sh}_M_sh{i}']      = M_i
            result[f'fw_n{n_sh}_r_sh{i}_kpc']  = r_i
            result[f'fw_n{n_sh}_sigma_sh{i}_kpc'] = sigma_i
            result[f'fw_n{n_sh}_M_BH_impl{i}'] = M_BH_impl
            result[f'fw_n{n_sh}_Hill_ratio{i}'] = hill_ratio
            result[f'fw_n{n_sh}_central_ratio{i}'] = central_ratio

    result['ratio_fw_over_burk'] = fw_best_chi2 / chi2_burk if chi2_burk > 0 else np.nan
    result['ratio_fw_over_nfw']  = fw_best_chi2 / chi2_nfw  if chi2_nfw > 0  else np.nan
    return result


def main():
    print("=" * 78)
    print("DGC-SCALED HILL-CONSTRAINED UP-TO-5-SHELL FITS")
    print("=" * 78)
    print(f"  DGC scaling:           M_BH = 10^{DGC_LOG10_K:.3f} * M_sh^{DGC_SLOPE}")
    print(f"  Hill constraint:       enabled={ENABLE_HILL_CONSTRAINT}, k_factor={HILL_K_FACTOR}")
    print(f"  M_central constraint:  enabled={ENABLE_MCENTRAL_CONSTRAINT}, max_frac={MCENTRAL_MAX_FRACTION:.1e}")
    print(f"  Large-shell pref:      enabled={ENABLE_LARGE_SHELL_PREFERENCE}, delta_BIC tol={BIC_TIE_THRESHOLD}")
    print(f"  Max shells:            {MAX_SHELLS}")
    print(f"  Caps: M={SHELL_M_MAX:.0e}, sigma/r={SHELL_WIDTH_MAX_FRAC}, r_grid={SHELL_R_MAX_GRID}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {OUTPUT_CSV}")
    print("=" * 78)

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: rotmod directory '{DATA_DIR}' not found"); sys.exit(1)
    if not os.path.exists(SAMPLE_CSV):
        print(f"ERROR: sample CSV '{SAMPLE_CSV}' not found"); sys.exit(1)

    sample = pd.read_csv(SAMPLE_CSV)
    target = sample[(sample['T'] >= T_MIN) & (sample['T'] <= T_MAX)].copy()
    print(f"\nTarget: {len(target)} galaxies with T in [{T_MIN}, {T_MAX}]")

    # >>> SMOKE TEST: uncomment to test on NGC 5055 only
    # target = target[target['Galaxy'] == 'NGC5055']
    # print(f"  SMOKE TEST: {target['Galaxy'].tolist()}")
    # <<<

    results = []
    t_total_start = time.time()
    with open(LOG_FILE, 'w') as log_fh:
        log_fh.write(f"DGC-Hill-5shell fits log\n")
        log_fh.write(f"DGC: M_BH = 10^{DGC_LOG10_K:.3f} * M_sh^{DGC_SLOPE}\n")
        log_fh.write(f"Hill: enabled={ENABLE_HILL_CONSTRAINT}, k={HILL_K_FACTOR}\n")
        log_fh.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write("=" * 72 + "\n")

        for i, row in target.iterrows():
            galaxy = row['Galaxy']
            T = int(row['T']); V_flat = float(row['Vflat'])
            logM_star = float(row['logM_star']); logM_halo = float(row['logM_halo'])
            try:
                result = fit_one_galaxy(galaxy, T, V_flat, logM_star, logM_halo, log_fh)
                if result is not None:
                    results.append(result)
            except Exception as e:
                msg = f"  [ERROR] {galaxy}: {e}"
                print(msg); log_fh.write(msg + '\n')
                continue
            elapsed = time.time() - t_total_start
            done = len(results)
            remaining = len(target) - i - 1
            avg = elapsed / max(done, 1)
            eta_s = avg * remaining
            print(f"  [{done}/{len(target)}] elapsed: {elapsed/60:.1f}m, ETA: {eta_s/60:.1f}m")

        log_fh.write(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write(f"Total: {(time.time() - t_total_start)/60:.1f}m\n")
        log_fh.write(f"Fit: {len(results)}/{len(target)}\n")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 78}\nDONE\n{'=' * 78}")
    print(f"Galaxies fit: {len(results)}/{len(target)}")
    print(f"Total time:   {(time.time() - t_total_start)/60:.1f}m")
    print(f"Output:       {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
