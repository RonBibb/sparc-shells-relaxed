"""
run_relaxed_caps_fits.py — Cap-relaxation sweep producer

Re-runs the canonical Burkert / NFW / framework (0/1/2 shell) fits across the
102-galaxy SPARC T=2-9 sample with all three architectural caps relaxed:

    M_shell upper bound:   5e10 M_sun  →  1e12 M_sun        (20× factor)
    sigma/r_shell cap:     0.4         →  0.8               (2× factor)
    r_shell upper bound:   {3, 6, 12} kpc grid  →  {12, 30, 80} kpc grid

Everything else identical to run_canonical_fits.py: Upsilon = (0.5, 0.7),
sigma_V floor = 1.0 km/s, exclusion rule V_obs^2 > V_bar^2, BIC selection
over n_shells in {0, 1, 2}, multi-restart structure, baryon decomposition,
chi^2 / BIC formulas.

Usage:
  Place this in the same directory as run_canonical_fits.py with:
    - Rotmod_LTG/                       (SPARC rotmod .dat files)
    - sparc_sample123.csv               (SPARC catalog)
  Then:
    python3 run_relaxed_caps_fits.py
  Output:
    - sparc_T2-T9_relaxed_caps_fits.csv (102 rows × same schema as canonical CSV)
    - run_relaxed_caps_fits.log

Sanity check before launching the full sweep: run two galaxies first as smoke
test by editing main() to slice target.head(2) — or fit NGC5371 and NGC6674
specifically (set sample = sample[sample['Galaxy'].isin(['NGC5371','NGC6674'])]).
Expected results from in-session reproduction:
  NGC5371: cap=0.8 yields n=1, r=4.25 kpc, M=3.35e10, sigma/r=0.789, chi^2_red~1.11
  NGC6674: cap=0.8 yields n=1, r=58.5 kpc, M=4.27e11, sigma/r=0.205, chi^2_red~0.74

Schema is IDENTICAL to canonical CSV so the two outputs can be joined on
'Galaxy' for direct comparison.
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
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PACKAGE_DATA = '/Users/ronbibb/Library/CloudStorage/OneDrive-Personal(2)/Documents/Academic/Rotmod_LTG'
_LOCAL_DATA = _os.path.join(_os.getcwd(), 'Rotmod_LTG')
DATA_DIR = _PACKAGE_DATA if _os.path.isdir(_PACKAGE_DATA) else _LOCAL_DATA
SAMPLE_CSV = '../data/sparc_sample123.csv'
OUTPUT_CSV = '../sparc_T2-T9_relaxed_caps_fits.csv'   # <<< CHANGED filename
LOG_FILE   = './run_relaxed_caps_fits.log'           # <<< CHANGED filename

G = 4.302e-6
SIGMA_V_FLOOR = 1.0
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7

T_MIN = 2
T_MAX = 9

# >>> CHANGED: relaxed cap parameters
SHELL_R_MAX_GRID = [12.0, 30.0, 80.0]   # was [3.0, 6.0, 12.0]
SHELL_WIDTH_MAX_FRAC = 0.8              # was 0.4
SHELL_M_MAX = 1e12                      # was 5e10 (hard-coded; now a constant)
# <<<

TIME_BUDGET_PER_FIT = 60


# ============================================================
# Physics  (UNCHANGED)
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
# Models  (UNCHANGED)
# ============================================================
def model_burkert(V2_bar):
    def vt(r, rho_0, a):
        return np.sqrt(np.maximum(V2_bar + burkert_v2(r, rho_0, a), 0))
    return vt

def model_nfw(V2_bar):
    def vt(r, rho_s, r_s):
        return np.sqrt(np.maximum(V2_bar + nfw_v2(r, rho_s, r_s), 0))
    return vt

def model_fw1(V2_bar):
    def vt(r, rho_0, a, M_sh, r_sh, sigma_frac):
        sigma_sh = sigma_frac * r_sh
        v2 = V2_bar + burkert_v2(r, rho_0, a) + shell_v2(r, M_sh, r_sh, sigma_sh)
        return np.sqrt(np.maximum(v2, 0))
    return vt

def model_fw2(V2_bar):
    def vt(r, rho_0, a, M1, r1, sf1, M2, r2, sf2):
        s1 = sf1 * r1; s2 = sf2 * r2
        v2 = (V2_bar + burkert_v2(r, rho_0, a)
              + shell_v2(r, M1, r1, s1) + shell_v2(r, M2, r2, s2))
        return np.sqrt(np.maximum(v2, 0))
    return vt


# ============================================================
# Fitters  (Burkert/NFW UNCHANGED; FW fitter changed bounds)
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


def fit_fw_n_shells(r, vobs, sig, V2_bar, n_shells, time_budget=TIME_BUDGET_PER_FIT):
    """
    Multi-restart framework fit with n_shells in {0, 1, 2}.
    Relaxed-cap version: M_sh ≤ 1e12, sigma_frac ≤ 0.8, r_sh ≤ {12, 30, 80} kpc.
    """
    if n_shells == 0:
        return fit_burkert(r, vobs, sig, V2_bar, time_budget)

    if n_shells == 1:
        vt = model_fw1(V2_bar)
        starts = []
        for r_max in SHELL_R_MAX_GRID:
            for rho0 in [1e7, 3e7, 1e8]:
                for a_init in [5.0, 10.0, 20.0]:
                    for r_frac in [0.3, 0.5, 0.7]:
                        r_sh_init = r_frac * r_max
                        # Initial M scaled up to encourage exploring the relaxed-mass regime
                        M_init = min(3e9 * (r_max / 12.0), 5e11)
                        starts.append(([rho0, a_init, M_init, r_sh_init, 0.25], r_max))
    else:  # n_shells == 2
        vt = model_fw2(V2_bar)
        starts = []
        for r_max in SHELL_R_MAX_GRID:
            for rho0 in [1e7, 3e7, 1e8]:
                for a_init in [5.0, 10.0, 20.0]:
                    r1_init = r_max / 3
                    r2_init = 2 * r_max / 3
                    M_init = min(3e9 * (r_max / 12.0), 5e11)
                    starts.append((
                        [rho0, a_init, M_init, r1_init, 0.25,
                                       M_init, r2_init, 0.25],
                        r_max))

    best_chi2 = np.inf
    best_p_physical = None
    t0 = time.time()

    for p0, r_max in starts:
        # >>> CHANGED: bounds use relaxed constants
        if n_shells == 1:
            lb = [1e3, 0.1, 1e6, 0.2, 0.01]
            ub = [1e10, 200.0, SHELL_M_MAX, r_max, SHELL_WIDTH_MAX_FRAC]
        else:
            lb = [1e3, 0.1, 1e6, 0.2, 0.01, 1e6, 0.2, 0.01]
            ub = [1e10, 200.0,
                  SHELL_M_MAX, r_max, SHELL_WIDTH_MAX_FRAC,
                  SHELL_M_MAX, r_max, SHELL_WIDTH_MAX_FRAC]
        # <<<

        try:
            p, _ = curve_fit(vt, r, vobs, p0=p0, bounds=(lb, ub),
                             sigma=sig, maxfev=20000)
            chi2 = float(np.sum(((vobs - vt(r, *p)) / sig)**2))
            if chi2 < best_chi2:
                best_chi2 = chi2
                if n_shells == 1:
                    rho0, a, M1, r1, sf1 = p
                    best_p_physical = np.array([rho0, a, M1, r1, sf1 * r1])
                else:
                    rho0, a, M1, r1, sf1, M2, r2, sf2 = p
                    best_p_physical = np.array([rho0, a, M1, r1, sf1 * r1,
                                                 M2, r2, sf2 * r2])
        except Exception:
            continue

        if time.time() - t0 > time_budget:
            break

    return best_p_physical, best_chi2


# ============================================================
# Helpers / driver / main  (UNCHANGED from canonical)
# ============================================================
def bic_of(chi2, n_pts_used, k):
    return chi2 + k * np.log(n_pts_used)

def chi2_red_of(chi2, n_pts_used, k):
    return chi2 / max(n_pts_used - k, 1)


def fit_one_galaxy(galaxy, T, V_flat, logM_star, logM_halo, log_fh):
    rotmod_path = os.path.join(DATA_DIR, f'{galaxy}_rotmod.dat')
    if not os.path.exists(rotmod_path):
        msg = f"  [SKIP] {galaxy}: rotmod file not found at {rotmod_path}"
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

    t0 = time.time()
    p_burk, chi2_burk = fit_burkert(r, vobs, sig, Vbar2)
    if p_burk is None:
        return None
    chi2r_burk = chi2_red_of(chi2_burk, n_used, 2)
    bic_burk = bic_of(chi2_burk, n_used, 2)
    print(f"    Burkert: chi^2={chi2_burk:.2f}, chi^2_r={chi2r_burk:.2f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    p_nfw, chi2_nfw = fit_nfw(r, vobs, sig, Vbar2)
    if p_nfw is None:
        return None
    chi2r_nfw = chi2_red_of(chi2_nfw, n_used, 2)
    bic_nfw = bic_of(chi2_nfw, n_used, 2)
    print(f"    NFW:     chi^2={chi2_nfw:.2f}, chi^2_r={chi2r_nfw:.2f} ({time.time()-t0:.1f}s)")

    fw_n0_chi2 = chi2_burk
    fw_n0_chi2r = chi2r_burk
    fw_n0_bic = bic_burk

    t0 = time.time()
    p_fw1, chi2_fw1 = fit_fw_n_shells(r, vobs, sig, Vbar2, 1)
    chi2r_fw1 = chi2_red_of(chi2_fw1, n_used, 5)
    bic_fw1 = bic_of(chi2_fw1, n_used, 5)
    print(f"    FW 1sh:  chi^2={chi2_fw1:.2f}, chi^2_r={chi2r_fw1:.2f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    p_fw2, chi2_fw2 = fit_fw_n_shells(r, vobs, sig, Vbar2, 2)
    chi2r_fw2 = chi2_red_of(chi2_fw2, n_used, 8)
    bic_fw2 = bic_of(chi2_fw2, n_used, 8)
    print(f"    FW 2sh:  chi^2={chi2_fw2:.2f}, chi^2_r={chi2r_fw2:.2f} ({time.time()-t0:.1f}s)")

    bics = [fw_n0_bic, bic_fw1, bic_fw2]
    chi2s = [fw_n0_chi2, chi2_fw1, chi2_fw2]
    chi2rs = [fw_n0_chi2r, chi2r_fw1, chi2r_fw2]
    fw_best_n_shells = int(np.argmin(bics))
    fw_best_chi2 = chi2s[fw_best_n_shells]
    fw_best_chi2r = chi2rs[fw_best_n_shells]
    fw_best_bic = bics[fw_best_n_shells]

    print(f"    BIC-best: n_shells={fw_best_n_shells}, chi^2_r={fw_best_chi2r:.2f}, BIC={fw_best_bic:.2f}")
    log_fh.write(f"  Burk: chi^2={chi2_burk:.2f}, chi^2_r={chi2r_burk:.2f}, BIC={bic_burk:.2f}\n")
    log_fh.write(f"  NFW:  chi^2={chi2_nfw:.2f}, chi^2_r={chi2r_nfw:.2f}, BIC={bic_nfw:.2f}\n")
    log_fh.write(f"  FW1:  chi^2={chi2_fw1:.2f}, chi^2_r={chi2r_fw1:.2f}, BIC={bic_fw1:.2f}\n")
    log_fh.write(f"  FW2:  chi^2={chi2_fw2:.2f}, chi^2_r={chi2r_fw2:.2f}, BIC={bic_fw2:.2f}\n")
    log_fh.write(f"  BIC-best: n_shells={fw_best_n_shells}\n")

    return {
        'Galaxy': galaxy, 'T': T, 'V_flat': V_flat,
        'logM_star': logM_star, 'logM_halo': logM_halo,
        'n_pts_total': n_total, 'n_pts_used': n_used, 'n_excluded': n_excluded,
        'is_clean': bool(n_excluded == 0),
        'burk_rho0': p_burk[0], 'burk_a_kpc': p_burk[1],
        'burk_chi2': chi2_burk, 'burk_chi2_red': chi2r_burk, 'burk_bic': bic_burk,
        'nfw_rho_s': p_nfw[0], 'nfw_r_s_kpc': p_nfw[1],
        'nfw_chi2': chi2_nfw, 'nfw_chi2_red': chi2r_nfw, 'nfw_bic': bic_nfw,
        'fw_n0_chi2': fw_n0_chi2, 'fw_n0_chi2_red': fw_n0_chi2r, 'fw_n0_bic': fw_n0_bic,
        'fw_n1_M_sh1': p_fw1[2] if p_fw1 is not None else np.nan,
        'fw_n1_r_sh1_kpc': p_fw1[3] if p_fw1 is not None else np.nan,
        'fw_n1_sigma_sh1_kpc': p_fw1[4] if p_fw1 is not None else np.nan,
        'fw_n1_chi2': chi2_fw1, 'fw_n1_chi2_red': chi2r_fw1, 'fw_n1_bic': bic_fw1,
        'fw_n2_M_sh1': p_fw2[2] if p_fw2 is not None else np.nan,
        'fw_n2_r_sh1_kpc': p_fw2[3] if p_fw2 is not None else np.nan,
        'fw_n2_sigma_sh1_kpc': p_fw2[4] if p_fw2 is not None else np.nan,
        'fw_n2_M_sh2': p_fw2[5] if p_fw2 is not None else np.nan,
        'fw_n2_r_sh2_kpc': p_fw2[6] if p_fw2 is not None else np.nan,
        'fw_n2_sigma_sh2_kpc': p_fw2[7] if p_fw2 is not None else np.nan,
        'fw_n2_chi2': chi2_fw2, 'fw_n2_chi2_red': chi2r_fw2, 'fw_n2_bic': bic_fw2,
        'fw_best_n_shells': fw_best_n_shells,
        'fw_best_chi2': fw_best_chi2, 'fw_best_chi2_red': fw_best_chi2r,
        'fw_best_bic': fw_best_bic,
        'ratio_fw_over_burk': fw_best_chi2 / chi2_burk,
        'ratio_fw_over_nfw': fw_best_chi2 / chi2_nfw,
    }


def main():
    print("=" * 72)
    print("RELAXED CAPS FITS PRODUCER")
    print(f"  M-cap:     {SHELL_M_MAX:.0e}  (canonical: 5e10)")
    print(f"  sigma/r:   {SHELL_WIDTH_MAX_FRAC}  (canonical: 0.4)")
    print(f"  r_sh grid: {SHELL_R_MAX_GRID}  (canonical: [3, 6, 12])")
    print(f"Data directory: {DATA_DIR}")
    print(f"Sample CSV:     {SAMPLE_CSV}")
    print(f"Output CSV:     {OUTPUT_CSV}")
    print(f"Log file:       {LOG_FILE}")
    print("=" * 72)

    if not os.path.isdir(DATA_DIR):
        print(f"\nERROR: rotmod directory not found at '{DATA_DIR}'")
        sys.exit(1)
    if not os.path.exists(SAMPLE_CSV):
        print(f"\nERROR: Sample CSV not found at '{SAMPLE_CSV}'")
        sys.exit(1)

    sample = pd.read_csv(SAMPLE_CSV)
    target = sample[(sample['T'] >= T_MIN) & (sample['T'] <= T_MAX)].copy()
    print(f"\nTarget sample: {len(target)} galaxies with T in [{T_MIN}, {T_MAX}]")
    if len(target) != 102:
        print(f"WARNING: Expected 102 galaxies, found {len(target)}.")

    # >>> SMOKE TEST: uncomment to verify on two reference galaxies before full sweep
    # target = target[target['Galaxy'].isin(['NGC5371','NGC6674'])]
    # print(f"  SMOKE TEST MODE — running only on {target['Galaxy'].tolist()}")
    # <<<

    results = []
    t_total_start = time.time()
    with open(LOG_FILE, 'w') as log_fh:
        log_fh.write(f"Relaxed-caps fits producer log\n")
        log_fh.write(f"M-cap = {SHELL_M_MAX:.0e}, sigma/r = {SHELL_WIDTH_MAX_FRAC}, "
                     f"r_sh grid = {SHELL_R_MAX_GRID}\n")
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
        log_fh.write(f"Total elapsed: {(time.time() - t_total_start)/60:.1f} minutes\n")
        log_fh.write(f"Galaxies fit: {len(results)}/{len(target)}\n")

    df_out = pd.DataFrame(results)
    column_order = [
        'Galaxy', 'T', 'V_flat',
        'burk_a_kpc', 'burk_bic', 'burk_chi2', 'burk_chi2_red', 'burk_rho0',
        'fw_best_bic', 'fw_best_chi2', 'fw_best_chi2_red', 'fw_best_n_shells',
        'fw_n0_bic', 'fw_n0_chi2', 'fw_n0_chi2_red',
        'fw_n1_M_sh1', 'fw_n1_bic', 'fw_n1_chi2', 'fw_n1_chi2_red',
        'fw_n1_r_sh1_kpc', 'fw_n1_sigma_sh1_kpc',
        'fw_n2_M_sh1', 'fw_n2_M_sh2', 'fw_n2_bic', 'fw_n2_chi2', 'fw_n2_chi2_red',
        'fw_n2_r_sh1_kpc', 'fw_n2_r_sh2_kpc', 'fw_n2_sigma_sh1_kpc', 'fw_n2_sigma_sh2_kpc',
        'is_clean', 'logM_halo', 'logM_star',
        'n_excluded', 'n_pts_total', 'n_pts_used',
        'nfw_bic', 'nfw_chi2', 'nfw_chi2_red', 'nfw_r_s_kpc', 'nfw_rho_s',
        'ratio_fw_over_burk', 'ratio_fw_over_nfw',
    ]
    df_out = df_out[column_order]
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 72}\nDONE\n{'=' * 72}")
    print(f"Galaxies fit: {len(results)}/{len(target)}")
    print(f"Total time:   {(time.time() - t_total_start)/60:.1f} minutes")
    print(f"Output:       {OUTPUT_CSV}")
    print(f"Log:          {LOG_FILE}")


if __name__ == '__main__':
    main()
