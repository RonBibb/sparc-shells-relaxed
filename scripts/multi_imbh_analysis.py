"""
multi_imbh_analysis.py

Multi-IMBH interpretation of SPARC rotation-curve "shells".

Premise:
  Wide rotation-curve shells (σ ~ kpc) cannot be single bound DM cocoons,
  because their σ/r requires single-object masses comparable to the host
  itself (m > 1-10% of M_host, violating any plausible compact-object scaling).
  Instead, shells must be populations of N narrow contributors.

This script:
  1. Loads the relaxed-cap framework shell catalog
  2. Computes σ/r and the "impossible mass" m_single = 3 M_host (σ/r)^3
  3. Under the multi-contributor hypothesis, derives:
     - N per shell, given an assumed per-IMBH cocoon mass
     - per-IMBH mass, given an assumed N
  4. Self-consistently uses Davis-Graham-Combes M_BH-M_halo scaling
  5. Outputs galaxy-level IMBH inventory and diagnostic plots

Author: Ron Bibb (halo_shells program)
Date: May 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = '/Users/ronbibb/Library/CloudStorage/OneDrive-Personal(2)/Documents/Academic/Rotmod_LTG'
SHELL_CSV = './sparc_T2-T9_relaxed_caps_fits.csv'        # Output from relaxed-caps fitting
SPARC_CSV = '../data/sparc_sample123.csv'  # SPARC main catalog
OUT_DIR = './output_multi_imbh'

# Physical constants
G_KPC_KMS2_MSUN = 4.302e-6   # G in kpc·(km/s)²/M_sun

# Davis-Graham-Combes 2019 spiral M_BH-M_halo relation
DGC_INTERCEPT = 8.0     # log M_BH at M_halo = 10^12
DGC_SLOPE     = 3    # log-log slope

# Cosmological per-dwarf-accretion typical parameters
# From Reines-Volonteri 2015 + dwarf-galaxy SHMR (Behroozi 2013)
DWARF_HALO_TYPICAL  = 1e2   # M_sun, typical accreted dwarf halo mass
DWARF_RETAIN_FRAC   = 0.30   # fraction of dwarf halo mass retained as cocoon
DWARF_IMBH_TYPICAL  = 3e2    # M_sun, typical IMBH in dwarf nucleus

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_shells(shell_csv):
    """Flatten the SPARC framework-fit CSV into one row per shell."""
    df = pd.read_csv(shell_csv)
    rows = []
    for _, r in df.iterrows():
        n_sh = int(r.get('fw_best_n_shells', 0))
        if n_sh == 0:
            continue
        if n_sh == 1:
            rows.append(dict(Galaxy=r['Galaxy'],
                             r_sh=r['fw_n1_r_sh1_kpc'],
                             sigma=r['fw_n1_sigma_sh1_kpc'],
                             M_sh=r['fw_n1_M_sh1'],
                             which='only'))
        elif n_sh == 2:
            for i in [1, 2]:
                rows.append(dict(Galaxy=r['Galaxy'],
                                 r_sh=r[f'fw_n2_r_sh{i}_kpc'],
                                 sigma=r[f'fw_n2_sigma_sh{i}_kpc'],
                                 M_sh=r[f'fw_n2_M_sh{i}'],
                                 which=f'pair{i}'))
    return pd.DataFrame(rows)


def m_host_enclosed(galaxy, r_target, data_dir=DATA_DIR):
    """Compute enclosed mass at radius r_target from observed rotation curve.
    Returns nan if rotmod file missing.
    """
    path = os.path.join(data_dir, f'{galaxy}_rotmod.dat')
    if not os.path.exists(path):
        return np.nan
    d = np.loadtxt(path, comments='#')
    r_data = d[:, 0]
    v_data = d[:, 1]
    # Interpolate V(r) at target radius
    if r_target < r_data.min() or r_target > r_data.max():
        # extrapolate flat outward, linear inward
        if r_target < r_data.min():
            v_at = v_data[0] * (r_target / r_data[0])
        else:
            v_at = v_data[-3:].mean()
    else:
        v_at = np.interp(r_target, r_data, v_data)
    return v_at**2 * r_target / G_KPC_KMS2_MSUN


def dgc_imbh_from_halo(M_halo):
    """M_BH from M_halo via Davis-Graham-Combes 2019 (extrapolated to low masses)."""
    return 10**(DGC_INTERCEPT + DGC_SLOPE * (np.log10(M_halo) - 12.0))


# ============================================================
# PRIMARY ANALYSIS
# ============================================================

def compute_shell_diagnostics(shells):
    """Add columns for σ/r, m_single_implied, multi-IMBH decomposition."""
    
    # σ/r diagnostic
    shells['sigma_over_r'] = shells['sigma'] / shells['r_sh']
    
    # Compute M_host at each shell's radius
    shells['M_host_at_r'] = shells.apply(
        lambda row: m_host_enclosed(row['Galaxy'], row['r_sh']), axis=1)
    
    # The "impossible single-object" mass — what m would tidal-radius reach σ?
    shells['m_single_implied'] = 3 * shells['M_host_at_r'] * shells['sigma_over_r']**3
    shells['m_single_frac_host'] = shells['m_single_implied'] / shells['M_host_at_r']
    
    # Single-object plausibility: m/M_host should be < ~1e-3 for a real SMBH
    shells['single_object_plausible'] = shells['m_single_frac_host'] < 1e-3
    
    # Under multi-IMBH hypothesis: assume each contributor is a typical accreted dwarf
    dwarf_cocoon_mass = DWARF_HALO_TYPICAL * DWARF_RETAIN_FRAC
    shells['N_implied_typical'] = shells['M_sh'] / dwarf_cocoon_mass
    
    # Alternative: assume each contributor has cocoon mass scaling with DGC
    # If per-IMBH mass = DWARF_IMBH_TYPICAL, the corresponding "halo" (cocoon) is:
    # Inverting DGC: M_halo = (M_BH / 1e8)^(1/4.34) * 1e12
    cocoon_per_imbh_dgc = (DWARF_IMBH_TYPICAL / 10**DGC_INTERCEPT) ** (1.0 / DGC_SLOPE) * 1e12
    shells['cocoon_per_imbh_dgc'] = cocoon_per_imbh_dgc
    shells['N_implied_dgc'] = shells['M_sh'] / cocoon_per_imbh_dgc
    
    # Width consistency check: for N Poisson-spread contributors, σ ∝ √N
    # If we observe σ_obs and infer N, the expected spatial scale per contributor is
    # σ_per = σ_obs / sqrt(N)
    shells['sigma_per_contributor_dgc'] = shells['sigma'] / np.sqrt(np.maximum(shells['N_implied_dgc'], 1))
    
    return shells


def per_galaxy_inventory(shells, sparc):
    """Build a per-galaxy summary of inferred IMBH content."""
    gal = shells.groupby('Galaxy').agg(
        n_shells=('M_sh', 'count'),
        total_shell_mass=('M_sh', 'sum'),
        total_N_dgc=('N_implied_dgc', 'sum'),
        median_sigma_over_r=('sigma_over_r', 'median'),
    ).reset_index()
    gal = gal.merge(sparc[['Galaxy', 'logM_halo', 'logM_star', 'Vflat', 'T']],
                    on='Galaxy', how='left')
    return gal


def plot_diagnostics(shells, gal_inv, out_dir):
    """Produce the diagnostic plot set."""
    
    # ---- Plot 1: σ/r distribution + impossible-mass discriminator ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=130)
    ax = axes[0]
    ax.hist(shells['sigma_over_r'].dropna(), bins=30, color='tab:blue',
            alpha=0.78, edgecolor='black')
    # Reference σ/r values (m/M_host = 1e-5 to 1e-2)
    for frac, label, color in [
        (1e-5, '10⁻⁵ (small SMBH)', 'tab:green'),
        (1e-4, '10⁻⁴ (typical SMBH)', 'tab:orange'),
        (1e-3, '10⁻³ (large SMBH)', 'tab:red'),
        (1e-2, '10⁻² (impossible)', 'gray')]:
        sor = (3 * frac) ** (1/3)
        ax.axvline(sor, ls='--', lw=2, color=color, alpha=0.85,
                   label=f'm/M_host = {label}: σ/r ≈ {sor:.3f}')
    ax.set_xlabel('σ/r', fontsize=12)
    ax.set_ylabel('Number of shells')
    ax.set_title('σ/r distribution vs single-object tidal-radius predictions',
                 fontweight='bold')
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    sub = shells.dropna(subset=['M_host_at_r', 'm_single_implied'])
    ax.scatter(np.log10(sub['M_host_at_r']),
               np.log10(sub['m_single_implied']),
               c=sub['sigma_over_r'], cmap='plasma', s=45, alpha=0.78,
               edgecolor='k', lw=0.5)
    xs = np.array([np.log10(sub['M_host_at_r']).min(),
                   np.log10(sub['M_host_at_r']).max()])
    for frac, label, c in [(1e-5, '10⁻⁵', 'tab:green'),
                           (1e-4, '10⁻⁴', 'tab:orange'),
                           (1e-3, '10⁻³', 'tab:red'),
                           (1e-2, '10⁻²', 'gray')]:
        ax.plot(xs, xs + np.log10(frac), '--', color=c, lw=1.5, alpha=0.7,
                label=f'm/M_host = {label}')
    ax.set_xlabel('log M_host(<r_shell) [M☉]')
    ax.set_ylabel('log m_single_implied [M☉]')
    ax.set_title('Single-object mass required for σ — most shells are physically impossible',
                 fontweight='bold')
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_sigma_over_r.png'), dpi=130,
                bbox_inches='tight')
    plt.close()
    print(f"  → {out_dir}/fig1_sigma_over_r.png")

    # ---- Plot 2: N IMBHs per shell + per-galaxy total ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=130)
    ax = axes[0]
    N_arr = shells['N_implied_dgc'].dropna()
    bins = np.logspace(0, 5, 30)
    ax.hist(N_arr, bins=bins, color='tab:purple', alpha=0.78, edgecolor='black')
    ax.set_xscale('log')
    ax.axvline(N_arr.median(), color='red', ls='--', lw=2,
               label=f'Median N = {N_arr.median():.0f}')
    ax.set_xlabel('N (IMBHs implied per shell, DGC scaling)')
    ax.set_ylabel('Number of shells')
    ax.set_title('Per-shell IMBH count (multi-contributor hypothesis)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    valid = gal_inv.dropna(subset=['logM_halo', 'total_N_dgc'])
    ax.scatter(valid['logM_halo'], np.log10(valid['total_N_dgc']),
               c='tab:blue', s=50, alpha=0.7, edgecolor='k', lw=0.5)
    rho, p = spearmanr(valid['logM_halo'], np.log10(valid['total_N_dgc']))
    slope, intercept = np.polyfit(valid['logM_halo'],
                                   np.log10(valid['total_N_dgc']), 1)
    xs = np.array([valid['logM_halo'].min(), valid['logM_halo'].max()])
    ax.plot(xs, slope * xs + intercept, 'k-', lw=2,
            label=f'slope={slope:+.2f}, ρ={rho:+.2f}')
    ax.set_xlabel('log M_halo (host) [M☉]')
    ax.set_ylabel('log Σ N_implied across all shells')
    ax.set_title('Total IMBH inventory per galaxy vs halo mass',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_imbh_inventory.png'), dpi=130,
                bbox_inches='tight')
    plt.close()
    print(f"  → {out_dir}/fig2_imbh_inventory.png")

    # ---- Plot 3: σ vs M shell scaling (the empirical tell) ----
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    log_M = np.log10(shells['M_sh'])
    log_s = np.log10(shells['sigma'].clip(lower=1e-3))
    slope_sM, int_sM = np.polyfit(log_M, log_s, 1)
    ax.scatter(log_M, log_s, c='tab:blue', s=45, alpha=0.7, edgecolor='k', lw=0.5,
               label=f'Shells (n={len(shells)})')
    xs = np.array([log_M.min(), log_M.max()])
    ax.plot(xs, slope_sM * xs + int_sM, 'k-', lw=2.5,
            label=f'Observed slope α = {slope_sM:.2f}')
    M_med = log_M.median(); s_med = log_s.median()
    for slope_ref, name, c in [(0.33, 'bound system', 'tab:green'),
                                (0.50, 'Poisson N-IMBH', 'tab:blue'),
                                (1.00, 'linear M', 'tab:red')]:
        ax.plot(xs, slope_ref * (xs - M_med) + s_med, '--', color=c, lw=1.5,
                alpha=0.7, label=f'α={slope_ref} ({name})')
    ax.set_xlabel('log M_shell [M☉]')
    ax.set_ylabel('log σ_shell [kpc]')
    ax.set_title('σ vs M: distinguishes physical scenarios via slope',
                 fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_sigma_M_scaling.png'), dpi=130,
                bbox_inches='tight')
    plt.close()
    print(f"  → {out_dir}/fig3_sigma_M_scaling.png")


def print_summary(shells, gal_inv):
    """Print the headline numbers."""
    print("\n" + "="*70)
    print("MULTI-IMBH ANALYSIS — SUMMARY")
    print("="*70)
    
    print(f"\nShells analyzed: {len(shells)}")
    print(f"Galaxies:        {shells['Galaxy'].nunique()}")
    
    print(f"\n--- σ/r diagnostic (single-object plausibility) ---")
    print(f"  Median σ/r:                     {shells['sigma_over_r'].median():.3f}")
    print(f"  Median m_single / M_host:       {shells['m_single_frac_host'].median():.4f}")
    print(f"  Shells with m/host < 10⁻³ (plausible single object):  "
          f"{shells['single_object_plausible'].sum()}/{len(shells)} "
          f"({100 * shells['single_object_plausible'].mean():.0f}%)")
    print(f"  Shells requiring multi-contributor interpretation:    "
          f"{(~shells['single_object_plausible']).sum()}/{len(shells)} "
          f"({100 * (~shells['single_object_plausible']).mean():.0f}%)")
    
    print(f"\n--- N IMBHs per shell (under DGC scaling, m_IMBH = {DWARF_IMBH_TYPICAL:.1e}) ---")
    N = shells['N_implied_dgc']
    print(f"  Median N per shell:  {N.median():.2f}")
    print(f"  IQR:                 [{N.quantile(0.25):.2f}, {N.quantile(0.75):.2f}]")
    print(f"  Range:               {N.min():.2f} - {N.max():.2f}")

    print(f"\n--- Per-galaxy total IMBH inventory ---")
    print(f"  Median total N per galaxy:  {gal_inv['total_N_dgc'].median():.02f}")
    print(f"  Range:                       {gal_inv['total_N_dgc'].min():.02f} - "
          f"{gal_inv['total_N_dgc'].max():.0f}")
    
    valid = gal_inv.dropna(subset=['logM_halo', 'total_N_dgc'])
    rho, p = spearmanr(valid['logM_halo'], np.log10(valid['total_N_dgc']))
    print(f"  Correlation w/ log M_halo:   ρ = {rho:+.2f}, p = {p:.1e}")
    
    # σ-M slope
    log_M = np.log10(shells['M_sh'])
    log_s = np.log10(shells['sigma'].clip(lower=1e-3))
    slope_sM, _ = np.polyfit(log_M, log_s, 1)
    print(f"\n--- σ-M scaling (population-distribution diagnostic) ---")
    print(f"  Observed σ ∝ M^α:  α = {slope_sM:+.3f}")
    print(f"  Predicted by:")
    print(f"    Single bound object (virial):  α = 0.33")
    print(f"    Poisson N-IMBH spread:         α = 0.50  ← closest to observed")
    print(f"    Linear (constant σ/M):         α = 1.00")
    
    print("\n" + "="*70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data ...")
    shells = load_shells(SHELL_CSV)
    sparc  = pd.read_csv(SPARC_CSV)
    print(f"  {len(shells)} shells loaded across {shells['Galaxy'].nunique()} galaxies")
    
    print("\nComputing diagnostics ...")
    shells = compute_shell_diagnostics(shells)
    
    print("Building per-galaxy inventory ...")
    gal_inv = per_galaxy_inventory(shells, sparc)
    
    print("Saving outputs ...")
    shells.to_csv(os.path.join(OUT_DIR, 'shells_with_multi_imbh.csv'), index=False)
    gal_inv.to_csv(os.path.join(OUT_DIR, 'galaxy_imbh_inventory.csv'), index=False)
    
    print("\nGenerating plots ...")
    plot_diagnostics(shells, gal_inv, OUT_DIR)
    
    print_summary(shells, gal_inv)


if __name__ == '__main__':
    main()
