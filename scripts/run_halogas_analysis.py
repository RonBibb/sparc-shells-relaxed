#!/usr/bin/env python3
"""
HALOGAS HI analysis for Paper 3.

Runs azimuthal MOM0 S/N analysis at shell radii for HALOGAS galaxies,
using the same pipeline as the THINGS analysis (things_radial_profiles.py).

Inputs (data/external/):
  <galaxy>-HR_mom0m.fits   HALOGAS DR1 high-res MOM0  (~15" beam)
  <galaxy>-HR_mom1m.fits   HALOGAS DR1 high-res MOM1  (NGC5055 only)

Outputs (data/processed/):
  halogas_snr_results.csv         per-shell S/N percentile & z-score
  halogas_azimuthal_ngc5055.csv   azimuthal MOM0 profile at shell radii

Usage:
  python scripts/run_halogas_analysis.py
  python scripts/run_halogas_analysis.py --data-dir data/external --n-bins 24

Author: Ron Bibb
Date:   2026-05-25
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import percentileofscore
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR   = 'data/external'
CATALOG    = 'data/processed/sparc_relaxed_caps_with_BH_decomp.csv'
OUTPUT_DIR = 'data/processed'

MOM0_FLOOR_PCTILE = 15   # mask pixels below this percentile of MOM0
MAX_SIGMA_KMS     = 80.0  # MOM2 sanity cap (km/s)

# Galaxy geometry (PA, inc from HALOGAS DR1 / NED / Heald+2011)
GEO = {
    # Shell-bearing HALOGAS galaxies in SPARC
    'NGC0891': {'ra': 35.63917, 'dec': 42.34917, 'pa': 23.,  'inc': 89., 'D_mpc': 9.5},
    'NGC1003': {'ra': 39.49125, 'dec': 40.87222, 'pa': 96.,  'inc': 74., 'D_mpc': 11.1},
    'NGC2403': {'ra':114.17308, 'dec': 65.60222, 'pa':124.,  'inc': 63., 'D_mpc': 3.2},
    'NGC5055': {'ra':198.95542, 'dec': 42.02972, 'pa':105.,  'inc': 59., 'D_mpc': 9.0},
    'NGC5585': {'ra':215.60417, 'dec': 28.74083, 'pa': 95.,  'inc': 54., 'D_mpc': 5.7},
    # Control — n=0 shells
    'NGC4559': {'ra':188.99417, 'dec': 27.95972, 'pa':150.,  'inc': 65., 'D_mpc': 10.0},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def fits_path(data_dir, galaxy, tag='HR_mom0m'):
    """Return path to a HALOGAS FITS file."""
    return os.path.join(data_dir, f'{galaxy}-{tag}.fits')


def deproject(galaxy, mom0_data, wcs):
    """Return (r_kpc, phi_deg, valid_mask) for each pixel."""
    g   = GEO[galaxy]
    asc = (1/3600.) * (np.pi/180.) * g['D_mpc'] * 1000.
    pa  = np.radians(g['pa'])
    ci  = max(np.cos(np.radians(g['inc'])), 0.1)

    yy, xx = np.indices(mom0_data.shape)
    sky    = wcs.pixel_to_world_values(xx, yy)
    dra    = (sky[0] - g['ra'])  * np.cos(np.radians(g['dec'])) * 3600.
    ddec   = (sky[1] - g['dec']) * 3600.

    x_maj =  dra * np.sin(pa) + ddec * np.cos(pa)
    y_min = -dra * np.cos(pa) + ddec * np.sin(pa)
    y_dep = y_min / ci

    r_kpc = np.sqrt(x_maj**2 + y_dep**2) * asc
    phi   = np.degrees(np.arctan2(y_dep, x_maj)) % 360.

    thr   = float(np.nanpercentile(mom0_data[mom0_data > 0], MOM0_FLOOR_PCTILE)) \
            if np.any(mom0_data > 0) else 0.
    valid = np.isfinite(mom0_data) & (mom0_data > thr)

    return r_kpc, phi, valid


def load_mom0(galaxy, data_dir, tag='HR_mom0m'):
    p = fits_path(data_dir, galaxy, tag)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Not found: {p}")
    print(f"    Loading: {os.path.basename(p)}  ({os.path.getsize(p)//1024} KB)")
    with fits.open(p, memmap=False) as h:
        data = np.squeeze(h[0].data).astype(float)
        wcs  = WCS(h[0].header, naxis=2)
    return data, wcs

# ── Analysis functions ───────────────────────────────────────────────────────

def radial_snr(galaxy, shells, data_dir):
    """
    For each shell radius: compute MOM0 percentile rank and z-score
    within the host galaxy's own radial distribution.
    Returns list of dicts.
    """
    try:
        mom0, wcs = load_mom0(galaxy, data_dir)
    except (FileNotFoundError, OSError, ValueError, Exception) as e:
        print(f"  [{galaxy}] Cannot open MOM0: {e} — skipping")
        return []

    r_kpc, _, valid = deproject(galaxy, mom0, wcs)

    # Build per-annulus medians
    r_bins = np.arange(0.3, 35, 0.5)
    ann_med = []
    for i in range(len(r_bins) - 1):
        m = (r_kpc >= r_bins[i]) & (r_kpc < r_bins[i+1]) & valid
        if m.sum() >= 15:
            ann_med.append(np.median(mom0[m]))
    ann_med = np.array(ann_med) if ann_med else np.array([np.nan])

    global_mean = np.nanmean(ann_med)
    global_std  = np.nanstd(ann_med)

    results = []
    for r_sh in shells:
        sig_sh = 0.25 * r_sh
        m_in   = valid & (r_kpc > r_sh - sig_sh) & (r_kpc < r_sh + sig_sh)
        m_off  = valid & (((r_kpc > r_sh-3*sig_sh) & (r_kpc < r_sh-sig_sh)) |
                          ((r_kpc > r_sh+sig_sh)   & (r_kpc < r_sh+3*sig_sh)))
        if m_in.sum() < 10 or m_off.sum() < 10:
            continue

        m0_in  = np.median(mom0[m_in])
        m0_off = np.median(mom0[m_off])
        ratio  = m0_in / m0_off if m0_off > 0 else np.nan
        pct    = percentileofscore(ann_med[np.isfinite(ann_med)], m0_in)
        z      = (m0_in - global_mean) / global_std if global_std > 0 else np.nan

        results.append({
            'galaxy':      galaxy,
            'r_sh':        r_sh,
            'MOM0_ratio':  ratio,
            'pct_in_gal':  pct,
            'z_score':     z,
            'n_pix_in':    int(m_in.sum()),
            'survey':      'HALOGAS',
        })

    return results


def azimuthal_profile(galaxy, r_sh, data_dir, n_bins=36, tag='HR_mom0m'):
    """
    Azimuthal MOM0 profile at shell radius r_sh.
    Returns DataFrame with phi_mid, ratio (in/off), n_pix columns.
    """
    try:
        mom0, wcs = load_mom0(galaxy, data_dir, tag)
    except (FileNotFoundError, OSError, ValueError, Exception) as e:
        print(f"  [{galaxy}] Cannot open {tag}: {e}")
        return pd.DataFrame()

    r_kpc, phi, valid = deproject(galaxy, mom0, wcs)
    sig_sh = 0.25 * r_sh

    in_sh = valid & (r_kpc > r_sh - sig_sh) & (r_kpc < r_sh + sig_sh)
    off   = valid & (r_kpc > r_sh + sig_sh)  & (r_kpc < r_sh + 3*sig_sh)

    phi_bins = np.linspace(0, 360, n_bins + 1)
    phi_mid  = (phi_bins[:-1] + phi_bins[1:]) / 2

    rows = []
    for i in range(n_bins):
        lo, hi  = phi_bins[i], phi_bins[i+1]
        m_in_p  = in_sh & (phi >= lo) & (phi < hi)
        m_off_p = off   & (phi >= lo) & (phi < hi)

        med_in  = np.median(mom0[m_in_p])  if m_in_p.sum()  >= 5 else np.nan
        med_off = np.median(mom0[m_off_p]) if m_off_p.sum() >= 5 else np.nan
        ratio   = med_in / med_off if (pd.notna(med_in) and
                                        pd.notna(med_off) and
                                        med_off > 0) else np.nan
        rows.append({
            'galaxy': galaxy, 'r_sh': r_sh, 'phi_mid': phi_mid[i],
            'med_in': med_in, 'med_off': med_off,
            'ratio': ratio, 'n_pix_in': int(m_in_p.sum()),
        })

    return pd.DataFrame(rows)

# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='HALOGAS HI analysis for Paper 3')
    p.add_argument('--data-dir',  default=DATA_DIR)
    p.add_argument('--catalog',   default=CATALOG)
    p.add_argument('--output-dir',default=OUTPUT_DIR)
    p.add_argument('--n-bins',    type=int, default=36,
                   help='Azimuthal bins (default 36 = 10 deg)')
    return p.parse_args()


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Diagnostic: show what files are actually in data_dir
    print(f"Data directory: {os.path.abspath(cfg.data_dir)}")
    if os.path.isdir(cfg.data_dir):
        fits_found = sorted(f for f in os.listdir(cfg.data_dir)
                            if f.endswith('.fits') or f.endswith('.FITS'))
        print(f"FITS files found ({len(fits_found)}):")
        for f in fits_found:
            print(f"  {f}")
    else:
        print(f"  *** Directory not found: {cfg.data_dir}")
        print(f"  Run from repo root: cd ~/path/to/sparc-shells-relaxed && python scripts/run_halogas_analysis.py")
        return
    print()

    cat = pd.read_csv(cfg.catalog)

    # Shell radii per galaxy
    shell_radii = {}
    for _, row in cat.iterrows():
        g = row['Galaxy']
        if g not in GEO: continue
        n = int(row['fw_best_n_shells'])
        radii = []
        for i in range(1, n+1):
            r = row.get(f'shell{i}_r_kpc')
            if pd.notna(r) and float(r) < 50.:  # skip pathological shells > 50 kpc
                radii.append(float(r))
        shell_radii[g] = radii

    # ── 1. Radial S/N for all galaxies ──────────────────────────────────────
    print("="*60)
    print("HALOGAS — radial S/N at shell radii")
    print("="*60)

    all_snr = []
    for galaxy in sorted(GEO.keys()):
        shells = shell_radii.get(galaxy, [])
        n_sh   = len(shells)
        role   = 'shell-bearing' if n_sh > 0 else 'CONTROL (n=0)'
        print(f"\n  {galaxy}  [{role}]  shells: {shells}")
        rows = radial_snr(galaxy, shells if shells else [], cfg.data_dir)
        for r in rows:
            print(f"    r={r['r_sh']:.1f}kpc  ratio={r['MOM0_ratio']:.3f}  "
                  f"pct={r['pct_in_gal']:.0f}%  z={r['z_score']:+.2f}σ")
        all_snr.extend(rows)

    snr_df = pd.DataFrame(all_snr)
    out1 = os.path.join(cfg.output_dir, 'halogas_snr_results.csv')
    snr_df.to_csv(out1, index=False)
    print(f"\nSaved: {out1}  ({len(snr_df)} rows)")

    # ── 2. Azimuthal profile for NGC5055 shell 2 (the X-1 shell) ────────────
    print("\n" + "="*60)
    print("HALOGAS — NGC5055 azimuthal profile at shell radii")
    print("="*60)

    az_rows = []
    for r_sh, tag in [(8.79, 'HR_mom0m'),   # shell 1
                      (14.87,'HR_mom0m'),    # shell 2 — X-1 graveyard
                      (14.87,'LR_mom0m')]:   # shell 2 low-res (deeper sensitivity)
        print(f"\n  NGC5055  r_sh={r_sh:.2f} kpc  ({tag})  "
              f"n_bins={cfg.n_bins}")
        df_az = azimuthal_profile('NGC5055', r_sh, cfg.data_dir,
                                  n_bins=cfg.n_bins, tag=tag)
        if len(df_az):
            df_az['tag'] = tag
            # Report peaks
            valid = df_az[df_az['ratio'].notna()]
            if len(valid):
                peak = valid.loc[valid['ratio'].idxmax()]
                print(f"    Peak: φ={peak['phi_mid']:.0f}°  ratio={peak['ratio']:.2f}x")
                print(f"    Mean ratio: {valid['ratio'].mean():.3f}  "
                      f"std: {valid['ratio'].std():.3f}")
            az_rows.append(df_az)

    if az_rows:
        az_df = pd.concat(az_rows, ignore_index=True)
        out2  = os.path.join(cfg.output_dir, 'halogas_azimuthal_ngc5055.csv')
        az_df.to_csv(out2, index=False)
        print(f"\nSaved: {out2}  ({len(az_df)} rows)")

    # ── 3. Summary ───────────────────────────────────────────────────────────
    if len(snr_df):
        print("\n" + "="*60)
        print("SUMMARY — shell radii above 80th percentile in host galaxy")
        print("="*60)
        above = snr_df[snr_df['pct_in_gal'] > 80]
        print(f"  {len(above)}/{len(snr_df)} shells above 80th percentile")
        for _, row in above.iterrows():
            print(f"  {row['galaxy']:10s}  r={row['r_sh']:.1f}  "
                  f"pct={row['pct_in_gal']:.0f}%  z={row['z_score']:+.2f}σ")

        # Control baseline
        print()
        print("Control galaxies (n=0 shells) — natural ISM variability:")
        controls = ['NGC4559']
        for g in controls:
            try:
                mom0, wcs = load_mom0(g, cfg.data_dir)
                r_kpc, _, valid = deproject(g, mom0, wcs)
                r_bins  = np.arange(0.3, 35, 0.5)
                ann_med = []
                for i in range(len(r_bins)-1):
                    m = (r_kpc >= r_bins[i]) & (r_kpc < r_bins[i+1]) & valid
                    if m.sum() >= 15:
                        ann_med.append(np.median(mom0[m]))
                if ann_med:
                    a = np.array(ann_med)
                    local_ratio = [a[i]/np.median(a[max(0,i-3):i+3])
                                   for i in range(2, len(a)-2)
                                   if np.median(a[max(0,i-3):i+3]) > 0]
                    print(f"  {g}: max local ratio = {np.max(local_ratio):.2f}x  "
                          f"90th pct = {np.percentile(local_ratio,90):.2f}x  "
                          f"(natural ISM ceiling)")
            except (FileNotFoundError, OSError, ValueError, Exception) as e:
                print(f"  {g}: skipped ({e})")


if __name__ == '__main__':
    main()
