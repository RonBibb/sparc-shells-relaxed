#!/usr/bin/env python3
"""
NGC2403 multi-tracer azimuthal analysis.

Runs azimuthal MOM0 profiles at all three shell radii using every
available tracer: THINGS NA, THINGS RO, HALOGAS HR, WISE W1.

NGC2403 X-1 position: r=2.93 kpc, phi=247 deg
Shell 2:              r=3.955 kpc, sigma=0.86 kpc  --> 1.2 sigma from X-1

Usage:
  python scripts/run_ngc2403_azimuthal.py
  python scripts/run_ngc2403_azimuthal.py --data-dir data/external --n-bins 36

Output: data/processed/ngc2403_azimuthal_multitracer.csv
        data/processed/ngc2403_azimuthal_multitracer.png

Author: Ron Bibb  |  2026-05-25
"""

import os, argparse, warnings, tarfile, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
warnings.filterwarnings('ignore')

DATA_DIR   = 'data/external'
OUTPUT_DIR = 'data/processed'

# NGC2403 geometry (fixed)
GEO = {'ra':114.17308, 'dec':65.60222, 'pa':124., 'inc':63., 'D_mpc':3.2}

# Shells from catalog
SHELLS = [
    {'shell': 1, 'r_kpc': 1.056, 'sigma': 0.47, 'skip': True},   # too inner
    {'shell': 2, 'r_kpc': 3.955, 'sigma': 0.86, 'skip': False},   # MAIN
    {'shell': 3, 'r_kpc': 5.982, 'sigma': 0.84, 'skip': False},   # secondary
]

# NGC2403 X-1 position (Walton+2022 / Swartz+2004)
ULX_PHI   = 247.0   # degrees
ULX_R_KPC = 2.93    # kpc  (1.2 sigma inside shell 2)

# Data files
TRACERS = [
    ('THINGS NA',  'NGC_2403_NA_MOM0_THINGS.FITS', 'things'),
    ('THINGS RO',  'NGC_2403_RO_MOM0_THINGS.FITS', 'things'),
    ('HALOGAS HR', 'NGC2403-HR_mom0m.fits',          'halogas'),
    ('WISE W1',    'WISE_W1_NGC2403.fits',            'wise'),
]


def load_fits(path, ftype):
    """Load a FITS image.  Returns (data, wcs)."""
    if ftype == 'wise':
        # unWISE returns a tar.gz bundle
        try:
            with tarfile.open(path, 'r:*') as tar:
                names = tar.getnames()
                img_name = next(
                    (n for n in names if 'img-m.fits' in n and not n.endswith('.gz')),
                    next((n for n in names if '.fits' in n), None)
                )
                if img_name is None:
                    raise ValueError(f"No FITS inside tar: {names}")
                member = tar.extractfile(img_name)
                with fits.open(io.BytesIO(member.read())) as h:
                    for ext in h:
                        if ext.data is not None and ext.data.ndim == 2:
                            return ext.data.astype(float), WCS(ext.header, naxis=2)
        except tarfile.TarError:
            pass  # fallthrough to plain FITS

    with fits.open(path, memmap=False) as h:
        for ext in h:
            d = ext.data
            if d is not None:
                if d.ndim == 2:
                    return d.astype(float), WCS(ext.header, naxis=2)
                if d.ndim >= 3:
                    d2 = np.squeeze(d)
                    if d2.ndim == 2:
                        return d2.astype(float), WCS(ext.header, naxis=2)
    raise ValueError(f"No 2D image found in {path}")


def deproject(data, wcs):
    """Return (r_kpc, phi_deg, valid_mask)."""
    g = GEO
    asc = (1/3600.)*(np.pi/180.)*g['D_mpc']*1000.
    pa  = np.radians(g['pa'])
    ci  = max(np.cos(np.radians(g['inc'])), 0.1)

    yy, xx = np.indices(data.shape)
    sky     = wcs.pixel_to_world_values(xx, yy)
    dra     = (sky[0] - g['ra'])  * np.cos(np.radians(g['dec'])) * 3600.
    ddec    = (sky[1] - g['dec']) * 3600.

    x_maj  =  dra*np.sin(pa) + ddec*np.cos(pa)
    y_min  = -dra*np.cos(pa) + ddec*np.sin(pa)
    y_dep  = y_min / ci

    r_kpc  = np.sqrt(x_maj**2 + y_dep**2) * asc
    phi    = np.degrees(np.arctan2(y_dep, x_maj)) % 360.

    floor  = float(np.nanpercentile(data[np.isfinite(data) & (data > 0)], 10)) \
             if np.any(data > 0) else 0.
    valid  = np.isfinite(data) & (data > floor) & (r_kpc > 0.2)
    return r_kpc, phi, valid


def azimuthal_profile(data, wcs, r_sh, n_bins=36):
    """Return azimuthal MOM0 profile at r_sh."""
    r_kpc, phi, valid = deproject(data, wcs)
    dsh   = 0.25 * r_sh
    in_sh = valid & (r_kpc > r_sh - dsh) & (r_kpc < r_sh + dsh)
    off   = valid & (r_kpc > r_sh + dsh) & (r_kpc < r_sh + 3*dsh)

    bins = np.linspace(0, 360, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m_i = in_sh & (phi >= bins[i]) & (phi < bins[i+1])
        m_o = off   & (phi >= bins[i]) & (phi < bins[i+1])
        vi  = np.median(data[m_i]) if m_i.sum() >= 5 else np.nan
        vo  = np.median(data[m_o]) if m_o.sum() >= 5 else np.nan
        rat = vi/vo if (pd.notna(vi) and pd.notna(vo) and vo > 0) else np.nan
        rows.append({'phi_mid': (bins[i]+bins[i+1])/2, 'ratio': rat,
                     'n_pix': int(m_i.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',   default=DATA_DIR)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--n-bins',     type=int, default=36)
    cfg = ap.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"NGC2403 multi-tracer azimuthal analysis")
    print(f"X-1 prediction: r={ULX_R_KPC:.2f} kpc  phi={ULX_PHI:.0f} deg")
    print()

    all_rows = []
    loaded   = {}

    # Load all tracers
    for label, fname, ftype in TRACERS:
        path = os.path.join(cfg.data_dir, fname)
        if not os.path.exists(path):
            print(f"  {label:12s}: NOT FOUND ({fname}) — skipping")
            continue
        try:
            data, wcs = load_fits(path, ftype)
            loaded[label] = (data, wcs)
            print(f"  {label:12s}: loaded  {data.shape}  "
                  f"({os.path.getsize(path)//1024} KB)")
        except Exception as e:
            print(f"  {label:12s}: ERROR — {e}")

    if not loaded:
        print("No data loaded. Exiting.")
        return

    print()

    # Run azimuthal profiles for each shell x tracer
    for sh in SHELLS:
        if sh['skip']:
            continue
        r_sh = sh['r_kpc']
        print(f"Shell {sh['shell']}  r={r_sh:.2f} kpc  (sigma={sh['sigma']:.2f})")
        for label, (data, wcs) in loaded.items():
            df = azimuthal_profile(data, wcs, r_sh, cfg.n_bins)
            valid = df[df['ratio'].notna()]
            if len(valid) == 0:
                print(f"  {label:12s}: no valid bins")
                continue
            pk = valid.loc[valid['ratio'].idxmax()]
            print(f"  {label:12s}: peak phi={pk['phi_mid']:.0f}°  "
                  f"ratio={pk['ratio']:.2f}x  "
                  f"mean={valid['ratio'].mean():.3f}")
            df['tracer'] = label
            df['shell']  = sh['shell']
            df['r_sh']   = r_sh
            all_rows.append(df)
        print()

    if not all_rows:
        print("No profiles generated.")
        return

    # Save CSV
    combined = pd.concat(all_rows, ignore_index=True)
    out_csv  = os.path.join(cfg.output_dir, 'ngc2403_azimuthal_multitracer.csv')
    combined.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(combined)} rows)")

    # ── Figure ────────────────────────────────────────────────────────────────
    shells_to_plot = [s for s in SHELLS if not s['skip']]
    fig, axes = plt.subplots(1, len(shells_to_plot),
                              figsize=(7.5*len(shells_to_plot), 6))
    if len(shells_to_plot) == 1:
        axes = [axes]

    colors = {'THINGS NA': '#2C6FAC', 'THINGS RO': '#5B9BD5',
              'HALOGAS HR': '#D95F02', 'WISE W1': '#228B22'}
    styles = {'THINGS NA': '-', 'THINGS RO': '--',
              'HALOGAS HR': '-', 'WISE W1': '-'}

    for ax, sh in zip(axes, shells_to_plot):
        r_sh = sh['r_kpc']
        sub  = combined[combined['r_sh'] == r_sh]

        for label in ['THINGS NA', 'THINGS RO', 'HALOGAS HR', 'WISE W1']:
            d = sub[sub['tracer']==label].sort_values('phi_mid')
            if len(d) == 0: continue
            col = colors.get(label, 'gray')
            ls  = styles.get(label, '-')
            ax.plot(d['phi_mid'], d['ratio'],
                    color=col, ls=ls, lw=2, alpha=0.85,
                    marker='o', ms=4, label=label)

        # X-1 prediction
        ax.axvline(ULX_PHI, color='red', lw=2.5, ls='-', zorder=6,
                   label=f'X-1 pred. φ={ULX_PHI:.0f}°')
        ax.fill_betweenx([0,25], ULX_PHI-15, ULX_PHI+15,
                          alpha=0.08, color='red')
        ax.axhline(1, color='k', lw=1, ls='--', alpha=0.3)
        ax.set_xlabel('Azimuthal angle φ (degrees)', fontsize=11)
        ax.set_ylabel('Ratio (in-shell / off-shell)', fontsize=11)
        ax.set_title(f'NGC2403 shell {sh["shell"]}\n'
                     f'r = {r_sh:.2f} kpc  (X-1 at 1.2σ)',
                     fontsize=11)
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 45))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('NGC2403 multi-tracer azimuthal analysis\n'
                 'Red line = NGC2403 X-1 predicted azimuth (φ=247°)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_png = os.path.join(cfg.output_dir, 'ngc2403_azimuthal_multitracer.png')
    plt.savefig(out_png, dpi=170, bbox_inches='tight')
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
