#!/usr/bin/env python3
"""
WISE W1 azimuthal analysis for SPARC shell galaxies.

Runs the same azimuthal profile analysis as THINGS/HALOGAS,
but using WISE W1 (3.4um, stellar mass) as the tracer.

Stellar mass is a physically independent tracer from HI:
  - HI excess = disturbed gas (could be ISM or accretion)
  - W1 excess = stellar overdensity (ONLY accreted/deposited stars)

Usage:
  python scripts/run_wise_analysis.py
  python scripts/run_wise_analysis.py --data-dir data/external --n-bins 36

Inputs (data/external/):
  WISE_W1_{galaxy}.fits   unWISE NEO7 cutouts (2.75"/pix, multi-extension)

Outputs (data/processed/):
  wise_snr_results.csv           per-shell S/N percentile + z-score
  wise_azimuthal_ngc5055.csv     azimuthal profile at shell radii for NGC5055

Author: Ron Bibb  |  2026-05-25
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import percentileofscore
warnings.filterwarnings('ignore')

DATA_DIR   = 'data/external'
CATALOG    = 'data/processed/sparc_relaxed_caps_with_BH_decomp.csv'
OUTPUT_DIR = 'data/processed'

PIXSCALE_ARCSEC = 2.75    # unWISE NEO7 pixel scale
MOM0_FLOOR_PCT  = 10      # mask faint background pixels
MAX_R_KPC       = 35.     # ignore shells beyond this radius

# Galaxy geometry — same as HALOGAS analysis
GEO = {
    'CamB':        {'ra': 70.333,  'dec': 67.441,  'pa':  0.,  'inc': 45., 'D_mpc': 3.3},
    'DDO064':      {'ra':142.427,  'dec': 31.443,  'pa': 65.,  'inc': 55., 'D_mpc': 6.9},
    'DDO154':      {'ra':193.523,  'dec': 27.148,  'pa': 50.,  'inc': 66., 'D_mpc': 3.7},
    'DDO161':      {'ra':191.726,  'dec':  4.143,  'pa':165.,  'inc': 72., 'D_mpc':  7.5},
    'DDO168':      {'ra':198.403,  'dec': 45.927,  'pa':  0.,  'inc': 48., 'D_mpc': 4.3},
    'DDO170':      {'ra':199.816,  'dec': 43.029,  'pa': 75.,  'inc': 55., 'D_mpc': 15.4},
    'ESO079-G014': {'ra': 84.635,  'dec':-68.750,  'pa': 80.,  'inc': 61., 'D_mpc': 27.2},
    'ESO116-G012': {'ra':  9.698,  'dec':-34.940,  'pa':105.,  'inc': 56., 'D_mpc': 10.1},
    'ESO444-G084': {'ra':197.821,  'dec':-28.998,  'pa': 35.,  'inc': 53., 'D_mpc': 4.3},
    'ESO563-G021': {'ra':131.917,  'dec':-20.368,  'pa': 25.,  'inc': 21., 'D_mpc': 60.0},
    'F561-1':      {'ra':141.801,  'dec': 21.920,  'pa':  0.,  'inc': 54., 'D_mpc': 67.0},
    'F563-1':      {'ra':143.440,  'dec': 20.172,  'pa': 55.,  'inc': 25., 'D_mpc': 46.8},
    'F563-V2':     {'ra':143.927,  'dec': 20.270,  'pa': 85.,  'inc': 27., 'D_mpc': 54.6},
    'F568-3':      {'ra':168.957,  'dec': 22.906,  'pa':  5.,  'inc': 40., 'D_mpc': 80.0},
    'IC2574':      {'ra':157.375,  'dec': 68.412,  'pa': 55.,  'inc': 75., 'D_mpc': 3.9},
    'IC4202':      {'ra':197.300,  'dec':-46.582,  'pa': 85.,  'inc': 73., 'D_mpc': 95.0},
    'KK98-251':    {'ra':352.104,  'dec': 52.182,  'pa':  0.,  'inc': 56., 'D_mpc': 7.6},
    'NGC0024':     {'ra':  9.708,  'dec':-24.963,  'pa': 45.,  'inc': 64., 'D_mpc': 7.3},
    'NGC0055':     {'ra':  3.723,  'dec':-39.197,  'pa':108.,  'inc': 78., 'D_mpc': 2.1},
    'NGC0100':     {'ra': 10.471,  'dec': 16.018,  'pa': 70.,  'inc': 80., 'D_mpc': 13.7},
    'NGC0247':     {'ra': 11.786,  'dec':-20.761,  'pa':170.,  'inc': 74., 'D_mpc': 3.4},
    'NGC0300':     {'ra': 13.723,  'dec':-37.684,  'pa':111.,  'inc': 40., 'D_mpc': 2.1},
    'NGC0801':     {'ra': 31.623,  'dec': 21.868,  'pa': 55.,  'inc': 77., 'D_mpc': 80.2},
    'NGC0891':     {'ra': 35.639,  'dec': 42.349,  'pa': 23.,  'inc': 89., 'D_mpc': 9.5},
    'NGC1003':     {'ra': 39.491,  'dec': 40.872,  'pa': 96.,  'inc': 74., 'D_mpc': 11.1},
    'NGC1090':     {'ra': 41.690,  'dec': -0.248,  'pa':155.,  'inc': 67., 'D_mpc': 36.0},
    'NGC2403':     {'ra':114.173,  'dec': 65.602,  'pa':124.,  'inc': 63., 'D_mpc': 3.2},
    'NGC2683':     {'ra':133.144,  'dec': 33.423,  'pa': 44.,  'inc': 78., 'D_mpc': 9.8},
    'NGC2841':     {'ra':140.511,  'dec': 50.977,  'pa':147.,  'inc': 74., 'D_mpc': 14.1},
    'NGC2903':     {'ra':143.042,  'dec': 21.500,  'pa': 17.,  'inc': 65., 'D_mpc': 8.9},
    'NGC2915':     {'ra':141.546,  'dec':-76.627,  'pa': 45.,  'inc': 55., 'D_mpc': 3.8},
    'NGC2976':     {'ra':146.817,  'dec': 67.890,  'pa':143.,  'inc': 65., 'D_mpc': 3.6},
    'NGC2366':     {'ra':112.829,  'dec': 69.215,  'pa': 30.,  'inc': 64., 'D_mpc': 3.4},
    'NGC3198':     {'ra':154.979,  'dec': 45.550,  'pa':215.,  'inc': 72., 'D_mpc': 14.5},
    'NGC3521':     {'ra':166.453,  'dec': -0.036,  'pa':163.,  'inc': 73., 'D_mpc': 11.2},
    'NGC3726':     {'ra':173.333,  'dec': 47.029,  'pa':195.,  'inc': 53., 'D_mpc': 17.0},
    'NGC3741':     {'ra':173.731,  'dec': 45.290,  'pa': 85.,  'inc': 60., 'D_mpc': 3.2},
    'NGC3769':     {'ra':174.035,  'dec': 47.894,  'pa':150.,  'inc': 73., 'D_mpc': 15.5},
    'NGC3877':     {'ra':176.528,  'dec': 47.494,  'pa': 35.,  'inc': 76., 'D_mpc': 17.0},
    'NGC3893':     {'ra':177.266,  'dec': 48.713,  'pa':163.,  'inc': 50., 'D_mpc': 18.1},
    'NGC3949':     {'ra':178.134,  'dec': 47.853,  'pa':120.,  'inc': 55., 'D_mpc': 19.1},
    'NGC3953':     {'ra':178.460,  'dec': 52.326,  'pa': 13.,  'inc': 63., 'D_mpc': 18.7},
    'NGC3992':     {'ra':179.706,  'dec': 53.373,  'pa': 68.,  'inc': 53., 'D_mpc': 23.7},
    'NGC4010':     {'ra':179.696,  'dec': 47.265,  'pa':170.,  'inc': 79., 'D_mpc': 16.0},
    'NGC4013':     {'ra':179.635,  'dec': 43.947,  'pa': 65.,  'inc': 89., 'D_mpc': 17.0},
    'NGC4051':     {'ra':180.790,  'dec': 44.531,  'pa':135.,  'inc': 40., 'D_mpc': 17.0},
    'NGC4085':     {'ra':181.285,  'dec': 50.353,  'pa': 80.,  'inc': 79., 'D_mpc': 19.0},
    'NGC4088':     {'ra':181.383,  'dec': 50.536,  'pa': 45.,  'inc': 68., 'D_mpc': 15.8},
    'NGC4100':     {'ra':181.506,  'dec': 49.574,  'pa':163.,  'inc': 71., 'D_mpc': 20.5},
    'NGC4138':     {'ra':182.370,  'dec': 43.688,  'pa':150.,  'inc': 38., 'D_mpc': 17.0},
    'NGC4157':     {'ra':182.713,  'dec': 50.489,  'pa': 65.,  'inc': 82., 'D_mpc': 15.6},
    'NGC4183':     {'ra':183.249,  'dec': 43.698,  'pa': 15.,  'inc': 80., 'D_mpc': 16.7},
    'NGC4217':     {'ra':183.967,  'dec': 47.092,  'pa': 48.,  'inc': 86., 'D_mpc': 19.6},
    'NGC4559':     {'ra':188.994,  'dec': 27.960,  'pa':150.,  'inc': 65., 'D_mpc': 10.0},
    'NGC5033':     {'ra':198.365,  'dec': 36.594,  'pa':170.,  'inc': 66., 'D_mpc': 18.7},
    'NGC5055':     {'ra':198.955,  'dec': 42.030,  'pa':105.,  'inc': 59., 'D_mpc': 9.0},
    'NGC5371':     {'ra':208.732,  'dec': 40.462,  'pa': 25.,  'inc': 50., 'D_mpc': 35.3},
    'NGC5585':     {'ra':215.604,  'dec': 28.741,  'pa': 95.,  'inc': 54., 'D_mpc': 5.7},
    'NGC5907':     {'ra':228.974,  'dec': 56.329,  'pa': 65.,  'inc': 87., 'D_mpc': 16.3},
    'NGC6503':     {'ra':274.900,  'dec': 70.144,  'pa':123.,  'inc': 74., 'D_mpc': 5.3},
    'NGC6946':     {'ra':308.718,  'dec': 60.154,  'pa': 65.,  'inc': 33., 'D_mpc': 6.8},
    'NGC7331':     {'ra':339.267,  'dec': 34.416,  'pa':171.,  'inc': 76., 'D_mpc': 14.7},
    'UGC02259':    {'ra': 42.386,  'dec': 27.866,  'pa': 80.,  'inc': 60., 'D_mpc': 10.8},
    'UGC04278':    {'ra':122.234,  'dec': 50.733,  'pa': 20.,  'inc': 83., 'D_mpc': 10.5},
    'UGC05005':    {'ra':140.635,  'dec': 29.758,  'pa':  0.,  'inc': 67., 'D_mpc': 50.5},
    'UGC05414':    {'ra':151.101,  'dec': 34.525,  'pa': 45.,  'inc': 57., 'D_mpc': 9.3},
    'UGC05716':    {'ra':159.008,  'dec': 43.029,  'pa': 55.,  'inc': 54., 'D_mpc': 7.1},
    'UGC05721':    {'ra':159.313,  'dec': 39.972,  'pa': 25.,  'inc': 55., 'D_mpc': 6.5},
    'UGC05750':    {'ra':160.226,  'dec': 25.982,  'pa': 75.,  'inc': 60., 'D_mpc': 56.1},
    'UGC06614':    {'ra':171.533,  'dec': 17.101,  'pa': 65.,  'inc': 32., 'D_mpc': 86.0},
    'UGC06628':    {'ra':172.107,  'dec': 19.559,  'pa':  0.,  'inc': 40., 'D_mpc': 14.0},
    'UGC06818':    {'ra':178.273,  'dec': 41.809,  'pa': 50.,  'inc': 58., 'D_mpc': 18.6},
}


def load_wise(galaxy, data_dir):
    """
    Load unWISE W1 image. Returns (data, wcs).
    unWISE API returns a tar.gz bundle containing img-m.fits + invvar/n/std.
    """
    import tarfile, io as _io
    path = os.path.join(data_dir, f'WISE_W1_{galaxy}.fits')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    # Try as tar.gz (unWISE cutout API format)
    try:
        with tarfile.open(path, 'r:*') as tar:
            names = tar.getnames()
            # Main image: img-m.fits (uncompressed inside the tar)
            img_name = next((n for n in names
                             if 'img-m.fits' in n and not n.endswith('.gz')), None)
            if img_name is None:
                img_name = next((n for n in names if '.fits' in n), None)
            if img_name is None:
                raise ValueError(f"No FITS found in tar: {names}")
            member = tar.extractfile(img_name)
            with fits.open(_io.BytesIO(member.read())) as h:
                img = None
                for ext in h:
                    if ext.data is not None and ext.data.ndim == 2:
                        img = ext.data.astype(float)
                        hdr = ext.header
                        break
                if img is None:
                    raise ValueError("No 2D extension in tar FITS")
                return img, WCS(hdr, naxis=2)
    except tarfile.TarError:
        pass  # Not a tar — try plain FITS

    # Fallback: plain FITS
    with fits.open(path, memmap=False) as h:
        for ext in h:
            if ext.data is not None and ext.data.ndim == 2:
                return ext.data.astype(float), WCS(ext.header, naxis=2)
    raise ValueError(f"No 2D image found in {path}")


def deproject(galaxy, data, wcs):
    """Return (r_kpc, phi_deg, valid_mask) for each pixel."""
    g = GEO[galaxy]
    asc = (1/3600.) * (np.pi/180.) * g['D_mpc'] * 1000.  # arcsec -> kpc
    pa = np.radians(g['pa'])
    ci = max(np.cos(np.radians(g['inc'])), 0.1)

    yy, xx = np.indices(data.shape)
    sky = wcs.pixel_to_world_values(xx, yy)
    dra  = (sky[0] - g['ra'])  * np.cos(np.radians(g['dec'])) * 3600.
    ddec = (sky[1] - g['dec']) * 3600.

    x_maj =  dra*np.sin(pa) + ddec*np.cos(pa)
    y_min = -dra*np.cos(pa) + ddec*np.sin(pa)
    y_dep = y_min / ci

    r_kpc = np.sqrt(x_maj**2 + y_dep**2) * asc
    phi   = np.degrees(np.arctan2(y_dep, x_maj)) % 360.

    # Valid pixels: finite, positive, above floor
    thr   = float(np.nanpercentile(data[np.isfinite(data) & (data > 0)],
                                    MOM0_FLOOR_PCT)) if np.any(data > 0) else 0.
    valid = np.isfinite(data) & (data > thr) & (r_kpc > 0.2)
    return r_kpc, phi, valid


def radial_snr(galaxy, shells, data_dir):
    try:
        data, wcs = load_wise(galaxy, data_dir)
    except (FileNotFoundError, Exception) as e:
        print(f"  [{galaxy}] {e} — skipping"); return []

    r_kpc, _, valid = deproject(galaxy, data, wcs)
    r_bins  = np.arange(0.5, MAX_R_KPC, 0.5)
    ann_med = []
    for i in range(len(r_bins)-1):
        m = valid & (r_kpc >= r_bins[i]) & (r_kpc < r_bins[i+1])
        if m.sum() >= 20:
            ann_med.append(np.median(data[m]))
    ann_med = np.array(ann_med) if ann_med else np.array([np.nan])

    mu  = np.nanmean(ann_med)
    sig = np.nanstd(ann_med)

    rows = []
    for r_sh in [s for s in shells if s < MAX_R_KPC]:
        dsh  = 0.25 * r_sh
        m_in = valid & (r_kpc > r_sh-dsh) & (r_kpc < r_sh+dsh)
        m_of = valid & (((r_kpc > r_sh-3*dsh) & (r_kpc < r_sh-dsh)) |
                         ((r_kpc > r_sh+dsh)   & (r_kpc < r_sh+3*dsh)))
        if m_in.sum() < 20 or m_of.sum() < 20: continue
        v_in = np.median(data[m_in])
        v_of = np.median(data[m_of])
        ratio = v_in/v_of if v_of > 0 else np.nan
        pct   = percentileofscore(ann_med[np.isfinite(ann_med)], v_in)
        z     = (v_in - mu)/sig if sig > 0 else np.nan
        rows.append({'galaxy': galaxy, 'r_sh': r_sh, 'W1_ratio': ratio,
                     'pct_in_gal': pct, 'z_score': z,
                     'n_pix_in': int(m_in.sum()), 'survey': 'WISE-W1'})
    return rows


def azimuthal_profile(galaxy, r_sh, data_dir, n_bins=36):
    try:
        data, wcs = load_wise(galaxy, data_dir)
    except Exception as e:
        print(f"  [{galaxy}] {e}"); return pd.DataFrame()
    r_kpc, phi, valid = deproject(galaxy, data, wcs)
    dsh   = 0.25 * r_sh
    in_sh = valid & (r_kpc > r_sh-dsh) & (r_kpc < r_sh+dsh)
    off   = valid & (r_kpc > r_sh+dsh) & (r_kpc < r_sh+3*dsh)
    bins  = np.linspace(0, 360, n_bins+1)
    rows  = []
    for i in range(n_bins):
        m_i = in_sh & (phi >= bins[i]) & (phi < bins[i+1])
        m_o = off   & (phi >= bins[i]) & (phi < bins[i+1])
        vi  = np.median(data[m_i]) if m_i.sum() >= 5 else np.nan
        vo  = np.median(data[m_o]) if m_o.sum() >= 5 else np.nan
        rat = vi/vo if (pd.notna(vi) and pd.notna(vo) and vo > 0) else np.nan
        rows.append({'galaxy': galaxy, 'r_sh': r_sh,
                     'phi_mid': (bins[i]+bins[i+1])/2,
                     'val_in': vi, 'val_off': vo,
                     'ratio': rat, 'n_pix_in': int(m_i.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',   default=DATA_DIR)
    ap.add_argument('--catalog',    default=CATALOG)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--n-bins',     type=int, default=36)
    cfg = ap.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cat = pd.read_csv(cfg.catalog)
    shell_radii = {}
    for _, row in cat.iterrows():
        g = row['Galaxy']
        if g not in GEO: continue
        n = int(row['fw_best_n_shells'])
        radii = [row.get(f'shell{i}_r_kpc') for i in range(1,n+1)]
        shell_radii[g] = [float(r) for r in radii
                          if pd.notna(r) and float(r) < MAX_R_KPC]

    # ── 1. Radial S/N ──────────────────────────────────────────────
    print("="*60)
    print("WISE W1 — radial S/N at shell radii (stellar mass tracer)")
    print("="*60)
    all_snr = []
    for galaxy in sorted(GEO.keys()):
        shells = shell_radii.get(galaxy, [])
        n_sh   = len(shells)
        role   = 'shell' if n_sh > 0 else 'CONTROL'
        found_file = os.path.exists(os.path.join(cfg.data_dir, f'WISE_W1_{galaxy}.fits'))
        if not found_file: continue
        print(f"\n  {galaxy:14s} [{role} n={n_sh}]  shells={shells}")
        rows = radial_snr(galaxy, shells, cfg.data_dir)
        for r in rows:
            print(f"    r={r['r_sh']:.1f}  ratio={r['W1_ratio']:.3f}  "
                  f"pct={r['pct_in_gal']:.0f}%  z={r['z_score']:+.2f}σ")
        all_snr.extend(rows)

    snr_df = pd.DataFrame(all_snr)
    out1 = os.path.join(cfg.output_dir, 'wise_snr_results.csv')
    snr_df.to_csv(out1, index=False)
    print(f"\nSaved: {out1}  ({len(snr_df)} rows)")

    # ── 2. NGC5055 azimuthal at shell 2 ────────────────────────────
    print("\n" + "="*60)
    print("WISE W1 — NGC5055 azimuthal at shell 2 (r=14.87 kpc)")
    print("="*60)
    az = azimuthal_profile('NGC5055', 14.87, cfg.data_dir, cfg.n_bins)
    if len(az):
        valid = az[az['ratio'].notna()]
        pk = valid.loc[valid['ratio'].idxmax()]
        print(f"  Peak: φ={pk['phi_mid']:.0f}°  ratio={pk['ratio']:.2f}x  "
              f"mean={valid['ratio'].mean():.3f}")
        out2 = os.path.join(cfg.output_dir, 'wise_azimuthal_ngc5055.csv')
        az.to_csv(out2, index=False)
        print(f"  Saved: {out2}")

    # ── 3. Control baseline ─────────────────────────────────────────
    print("\n" + "="*60)
    print("WISE W1 — control baseline (n=0 galaxies)")
    print("Stellar mass variability << ISM variability expected")
    print("="*60)
    controls = [g for g in GEO if shell_radii.get(g, []) == []]
    ctrl_maxes = []
    for g in sorted(controls)[:8]:
        try:
            data, wcs = load_wise(g, cfg.data_dir)
            r_kpc, _, valid = deproject(g, data, wcs)
            r_bins = np.arange(0.5, 20, 0.5)
            meds = [np.median(data[valid & (r_kpc>=r_bins[i]) & (r_kpc<r_bins[i+1])])
                    for i in range(len(r_bins)-1)
                    if (valid & (r_kpc>=r_bins[i]) & (r_kpc<r_bins[i+1])).sum() >= 20]
            if len(meds) < 4: continue
            a = np.array(meds)
            ratios = [a[i]/np.median(a[max(0,i-3):i+3])
                      for i in range(2,len(a)-2)
                      if np.median(a[max(0,i-3):i+3]) > 0]
            if ratios:
                mx = np.max(ratios)
                ctrl_maxes.append(mx)
                print(f"  {g:14s}: max ratio = {mx:.3f}x  "
                      f"(90th pctile = {np.percentile(ratios,90):.3f}x)")
        except: pass
    if ctrl_maxes:
        print(f"\n  W1 ISM ceiling: {np.median(ctrl_maxes):.3f}x (median of controls)")
        print(f"  cf. HI ceiling: ~1.22x  (from THINGS)")

    if len(snr_df):
        above = snr_df[snr_df['pct_in_gal'] > 80]
        print(f"\n  Shells above 80th pctile: {len(above)}/{len(snr_df)}")
        for _, r in above.sort_values('z_score',ascending=False).iterrows():
            print(f"    {r['galaxy']:14s} r={r['r_sh']:.1f}  "
                  f"pct={r['pct_in_gal']:.0f}%  z={r['z_score']:+.2f}σ  "
                  f"ratio={r['W1_ratio']:.3f}x")


if __name__ == '__main__':
    main()
