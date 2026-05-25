#!/usr/bin/env python3
"""
Download unWISE W1 (3.4um) cutouts for SPARC galaxies.

Uses the unWISE cutout API (unwise.me) — no authentication needed.
W1 traces OLD STELLAR MASS — directly analogous to Spitzer S4G 3.6um.
Same azimuthal analysis as THINGS/HALOGAS, independent tracer.

URL format:
  https://unwise.me/cutout_fits?version=neo7&ra={RA}&dec={DEC}&size={SIZE}&bands=1

Run from repo root:
  python scripts/download_wise_w1.py          # all SPARC shell-bearing + controls
  python scripts/download_wise_w1.py --shells-only
  python scripts/download_wise_w1.py --controls-only
  python scripts/download_wise_w1.py --dry-run

Output: data/external/WISE_W1_{galaxy}.fits
Pixel scale: 2.75 arcsec/pixel
Default size: 400 pixels = 18.3 arcmin (covers any SPARC galaxy)

Author: Ron Bibb | 2026-05-25
"""

import os, time, argparse, requests
import pandas as pd
import numpy as np

UNWISE_URL  = "https://unwise.me/cutout_fits"
OUTPUT_DIR  = "data/external"
SIZE_PIX    = 400      # pixels — 18.3 arcmin at 2.75"/pix, covers all SPARC at D<50Mpc
VERSION     = "neo7"   # latest unWISE reprocessing

# Full SPARC catalog — all shell-bearing AND n=0 controls
# Coordinates from NED / SPARC catalog
GALAXIES = [
    # ── Shell-bearing (n_shells > 0) ──────────────────────────────────────────
    ('CamB',       70.33333, 67.44056,   1, 'shell'),
    ('D512-2',    216.57833, 49.59083,   1, 'shell'),
    ('D564-8',    107.04792, 51.29444,   1, 'shell'),
    ('D631-7',    132.07333, 47.10444,   1, 'shell'),
    ('DDO064',    142.42750, 31.44333,   1, 'shell'),
    ('DDO154',    193.52292, 27.14806,   1, 'shell'),
    ('DDO161',    191.72583,  4.14278,   1, 'shell'),
    ('DDO168',    198.40292, 45.92694,   1, 'shell'),
    ('DDO170',    199.81625, 43.02917,   1, 'shell'),
    ('ESO079-G014', 84.63500,-68.75000,  1, 'shell'),
    ('ESO116-G012',  9.69833,-34.94028,  1, 'shell'),
    ('ESO444-G084', 197.82083,-28.99833, 1, 'shell'),
    ('ESO563-G021', 131.91667,-20.36806, 2, 'shell'),
    ('F561-1',    141.80083, 21.91972,   1, 'shell'),
    ('F563-1',    143.43958, 20.17167,   2, 'shell'),
    ('F563-V2',   143.92708, 20.27000,   2, 'shell'),
    ('F568-3',    168.95708, 22.90583,   1, 'shell'),
    ('IC2574',    157.37500, 68.41222,   1, 'shell'),
    ('IC4202',    197.30000,-46.58167,   1, 'shell'),
    ('KK98-251',  352.10417, 52.18222,   1, 'shell'),
    ('NGC0024',     9.70833,-24.96333,   2, 'shell'),
    ('NGC0055',     3.72292,-39.19694,   1, 'shell'),
    ('NGC0100',    10.47083, 16.01833,   1, 'shell'),
    ('NGC0247',    11.78583,-20.76111,   1, 'shell'),
    ('NGC0300',    13.72292,-37.68444,   1, 'shell'),
    ('NGC0801',    31.62333, 21.86750,   2, 'shell'),
    ('NGC0891',    35.63917, 42.34917,   1, 'shell'),
    ('NGC1003',    39.49125, 40.87222,   2, 'shell'),
    ('NGC1090',    41.68958,-0.24806,    2, 'shell'),
    ('NGC2403',   114.17308, 65.60222,   3, 'shell'),
    ('NGC2683',   133.14375, 33.42250,   1, 'shell'),
    ('NGC2841',   140.51083, 50.97667,   2, 'shell'),
    ('NGC2903',   143.04208, 21.50000,   1, 'shell'),
    ('NGC2915',   141.54583,-76.62667,   1, 'shell'),
    ('NGC3198',   154.97875, 45.55000,   1, 'shell'),
    ('NGC3521',   166.45250, -0.03611,   1, 'shell'),
    ('NGC3726',   173.33250, 47.02917,   1, 'shell'),
    ('NGC3741',   173.73125, 45.28972,   1, 'shell'),
    ('NGC4010',   179.69583, 47.26500,   1, 'shell'),
    ('NGC4013',   179.63458, 43.94722,   1, 'shell'),
    ('NGC5033',   198.36458, 36.59361,   2, 'shell'),
    ('NGC5055',   198.95542, 42.02972,   2, 'shell'),
    ('NGC5371',   208.73167, 40.46222,   1, 'shell'),
    ('NGC5585',   215.60417, 28.74083,   3, 'shell'),
    ('NGC5907',   228.97375, 56.32889,   2, 'shell'),
    ('NGC6503',   274.90042, 70.14444,   2, 'shell'),
    ('NGC6946',   308.71792, 60.15361,   2, 'shell'),
    ('NGC7331',   339.26704, 34.41564,   2, 'shell'),
    ('UGC02259',   42.38583, 27.86611,   1, 'shell'),
    ('UGC04278',  122.23417, 50.73278,   1, 'shell'),
    ('UGC05005',  140.63500, 29.75750,   1, 'shell'),
    ('UGC05414',  151.10083, 34.52500,   2, 'shell'),
    ('UGC05716',  159.00750, 43.02889,   1, 'shell'),
    ('UGC05721',  159.31333, 39.97222,   1, 'shell'),
    ('UGC05750',  160.22583, 25.98194,   1, 'shell'),
    ('UGC06614',  171.53292, 17.10139,   2, 'shell'),
    ('UGC06628',  172.10708, 19.55889,   1, 'shell'),
    ('UGC06818',  178.27333, 41.80917,   1, 'shell'),
    # ── Controls (n_shells = 0) ────────────────────────────────────────────────
    ('DDO064',    142.42750, 31.44333,   0, 'control'),  # also appears as shell? check
    ('NGC2366',   112.82917, 69.21528,   0, 'control'),
    ('NGC2976',   146.81667, 67.89028,   0, 'control'),
    ('NGC3769',   174.03500, 47.89361,   0, 'control'),
    ('NGC3877',   176.52750, 47.49417,   0, 'control'),
    ('NGC3893',   177.26583, 48.71278,   0, 'control'),
    ('NGC3949',   178.13375, 47.85278,   0, 'control'),
    ('NGC3953',   178.46042, 52.32556,   0, 'control'),
    ('NGC3992',   179.70625, 53.37306,   0, 'control'),
    ('NGC4051',   180.79000, 44.53139,   0, 'control'),
    ('NGC4085',   181.28500, 50.35306,   0, 'control'),
    ('NGC4088',   181.38333, 50.53639,   0, 'control'),
    ('NGC4100',   181.50583, 49.57444,   0, 'control'),
    ('NGC4138',   182.36958, 43.68833,   0, 'control'),
    ('NGC4157',   182.71250, 50.48917,   0, 'control'),
    ('NGC4183',   183.24917, 43.69833,   0, 'control'),
    ('NGC4217',   183.96667, 47.09167,   0, 'control'),
    ('NGC4559',   188.99417, 27.95972,   0, 'control'),
    ('NGC7331',   339.26704, 34.41564,   0, 'control'),  # also shell? use SPARC n
]

def download_wise_w1(ra, dec, size=SIZE_PIX, version=VERSION):
    """Download unWISE W1 FITS cutout."""
    params = {'version': version, 'ra': ra, 'dec': dec,
              'size': size, 'bands': '1'}
    r = requests.get(UNWISE_URL, params=params, timeout=60, stream=True)
    if r.status_code != 200:
        return None, r.status_code
    data = b''.join(r.iter_content(65536))
    return data, r.status_code

def main():
    ap = argparse.ArgumentParser(description='Download unWISE W1 cutouts for SPARC galaxies')
    ap.add_argument('--output-dir',    default=OUTPUT_DIR)
    ap.add_argument('--dry-run',       action='store_true')
    ap.add_argument('--shells-only',   action='store_true')
    ap.add_argument('--controls-only', action='store_true')
    ap.add_argument('--size',          type=int, default=SIZE_PIX,
                    help='Cutout size in pixels (2.75 arcsec/pixel)')
    ap.add_argument('--catalog',       default='data/processed/sparc_relaxed_caps_with_BH_decomp.csv',
                    help='SPARC catalog to get n_shells from')
    cfg = ap.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Use catalog n_shells if available (overrides hardcoded values)
    try:
        cat = pd.read_csv(cfg.catalog)
        n_shells_map = dict(zip(cat['Galaxy'], cat['fw_best_n_shells']))
    except FileNotFoundError:
        n_shells_map = {}

    # Build galaxy list from catalog if available, else use hardcoded list
    if n_shells_map:
        import numpy as np
        # Just use the full SPARC galaxy list from catalog + hardcoded coords
        coords = {g[0]: (g[1], g[2]) for g in GALAXIES}
        galaxies = []
        for _, row in cat.iterrows():
            name = row['Galaxy']
            if name in coords:
                n = int(row['fw_best_n_shells'])
                role = 'shell' if n > 0 else 'control'
                galaxies.append((name, coords[name][0], coords[name][1], n, role))
    else:
        galaxies = GALAXIES

    if cfg.shells_only:   galaxies = [g for g in galaxies if g[4]=='shell']
    if cfg.controls_only: galaxies = [g for g in galaxies if g[4]=='control']
    # Deduplicate
    seen = set(); galaxies_dedup = []
    for g in galaxies:
        if g[0] not in seen:
            seen.add(g[0]); galaxies_dedup.append(g)
    galaxies = galaxies_dedup

    fov_arcmin = cfg.size * 2.75 / 60
    print(f"unWISE W1 download — {len(galaxies)} galaxies")
    print(f"  Cutout: {cfg.size}×{cfg.size} px  = {fov_arcmin:.1f} arcmin FOV")
    print(f"  Output: {os.path.abspath(cfg.output_dir)}\n")

    ok = skip = fail = 0
    for name, ra, dec, n_shells, role in galaxies:
        tag     = f"{'SH' if role=='shell' else 'CT'} n={n_shells}"
        outpath = os.path.join(cfg.output_dir, f"WISE_W1_{name}.fits")

        if os.path.exists(outpath) and os.path.getsize(outpath) > 10000:
            print(f"  {name:14s} [{tag}]  exists ({os.path.getsize(outpath)//1024} KB)")
            skip += 1; continue

        if cfg.dry_run:
            url = f"{UNWISE_URL}?version={VERSION}&ra={ra}&dec={dec}&size={cfg.size}&bands=1"
            print(f"  {name:14s} [{tag}]  {url}"); continue

        print(f"  {name:14s} [{tag}]  downloading...", end='', flush=True)
        data, status = download_wise_w1(ra, dec, cfg.size)
        if data and len(data) > 10000:
            with open(outpath, 'wb') as f: f.write(data)
            print(f"  {len(data)//1024} KB  OK"); ok += 1
        else:
            print(f"  FAILED (HTTP {status})"); fail += 1
        time.sleep(0.15)   # be polite to unwise.me

    print(f"\n{'='*55}")
    print(f"Done: {ok} downloaded  {skip} skipped  {fail} failed")

if __name__ == '__main__':
    main()
