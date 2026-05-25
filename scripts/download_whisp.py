#!/usr/bin/env python3
"""
Download WHISP HI moment maps for SPARC galaxies.

Queries the WHISP VO service at vo.astron.nl and downloads MOM0
(integrated intensity) FITS files for all SPARC-WHISP overlap galaxies.

Run from repo root:
  python scripts/download_whisp.py
  python scripts/download_whisp.py --output-dir data/external --dry-run

Requirements:
  pip install pyvo astropy requests

Author: Ron Bibb
Date:   2026-05-25
"""

import os
import argparse
import requests
import time
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# ── SPARC-WHISP galaxy list ───────────────────────────────────────────────────
# Confirmed WHISP galaxies (Swaters+2002, Noordermeer+2005, van der Hulst+2002)
# that are also in the SPARC relaxed-cap catalog
# n_shells = number of shells detected in Paper 3 relaxed-cap fit
GALAXIES = [
    # name          RA (deg)    Dec (deg)   n_shells  role
    # Shell-bearing
    ('DDO168',      198.40292,  45.92694,   1,  'shell'),
    ('DDO170',      199.81625,  43.02917,   1,  'shell'),
    ('NGC2403',     114.17308,  65.60222,   3,  'shell'),
    ('NGC2841',     140.51083,  50.97667,   2,  'shell'),
    ('NGC3198',     154.97875,  45.55000,   1,  'shell'),
    ('NGC3726',     173.33250,  47.02917,   1,  'shell'),
    ('NGC4010',     179.69583,  47.26500,   1,  'shell'),
    ('NGC4013',     179.63458,  43.94722,   1,  'shell'),
    ('NGC5033',     198.36458,  36.59361,   2,  'shell'),
    ('NGC5055',     198.95542,  42.02972,   2,  'shell'),
    ('NGC5585',     215.60417,  28.74083,   3,  'shell'),
    ('NGC6503',     274.90042,  70.14444,   2,  'shell'),
    ('NGC6946',     308.71792,  60.15361,   2,  'shell'),
    ('UGC04278',    122.23417,  50.73278,   1,  'shell'),
    # Controls (n=0 shells) — establish ISM variability baseline
    ('NGC2683',     133.14375,  33.42250,   0,  'control'),
    ('NGC3769',     174.03500,  47.89361,   0,  'control'),
    ('NGC3877',     176.52750,  47.49417,   0,  'control'),
    ('NGC3893',     177.26583,  48.71278,   0,  'control'),
    ('NGC3949',     178.13375,  47.85278,   0,  'control'),
    ('NGC3953',     178.46042,  52.32556,   0,  'control'),
    ('NGC3992',     179.70625,  53.37306,   0,  'control'),
    ('NGC4051',     180.79000,  44.53139,   0,  'control'),
    ('NGC4085',     181.28500,  50.35306,   0,  'control'),
    ('NGC4088',     181.38333,  50.53639,   0,  'control'),
    ('NGC4100',     181.50583,  49.57444,   0,  'control'),
    ('NGC4138',     182.36958,  43.68833,   0,  'control'),
    ('NGC4157',     182.71250,  50.48917,   0,  'control'),
    ('NGC4183',     183.24917,  43.69833,   0,  'control'),
    ('NGC4217',     183.96667,  47.09167,   0,  'control'),
    ('NGC4559',     188.99417,  27.95972,   0,  'control'),
]

OUTPUT_DIR = 'data/external'
WHISP_VO   = 'https://vo.astron.nl/whisp/q/cube/siap.xml'


def query_whisp_vo(ra, dec, size_deg=0.5):
    """
    Query the WHISP VO SIAP service for moment map URLs near (ra, dec).
    Returns list of (url, format, description) tuples.
    """
    params = {
        'POS': f'{ra:.5f},{dec:.5f}',
        'SIZE': f'{size_deg}',
        'FORMAT': 'image/fits',
    }
    try:
        r = requests.get(WHISP_VO, params=params, timeout=20)
        if r.status_code != 200:
            return []
        # Parse VOTable response
        from astropy.io.votable import parse_single_table
        from io import BytesIO
        table = parse_single_table(BytesIO(r.content))
        urls = []
        for row in table.array:
            url  = str(row['access_url']).strip()
            fmt  = str(row.get('access_format', '')).strip()
            desc = str(row.get('image_title', '')).strip()
            if url and 'fits' in url.lower():
                urls.append((url, fmt, desc))
        return urls
    except Exception as e:
        return []


def download_file(url, outpath, verbose=True):
    """Download a file from url to outpath."""
    try:
        r = requests.get(url, timeout=60, stream=True)
        if r.status_code != 200:
            if verbose: print(f"      HTTP {r.status_code}")
            return False
        size = 0
        with open(outpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                size += len(chunk)
        if verbose: print(f"      OK  {size//1024} KB")
        return True
    except Exception as e:
        if verbose: print(f"      Error: {e}")
        return False


def whisp_direct_url(galaxy):
    """
    Construct direct URL for WHISP moment maps.
    Based on the WHISP website file naming convention.
    """
    # WHISP website hosts files at:
    # https://www.astron.nl/whispimage/<GALAXY>_hi_mom0.fits.gz
    # Naming: galaxy names are standardized
    name = galaxy.replace(' ','').upper()
    base = f"https://www.astron.nl/whispimage"
    return {
        'mom0': f"{base}/{name}_hi_mom0.fits.gz",
        'mom1': f"{base}/{name}_hi_mom1.fits.gz",
    }


def main():
    ap = argparse.ArgumentParser(description='Download WHISP moment maps for SPARC galaxies')
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--dry-run',    action='store_true',
                    help='Print URLs without downloading')
    ap.add_argument('--shells-only', action='store_true',
                    help='Download only shell-bearing galaxies')
    ap.add_argument('--controls-only', action='store_true',
                    help='Download only control (n=0) galaxies')
    cfg = ap.parse_args()

    os.makedirs(cfg.output_dir, exist_ok=True)

    gals = GALAXIES
    if cfg.shells_only:   gals = [g for g in gals if g[4]=='shell']
    if cfg.controls_only: gals = [g for g in gals if g[4]=='control']

    print(f"WHISP download — {len(gals)} galaxies")
    print(f"  Output: {os.path.abspath(cfg.output_dir)}")
    print(f"  Shell-bearing: {sum(1 for g in gals if g[4]=='shell')}")
    print(f"  Controls:      {sum(1 for g in gals if g[4]=='control')}")
    print()

    success, skipped, failed = 0, 0, 0

    for name, ra, dec, n_shells, role in gals:
        tag    = f"[{'SH' if role=='shell' else 'CT'}  n={n_shells}]"
        outname = f"WHISP_{name}_mom0.fits"
        outpath = os.path.join(cfg.output_dir, outname)

        if os.path.exists(outpath) and os.path.getsize(outpath) > 10000:
            print(f"  {name:12s} {tag}  already exists — skipping")
            skipped += 1
            continue

        print(f"  {name:12s} {tag}  querying WHISP VO...")

        if cfg.dry_run:
            urls = whisp_direct_url(name)
            print(f"      mom0: {urls['mom0']}")
            continue

        # Method 1: WHISP VO SIAP query
        urls = query_whisp_vo(ra, dec)
        mom0_url = None
        for url, fmt, desc in urls:
            if 'mom0' in url.lower() or 'mom0' in desc.lower():
                mom0_url = url
                break
            elif 'integrated' in desc.lower() or 'zeroth' in desc.lower():
                mom0_url = url
                break

        # Method 2: direct URL fallback
        if not mom0_url:
            direct = whisp_direct_url(name)
            mom0_url = direct['mom0']

        print(f"      URL: {mom0_url}")
        ok = download_file(mom0_url, outpath)
        if ok:
            # Verify it's a real FITS file
            if os.path.getsize(outpath) < 5000:
                print(f"      WARNING: file too small ({os.path.getsize(outpath)} bytes) — may be error page")
                os.remove(outpath)
                ok = False

        if ok:
            success += 1
        else:
            failed += 1
            if os.path.exists(outpath): os.remove(outpath)

        time.sleep(0.5)  # be polite

    print()
    print(f"{'='*50}")
    print(f"Done: {success} downloaded, {skipped} skipped, {failed} failed")
    if failed > 0:
        print(f"  For failed galaxies, try the WHISP VO manually:")
        print(f"  https://vo.astron.nl/whisp/q/cube/form")
        print(f"  Or the WHISP website: https://www.astron.nl/whisp/")

    print()
    print(f"Output files:")
    for f in sorted(os.listdir(cfg.output_dir)):
        if f.startswith('WHISP_') and f.endswith('.fits'):
            sz = os.path.getsize(os.path.join(cfg.output_dir, f))
            print(f"  {f}  {sz//1024} KB")


if __name__ == '__main__':
    main()
