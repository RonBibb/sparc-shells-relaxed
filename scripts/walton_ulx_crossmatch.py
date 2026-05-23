#!/usr/bin/env python3
"""
Cross-match Walton+ 2022 ULX catalogue against the 5-shell SPARC catalog.

For each shell-bearing galaxy:
  1. Find all Walton+ 2022 ULX entries in that galaxy
  2. Compute projected galactocentric radius of each ULX in kpc
     (deprojected if inclination is available; on-sky projected otherwise)
  3. For each ULX, find the nearest shell and report Δr
  4. Output: per-shell match status + per-galaxy summary

Inputs:
  - WALTON_CSV:  CSV/FITS conversion of Walton+ 2022 (J/MNRAS/509/1587)
                 Download: https://dwalton354.wixsite.com/djwalton/ulxcat
                 Or:       https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/509/1587
  - SHELL_CSV:   the 5-shell catalog (sparc_relaxed_caps_with_BH_decomp.csv)
  - SPARC_TABLE: SPARC Lelli+2016 master table with Galaxy, RA, DEC, D, Inc, PA
                 Download: http://astroweb.cwru.edu/SPARC/  (Table 1)

Outputs:
  - ulx_shell_matches.csv: one row per (ULX, nearest shell) pair
  - ulx_galaxy_summary.csv: one row per shell-bearing galaxy with ULX counts

Author: generated for Ron Bibb's Paper 4 cross-tracer test
"""

import os
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
WALTON_CSV  = 'walton_2022_ulx.csv'          # converted from FITS/Vizier
SHELL_CSV   = '../data/sparc_relaxed_caps_with_BH_decomp.csv'
SPARC_TABLE = 'sparc_lelli_table1.csv'       # see notes at bottom
OUT_MATCHES = 'ulx_shell_matches.csv'
OUT_SUMMARY = 'ulx_galaxy_summary.csv'

# Match tolerance for "ULX coincides with a shell" (in units of shell sigma)
SIGMA_MATCH_THRESHOLD = 1.0      # within 1 sigma_shell counts as a hit
KPC_FALLBACK_THRESHOLD = 2.0     # if sigma_shell <0.5 kpc, use 2 kpc absolute

# Hard cut: ULX must be inside the galaxy (avoid background AGN contamination)
MAX_R_KPC_BY_GALAXY_FRAC = 1.5   # within 1.5 × largest shell radius

# ============================================================================
# WALTON CATALOG NAME NORMALIZATION
# ============================================================================
# Walton+ 2022 uses HyperLEDA names; SPARC uses 8-char NGC/UGC/IC format.
# Build a normalizer so "NGC 5055" / "NGC5055" / "PGC46153" all match.

def norm_galaxy(name):
    """Normalize galaxy name to SPARC convention (e.g. NGC5055, UGC02487)."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip().upper().replace(' ', '').replace('_', '')
    # Pad NGC/UGC/IC numeric portion to 4 digits, IC to 4
    for prefix in ['NGC', 'UGC', 'IC', 'PGC', 'ESO']:
        if s.startswith(prefix):
            num = s[len(prefix):]
            # Strip leading zeros then re-pad to 4
            try:
                n = int(''.join(c for c in num if c.isdigit()))
                if prefix in ('NGC','UGC','IC'):
                    return f"{prefix}{n:04d}"
                return f"{prefix}{n}"
            except ValueError:
                return s
    return s

# ============================================================================
# COORDINATE CONVERSION
# ============================================================================
def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation in arcsec between two J2000 positions (degrees)."""
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    d = 2 * np.arcsin(np.sqrt(
        np.sin((dec2 - dec1)/2)**2 +
        np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1)/2)**2
    ))
    return np.degrees(d) * 3600.0

def deproject_position(ra_ulx, dec_ulx, ra_gal, dec_gal, distance_mpc, inc_deg, pa_deg):
    """
    Convert ULX sky position to galactocentric radius in kpc, deprojected
    by inclination and position angle of the host disk.
    
    Returns: r_galactocentric_kpc
    """
    # Sky-plane offsets in arcsec
    dra  = (ra_ulx - ra_gal) * np.cos(np.radians(dec_gal)) * 3600.0
    ddec = (dec_ulx - dec_gal) * 3600.0
    
    # Rotate to align with galaxy major axis (PA = angle of major axis E of N)
    pa = np.radians(pa_deg)
    x_maj =  dra * np.sin(pa) + ddec * np.cos(pa)  # along major axis
    y_min = -dra * np.cos(pa) + ddec * np.sin(pa)  # along minor axis (sky)
    
    # Deproject minor-axis offset by inclination
    inc = np.radians(inc_deg)
    y_deproj = y_min / np.cos(inc) if np.cos(inc) > 0.01 else y_min
    
    # Galactocentric radius (in arcsec, deprojected)
    r_arcsec = np.sqrt(x_maj**2 + y_deproj**2)
    
    # Convert to kpc using distance
    r_kpc = r_arcsec * (1.0/3600.0) * (np.pi/180.0) * (distance_mpc * 1000.0)
    return r_kpc

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Load 5-shell catalog
    shells = pd.read_csv(SHELL_CSV)
    shell_bearing = shells[shells['fw_best_n_shells'] > 0].copy()
    shell_bearing['galaxy_norm'] = shell_bearing['Galaxy'].apply(norm_galaxy)
    print(f"Shell-bearing galaxies: {len(shell_bearing)}")
    
    # Load Walton ULX catalog
    if not os.path.exists(WALTON_CSV):
        print(f"ERROR: Walton catalog not found at {WALTON_CSV}")
        print("Download from https://dwalton354.wixsite.com/djwalton/ulxcat")
        print("Convert FITS to CSV with: astropy Table.read(fits).write(csv)")
        print("Required columns: HostName, RAJ2000, DEJ2000, Lpeak, ULXcand")
        return
    
    walton = pd.read_csv(WALTON_CSV)
    print(f"Walton+ 2022 ULX candidates: {len(walton)}")
    walton['galaxy_norm'] = walton['HostName'].apply(norm_galaxy)
    
    # Load SPARC master table for galaxy positions + inclinations
    if not os.path.exists(SPARC_TABLE):
        print(f"WARNING: SPARC master table not at {SPARC_TABLE}")
        print("  Will use on-sky projected radius (no inclination correction)")
        sparc_meta = None
    else:
        sparc_meta = pd.read_csv(SPARC_TABLE)
        sparc_meta['galaxy_norm'] = sparc_meta['Galaxy'].apply(norm_galaxy)
    
    # Cross-match
    matches = []
    summary = []
    
    for _, gal_row in shell_bearing.iterrows():
        gnorm = gal_row['galaxy_norm']
        n_sh = int(gal_row['fw_best_n_shells'])
        
        # Extract shells
        shells_in_gal = []
        for i in range(1, n_sh+1):
            shells_in_gal.append({
                'idx': i,
                'r_kpc': gal_row[f'shell{i}_r_kpc'],
                'M_sh': gal_row[f'shell{i}_M'],
                'sigma_kpc': gal_row[f'shell{i}_sigma_kpc'],
                'n_BH': int(gal_row[f'shell{i}_decomp_n_BH']),
                'acceptable': bool(gal_row[f'shell{i}_decomp_acceptable'])
            })
        
        # Find ULXs in this galaxy
        ulxs = walton[walton['galaxy_norm'] == gnorm]
        
        if sparc_meta is not None:
            meta = sparc_meta[sparc_meta['galaxy_norm'] == gnorm]
            if len(meta) == 0:
                D_mpc, inc, pa, ra_gal, dec_gal = None, None, None, None, None
            else:
                m = meta.iloc[0]
                D_mpc  = m.get('Distance_Mpc', m.get('D', np.nan))
                inc    = m.get('Inclination',  m.get('Inc', np.nan))
                pa     = m.get('PA', np.nan)
                ra_gal = m.get('RA_deg', m.get('RA', np.nan))
                dec_gal= m.get('DEC_deg', m.get('DE', np.nan))
        else:
            D_mpc, inc, pa, ra_gal, dec_gal = None, None, None, None, None
        
        n_ulx = len(ulxs)
        n_matched = 0
        
        for _, ulx in ulxs.iterrows():
            ra_u  = ulx['RAJ2000']
            dec_u = ulx['DEJ2000']
            
            # Compute galactocentric radius
            if ra_gal is not None and not np.isnan(D_mpc):
                if inc is not None and pa is not None and not np.isnan(inc):
                    r_kpc = deproject_position(ra_u, dec_u, ra_gal, dec_gal,
                                                D_mpc, inc, pa)
                    r_method = 'deprojected'
                else:
                    sep_arcsec = angular_sep_arcsec(ra_u, dec_u, ra_gal, dec_gal)
                    r_kpc = sep_arcsec * (1/3600) * (np.pi/180) * D_mpc * 1000
                    r_method = 'projected'
            else:
                r_kpc = np.nan
                r_method = 'no_distance'
            
            # Sanity cut: ULX should be inside the galaxy
            max_shell_r = max(s['r_kpc'] for s in shells_in_gal)
            inside_disk = (not np.isnan(r_kpc)) and (r_kpc < max_shell_r * MAX_R_KPC_BY_GALAXY_FRAC)
            
            # Find nearest shell
            best_dr = np.inf
            best_shell = None
            for s in shells_in_gal:
                if np.isnan(r_kpc):
                    continue
                dr = abs(r_kpc - s['r_kpc'])
                if dr < best_dr:
                    best_dr = dr
                    best_shell = s
            
            # Match criterion: |Δr| < max(sigma_shell, fallback)
            is_match = False
            if best_shell is not None:
                tol = max(best_shell['sigma_kpc'] * SIGMA_MATCH_THRESHOLD,
                          KPC_FALLBACK_THRESHOLD if best_shell['sigma_kpc'] < 0.5 else 0)
                is_match = best_dr <= tol
                if is_match:
                    n_matched += 1
            
            matches.append({
                'Galaxy': gal_row['Galaxy'],
                'ULX_name': ulx.get('Name', ulx.get('SrcID', '')),
                'ULX_RA': ra_u,
                'ULX_DEC': dec_u,
                'Lpeak_erg_s': ulx.get('Lpeak', np.nan),
                'r_ulx_kpc': r_kpc,
                'r_method': r_method,
                'inside_disk': inside_disk,
                'nearest_shell_idx': best_shell['idx'] if best_shell else None,
                'shell_r_kpc': best_shell['r_kpc'] if best_shell else None,
                'shell_sigma_kpc': best_shell['sigma_kpc'] if best_shell else None,
                'shell_M_sh': best_shell['M_sh'] if best_shell else None,
                'shell_n_BH': best_shell['n_BH'] if best_shell else None,
                'shell_acceptable': best_shell['acceptable'] if best_shell else None,
                'delta_r_kpc': best_dr if best_shell else None,
                'delta_r_sigma': best_dr / best_shell['sigma_kpc'] if best_shell else None,
                'is_match': is_match
            })
        
        summary.append({
            'Galaxy': gal_row['Galaxy'],
            'n_shells': n_sh,
            'n_ulx_total': n_ulx,
            'n_ulx_matched': n_matched,
            'chi2_r': gal_row['fw_best_chi2_red'],
            'shell_r_kpc_list': [s['r_kpc'] for s in shells_in_gal],
            'has_match': n_matched > 0
        })
    
    # Write outputs
    matches_df = pd.DataFrame(matches)
    summary_df = pd.DataFrame(summary)
    matches_df.to_csv(OUT_MATCHES, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    
    # Print summary
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total ULX-shell pairs evaluated: {len(matches_df)}")
    print(f"  Inside-disk ULXs: {matches_df['inside_disk'].sum()}")
    print(f"  ULX-shell matches (within 1 sigma): {matches_df['is_match'].sum()}")
    print()
    print(f"Shell-bearing galaxies: {len(summary_df)}")
    print(f"  With >= 1 ULX from Walton+ 2022: {(summary_df['n_ulx_total']>0).sum()}")
    print(f"  With >= 1 ULX-shell positional match: {summary_df['has_match'].sum()}")
    print()
    print("Wrote:", OUT_MATCHES, "and", OUT_SUMMARY)

# ============================================================================
# DOWNLOAD NOTES
# ============================================================================
# WALTON+ 2022 CATALOG:
#   Option A (recommended):
#     1. Download: https://dwalton354.wixsite.com/djwalton/ulxcat
#        Click "ULX Catalogue Files" -> downloads final_ULX_catalogue_files.tar.gz
#     2. Extract: tar -xzf final_ULX_catalogue_files.tar.gz
#     3. Convert FITS to CSV:
#          from astropy.table import Table
#          t = Table.read('master_ULX_catalogue.fits')
#          t.write('walton_2022_ulx.csv', format='csv')
#
#   Option B (via Vizier):
#     1. Visit: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/509/1587
#     2. Use "FTP" -> table "master" -> save as CSV
#
# SPARC MASTER TABLE (for galaxy distances, inclinations, PAs):
#   1. Visit http://astroweb.cwru.edu/SPARC/
#   2. Download SPARC_Lelli2016c.mrt (machine-readable)
#   3. Convert to CSV - the key columns are:
#        Galaxy, T, D (distance, Mpc), Inc (inclination, deg), PA (position angle, deg)
#      The RA/Dec are not in Table 1 but can be obtained from NED by galaxy name.
#
# COLUMN NAME ASSUMPTIONS (you may need to edit):
#   Walton CSV columns expected:  HostName, RAJ2000 (deg), DEJ2000 (deg), Lpeak, ULXcand
#   SPARC table expected: Galaxy, Distance_Mpc, Inclination, PA, RA_deg, DEC_deg

if __name__ == '__main__':
    main()
