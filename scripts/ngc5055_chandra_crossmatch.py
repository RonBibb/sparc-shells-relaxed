"""
NGC 5055 Chandra Cross-Match v2 - column name fix.

Same predictions, fixed to use lowercase column names that HEASARC actually returns.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.heasarc import Heasarc
import warnings
warnings.filterwarnings('ignore')

# ========== NGC 5055 PARAMETERS ==========
NGC5055_RA   = "13h15m49.25s"
NGC5055_DEC  = "+42d01m49.3s"
NGC5055_DIST_MPC = 9.04
NGC5055_R25_KPC  = 11.6
center = SkyCoord(NGC5055_RA, NGC5055_DEC, frame='icrs')

# Shell predictions
SHELLS = {
    'inner': {'r_kpc': 8.79,  'M_sh': 1.53e10, 'sigma_kpc': 1.69,
              'offset_arcmin': 3.34, 'sigma_arcmin': 0.64},
    'outer': {'r_kpc': 14.87, 'M_sh': 4.28e10, 'sigma_kpc': 3.23,
              'offset_arcmin': 5.65, 'sigma_arcmin': 1.23},
}

print("="*78)
print("NGC 5055 Chandra Cross-Match — Inner shell IMBH prediction test")
print("="*78)
print(f"Inner shell predicted at: {SHELLS['inner']['offset_arcmin']:.2f}' +/- {SHELLS['inner']['sigma_arcmin']:.2f}' (1-sigma)")
print(f"                          range: {SHELLS['inner']['offset_arcmin']-SHELLS['inner']['sigma_arcmin']:.2f}' to {SHELLS['inner']['offset_arcmin']+SHELLS['inner']['sigma_arcmin']:.2f}'")
print(f"Outer shell (known X-1):  {SHELLS['outer']['offset_arcmin']:.2f}' (already matched)")
print()

# Query HEASARC CHNGPSCLIU
heasarc = Heasarc()
results = heasarc.query_region(
    position=center,
    radius=8.0 * u.arcmin,
    mission='chngpscliu',
)

print(f"Total sources within 8' of NGC 5055 center: {len(results)}")

# Filter to NGC 5055 association
in_ngc5055 = np.array(['5055' in str(name) for name in results['alt_name']])
ngc_sources = results[in_ngc5055]
print(f"Sources alt_name-associated with NGC 5055: {len(ngc_sources)}")
print()

# Sort by nuclear offset
ngc_sources.sort('nuclear_offset')

# Show all NGC 5055 sources
print("="*78)
print("All NGC 5055 sources, sorted by nuclear offset:")
print("="*78)
print(f"{'alt_name':<25} {'offset(arcmin)':>14} {'offset(kpc)':>12} {'log Lx':>10}")
print("-"*78)
for s in ngc_sources:
    offset_arcmin = float(s['nuclear_offset'])
    # arcmin to kpc: arcsec * (D_Mpc * 1000) / 206265
    offset_kpc = (offset_arcmin * 60) * NGC5055_DIST_MPC * 1000 / 206265
    try:
        log_lx = np.log10(float(s['lx_max']))
        lx_str = f"{log_lx:.2f}"
    except (ValueError, TypeError):
        lx_str = "N/A"
    name = str(s['alt_name'])
    # Mark inner-shell match candidates
    marker = ""
    if SHELLS['inner']['offset_arcmin'] - SHELLS['inner']['sigma_arcmin'] <= offset_arcmin <= SHELLS['inner']['offset_arcmin'] + SHELLS['inner']['sigma_arcmin']:
        marker = "  <-- INNER SHELL CANDIDATE"
    elif SHELLS['outer']['offset_arcmin'] - SHELLS['outer']['sigma_arcmin'] <= offset_arcmin <= SHELLS['outer']['offset_arcmin'] + SHELLS['outer']['sigma_arcmin']:
        marker = "  <-- outer shell region (X-1)"
    print(f"{name:<25} {offset_arcmin:>14.3f} {offset_kpc:>12.2f} {lx_str:>10}{marker}")

# Summary
print()
print("="*78)
print("INNER SHELL MATCH ANALYSIS")
print("="*78)
inner_candidates = ngc_sources[
    (ngc_sources['nuclear_offset'] >= SHELLS['inner']['offset_arcmin'] - SHELLS['inner']['sigma_arcmin']) &
    (ngc_sources['nuclear_offset'] <= SHELLS['inner']['offset_arcmin'] + SHELLS['inner']['sigma_arcmin'])
]
if len(inner_candidates) == 0:
    print("NO MATCH within 1-sigma of inner shell position (3.34' +/- 0.64')")
    # Find nearest
    diffs = np.abs(ngc_sources['nuclear_offset'] - SHELLS['inner']['offset_arcmin'])
    i_nearest = np.argmin(diffs)
    print(f"Nearest source: {ngc_sources['alt_name'][i_nearest]} at {float(ngc_sources['nuclear_offset'][i_nearest]):.3f}'")
    print(f"  Delta from prediction: {float(diffs[i_nearest]):.3f}' (= {float(diffs[i_nearest])/SHELLS['inner']['sigma_arcmin']:.2f} sigma)")
else:
    print(f"{len(inner_candidates)} MATCH(es) found:")
    for s in inner_candidates:
        offset = float(s['nuclear_offset'])
        try:
            log_lx = np.log10(float(s['lx_max']))
            lx_str = f"log Lx = {log_lx:.2f}"
        except:
            lx_str = "Lx unknown"
        print(f"  {s['alt_name']} at {offset:.3f}' ({lx_str})")
        # Compare to expected sub-Eddington luminosity for 3.83e6 M_sun BH
        # Eddington L ~ 1.3e38 * (M/M_sun) = 5e44 erg/s
        # If sub-Eddington fraction ~10^-4 to 10^-3: L_X ~ 10^40-10^41
        print(f"    Framework prediction: M_BH ~ 3.83e6 M_sun -> L_Edd ~ 5e44 erg/s")
        print(f"    Sub-Eddington 10^-3 to 10^-5 -> L_X ~ 10^39 to 10^41 erg/s")
