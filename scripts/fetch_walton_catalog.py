#!/usr/bin/env python3
"""
Fetch Walton+ 2022 ULX catalog from Vizier and save as CSV.

The catalog ID is J/MNRAS/509/1587. Two tables of interest:
  - 'master':    unique ULX candidates (the deduplicated 'master' list)
  - 'xmm', 'csc2', 'sxps': per-mission tables (you typically want master)

Output: walton_2022_ulx.csv with normalized column names matching the
        cross-match script's expectations.
"""

from astroquery.vizier import Vizier
import pandas as pd

VIZIER_CAT = 'J/MNRAS/509/1587'

def main():
    Vizier.ROW_LIMIT = -1   # no row cap
    print(f"Querying Vizier for {VIZIER_CAT} ...")
    cats = Vizier.get_catalogs(VIZIER_CAT)
    
    print(f"Got {len(cats)} tables:")
    for i, t in enumerate(cats):
        print(f"  [{i}] {t.meta.get('name', 'unnamed')}  rows={len(t)}  cols={t.colnames}")
    
    # The 'master' table is the deduplicated list; try to find it
    target = None
    for t in cats:
        name = t.meta.get('name', '')
        if 'master' in name.lower():
            target = t
            print(f"\nUsing table: {name}")
            break
    if target is None:
        # fallback to the largest table
        target = max(cats, key=len)
        print(f"\n'master' not found; using largest table with {len(target)} rows")
    
    df = target.to_pandas()
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Rows: {len(df)}")
    
    # Map Vizier column names to script expectations.
    # Walton uses HostName/RAJ2000/DEJ2000/LpkX (or similar). Vizier renames sometimes.
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('host', 'hostname', 'galaxy', 'name', 'gal'):
            rename_map[c] = 'HostName'
        if cl in ('raj2000', '_raj2000', 'ra'):
            rename_map[c] = 'RAJ2000'
        if cl in ('dej2000', '_dej2000', 'dec', 'de'):
            rename_map[c] = 'DEJ2000'
        if 'lpk' in cl or 'lpeak' in cl or 'lx' in cl:
            if 'Lpeak' not in rename_map.values():
                rename_map[c] = 'Lpeak'
    df = df.rename(columns=rename_map)
    
    print(f"\nFinal columns: {df.columns.tolist()}")
    df.to_csv('walton_2022_ulx.csv', index=False)
    print(f"\nWrote walton_2022_ulx.csv ({len(df)} rows)")
    print("\nNow re-run: python walton_ulx_crossmatch.py")

if __name__ == '__main__':
    main()
