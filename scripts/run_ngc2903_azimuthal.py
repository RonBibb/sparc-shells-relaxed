#!/usr/bin/env python3
"""
NGC2903 multi-tracer azimuthal analysis.

Shell 1: r=7.031 kpc, sigma=2.11 kpc
ULX (4XMM+CSC2): r=7.10 kpc, phi=77.8 deg  (Δr=0.0 sigma)

Usage: python scripts/run_ngc2903_azimuthal.py
Output: data/processed/ngc2903_azimuthal_multitracer.{csv,png}
"""
import os, argparse, warnings, tarfile, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
warnings.filterwarnings('ignore')

DATA_DIR   = 'data/external'
OUTPUT_DIR = 'data/processed'

GEO     = {'ra':143.042052,'dec':21.501566,'pa':17.,'inc':65.,'D_mpc':8.9}
SHELLS  = [{'shell':1,'r_kpc':7.031,'sigma':2.11}]
ULX_PHI = 77.8   # 4XMM+CSC2, Δr=0.0σ
ULX_R   = 7.10

TRACERS = [
    ('THINGS NA', 'NGC_2903_NA_MOM0_THINGS.FITS', 'things'),
    ('THINGS RO', 'NGC_2903_RO_MOM0_THINGS.FITS', 'things'),
    ('WISE W1',   'WISE_W1_NGC2903.fits',           'wise'),
]

def load_fits(path, ftype):
    if ftype == 'wise':
        try:
            with tarfile.open(path,'r:*') as tar:
                nm = next((n for n in tar.getnames()
                           if 'img-m.fits' in n and not n.endswith('.gz')), None)
                if nm:
                    with fits.open(io.BytesIO(tar.extractfile(nm).read())) as h:
                        for e in h:
                            if e.data is not None and e.data.ndim==2:
                                return e.data.astype(float), WCS(e.header,naxis=2)
        except tarfile.TarError: pass
    with fits.open(path, memmap=False) as h:
        for e in h:
            d = e.data
            if d is not None:
                d2 = np.squeeze(d)
                if d2.ndim==2:
                    return d2.astype(float), WCS(e.header,naxis=2)
    raise ValueError(f"No 2D image: {path}")

def deproject(data, wcs):
    g=GEO; asc=(1/3600.)*(np.pi/180.)*g['D_mpc']*1000.
    pa=np.radians(g['pa']); ci=max(np.cos(np.radians(g['inc'])),0.1)
    yy,xx=np.indices(data.shape); sky=wcs.pixel_to_world_values(xx,yy)
    dra=(sky[0]-g['ra'])*np.cos(np.radians(g['dec']))*3600.
    ddec=(sky[1]-g['dec'])*3600.
    xm=dra*np.sin(pa)+ddec*np.cos(pa); ym=(-dra*np.cos(pa)+ddec*np.sin(pa))/ci
    r=np.sqrt(xm**2+ym**2)*asc; phi=np.degrees(np.arctan2(ym,xm))%360.
    floor=float(np.nanpercentile(data[np.isfinite(data)&(data>0)],10)) if np.any(data>0) else 0.
    return r, phi, np.isfinite(data)&(data>floor)&(r>0.2)

def az_profile(data, wcs, r_sh, n_bins=36):
    r,phi,v=deproject(data,wcs); dsh=0.25*r_sh
    in_sh=v&(r>r_sh-dsh)&(r<r_sh+dsh); off=v&(r>r_sh+dsh)&(r<r_sh+3*dsh)
    bins=np.linspace(0,360,n_bins+1); rows=[]
    for i in range(n_bins):
        mi=in_sh&(phi>=bins[i])&(phi<bins[i+1])
        mo=off  &(phi>=bins[i])&(phi<bins[i+1])
        vi=np.median(data[mi]) if mi.sum()>=5 else np.nan
        vo=np.median(data[mo]) if mo.sum()>=5 else np.nan
        rat=vi/vo if (pd.notna(vi) and pd.notna(vo) and vo>0) else np.nan
        rows.append({'phi_mid':(bins[i]+bins[i+1])/2,'ratio':rat,'n_pix':int(mi.sum())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default=DATA_DIR)
    ap.add_argument('--output-dir',default=OUTPUT_DIR)
    ap.add_argument('--n-bins',type=int,default=36)
    cfg=ap.parse_args(); os.makedirs(cfg.output_dir,exist_ok=True)
    print(f"NGC2903  ULX phi={ULX_PHI}°  r={ULX_R} kpc  (Δr=0.0σ from shell 1)")
    loaded={}
    for label,fname,ftype in TRACERS:
        path=os.path.join(cfg.data_dir,fname)
        if not os.path.exists(path): print(f"  {label}: NOT FOUND"); continue
        try:
            data,wcs=load_fits(path,ftype); loaded[label]=(data,wcs)
            print(f"  {label}: loaded {data.shape}")
        except Exception as e: print(f"  {label}: {e}")
    rows=[]
    for sh in SHELLS:
        r_sh=sh['r_kpc']
        print(f"\nShell {sh['shell']}  r={r_sh:.2f} kpc")
        for label,(data,wcs) in loaded.items():
            df=az_profile(data,wcs,r_sh,cfg.n_bins)
            v=df[df['ratio'].notna()]
            if len(v)==0: continue
            pk=v.loc[v['ratio'].idxmax()]
            print(f"  {label:12s}: peak phi={pk['phi_mid']:.0f}°  ratio={pk['ratio']:.2f}x")
            df['tracer']=label; df['shell']=sh['shell']; df['r_sh']=r_sh
            rows.append(df)
    if not rows: return
    combined=pd.concat(rows,ignore_index=True)
    out_csv=os.path.join(cfg.output_dir,'ngc2903_azimuthal_multitracer.csv')
    combined.to_csv(out_csv,index=False); print(f"\nSaved: {out_csv}")
    # Figure
    fig,ax=plt.subplots(figsize=(9,5.5))
    C={'THINGS NA':'#2C6FAC','THINGS RO':'#5B9BD5','WISE W1':'#228B22'}
    LS={'THINGS NA':'-','THINGS RO':'--','WISE W1':'-'}
    for label in ['THINGS NA','THINGS RO','WISE W1']:
        d=combined[combined['tracer']==label].sort_values('phi_mid')
        if len(d)==0: continue
        ax.plot(d['phi_mid'],d['ratio'],color=C.get(label,'gray'),
                ls=LS.get(label,'-'),lw=2,marker='o',ms=4,alpha=0.85,label=label)
    ax.axvline(ULX_PHI,color='red',lw=2.5,label=f'ULX phi={ULX_PHI:.0f}° (Δr=0.0σ)')
    ax.fill_betweenx([0,30],ULX_PHI-15,ULX_PHI+15,alpha=0.08,color='red')
    ax.axhline(1,color='k',lw=1,ls='--',alpha=0.3)
    ax.set_xlabel('Azimuthal angle φ (degrees)',fontsize=11)
    ax.set_ylabel('Ratio (in-shell / off-shell)',fontsize=11)
    ax.set_title('NGC2903 shell 1 (r=7.03 kpc)\nULX at Δr=0.0σ, φ=77.8°',fontsize=12)
    ax.set_xlim(0,360); ax.set_xticks(range(0,361,45)); ax.set_ylim(bottom=0)
    ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
    out_png=os.path.join(cfg.output_dir,'ngc2903_azimuthal_multitracer.png')
    plt.tight_layout(); plt.savefig(out_png,dpi=170,bbox_inches='tight')
    print(f"Saved: {out_png}")

if __name__=='__main__': main()
