#!/usr/bin/env python3
"""
NGC6946 multi-tracer azimuthal analysis.

Shell 1: r=4.249 kpc, sigma=1.12 kpc
  ULX U3: r=3.95 kpc, phi=73.6 deg  (Δr=0.3σ)  — best match
  ULX U2: r=4.73 kpc, phi=142.3 deg (Δr=0.4σ)  — second match

Shell 2: r=11.07 kpc, sigma=2.50 kpc
  ULX U10: r=8.89 kpc, phi=274.0 deg (Δr=0.9σ)

Usage: python scripts/run_ngc6946_azimuthal.py
Output: data/processed/ngc6946_azimuthal_multitracer.{csv,png}
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

GEO = {'ra':308.718,'dec':60.153917,'pa':65.,'inc':33.,'D_mpc':7.73}
SHELLS = [
    {'shell':1,'r_kpc':4.249,'sigma':1.12,
     'ulxs':[('U3',73.6,0.3),('U2',142.3,0.4)]},
    {'shell':2,'r_kpc':11.07,'sigma':2.50,
     'ulxs':[('U10',274.0,0.9)]},
]
TRACERS = [
    ('THINGS NA', 'NGC_6946_NA_MOM0_THINGS.FITS', 'things'),
    ('THINGS RO', 'NGC_6946_RO_MOM0_THINGS.FITS', 'things'),
    ('WISE W1',   'WISE_W1_NGC6946.fits',           'wise'),
]

def load_fits(path, ftype):
    if ftype == 'wise':
        try:
            with tarfile.open(path,'r:*') as tar:
                nm=next((n for n in tar.getnames()
                         if 'img-m.fits' in n and not n.endswith('.gz')),None)
                if nm:
                    with fits.open(io.BytesIO(tar.extractfile(nm).read())) as h:
                        for e in h:
                            if e.data is not None and e.data.ndim==2:
                                return e.data.astype(float),WCS(e.header,naxis=2)
        except tarfile.TarError: pass
    with fits.open(path,memmap=False) as h:
        for e in h:
            d=e.data
            if d is not None:
                d2=np.squeeze(d)
                if d2.ndim==2: return d2.astype(float),WCS(e.header,naxis=2)
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
    return r,phi,np.isfinite(data)&(data>floor)&(r>0.2)

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
    print("NGC6946 multi-tracer azimuthal analysis")
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
    out_csv=os.path.join(cfg.output_dir,'ngc6946_azimuthal_multitracer.csv')
    combined.to_csv(out_csv,index=False); print(f"\nSaved: {out_csv}")
    # Figure
    shells_plot=[s for s in SHELLS]
    fig,axes=plt.subplots(1,len(shells_plot),figsize=(8*len(shells_plot),5.5))
    if len(shells_plot)==1: axes=[axes]
    C={'THINGS NA':'#2C6FAC','THINGS RO':'#5B9BD5','WISE W1':'#228B22'}
    LS={'THINGS NA':'-','THINGS RO':'--','WISE W1':'-'}
    ULX_COLORS=['red','darkred','crimson']
    for ax,sh in zip(axes,shells_plot):
        r_sh=sh['r_kpc']
        sub=combined[combined['r_sh']==r_sh]
        for label in ['THINGS NA','THINGS RO','WISE W1']:
            d=sub[sub['tracer']==label].sort_values('phi_mid')
            if len(d)==0: continue
            ax.plot(d['phi_mid'],d['ratio'],color=C.get(label,'gray'),
                    ls=LS.get(label,'-'),lw=2,marker='o',ms=4,alpha=0.85,label=label)
        for i,(name,phi,dr) in enumerate(sh['ulxs']):
            col=ULX_COLORS[i%len(ULX_COLORS)]
            ax.axvline(phi,color=col,lw=2,ls='-' if i==0 else '--',
                       label=f'{name} φ={phi:.0f}° (Δr={dr:.1f}σ)',zorder=6)
            ax.fill_betweenx([0,30],phi-15,phi+15,alpha=0.07,color=col)
        ax.axhline(1,color='k',lw=1,ls='--',alpha=0.3)
        ax.set_xlabel('Azimuthal angle φ (degrees)',fontsize=11)
        ax.set_ylabel('Ratio (in-shell / off-shell)',fontsize=11)
        ulx_str=' | '.join([f"{n} φ={p:.0f}°" for n,p,_ in sh['ulxs']])
        ax.set_title(f"NGC6946 shell {sh['shell']} (r={r_sh:.2f} kpc)\nULX: {ulx_str}",
                     fontsize=11)
        ax.set_xlim(0,360); ax.set_xticks(range(0,361,45)); ax.set_ylim(bottom=0)
        ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('NGC6946 multi-tracer azimuthal analysis',fontsize=12,y=1.01)
    plt.tight_layout()
    out_png=os.path.join(cfg.output_dir,'ngc6946_azimuthal_multitracer.png')
    plt.savefig(out_png,dpi=170,bbox_inches='tight'); print(f"Saved: {out_png}")

if __name__=='__main__': main()
