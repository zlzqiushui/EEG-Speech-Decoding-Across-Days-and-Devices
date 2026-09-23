"""Figures 3 and 4 from the local v5/v6 manuscript plotting code."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import reanalyze_submission as a


def make_figures(output):
    OUT = Path(output)
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    penalties = pd.read_csv(a.OUT/'paired_generalization_penalty.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
     'axes.labelsize':9.5,'axes.titlesize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
     'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,
     'axes.spines.top':False,'axes.spines.right':False})
    features=[('envelope','Envelope'),('mel_10','Mel-10'),('wav2vecbase9','wav2vec 2.0')]
    shifts=[['day1_to_day2','day2_to_day1'],['neurascan_to_brk']]
    def save(fig,name):
        fig.savefig(OUT/f'figures/{name}.pdf')
        fig.savefig(OUT/f'figures/{name}.png',dpi=220)
        plt.close(fig)

    # No connecting lines across categorical shift conditions; directly aligned estimates.
    fig,axes=plt.subplots(2,1,figsize=(3.386,2.9),sharex=True)
    x=np.array([0,1,2,3.5,4.5,5.5])
    d=penalties.set_index(['shift','feature']).loc[[(s,l) for s in ['cross-day','day-plus-device'] for _,l in features]]
    ax=axes[0]
    ax.errorbar(x-.12,d.target_mean,yerr=d.target_sem,fmt='o',ms=3.5,color='#0072B2',capsize=2,lw=1,label='Target-trained')
    ax.errorbar(x+.12,d.zero_mean,yerr=d.zero_sem,fmt='s',ms=3.5,color='#D55E00',mfc='white',capsize=2,lw=1,label='Zero-shot')
    ax.axhline(20,ls=':',color='#777777',lw=.8)
    ax.set(ylim=(17,73),yticks=[20,40,60],ylabel='Accuracy (%)')
    ax.text(.02,.96,'(a) Mean ± SEM',transform=ax.transAxes,va='top')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,handletextpad=.2,columnspacing=.7,bbox_to_anchor=(.55,1.01))
    ax=axes[1]
    ax.errorbar(x,d.mean_diff,yerr=np.vstack([d.mean_diff-d.ci_low,d.ci_high-d.mean_diff]),fmt='o',ms=3.5,color='#0072B2',capsize=2,lw=1)
    ax.axhline(0,color='#777777',lw=.8)
    ax.set(ylim=(-2,30),yticks=[0,10,20],ylabel='Loss (pp)')
    ax.text(.02,.98,'(b) Paired loss, 95% CI',transform=ax.transAxes,va='top')
    ax.set_xticks(x, ['E','M','W']*2)
    ax.text(.245,-.38,'Cross-day',transform=ax.transAxes,ha='center')
    ax.text(.78,-.38,'Day + device',transform=ax.transAxes,ha='center')
    for ax in axes:
        ax.set_xlim(-.5,6);ax.axvline(2.75,color='#bbbbbb',lw=.6)
        ax.grid(axis='y',alpha=.2);ax.tick_params(pad=2,length=2)
    axes[0].tick_params(axis='x',bottom=False)
    fig.subplots_adjust(left=.17,right=.99,top=.84,bottom=.19,hspace=.13)
    save(fig,'fig3_generalization')

    # Six panels retain the original row/column arrangement, with readable type.
    fig,axes=plt.subplots(3,2,figsize=(3.386,4.1),sharex=True,sharey=True)
    ratios=np.array([.2,.4,.6,.8,1.])
    styles={2:('#0072B2','o','-'),3:('#009E73','s','--'),4:('#CC79A7','^','-.')}
    rows=[]
    for i,(f,label) in enumerate(features):
     for j,transfers in enumerate(shifts):
        ax=axes[i,j];frames=[a.read_b(f,t) for t in transfers]
        def vals(s,r):
            v=pd.concat([df[a.col(s,r)] for df in frames],axis=1)
            assert v.shape==(25,len(frames)) and not v.isna().any().any()
            return v.mean(axis=1)
        for s,(c,m,ls) in styles.items():
            vs=[vals(s,r) for r in ratios];means=np.array([v.mean() for v in vs]);ses=np.array([v.sem() for v in vs])
            ax.plot(ratios,means,color=c,marker=m,ls=ls,ms=3,lw=1)
            ax.fill_between(ratios,means-ses,means+ses,color=c,alpha=.13,lw=0)
        ax.axhline(vals(1,1).mean(),color='#555555',ls='--',lw=.9)
        v=vals(5,1);ax.errorbar(1,v.mean(),yerr=v.sem(),fmt='D',mfc='white',color='#A46700',ms=4,capsize=2,lw=.9)
        ax.axhline(20,color='#777777',ls=':',lw=.8)
        ax.set(xlim=(.15,1.055),ylim=(16,73),xticks=[.2,.6,1],yticks=[20,40,60])
        ax.text(.025,.98,chr(97+i*2+j)+') '+['Env.','Mel-10','w2v'][i],transform=ax.transAxes,va='top')
        if i==0: ax.set_title(['Cross-day','Day + device'][j],pad=7)
        ax.grid(axis='y',alpha=.2,lw=.5);ax.tick_params(pad=2,length=2)
        for s in range(1,6):
         for r in (ratios if s in (2,3,4) else [1.]):
            v=vals(s,r);rows.append(dict(feature=f,shift=['cross-day','day-plus-device'][j],strategy=s,ratio=r,n=len(v),mean=v.mean(),sem=v.sem()))
    handles=[Line2D([],[],color='#555555',ls='--',label='S1')]+[Line2D([],[],color=c,marker=m,ls=ls,label=f'S{s}',ms=3) for s,(c,m,ls) in styles.items()]+[Line2D([],[],color='#A46700',marker='D',mfc='white',ls='none',label='S5',ms=4),Line2D([],[],color='#777777',ls=':',label='Chance')]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,1.01),columnspacing=.75,handlelength=1.25,handletextpad=.3)
    fig.supylabel('Accuracy (%)',fontsize=9.5,x=.005)
    fig.supxlabel('Target training-data ratio',fontsize=9.5,y=.005)
    fig.subplots_adjust(left=.16,right=.99,top=.79,bottom=.105,hspace=.15,wspace=.14)
    save(fig,'fig4_calibration')
    pd.DataFrame(rows).to_csv(OUT/'statistics/figure4_source_values.csv',index=False)
    # Direction-specific descriptive audit: no new hypothesis tests.
    direction=[]
    for f,label in features:
     for t in ['day1_to_day2','day2_to_day1','neurascan_to_brk']:
        b=a.read_b(f,t)
        direction.append(dict(feature=label,transfer=t,
          sparse_s2_s1=(b[a.col(2,.2)]-b[a.col(1,1)]).mean(),
          sparse_s2_s3=(b[a.col(2,.2)]-b[a.col(3,.2)]).mean(),
          full_s2_s3=(b[a.col(2,1)]-b[a.col(3,1)]).mean()))
    pd.DataFrame(direction).to_csv(OUT/'statistics/directional_descriptive.csv',index=False)
