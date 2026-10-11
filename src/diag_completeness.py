"""Діагностика двох зауважень рецензента до перевірки kᵐ.

1. MAX_STEPS: скільки юнітів обриваються на межі кроків, а не через вичерпання ієрархії?
2. pool[:14]: евристика бере лише 14 кодів з найменшим одиничним support. Скільки порушень
   вона пропускає проти ПОВНОГО перебору всіх сполучень?

Вихід: results/unit/diag_completeness.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from kmanon import gen_code

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT/"results"/"unit"
K, MAXLVL, M = 5, 2, 3
N_SAMPLE, SEED = 150, 20261008

rng = np.random.default_rng(SEED)
dx = pd.read_csv(ROOT/"raw"/"hosp"/"diagnoses_icd.csv.gz", dtype={"icd_code": str})
dx["code"] = dx.icd_version.astype(str)+":"+dx.icd_code.str.strip()
pts = pd.read_csv(ROOT/"raw"/"hosp"/"patients.csv.gz", usecols=["subject_id","gender","anchor_age"])
pts["band"] = (pts.anchor_age//10*10).astype(int).astype(str)+"s"
u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)}).join(
    pts.set_index("subject_id")[["gender","band"]], how="inner")
u = u[u.gender.notna() & u.band.notna()]
idx = {l: defaultdict(set) for l in range(MAXLVL+1)}
for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
    for c in cs:
        for l in range(MAXLVL+1): idx[l][(g,b,gen_code(c,l))].add(sid)

def sup(g,b,items):
    S=[idx[l][(g,b,gen_code(c,l))] for c,l in items]
    if not S: return len(u)
    S.sort(key=len); out=S[0]
    for s in S[1:]:
        out = out & s
        if len(out) < K: return len(out)
    return len(out)

def worst(g,b,cs,lvl,m,pool_size):
    s1={c: sup(g,b,[(c,lvl[c])]) for c in cs}
    w,ws=min(s1.values()),[min(s1,key=s1.get)]
    pool=sorted(cs,key=lambda c:s1[c])[:pool_size]
    for j in range(2,m+1):
        for comb in combinations(pool,j):
            v=sup(g,b,[(c,lvl[c]) for c in comb])
            if v<w: w,ws=v,list(comb)
            if w==0: break
    return w,ws

sample=rng.choice(u.index.values,N_SAMPLE,replace=False)
rows=[]
for n,sid in enumerate(sample):
    if n%25==0: print(f"  {n}/{N_SAMPLE}",flush=True)
    cs=sorted(u.at[sid,"codes"]); g,b=u.at[sid,"gender"],u.at[sid,"band"]
    if len(cs)<M or len(cs)>22: continue      # >22 кодів: повний перебір надто дорогий
    lvl={c:0 for c in cs}
    w14,_=worst(g,b,cs,lvl,M,14)
    wfull,_=worst(g,b,cs,lvl,M,len(cs))
    # прогін з великим лімітом кроків
    steps=0; lv=dict(lvl); limit=2*len(cs)+5
    while steps<limit:
        w,ws=worst(g,b,cs,lv,M,14)
        if w>=K: break
        best,gain=None,-1
        for c in ws:
            if lv[c]>=MAXLVL: continue
            t=dict(lv); t[c]+=1; v=sup(g,b,[(x,t[x]) for x in ws])
            if v>gain: best,gain=c,v
        if best is None: break
        lv[best]+=1; steps+=1
    rows.append(dict(sid=sid,n_codes=len(cs),worst_pool14=w14,worst_full=wfull,
                     missed=wfull<w14, steps_used=steps, hit_40=steps>=40,
                     reached=worst(g,b,cs,lv,M,14)[0]>=K))
d=pd.DataFrame(rows); d.to_csv(RES/"diag_completeness.csv",index=False)
print(f"\nn={len(d)}, медіана кодів {d.n_codes.median():.0f}")
print(f"pool[:14] пропускає гірше порушення: {d.missed.mean():.1%} юнітів")
print(f"  медіана worst: pool14 {d.worst_pool14.median():.0f} проти повного {d.worst_full.median():.0f}")
print(f"кроків використано: медіана {d.steps_used.median():.0f}, максимум {d.steps_used.max()}")
print(f"досягли б межі MAX_STEPS=40: {d.hit_40.mean():.1%}")
print(f"reached при знятому ліміті: {d.reached.mean():.1%}")
