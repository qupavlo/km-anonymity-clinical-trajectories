"""Does the racial gradient in refusals survive adjustment for trajectory complexity?

Reviewer: refusal is driven by the number of codes and admissions, and those correlate with race
in MIMIC. We check with a logistic regression: race + number of codes + admissions + rare codes.
"""
import numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm

R = Path(__file__).resolve().parents[1]/"results"/"unit"
d = pd.read_csv(R/"who_fails.csv")
d["refused"] = (~d.protected).astype(int)
d["race_g"] = d.race.fillna("UNKNOWN").str.upper().map(
    lambda s: "BLACK" if "BLACK" in s else "HISPANIC" if "HISPANIC" in s
    else "WHITE" if "WHITE" in s else "ASIAN" if "ASIAN" in s else "OTHER")
print("refusal rate by group (unadjusted):")
g = d.groupby("race_g").refused.agg(["mean","size"])
for k,r in g.sort_values("mean",ascending=False).iterrows():
    p,n = r["mean"], int(r["size"]); se=np.sqrt(p*(1-p)/n)
    print(f"  {k:10s} {p:6.1%}  n={n:5d}  95% CI [{max(0,p-1.96*se):.1%}, {min(1,p+1.96*se):.1%}]")

X = pd.get_dummies(d.race_g, prefix="race", drop_first=False).drop(columns=["race_WHITE"])
for c in ("n_codes","n_adm","n_rare10"): X[c]=d[c].astype(float)
X = sm.add_constant(X.astype(float))
m1 = sm.Logit(d.refused, sm.add_constant(pd.get_dummies(d.race_g).drop(columns=["WHITE"]).astype(float))).fit(disp=0)
m2 = sm.Logit(d.refused, X).fit(disp=0)
print("\nunadjusted (ethnicity only), odds ratio vs WHITE:")
for k in m1.params.index:
    if k=="const": continue
    print(f"  {k:12s} OR={np.exp(m1.params[k]):5.2f}  p={m1.pvalues[k]:.3f}")
print("\nadjusted for number of codes, admissions and rare codes:")
for k in m2.params.index:
    if k=="const": continue
    print(f"  {k:12s} OR={np.exp(m2.params[k]):5.2f}  p={m2.pvalues[k]:.3f}")
