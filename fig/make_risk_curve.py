"""Figure 1: the risk curve. Single column, vector, legible in black and white.

(a) median candidate-set size as a function of adversary knowledge (log scale)
(b) share identified when one attribute is added to two known diagnoses

Colour never carries information: identity is carried by position, marker shape and direct labels.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.use("Agg")
RES = Path(__file__).resolve().parents[3] / "experiments" / "mimic" / "results" / "unit"
OUT = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

INK, MUTED, GRID = "#1a1a1a", "#555555", "#d4d4d4"

link = pd.read_csv(RES / "linkage_symmetric.csv")
attr = pd.read_csv(RES / "attributes_symmetric.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0),
                               gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.62})

# ---- (a) the risk curve -----------------------------------------------------
# Values are read from the CSV rather than hard-coded: otherwise re-running the
# experiment would silently leave the figure disagreeing with the paper.
base = float(link.loc[link.policy == "-", "median"].iloc[0])


def curve(policy):
    d = link[link.policy == policy].sort_values("q")
    return [0] + d.q.tolist(), [base] + d["median"].tolist()


xs, yc = curve("common")
_, yr = curve("rare")
ax1.set_yscale("log")
ax1.axhspan(1, 5, color=GRID, alpha=0.45, lw=0)
ax1.text(-0.18, 2.2, "$K<5$", fontsize=6.5, color=MUTED, ha="left", va="center")
ax1.plot(xs, yc, "-o", color=INK, lw=1.4, ms=4.5, mfc="white", mew=1.2, zorder=3,
         clip_on=False, label="$q$ most common")
ax1.plot(xs, yr, "--s", color=MUTED, lw=1.4, ms=4.0, mfc=MUTED, mew=0, zorder=3,
         clip_on=False, label="$q$ rarest")
for x, y in zip(xs, yc):
    ax1.annotate(f"{y:,.0f}".replace(",", " "), (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=7, color=INK)
# label q=1 only: at q=2,3 the value is 1 and can be read off the axis, while a
# label there unavoidably collides with the tick labels
for x, y in list(zip(xs, yr))[1:2]:
    # label above the marker: below it would collide with the axis labels
    ax1.annotate(f"{y:,.0f}".replace(",", " "), (x, y), textcoords="offset points",
                 xytext=(9, 1), ha="left", va="center", fontsize=7, color=MUTED)
ax1.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.6,
           borderaxespad=0.1)
ax1.set_xticks(xs)
ax1.set_xticklabels(["sex,\nage band", "$q{=}1$", "$q{=}2$", "$q{=}3$"])
ax1.set_ylabel("median candidate set")
ax1.set_ylim(0.7, 60000)
ax1.set_xlim(-0.25, 3.25)
ax1.grid(axis="y", color=GRID, lw=0.4, alpha=0.7)
ax1.set_axisbelow(True)
ax1.set_title("(a) risk as a function of adversary knowledge", loc="left", pad=11)

# ---- (b) contribution of individual attributes ------------------------------
d = attr.copy()
d["name"] = d.extra.str.replace("— (diagnoses only)", "two diagnoses only", regex=False)
d = d.sort_values("unique")
hi = d.unique >= 0.15                       # attributes that break the protection
base = float(d[d.name == "two diagnoses only"].unique.iloc[0] * 100)
ax2.axvline(base, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
bars = ax2.barh(d.name, d.unique * 100, height=0.62,
                color=[INK if h else "#b8b8b8" for h in hi], lw=0, zorder=3)
ax2.spines["left"].set_visible(False)
ax2.axvline(0, color=INK, lw=0.6, zorder=4)
for b, v in zip(bars, d.unique * 100):
    ax2.text(v + 1.6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center",
             fontsize=7, color=INK)
ax2.set_xlabel("patients identified (%)")
ax2.set_xlim(0, 112)
ax2.set_xticks([0, 25, 50, 75, 100])
ax2.grid(axis="x", color=GRID, lw=0.4, alpha=0.7)
ax2.set_axisbelow(True)
ax2.tick_params(axis="y", length=0)
ax2.text(base + 1.5, -0.95, "baseline", fontsize=6.5, color=MUTED, va="center")
ax2.set_title("(b) one further attribute, added to two known diagnoses",
              loc="left", pad=8)

fig.savefig(OUT / "risk_curve.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "risk_curve.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote risk_curve.pdf and .png")
