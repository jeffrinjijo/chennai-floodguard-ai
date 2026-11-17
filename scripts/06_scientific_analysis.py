"""Step 6 - uncertainty and sanity analysis of the susceptibility scores (numpy/pandas/matplotlib only).

Reads data/processed/predictions.csv (written by 04_train_models.py; labelled cells carry OUT-OF-FOLD scores)
and reports:
  1. Block-bootstrap 95% CIs for ROC-AUC / PR-AUC (resampling 5 km blocks, the unit of the spatial CV)
  2. Enrichment: observed recurrent-flood rate per risk class and per score quintile; precision/recall/lift at fixed budgets
  3. Monotonic sanity check: ambiguous cells (flooded once) should score between never-flooded and recurrent cells
  4. Spatial autocorrelation (Moran's I, kNN weights, permutation p) of the labels and of the model residuals
  5. Event overlap (phi / Jaccard) between the three flood records
  6. Descriptive statistics of the features

Outputs: reports/scientific_analysis.md, reports/figures/roc_pr_bootstrap.png, reports/figures/enrichment.png
Run: python scripts/06_scientific_analysis.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, BLOCK_KM, N_BOOT, N_PERM, KNN = 42, 5, 2000, 999, 8
rng = np.random.default_rng(SEED)

s = pd.read_csv(ROOT / "data/processed/predictions.csv")
dy = BLOCK_KM / 111.0
dx = BLOCK_KM / (111.0 * np.cos(np.radians(s.lat.mean())))
s["block"] = (((s.lon - s.lon.min()) // dx).astype(int).astype(str) + "_" +
              ((s.lat - s.lat.min()) // dy).astype(int).astype(str))
L = s[s.recurrent_label >= 0].reset_index(drop=True)
y = L.recurrent_label.values.astype(int)
p = L.score.values


# ---------- metrics without sklearn ----------
def auc(yt, sc):
    r = pd.Series(sc).rank().values
    n1, n0 = yt.sum(), (1 - yt).sum()
    return float((r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(yt, sc):
    o = np.argsort(-sc, kind="mergesort")
    yy = yt[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def roc_curve(yt, sc):
    o = np.argsort(-sc, kind="mergesort")
    yy = yt[o]
    tpr = np.r_[0, np.cumsum(yy) / yy.sum()]
    fpr = np.r_[0, np.cumsum(1 - yy) / (1 - yy).sum()]
    return fpr, tpr


# ---------- 1. block bootstrap ----------
blocks = {b: np.where(L.block.values == b)[0] for b in L.block.unique()}
keys = list(blocks)
base_rule = -L.elev_mean.values          # fixed, un-fitted rule: lower elevation = riskier
rows = {"XGBoost (out-of-fold)": p, "Elevation rule (lower = riskier)": base_rule}
boot = {k: {"auc": [], "ap": []} for k in rows}
diff = []
grid = np.linspace(0, 1, 101)
tpr_band = []
for _ in range(N_BOOT):
    pick = rng.choice(len(keys), len(keys), replace=True)
    idx = np.concatenate([blocks[keys[i]] for i in pick])
    yt = y[idx]
    if yt.sum() in (0, len(yt)):
        continue
    a = {}
    for k, sc in rows.items():
        a[k] = auc(yt, sc[idx])
        boot[k]["auc"].append(a[k])
        boot[k]["ap"].append(ap(yt, sc[idx]))
    diff.append(a["XGBoost (out-of-fold)"] - a["Elevation rule (lower = riskier)"])
    f, t = roc_curve(yt, p[idx])
    tpr_band.append(np.interp(grid, f, t))
ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
diff = np.array(diff)

# ---------- 2. enrichment ----------
prev = y.mean()
cls = L.groupby("risk_class").agg(n=("score", "size"), recurrent=("recurrent_label", "sum"))
cls["rate"] = cls.recurrent / cls.n
cls["lift"] = cls.rate / prev
L["quint"] = pd.qcut(L.score.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
qn = L.groupby("quint").agg(n=("score", "size"), recurrent=("recurrent_label", "sum"))
qn["rate"] = qn.recurrent / qn.n
budget_rows = []
order = np.argsort(-p)
for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
    k = int(round(frac * len(L)))
    tp = y[order[:k]].sum()
    budget_rows.append((frac, k, tp / k, tp / y.sum(), (tp / k) / prev))

# ---------- 3. monotonic sanity ----------
s["n_ev"] = s.n_events.astype(int)
mono = s.groupby("n_ev").score.agg(["size", "mean", "median"])
# note: labelled cells carry out-of-fold scores, ambiguous cells (n_events == 1) carry final-model scores
amb_vs_neg = auc(np.r_[np.ones((s.n_ev == 1).sum()), np.zeros((s.n_ev == 0).sum())].astype(int),
                 np.r_[s.score[s.n_ev == 1], s.score[s.n_ev == 0]])
pos_vs_amb = auc(np.r_[np.ones((s.n_ev >= 2).sum()), np.zeros((s.n_ev == 1).sum())].astype(int),
                 np.r_[s.score[s.n_ev >= 2], s.score[s.n_ev == 1]])


# ---------- 4. Moran's I ----------
def knn_idx(df, k):
    x = (df.lon.values - df.lon.mean()) * 111.32 * np.cos(np.radians(df.lat.mean()))
    yv = (df.lat.values - df.lat.mean()) * 110.57
    d = np.hypot(x[:, None] - x[None], yv[:, None] - yv[None])
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def morans(z, nbr):
    z = z - z.mean()
    return float((z * z[nbr].mean(1)).sum() / (z * z).sum())


def moran_p(z, nbr):
    obs = morans(z, nbr)
    perm = np.array([morans(rng.permutation(z), nbr) for _ in range(N_PERM)])
    return obs, float((1 + (perm >= obs).sum()) / (N_PERM + 1))


nbr_L = knn_idx(L, KNN)
nbr_S = knn_idx(s, KNN)
I_label, p_label = moran_p(y.astype(float), nbr_L)
I_resid, p_resid = moran_p(y - p, nbr_L)
I_score, p_score = moran_p(s.score.values, nbr_S)
I_elev, p_elev = moran_p(s.elev_mean.values, nbr_S)

# ---------- 5. event overlap ----------
ev = pd.DataFrame({"2005": s.pos_2005, "2015": s.e2015, "2020": s.pos_hot_2020})
phi = ev.corr()
jac = pd.DataFrame({a: {b: ((ev[a] & ev[b]).sum() / max(((ev[a] | ev[b]).sum()), 1)) for b in ev} for a in ev})

# ---------- 6. feature stats ----------
feat_cols = [c for c in ["elev_mean", "elev_min", "elev_std", "slope_mean", "rel_elev", "dist_sea_m", "dist_inland_water_m",
                         "frac_tree", "frac_shrub", "frac_grass", "frac_crop", "frac_built", "frac_bare", "frac_water",
                         "frac_wetland"] if c in s.columns]
desc = s[feat_cols].describe().T[["mean", "std", "min", "50%", "max"]]
miss = s[feat_cols].isna().mean()

# ---------- figures ----------
fig_dir = ROOT / "reports/figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
tb = np.array(tpr_band)
f0, t0 = roc_curve(y, p)
fe, te = roc_curve(y, base_rule)
ax[0].fill_between(grid, np.percentile(tb, 2.5, 0), np.percentile(tb, 97.5, 0), color="#d35400", alpha=.2, label="XGBoost 95% block-bootstrap band")
ax[0].plot(f0, t0, color="#d35400", lw=2, label=f"XGBoost OOF (AUC {auc(y, p):.2f})")
ax[0].plot(fe, te, color="#2c3e50", lw=1.5, ls="--", label=f"Elevation rule (AUC {auc(y, base_rule):.2f})")
ax[0].plot([0, 1], [0, 1], color="#999", lw=1, ls=":")
ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC, spatially out-of-fold"); ax[0].legend(loc="lower right", fontsize=8)
o = np.argsort(-p); yy = y[o]
ax[1].plot(np.cumsum(yy) / np.arange(1, len(yy) + 1), color="#d35400", lw=2, label="XGBoost OOF")
ax[1].axhline(prev, color="#999", ls=":", label=f"chance = prevalence {prev:.2f}")
ax[1].set(xlabel="Cells flagged (ranked by score)", ylabel="Precision", title="Precision as the flagged list grows"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(fig_dir / "roc_pr_bootstrap.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(cls.index.astype(str), cls.rate, color=["#fee8c8", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"])
ax[0].axhline(prev, color="#555", ls=":"); ax[0].set(xlabel="Risk class (1 low - 5 very high)", ylabel="Observed recurrent-flood rate", title="Observed rate by risk class (labelled cells)")
for i, (r, n) in enumerate(zip(cls.rate, cls.n)):
    ax[0].text(i, r + .01, f"n={n}", ha="center", fontsize=8)
ax[1].bar(mono.index.astype(str), mono["mean"], color=["#9ec5e8", "#fdbb84", "#e34a33", "#b30000"][:len(mono)])
ax[1].set(xlabel="Number of flood events recorded in the cell", ylabel="Mean score", title="Mean score by events recorded (3-event bar: n=9)")
for i, (m, n) in enumerate(zip(mono["mean"], mono["size"])):
    ax[1].text(i, m + .01, f"n={n}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(fig_dir / "enrichment.png", dpi=140); plt.close()

# ---------- report ----------
f2 = lambda v: f"{v:.2f}"
R = ["# Scientific analysis", "",
     f"Labelled cells: {len(L)} ({int(y.sum())} recurrent, {int((1-y).sum())} never-flooded) in {len(keys)} spatial blocks of {BLOCK_KM} km. "
     f"Scores are out-of-fold from 5-fold spatial block CV. Bootstrap: {N_BOOT} resamples of blocks (not cells), seed {SEED}.", "",
     "## 1. Uncertainty (block bootstrap, 95% percentile interval)", "",
     "| ranking | ROC-AUC | PR-AUC |", "|---|---|---|"]
for k in rows:
    a, b = ci(boot[k]["auc"]), ci(boot[k]["ap"])
    pt = (auc(y, rows[k]), ap(y, rows[k]))
    R.append(f"| {k} | {f2(pt[0])} [{f2(a[0])}, {f2(a[1])}] | {f2(pt[1])} [{f2(b[0])}, {f2(b[1])}] |")
d_lo, d_hi = ci(diff)
R += ["", f"Paired difference in ROC-AUC (XGBoost minus elevation rule): {diff.mean():+.2f}, 95% interval [{d_lo:+.2f}, {d_hi:+.2f}]. "
      + ("The interval excludes zero." if d_lo > 0 or d_hi < 0 else "The interval includes zero, so the gain over elevation alone is not established."),
      f"Chance PR-AUC = prevalence = {prev:.2f}.", "",
      "![ROC and precision](figures/roc_pr_bootstrap.png)", "",
      "## 2. Does a higher class really mean more recurrent flooding?", "",
      "| risk class | labelled cells | recurrent | observed rate | lift vs prevalence |", "|---|---|---|---|---|"]
for k, r in cls.iterrows():
    R.append(f"| {k} | {int(r.n)} | {int(r.recurrent)} | {r.rate:.2f} | {r.lift:.2f}x |")
R += ["", "Budget view: if only the top share of labelled cells could be inspected,", "",
      "| share flagged | cells | precision | recall | lift |", "|---|---|---|---|---|"]
for fr, k, pr, rc, lf in budget_rows:
    R.append(f"| {fr:.0%} | {k} | {pr:.2f} | {rc:.2f} | {lf:.2f}x |")
R += ["", "![Enrichment](figures/enrichment.png)", "",
      "## 3. Monotonic sanity check on the cells the model never trained on",
      "", "Cells flooded in exactly one event were excluded from training. A sensible score should place them between never-flooded and recurrent cells.", "",
      "| events recorded | cells | mean score | median score |", "|---|---|---|---|"]
for k, r in mono.iterrows():
    R.append(f"| {int(k)} | {int(r['size'])} | {r['mean']:.2f} | {r['median']:.2f} |")
R += ["", f"AUC of single-event cells vs never-flooded cells: {amb_vs_neg:.2f}. AUC of recurrent vs single-event cells: {pos_vs_amb:.2f}. "
      "(Labelled cells use out-of-fold scores; single-event cells use the final model, so this is a consistency check, not an extra validation.)", "",
      "## 4. Spatial autocorrelation (Moran's I, 8 nearest neighbours, 999 permutations)", "",
      "| variable | Moran's I | permutation p |", "|---|---|---|",
      f"| recurrent-flood label | {I_label:.2f} | {p_label:.3f} |",
      f"| model residual (label minus out-of-fold score) | {I_resid:.2f} | {p_resid:.3f} |",
      f"| model score | {I_score:.2f} | {p_score:.3f} |",
      f"| mean elevation | {I_elev:.2f} | {p_elev:.3f} |", "",
      "A strongly positive I for the label confirms neighbouring cells are not independent, which is why random CV leaks and spatial block CV is the headline. "
      + ("Residuals still show positive autocorrelation, so the model misses spatially structured causes (drainage, drain capacity, local engineering)."
         if I_resid > 0.1 and p_resid < 0.05 else "Residual autocorrelation is small, so little spatial structure is left unexplained."), "",
      "## 5. Agreement between the three flood records", "",
      "Phi coefficient (correlation of the binary footprints):", "", phi.round(2).to_markdown() if hasattr(phi, "to_markdown") else phi.round(2).to_string(), "",
      "Jaccard overlap:", "", jac.round(2).to_markdown() if hasattr(jac, "to_markdown") else jac.round(2).to_string(), "",
      "Low or negative agreement means the records see different things (satellite water extent vs reported stagnation vs hotspot list), which limits what any model can learn.", "",
      "## 6. Feature statistics (land cells)", "", f"Missing values: {int((miss > 0).sum())} of {len(feat_cols)} features have any missing cell (max {miss.max():.1%}).", ""]
R.append(desc.round(2).to_markdown() if hasattr(desc, "to_markdown") else desc.round(2).to_string())
(ROOT / "reports/scientific_analysis.md").write_text("\n".join(R), encoding="utf-8")
print("\n".join(R))

# ---------- machine-readable metrics for the dashboard ----------
import json
import re

loeo = []
mr = (ROOT / "reports/model_report.md").read_text(encoding="utf-8")
for m in re.finditer(r"^\| (20\d\d) \| (Elevation only \(LR\)|XGBoost) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", mr, re.M):
    loeo.append({"event": m.group(1), "model": m.group(2), "prevalence": float(m.group(3)),
                 "roc_auc": float(m.group(4)), "pr_auc": float(m.group(5))})
metrics = {
    "labelled_cells": int(len(L)), "recurrent_cells": int(y.sum()), "never_cells": int((1 - y).sum()),
    "ambiguous_cells": int((s.n_ev == 1).sum()), "land_cells": int(len(s)), "blocks": len(keys), "prevalence": round(float(prev), 3),
    "roc_auc": {"xgb": round(auc(y, p), 3), "xgb_ci": [round(v, 3) for v in ci(boot["XGBoost (out-of-fold)"]["auc"])],
                "elev": round(auc(y, base_rule), 3), "elev_ci": [round(v, 3) for v in ci(boot["Elevation rule (lower = riskier)"]["auc"])],
                "gain": round(float(diff.mean()), 3), "gain_ci": [round(v, 3) for v in ci(diff)]},
    "pr_auc": {"xgb": round(ap(y, p), 3), "elev": round(ap(y, base_rule), 3)},
    "class_rates": [{"risk_class": int(k), "n": int(r.n), "recurrent": int(r.recurrent), "rate": round(float(r.rate), 3)} for k, r in cls.iterrows()],
    "budget": [{"share": fr, "precision": round(float(pr), 3), "recall": round(float(rc), 3), "lift": round(float(lf), 2)} for fr, k, pr, rc, lf in budget_rows],
    "mean_score_by_events": {str(int(k)): round(float(r["mean"]), 3) for k, r in mono.iterrows()},
    "moran": {"label": round(I_label, 3), "residual": round(I_resid, 3)},
    "phi_2005_2015": round(float(phi.loc["2005", "2015"]), 2),
    "loeo": loeo,
}
out = ROOT / "backend/data/metrics.json"
out.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
print(f"\nWrote {out.relative_to(ROOT)}")
