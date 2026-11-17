"""Step 5 - recurrent flood-proneness model with spatial cross-validation.

Label (per land cell, from the 2005, 2015 and 2020 flood records; 2015 counts once whether it
comes from the satellite extent or the stagnation points):
    n_events >= 2  -> positive  (flooded in at least two events = recurrent)
    n_events == 0  -> negative  (never seen flooded - may include unseen floods)
    n_events == 1  -> ambiguous (excluded from training/evaluation, still scored)

Evaluation
  * spatial block CV (5 km blocks held out together)  <- headline
  * random CV, shown only to demonstrate how much it overstates skill
  * leave-one-event-out stress test (train on other events' labels, test on the held-out event)

Outputs: reports/model_report.md, reports/figures/*.png, data/processed/predictions.csv,
         models/xgb_susceptibility.json

Run: python scripts/04_train_models.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
SEED = CFG["seed"]
BLOCK_KM, N_FOLDS = 5, 5

# ---------------- data + label ----------------
df = pd.read_csv(ROOT / "data/processed/features.csv")
s = df[df.in_study == 1].reset_index(drop=True)
s["e2015"] = ((s.pos_2015 == 1) | (s.pos_stag_2015 == 1)).astype(int)
s["n_events"] = s.pos_2005 + s.e2015 + s.pos_hot_2020
EVENTS = {"2005": s.pos_2005.values, "2015": s.e2015.values, "2020": s.pos_hot_2020.values}

FEATS = ["elev_mean", "elev_min", "elev_std", "slope_mean", "rel_elev", "dist_sea_m",
         "dist_inland_water_m"] + [f"frac_{n}" for n in
                                   ("tree", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "mangrove")]
# NOTE: explicit list on purpose - frac_2005 / frac_2015 are LABEL columns and must never be features.
FEATS = [c for c in FEATS if c in s.columns and s[c].nunique(dropna=True) > 1]
assert not any(c in FEATS for c in ("frac_2005", "frac_2015", "n_stag_2015", "n_hot_2020")), "label leakage"

dy = BLOCK_KM / 111.0
dx = BLOCK_KM / (111.0 * np.cos(np.radians(s.lat.mean())))
s["block"] = (((s.lon - s.lon.min()) // dx).astype(int).astype(str) + "_" +
              ((s.lat - s.lat.min()) // dy).astype(int).astype(str))

lab_mask = (s.n_events >= 2) | (s.n_events == 0)
L = s[lab_mask].reset_index(drop=True)
y = (L.n_events >= 2).astype(int).values
groups = L.block.values
print(f"land cells {len(s)}, labelled {len(L)} (pos {y.sum()}, neg {(1 - y).sum()}), "
      f"ambiguous {len(s) - len(L)}, blocks {L.block.nunique()}, features {len(FEATS)}")


# ---------------- models ----------------
def _lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def _rf(_y):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=SEED))


def _xgb(yy):
    pos = max(int(np.sum(yy)), 1)
    return XGBClassifier(objective="binary:logistic", n_estimators=300, learning_rate=0.03,
                         max_depth=3, min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=(len(yy) - pos) / pos, eval_metric="logloss",
                         random_state=SEED, n_jobs=2)


MODELS = {
    "Elevation only (LR)": (["elev_mean"], lambda yy: _lr()),
    "Logistic Regression": (FEATS, lambda yy: _lr()),
    "Random Forest": (FEATS, _rf),
    "XGBoost": (FEATS, _xgb),
}


def ece(yt, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(yt[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(yt, p):
    k = int(yt.sum())
    top = np.argsort(-p)[:k]
    return {"roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
            "brier": brier_score_loss(yt, p), "ece": ece(yt, p), "p_at_k": float(yt[top].mean())}


def oof(name, splits):
    cols, make = MODELS[name]
    X, p = L[cols], np.zeros(len(L))
    for tr, te in splits:
        m = make(y[tr]).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


spatial = list(GroupKFold(n_splits=N_FOLDS).split(L, y, groups))
randomk = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(L))

rows, oof_spatial = [], {}
for name in MODELS:
    ps, pr = oof(name, spatial), oof(name, randomk)
    oof_spatial[name] = ps
    ms, mr = metrics(y, ps), metrics(y, pr)
    fold_auc = [roc_auc_score(y[te], ps[te]) for _, te in spatial if 0 < y[te].sum() < len(te)]
    rows.append({"model": name, **{f"sp_{k}": v for k, v in ms.items()},
                 "sp_auc_fold_mean": np.mean(fold_auc), "sp_auc_fold_sd": np.std(fold_auc),
                 "rand_roc_auc": mr["roc_auc"], "rand_pr_auc": mr["pr_auc"]})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

# ---------------- leave-one-event-out stress test (inside spatial folds) ----------------
all_idx = np.arange(len(s))
s_groups = s.block.values
loeo_rows = []
for held, yt_ev in EVENTS.items():
    train_lab = np.max([v for k, v in EVENTS.items() if k != held], axis=0)   # flooded in any other event
    for name in ["Elevation only (LR)", "XGBoost"]:
        cols, make = MODELS[name]
        p = np.zeros(len(s))
        for tr, te in GroupKFold(n_splits=N_FOLDS).split(s, train_lab, s_groups):
            m = make(train_lab[tr]).fit(s.iloc[tr][cols], train_lab[tr])
            p[te] = m.predict_proba(s.iloc[te][cols])[:, 1]
        loeo_rows.append({"held_out": held, "model": name, "prevalence": yt_ev.mean(),
                          "roc_auc": roc_auc_score(yt_ev, p), "pr_auc": average_precision_score(yt_ev, p)})
loeo = pd.DataFrame(loeo_rows)
print(loeo.round(3).to_string(index=False))

# ---------------- final model, predictions, SHAP ----------------
cols, make = MODELS["XGBoost"]
final = make(y).fit(L[cols], y)
final.save_model(str(ROOT / "models/xgb_susceptibility.json"))
s["score"] = final.predict_proba(s[cols])[:, 1]                 # model scored every land cell
oof_series = pd.Series(oof_spatial["XGBoost"], index=L.index)
lab_rows = s.index[lab_mask.values]
s.loc[lab_rows, "score"] = oof_series.values                    # labelled cells: out-of-fold score
s["recurrent_label"] = np.where(s.n_events >= 2, 1, np.where(s.n_events == 0, 0, -1))
s["risk_class"] = pd.qcut(s.score.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
(ROOT / "data/processed").mkdir(exist_ok=True)
s.drop(columns=["block"]).to_csv(ROOT / "data/processed/predictions.csv", index=False)

fig_dir = ROOT / "reports/figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 8), sharex=True, sharey=True)
sc = ax[0].scatter(s.lon, s.lat, c=s.score, s=14, marker="s", cmap="YlOrRd", vmin=0, vmax=1)
ax[0].set_title("Model susceptibility score\n(out-of-sample)")
plt.colorbar(sc, ax=ax[0], shrink=0.6, label="score (relative, not a probability)")
colour = s.recurrent_label.map({1: "#c0392b", 0: "#9ec5e8", -1: "#dddddd"})
ax[1].scatter(s.lon, s.lat, c=colour, s=14, marker="s")
ax[1].set_title("Recurrent-flood label\nred = flooded in 2+ events, blue = never, grey = once")
for a in ax:
    a.set_aspect("equal"); a.set_xlabel("lon"); a.set_ylabel("lat")
plt.tight_layout(); plt.savefig(fig_dir / "susceptibility_map.png", dpi=130); plt.close()

shap_note = ""
try:
    import shap
    sv = shap.TreeExplainer(final)(L[cols])
    shap.summary_plot(sv.values, L[cols], show=False, max_display=12)
    plt.tight_layout(); plt.savefig(fig_dir / "shap_summary.png", dpi=130); plt.close()
    imp = pd.Series(np.abs(sv.values).mean(0), index=cols).sort_values(ascending=False)
    shap_note = "\n## Mean |SHAP| (top 8)\n\n" + "\n".join(f"- {k}: {v:.3f}" for k, v in imp.head(8).items())
except Exception as e:                                           # SHAP must never kill the run
    shap_note = f"\n_SHAP skipped: {e!r}_"

# ---------------- report ----------------
f = lambda v: f"{v:.2f}"
R = ["# Model report", "",
     f"Land cells: {len(s):,}. Training/eval cells: **{len(L):,}** ({int(y.sum())} recurrent, {int((1 - y).sum())} never-flooded). "
     f"Ambiguous (flooded in exactly 1 event): {len(s) - len(L):,}, scored but not used for training.",
     f"Spatial CV = {N_FOLDS} folds over {BLOCK_KM} km blocks. Prevalence (chance PR-AUC) = {y.mean():.2f}.", "",
     "## Headline: spatial cross-validation", "",
     "| model | ROC-AUC | PR-AUC | P@k | Brier | ECE | fold AUC (mean ± sd) |", "|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    R.append(f"| {r.model} | {f(r.sp_roc_auc)} | {f(r.sp_pr_auc)} | {f(r.sp_p_at_k)} | {f(r.sp_brier)} | {f(r.sp_ece)} | "
             f"{f(r.sp_auc_fold_mean)} ± {f(r.sp_auc_fold_sd)} |")
R += ["", "## Random CV vs spatial CV (leakage check)", "", "| model | random ROC-AUC | spatial ROC-AUC |", "|---|---|---|"]
for _, r in res.iterrows():
    R.append(f"| {r.model} | {f(r.rand_roc_auc)} | {f(r.sp_roc_auc)} |")
R += ["", "A big gap means random CV is leaking neighbouring cells; trust the spatial numbers.", "",
      "## Stress test: leave-one-event-out", "",
      "Train on cells flooded in any OTHER event, test on the held-out event (spatial folds, so no neighbour leakage).", "",
      "| held-out event | model | prevalence | ROC-AUC | PR-AUC |", "|---|---|---|---|---|"]
for _, r in loeo.iterrows():
    R.append(f"| {r.held_out} | {r.model} | {f(r.prevalence)} | {f(r.roc_auc)} | {f(r.pr_auc)} |")
R += [shap_note, "", "## Read this before quoting numbers", "",
      "- The score ranks cells by recurrent flood-proneness. It is NOT a flood probability and NOT a warning.",
      "- Negatives mean 'not seen flooded' in three imperfect records, so some are really positives.",
      "- ROC-AUC near 0.5 = no skill. Report spatial-CV numbers, and the leave-one-event-out table as honest context.",
      "- Only terrain and land-cover features; no rainfall, drainage network or drain capacity.",
      "- Cells were labelled from 3 events only; treat results as a prototype."]
(ROOT / "reports/model_report.md").write_text("\n".join(R), encoding="utf-8")
print("\nWrote reports/model_report.md, reports/figures/*.png, data/processed/predictions.csv")
