"""Step 7 - classification metrics at a fixed decision threshold (accuracy, precision, recall, F1) for every model.

Same data, label, 5 km spatial blocks and 5-fold GroupKFold as 04_train_models.py, so the numbers are comparable.
  * Logistic regression, random forest and the elevation-only baseline are refitted here (same settings as step 4).
  * XGBoost uses the out-of-fold scores that step 4 already wrote to data/processed/predictions.csv for labelled cells,
    so this script needs no xgboost install and cannot drift from the headline model.
Decision rule: predict "recurrent" when score >= 0.5. All models are trained with balanced class weights, so 0.5 is the
natural boundary and nothing is tuned on the evaluation folds. A second table repeats the metrics for a "flag the top
20% of cells" rule, the practical screening use.

Outputs: reports/threshold_metrics.md, reports/figures/confusion_matrix_xgb.png
Run: python scripts/07_threshold_metrics.py   (after scripts/04_train_models.py)
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEED = yaml.safe_load(open(ROOT / "configs/config.yaml"))["seed"]
BLOCK_KM, N_FOLDS = 5, 5

df = pd.read_csv(ROOT / "data/processed/features.csv")
s = df[df.in_study == 1].reset_index(drop=True)
s["e2015"] = ((s.pos_2015 == 1) | (s.pos_stag_2015 == 1)).astype(int)
s["n_events"] = s.pos_2005 + s.e2015 + s.pos_hot_2020
FEATS = ["elev_mean", "elev_min", "elev_std", "slope_mean", "rel_elev", "dist_sea_m", "dist_inland_water_m"] + \
        [f"frac_{n}" for n in ("tree", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "mangrove")]
FEATS = [c for c in FEATS if c in s.columns and s[c].nunique(dropna=True) > 1]
dy = BLOCK_KM / 111.0
dx = BLOCK_KM / (111.0 * np.cos(np.radians(s.lat.mean())))
s["block"] = (((s.lon - s.lon.min()) // dx).astype(int).astype(str) + "_" + ((s.lat - s.lat.min()) // dy).astype(int).astype(str))
L = s[(s.n_events >= 2) | (s.n_events == 0)].reset_index(drop=True)
y = (L.n_events >= 2).astype(int).values
spatial = list(GroupKFold(n_splits=N_FOLDS).split(L, y, L.block.values))


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED))


def oof(cols, make):
    p = np.zeros(len(L))
    for tr, te in spatial:
        p[te] = make().fit(L.loc[tr, cols], y[tr]).predict_proba(L.loc[te, cols])[:, 1]
    return p


scores = {"Elevation only (LR)": oof(["elev_mean"], lr), "Logistic Regression": oof(FEATS, lr), "Random Forest": oof(FEATS, rf)}
pred = pd.read_csv(ROOT / "data/processed/predictions.csv").set_index("cell_id")
xgb = pred.loc[L.cell_id, "score"].values if "cell_id" in L.columns else None
assert xgb is not None and len(xgb) == len(L), "predictions.csv must contain the labelled cells"
scores["XGBoost"] = xgb
order = ["Elevation only (LR)", "Logistic Regression", "Random Forest", "XGBoost"]


def row(name, yhat, p):
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"model": name, "accuracy": accuracy_score(y, yhat), "balanced_acc": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "mcc": matthews_corrcoef(y, yhat), "roc_auc": roc_auc_score(y, p),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


A = pd.DataFrame([row(m, (scores[m] >= 0.5).astype(int), scores[m]) for m in order])
k = int(round(0.20 * len(L)))
B = pd.DataFrame([row(m, (np.argsort(np.argsort(-scores[m])) < k).astype(int), scores[m]) for m in order])
majority = max(y.mean(), 1 - y.mean())

f = lambda v: f"{v:.2f}"
R = ["# Classification metrics at a fixed threshold", "",
     f"Labelled cells: {len(L)} ({int(y.sum())} recurrent, {int((1 - y).sum())} never recorded). 5-fold spatial block CV on {BLOCK_KM} km blocks. "
     f"Accuracy of always predicting the majority class: **{majority:.2f}**. ROC-AUC is threshold-free and is repeated for reference.", "",
     "## A. Decision rule: predict recurrent when score >= 0.5", "",
     "| model | accuracy | balanced acc. | precision | recall | F1 | MCC | ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
R += [f"| {r.model} | {f(r.accuracy)} | {f(r.balanced_acc)} | {f(r.precision)} | {f(r.recall)} | {f(r.f1)} | {f(r.mcc)} | {f(r.roc_auc)} |" for r in A.itertuples()]
R += ["", "Confusion matrices (rows = truth; counts of cells):", "", "| model | TN | FP | FN | TP |", "|---|---|---|---|---|"]
R += [f"| {r.model} | {r.tn} | {r.fp} | {r.fn} | {r.tp} |" for r in A.itertuples()]
R += ["", "## B. Screening rule: flag the top 20% of cells by score", "",
      f"Flags {k} of {len(L)} labelled cells.", "",
      "| model | accuracy | precision | recall | F1 | MCC |", "|---|---|---|---|---|---|"]
R += [f"| {r.model} | {f(r.accuracy)} | {f(r.precision)} | {f(r.recall)} | {f(r.f1)} | {f(r.mcc)} |" for r in B.itertuples()]
R += ["", "## Reading these numbers", "",
      "- Accuracy alone is misleading on imbalanced classes; compare it with the majority-class baseline and read balanced accuracy, F1 and MCC next to it.",
      "- The threshold is fixed in advance (0.5, or a fixed top-20% budget); nothing was tuned on the evaluation folds.",
      "- Labels are noisy (some never-recorded cells really flooded), so precision is probably understated and these figures are conservative for the true task.",
      "- Spatial-CV numbers describe skill on unseen neighbourhoods of the same three events, not on a new storm (see the leave-one-event-out table in model_report.md)."]
(ROOT / "reports/threshold_metrics.md").write_text("\n".join(R), encoding="utf-8")

cm = confusion_matrix(y, (scores["XGBoost"] >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=14)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["never", "recurrent"], yticklabels=["never", "recurrent"],
       xlabel="predicted", ylabel="actual", title="XGBoost, spatial CV, threshold 0.5")
plt.tight_layout(); (ROOT / "reports/figures").mkdir(exist_ok=True, parents=True); plt.savefig(ROOT / "reports/figures/confusion_matrix_xgb.png", dpi=150); plt.close()
print("\n".join(R))
