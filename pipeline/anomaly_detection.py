"""
anomaly_detection.py
----------------------
Two anomaly detectors, both scored against the SAME test set for
a fair comparison:

  1. Statistical baseline: simple combined Z-score threshold
     (a naive rule-based approach - what most legacy systems do)

  2. ML model: Isolation Forest trained on each user's deviation
     features (login-hour z-score, duration z-score, new-device flag,
     session-frequency z-score, off-hours flag)

Train/test split: baseline period (first half of each user's history)
is used for "normal" training data; later period is the test set,
which is where all injected malicious insider behavior lives.

Outputs:
  - data/predictions.csv          (per-session scores + labels)
  - outputs/evaluation_report.txt
  - outputs/roc_curve.png
  - outputs/score_distribution.png
  - outputs/confusion_matrix.png
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_fscore_support,
                              confusion_matrix, average_precision_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

feat_df = pd.read_csv("../data/session_features.csv", parse_dates=["date"])

FEATURES = ["login_hour_zscore", "duration_zscore", "is_new_device",
            "sessions_today_zscore", "is_off_hours"]

train_df = feat_df[feat_df["is_baseline_period"] == 1].copy()
test_df = feat_df[feat_df["is_baseline_period"] == 0].copy()

print(f"Train (baseline/normal) rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}  |  Malicious in test: {test_df['is_malicious_gt'].sum()}")

# ============================================================
# 1. STATISTICAL BASELINE  (naive combined z-score rule)
# ============================================================
# Flag as anomalous if ANY deviation feature crosses a hard threshold.
def statistical_score(row):
    score = 0
    score += abs(row["login_hour_zscore"])
    score += abs(row["duration_zscore"])
    score += 3 * row["is_new_device"]        # weight device change heavily
    score += abs(row["sessions_today_zscore"])
    score += 2 * row["is_off_hours"]
    return score

test_df["stat_score"] = test_df.apply(statistical_score, axis=1)
stat_threshold = train_df.apply(statistical_score, axis=1).quantile(0.97)
test_df["stat_pred"] = (test_df["stat_score"] > stat_threshold).astype(int)

# ============================================================
# 2. ISOLATION FOREST (unsupervised ML model)
# ============================================================
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,     # assume ~5% of test sessions might be anomalous
    random_state=42,
    max_samples="auto",
)
iso.fit(train_df[FEATURES])

# decision_function: higher = more normal. We flip sign so higher = more anomalous.
test_df["iso_raw_score"] = -iso.decision_function(test_df[FEATURES])
test_df["iso_pred"] = (iso.predict(test_df[FEATURES]) == -1).astype(int)

# normalize iso score to 0-100 risk score for interpretability
min_s, max_s = test_df["iso_raw_score"].min(), test_df["iso_raw_score"].max()
test_df["risk_score"] = ((test_df["iso_raw_score"] - min_s) / (max_s - min_s) * 100).round(1)

# ============================================================
# 3. EVALUATION
# ============================================================
y_true = test_df["is_malicious_gt"]

def evaluate(y_true, y_pred, y_score, name):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    return {"name": name, "precision": p, "recall": r, "f1": f1, "roc_auc": auc, "avg_precision": ap, "cm": cm}

stat_results = evaluate(y_true, test_df["stat_pred"], test_df["stat_score"], "Statistical Z-score Baseline")
iso_results = evaluate(y_true, test_df["iso_pred"], test_df["iso_raw_score"], "Isolation Forest")

report_lines = []
report_lines.append("="*70)
report_lines.append("INSIDER THREAT DETECTION - MODEL EVALUATION REPORT")
report_lines.append("="*70)
report_lines.append(f"\nTest set size: {len(test_df)} sessions | Malicious (ground truth): {y_true.sum()}")
report_lines.append(f"Malicious users: {sorted(test_df[test_df.is_malicious_gt==1]['user'].unique())}\n")

for res in [stat_results, iso_results]:
    report_lines.append("-"*70)
    report_lines.append(f"MODEL: {res['name']}")
    report_lines.append("-"*70)
    report_lines.append(f"  Precision      : {res['precision']:.3f}")
    report_lines.append(f"  Recall         : {res['recall']:.3f}")
    report_lines.append(f"  F1-score       : {res['f1']:.3f}")
    report_lines.append(f"  ROC-AUC        : {res['roc_auc']:.3f}")
    report_lines.append(f"  Avg Precision  : {res['avg_precision']:.3f}")
    tn, fp, fn, tp = res["cm"].ravel()
    report_lines.append(f"  Confusion Matrix -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    report_lines.append(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    report_lines.append("")

improvement = ((iso_results["f1"] - stat_results["f1"]) / max(stat_results["f1"], 1e-9)) * 100
report_lines.append("="*70)
report_lines.append(f"Isolation Forest F1 improvement over statistical baseline: {improvement:+.1f}%")
report_lines.append("="*70)

report_text = "\n".join(report_lines)
print(report_text)

with open("../results/evaluation_report.txt", "w") as f:
    f.write(report_text)

# ============================================================
# 4. PLOTS
# ============================================================

# ROC curve comparison
plt.figure(figsize=(6,5))
for res, score_col, style in [(stat_results, "stat_score", "--"), (iso_results, "iso_raw_score", "-")]:
    fpr, tpr, _ = roc_curve(y_true, test_df[score_col])
    plt.plot(fpr, tpr, style, label=f"{res['name']} (AUC={res['roc_auc']:.3f})", linewidth=2)
plt.plot([0,1],[0,1],':', color='gray', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Insider Threat Detection Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../results/roc_curve.png", dpi=150)
plt.close()

# Risk score distribution: normal vs malicious
plt.figure(figsize=(7,5))
plt.hist(test_df[test_df.is_malicious_gt==0]["risk_score"], bins=30, alpha=0.6, label="Normal sessions", color="#4C72B0")
plt.hist(test_df[test_df.is_malicious_gt==1]["risk_score"], bins=30, alpha=0.7, label="Malicious insider sessions", color="#C44E52")
plt.xlabel("Isolation Forest Risk Score (0-100)")
plt.ylabel("Number of sessions")
plt.title("Risk Score Distribution: Normal vs Malicious Sessions")
plt.legend()
plt.tight_layout()
plt.savefig("../results/score_distribution.png", dpi=150)
plt.close()

# Confusion matrix heatmap for Isolation Forest
plt.figure(figsize=(4.5,4))
cm = iso_results["cm"]
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                  color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
plt.xticks([0,1], ["Normal","Malicious"])
plt.yticks([0,1], ["Normal","Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Isolation Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png", dpi=150)
plt.close()

test_df.to_csv("../data/predictions.csv", index=False)
print("\nSaved: predictions.csv, evaluation_report.txt, roc_curve.png, score_distribution.png, confusion_matrix.png")
