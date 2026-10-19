"""
Model Evaluation and Diagnostic Metrics Module.
Calculates clinical screening metrics: Sensitivity (Recall), Specificity,
Precision, F1, ROC-AUC, PR-AUC, Brier score, and calibration curves.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import FIGURES_DIR, RANDOM_SEED


def evaluate_binary_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics with clinical sensitivity emphasis.
    """
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            dfunc = model.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-dfunc))
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion Matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0.0
        fn_rate = 0.0
        fp_rate = 0.0

    # ROC-AUC & PR-AUC
    roc_auc = None
    pr_auc = None
    brier = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
            prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(rec_arr, prec_arr)
            brier = brier_score_loss(y_test, y_prob)
        except Exception:
            pass

    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
        "brier_score": round(float(brier), 4) if brier is not None else None,
        "false_negative_rate": round(float(fn_rate), 4),
        "false_positive_rate": round(float(fp_rate), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }


def evaluate_multiclass_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "MulticlassModel"
) -> Dict[str, Any]:
    """
    Evaluates multi-class models (e.g. BRFSS 012 Prediabetes distinction).
    """
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    roc_auc_ovr = None
    if y_prob is not None:
        try:
            roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            pass

    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc_ovr": round(float(roc_auc_ovr), 4) if roc_auc_ovr is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }


def run_cross_validation(
    pipeline: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> Dict[str, Any]:
    """
    Executes Stratified K-Fold cross-validation and computes mean & std across folds.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=True)

    summary = {}
    for metric in scoring:
        test_scores = scores[f"test_{metric}"]
        train_scores = scores[f"train_{metric}"]
        summary[metric] = {
            "mean": round(float(np.mean(test_scores)), 4),
            "std": round(float(np.std(test_scores)), 4),
            "train_mean": round(float(np.mean(train_scores)), 4),
            "train_std": round(float(np.std(train_scores)), 4),
            "overfit_gap": round(float(np.mean(train_scores) - np.mean(test_scores)), 4),
        }
    return summary


def plot_model_diagnostic_charts(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, save_prefix: str
) -> None:
    """
    Generates ROC curve, PR curve, Calibration curve, and Confusion Matrix heatmap.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 1. Diagnostic 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_val = auc(fpr, tpr)
    axes[0, 0].plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC (AUC = {roc_val:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0, 0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0, 0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0, 0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0, 0].legend(loc="lower right")

    # Panel B: Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_val = auc(rec, prec)
    axes[0, 1].plot(rec, prec, color="#10B981", lw=2.5, label=f"PR (AUC = {pr_val:.3f})")
    baseline = np.mean(y_test)
    axes[0, 1].axhline(baseline, color="gray", linestyle="--", label=f"Baseline Prevalence ({baseline:.1%})")
    axes[0, 1].set_title("Precision-Recall (PR) Curve")
    axes[0, 1].set_xlabel("Recall (Sensitivity)")
    axes[0, 1].set_ylabel("Precision (Positive Predictive Value)")
    axes[0, 1].legend(loc="upper right")

    # Panel C: Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_test, y_prob)
    axes[1, 0].plot(prob_pred, prob_true, marker="s", color="#8B5CF6", lw=2, label=f"Model (Brier={brier:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    axes[1, 0].set_title("Probability Calibration Curve")
    axes[1, 0].set_xlabel("Mean Predicted Probability")
    axes[1, 0].set_ylabel("Observed Fraction of Positives")
    axes[1, 0].legend(loc="upper left")

    # Panel D: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    labels = [[f"TN: {tn:,}\n({tn/(tn+fp):.1%} Spec)", f"FP: {fp:,}\n(Type I Err)"],
              [f"FN: {fn:,}\n(Type II Err - Missed!)", f"TP: {tp:,}\n({tp/(tp+fn):.1%} Sens)"]]
    axes[1, 1].matshow(cm, cmap="Blues", alpha=0.3)
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, labels[i][j], ha="center", va="center", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(["Pred Negative", "Pred Positive"])
    axes[1, 1].set_yticklabels(["Actual Negative", "Actual Positive"])
    axes[1, 1].set_title("Clinical Confusion Matrix & Error Rates")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{save_prefix}_diagnostics.png")
    plt.close()
