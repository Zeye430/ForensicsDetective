#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2


RESULTS_DIR = "results"
METRICS_CSV       = os.path.join(RESULTS_DIR, "performance_metrics.csv")
PREDICTIONS_CSV   = os.path.join(RESULTS_DIR, "predictions.csv")
FULL_METRICS_CSV  = os.path.join(RESULTS_DIR, "full_metrics.csv")
SUMMARY_CSV       = os.path.join(RESULTS_DIR, "robustness_summary.csv")
SIGNIFICANCE_CSV  = os.path.join(RESULTS_DIR, "statistical_significance.csv")
ROBUSTNESS_PLOT_DIR = os.path.join(RESULTS_DIR, "robustness_plots")
CM_DIR              = os.path.join(RESULTS_DIR, "confusion_matrices")

CLASS_NAMES = ["Word", "Google", "Python"]

CONDITION_ORDER = [
    "original",
    "gaussian_noise",
    "jpeg_compression",
    "downsample_dpi",
    "random_crop",
    "reduce_bit_depth",
]

DISPLAY_NAMES = {
    "original":         "Original",
    "gaussian_noise":   "Gaussian Noise",
    "jpeg_compression": "JPEG Compression",
    "downsample_dpi":   "DPI Downsample",
    "random_crop":      "Random Crop",
    "reduce_bit_depth": "Bit-Depth Reduction",
}


def load_metrics():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"metrics file not found: {METRICS_CSV}")
    df = pd.read_csv(METRICS_CSV)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df.sort_values(["classifier", "condition"])


def compute_accuracy_drop(df):
    rows = []
    for clf in df["classifier"].unique():
        clf_df = df[df["classifier"] == clf]
        baseline = clf_df[clf_df["condition"] == "original"]["accuracy"].iloc[0]
        for _, row in clf_df.iterrows():
            rows.append({
                "classifier":        clf,
                "condition":         row["condition"],
                "display_condition": DISPLAY_NAMES[row["condition"]],
                "accuracy":          row["accuracy"],
                "baseline_accuracy": baseline,
                "accuracy_drop":     round(baseline - row["accuracy"], 4),
            })
    return pd.DataFrame(rows)


def compute_prf1(df):
    rows = []
    for _, row in df.iterrows():
        cm = np.array([
            [row["cm_00_word_as_word"],   row["cm_01_word_as_google"],   row["cm_02_word_as_python"]],
            [row["cm_10_google_as_word"], row["cm_11_google_as_google"], row["cm_12_google_as_python"]],
            [row["cm_20_python_as_word"], row["cm_21_python_as_google"], row["cm_22_python_as_python"]],
        ])
        ps, rs, fs = [], [], []
        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            ps.append(round(p, 4)); rs.append(round(r, 4)); fs.append(round(f, 4))

        rows.append({
            "classifier": row["classifier"],
            "condition":  row["condition"],
            "accuracy":   row["accuracy"],
            "precision_word":   ps[0], "recall_word":   rs[0], "f1_word":   fs[0],
            "precision_google": ps[1], "recall_google": rs[1], "f1_google": fs[1],
            "precision_python": ps[2], "recall_python": rs[2], "f1_python": fs[2],
            "macro_precision":  round(float(np.mean(ps)), 4),
            "macro_recall":     round(float(np.mean(rs)), 4),
            "macro_f1":         round(float(np.mean(fs)), 4),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(FULL_METRICS_CSV, index=False)
    print(f"full metrics saved to {FULL_METRICS_CSV}")
    return result_df


def plot_robustness_curves(summary_df):
    plt.figure(figsize=(10, 6))
    for clf in summary_df["classifier"].unique():
        sub = summary_df[summary_df["classifier"] == clf].sort_values("condition")
        plt.plot(sub["display_condition"], sub["accuracy"], marker="o", linewidth=2, label=clf)

    plt.title("Classifier Accuracy Across Augmentation Conditions")
    plt.xlabel("Test Condition")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()

    out = os.path.join(ROBUSTNESS_PLOT_DIR, "robustness_curves.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"saved {out}")


def plot_accuracy_drop(summary_df):
    plt.figure(figsize=(10, 6))
    plot_df = summary_df[summary_df["condition"] != "original"]
    sns.barplot(data=plot_df, x="display_condition", y="accuracy_drop", hue="classifier")
    plt.title("Accuracy Drop per Augmentation Type")
    plt.xlabel("Augmentation")
    plt.ylabel("Drop from Baseline")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()

    out = os.path.join(ROBUSTNESS_PLOT_DIR, "accuracy_drop_by_augmentation.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"saved {out}")


def generate_confusion_matrix_plots(df):
    for _, row in df.iterrows():
        cm = np.array([
            [row["cm_00_word_as_word"],   row["cm_01_word_as_google"],   row["cm_02_word_as_python"]],
            [row["cm_10_google_as_word"], row["cm_11_google_as_google"], row["cm_12_google_as_python"]],
            [row["cm_20_python_as_word"], row["cm_21_python_as_google"], row["cm_22_python_as_python"]],
        ])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{row['classifier']} — {DISPLAY_NAMES.get(row['condition'], row['condition'])}")
        plt.tight_layout()
        fig.savefig(os.path.join(CM_DIR, f"{row['classifier']}_{row['condition']}.png"), dpi=150)
        plt.close(fig)
    print(f"confusion matrices saved to {CM_DIR}/")


def _mcnemar_p(correct_a, correct_b):
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    if n01 + n10 == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(chi2.sf(stat, df=1))


def run_statistical_tests():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"predictions.csv not found, skipping significance tests")
        return None

    pred_df = pd.read_csv(PREDICTIONS_CSV)
    classifiers = pred_df["classifier"].unique()
    conditions  = pred_df["condition"].unique()

    rows = []
    for condition in conditions:
        cond_df = pred_df[pred_df["condition"] == condition]
        for i, clf_a in enumerate(classifiers):
            for clf_b in classifiers[i + 1:]:
                da = cond_df[cond_df["classifier"] == clf_a].sort_values("sample_idx")
                db = cond_df[cond_df["classifier"] == clf_b].sort_values("sample_idx")
                if len(da) != len(db):
                    continue
                ca = da["true_label"].values == da["pred_label"].values
                cb = db["true_label"].values == db["pred_label"].values
                p = _mcnemar_p(ca, cb)
                rows.append({
                    "condition":          condition,
                    "classifier_a":       clf_a,
                    "classifier_b":       clf_b,
                    "p_value":            round(p, 6),
                    "significant_at_0.05": p < 0.05,
                })

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(SIGNIFICANCE_CSV, index=False)
    print(f"McNemar results saved to {SIGNIFICANCE_CSV}")
    print(sig_df.to_string(index=False))
    return sig_df


def save_largest_degradation(summary_df):
    rows = []
    for clf in summary_df["classifier"].unique():
        sub = summary_df[(summary_df["classifier"] == clf) & (summary_df["condition"] != "original")]
        if sub.empty:
            continue
        worst = sub.sort_values("accuracy_drop", ascending=False).iloc[0]
        rows.append({
            "classifier":       clf,
            "worst_condition":  worst["display_condition"],
            "accuracy_drop":    worst["accuracy_drop"],
            "accuracy":         worst["accuracy"],
        })
    degradation_df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "largest_degradation.csv")
    degradation_df.to_csv(out, index=False)
    print(f"largest degradation saved to {out}")
    print(degradation_df.to_string(index=False))


def main():
    Path(ROBUSTNESS_PLOT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CM_DIR).mkdir(parents=True, exist_ok=True)

    df = load_metrics()

    full_df = compute_prf1(df)
    print(full_df[["classifier", "condition", "accuracy", "macro_precision", "macro_recall", "macro_f1"]])

    generate_confusion_matrix_plots(df)

    run_statistical_tests()

    summary_df = compute_accuracy_drop(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    plot_robustness_curves(summary_df)
    plot_accuracy_drop(summary_df)

    save_largest_degradation(summary_df)

    print("\ndone.")


if __name__ == "__main__":
    main()
