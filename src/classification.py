#!/usr/bin/env python3

import os
import pickle
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


TARGET_SIZE = (200, 200)
MAX_SAMPLES_PER_CLASS = None
TEST_SIZE = 0.2
RANDOM_STATE = 42

WORD_DIR = "word_pdfs_png"
GOOGLE_DIR = "google_docs_pdfs_png"
PYTHON_DIR = "python_pdfs_png"
AUGMENTED_ROOT = "augmented_images"

CONDITIONS = [
    "original",
    "gaussian_noise",
    "jpeg_compression",
    "downsample_dpi",
    "random_crop",
    "reduce_bit_depth",
]

RESULTS_DIR = "results"
METRICS_CSV = os.path.join(RESULTS_DIR, "performance_metrics.csv")


def load_image_vector(img_path, target_size=TARGET_SIZE):
    img = Image.open(img_path).convert("L")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).flatten()


def load_dataset(word_dir=WORD_DIR, google_dir=GOOGLE_DIR, python_dir=PYTHON_DIR,
                 max_samples_per_class=None, target_size=TARGET_SIZE):
    X, y, records = [], [], []

    sources = [
        (word_dir,   0, "word"),
        (google_dir, 1, "google"),
        (python_dir, 2, "python"),
    ]

    for src_dir, label, src_name in sources:
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        print(f"loading {len(files)} {src_name} images...")
        for filename in files:
            try:
                vec = load_image_vector(os.path.join(src_dir, filename), target_size)
                X.append(vec)
                y.append(label)
                records.append((src_name, filename))
            except Exception as e:
                print(f"  skipping {filename}: {e}")

    X, y = np.array(X), np.array(y)
    print(f"total: {X.shape[0]} samples  word={np.sum(y==0)} google={np.sum(y==1)} python={np.sum(y==2)}")
    return X, y, records


def resolve_image_path(source_name, filename, condition):
    if condition == "original":
        dir_map = {"word": WORD_DIR, "google": GOOGLE_DIR, "python": PYTHON_DIR}
        return os.path.join(dir_map[source_name], filename)
    return os.path.join(AUGMENTED_ROOT, source_name, condition, filename)


def load_condition_dataset(records, condition, target_size=TARGET_SIZE):
    X, y = [], []
    label_map = {"word": 0, "google": 1, "python": 2}
    for source_name, filename in records:
        img_path = resolve_image_path(source_name, filename, condition)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"missing: {img_path}")
        X.append(load_image_vector(img_path, target_size))
        y.append(label_map[source_name])
    return np.array(X), np.array(y)


def train_svm(X_train, y_train):
    print("training SVM...")
    t = time.time()
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    elapsed = time.time() - t
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def train_sgd(X_train, y_train):
    print("training SGD...")
    t = time.time()
    model = SGDClassifier(loss="hinge", alpha=0.01, max_iter=1000, tol=1e-3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    elapsed = time.time() - t
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def train_rf(X_train, y_train):
    print("training Random Forest...")
    t = time.time()
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = time.time() - t
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def train_xgb(X_train, y_train):
    print("training XGBoost...")
    t = time.time()
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def evaluate(model, name, condition, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  {name} / {condition}: accuracy={acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Word", "Google", "Python"]))
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm, y_pred


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y, records = load_dataset(max_samples_per_class=MAX_SAMPLES_PER_CLASS)

    X_train, X_test, y_train, y_test, train_records, test_records = train_test_split(
        X, y, records, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"train={X_train.shape[0]}  test={X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    svm,  svm_t  = train_svm(X_train_sc, y_train)
    sgd,  sgd_t  = train_sgd(X_train_sc, y_train)
    rf,   rf_t   = train_rf(X_train_sc, y_train)
    xgb,  xgb_t  = train_xgb(X_train_sc, y_train)

    for name, model in [("svm", svm), ("sgd", sgd), ("rf", rf), ("xgb", xgb), ("scaler", scaler)]:
        with open(f"{name}_model.pkl" if name != "scaler" else "scaler.pkl", "wb") as f:
            pickle.dump(model, f)

    classifiers = [
        ("SVM",         svm,  svm_t),
        ("SGD",         sgd,  sgd_t),
        ("RandomForest", rf,  rf_t),
        ("XGBoost",     xgb,  xgb_t),
    ]

    results = []
    pred_rows = []

    for condition in CONDITIONS:
        if condition == "original":
            X_cond, y_cond = X_test_sc, y_test
        else:
            print(f"\nloading augmented test set: {condition}")
            X_raw, y_cond = load_condition_dataset(test_records, condition)
            X_cond = scaler.transform(X_raw)

        for clf_name, model, train_time in classifiers:
            acc, cm, y_pred = evaluate(model, clf_name, condition, X_cond, y_cond)

            results.append({
                "classifier":       clf_name,
                "condition":        condition,
                "train_time_sec":   round(train_time, 4),
                "accuracy":         round(acc, 4),
                "cm_00_word_as_word":    int(cm[0][0]),
                "cm_01_word_as_google":  int(cm[0][1]),
                "cm_02_word_as_python":  int(cm[0][2]),
                "cm_10_google_as_word":  int(cm[1][0]),
                "cm_11_google_as_google":int(cm[1][1]),
                "cm_12_google_as_python":int(cm[1][2]),
                "cm_20_python_as_word":  int(cm[2][0]),
                "cm_21_python_as_google":int(cm[2][1]),
                "cm_22_python_as_python":int(cm[2][2]),
            })

            for idx, (true, pred) in enumerate(zip(y_cond, y_pred)):
                pred_rows.append({
                    "classifier":  clf_name,
                    "condition":   condition,
                    "sample_idx":  idx,
                    "true_label":  int(true),
                    "pred_label":  int(pred),
                })

    pd.DataFrame(results).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    print(f"\nresults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
