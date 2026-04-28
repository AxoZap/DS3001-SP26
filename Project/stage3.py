# Assisted By Codex
import argparse
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parent
STAGE2_DIR = SCRIPTS_ROOT
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = SCRIPTS_ROOT / "stage3_outputs"
ArcFaceExtractor = None
VGG19Extractor = None
GALLERY_DIR = None
PROBE_DIR = None
SUPPORTED_EXTS = None
identity_sort_key = None
normalize_identity_name = None

# Optional path overrides for the rubric script.
# Leave these as None to use the default folders from stage2.py.
# Example:
# GALLERY_DIR_OVERRIDE = r"C:\path\to\Gallery Set"
# PROBE_DIR_OVERRIDE = r"C:\path\to\Probe Set"
GALLERY_DIR_OVERRIDE = None
PROBE_DIR_OVERRIDE = None


def load_stage2_dependencies():
    global ArcFaceExtractor
    global VGG19Extractor
    global GALLERY_DIR
    global PROBE_DIR
    global SUPPORTED_EXTS
    global identity_sort_key
    global normalize_identity_name

    if ArcFaceExtractor is not None:
        return

    try:
        from stage2 import (
            ArcFaceExtractor as _ArcFaceExtractor,
            GALLERY_DIR as _GALLERY_DIR,
            PROBE_DIR as _PROBE_DIR,
            SUPPORTED_EXTS as _SUPPORTED_EXTS,
            VGG19Extractor as _VGG19Extractor,
            identity_sort_key as _identity_sort_key,
            normalize_identity_name as _normalize_identity_name,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency while importing stage2.py. "
            "Install the Stage 2 requirements first "
            f"(current failure: {exc})."
        ) from exc

    ArcFaceExtractor = _ArcFaceExtractor
    VGG19Extractor = _VGG19Extractor
    GALLERY_DIR = (
        Path(GALLERY_DIR_OVERRIDE).expanduser().resolve()
        if GALLERY_DIR_OVERRIDE
        else _GALLERY_DIR
    )
    PROBE_DIR = (
        Path(PROBE_DIR_OVERRIDE).expanduser().resolve()
        if PROBE_DIR_OVERRIDE
        else _PROBE_DIR
    )
    SUPPORTED_EXTS = _SUPPORTED_EXTS
    identity_sort_key = _identity_sort_key
    normalize_identity_name = _normalize_identity_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 Part 1: condition/severity classification and supervised re-identification."
    )
    parser.add_argument(
        "--condition-feature-set",
        choices=["arcface", "vgg", "both"],
        default="both",
        help="Feature set used for condition/severity classification.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of the condition dataset reserved for held-out testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test splitting.",
    )
    parser.add_argument(
        "--reid-probe-test-size",
        type=float,
        default=0.40,
        help="Fraction of known probe images held out for VGG re-identification testing.",
    )
    return parser.parse_args()


def extract_condition_and_severity(image_path):
    parts = image_path.relative_to(PROBE_DIR).parts
    condition_group = parts[1] if len(parts) > 1 else "Unknown"
    stem = image_path.stem.lower()

    if condition_group == "Blur":
        condition = "blur"
        if "bb_" in stem:
            severity = "slight"
        elif "slight" in stem:
            severity = "moderate"
        elif "very" in stem:
            severity = "severe"
        else:
            severity = "moderate"
    elif condition_group == "Face Covering":
        condition = "face_covering"
        if "uncovered" in stem:
            severity = "slight"
        elif "very" in stem:
            severity = "severe"
        else:
            severity = "moderate"
    elif condition_group == "Facial Orientations":
        condition = "facial_orientation"
        if "very" in stem:
            severity = "severe"
        elif "tilted" in stem:
            severity = "moderate"
        elif "low" in stem or "slight" in stem:
            severity = "slight"
        else:
            severity = "moderate"
    else:
        condition = "unknown"
        severity = "unknown"

    return condition, severity


def collect_probe_condition_dataset():
    records = []
    for image_path in sorted(PROBE_DIR.rglob("*")):
        if image_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        identity = normalize_identity_name(image_path.parts[-3])
        condition, severity = extract_condition_and_severity(image_path)
        records.append(
            {
                "path": image_path,
                "identity": identity or "Unknown",
                "condition": condition,
                "severity": severity,
                "joint_label": f"{condition}__{severity}",
            }
        )
    return records


def collect_identity_datasets():
    gallery_records = []
    for folder in sorted(GALLERY_DIR.iterdir()):
        if not folder.is_dir():
            continue
        identity = normalize_identity_name(folder.name)
        if identity is None:
            continue
        for image_path in sorted(folder.rglob("*")):
            if image_path.suffix.lower() in SUPPORTED_EXTS:
                gallery_records.append({"path": image_path, "identity": identity})

    probe_records = []
    for image_path in sorted(PROBE_DIR.rglob("*")):
        if image_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        identity = normalize_identity_name(image_path.parts[-3])
        if identity is None:
            continue
        if identity == "Identity10":
            identity = "UnknownProbe"
        condition, severity = extract_condition_and_severity(image_path)
        probe_records.append(
            {
                "path": image_path,
                "identity": identity,
                "condition": condition,
                "severity": severity,
            }
        )

    return gallery_records, probe_records


def extract_feature_table(records, extractor, feature_name):
    valid_rows = []
    for record in records:
        embedding = extractor.extract(str(record["path"]))
        if embedding is None:
            continue
        enriched = dict(record)
        enriched[feature_name] = embedding.astype(np.float32)
        valid_rows.append(enriched)

    print(
        f"[{feature_name}] extracted {len(valid_rows)}/{len(records)} usable embeddings."
    )
    return valid_rows


def merge_feature_tables(primary_rows, secondary_rows, primary_name, secondary_name):
    secondary_map = {str(row["path"]): row for row in secondary_rows}
    merged = []
    dropped = 0

    for row in primary_rows:
        match = secondary_map.get(str(row["path"]))
        if match is None:
            dropped += 1
            continue
        combined = dict(row)
        combined[secondary_name] = match[secondary_name]
        merged.append(combined)

    if dropped:
        print(
            f"[merge] dropped {dropped} samples that were missing {secondary_name} features."
        )
    print(f"[merge] paired {len(merged)} samples across {primary_name} and {secondary_name}.")
    return merged


def stack_features(rows, feature_columns):
    return np.vstack(
        [np.concatenate([row[col] for col in feature_columns]) for row in rows]
    )


def train_multiclass_logreg(X_train, y_train, random_state):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    ).fit(X_train, y_train)


def sanitize_label(text):
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def summarize_failure_cases(rows, y_true, y_pred, title, limit=10):
    mismatches = []
    for row, true_label, pred_label in zip(rows, y_true, y_pred):
        if true_label == pred_label:
            continue
        mismatches.append(
            {
                "path": str(row["path"]),
                "identity": row.get("identity", "Unknown"),
                "condition": row.get("condition", "unknown"),
                "severity": row.get("severity", "unknown"),
                "true": str(true_label),
                "pred": str(pred_label),
            }
        )

    print(f"\n[{title}] Failure cases: {len(mismatches)}")
    if not mismatches:
        print("No misclassifications on this split.")
        return

    pair_counts = Counter((item["true"], item["pred"]) for item in mismatches)
    print("Most common true -> predicted confusions:")
    for (true_label, pred_label), count in pair_counts.most_common(min(5, len(pair_counts))):
        print(f"- {true_label} -> {pred_label}: {count}")

    print("Example misclassified files:")
    for item in mismatches[:limit]:
        print(
            f"- true={item['true']} pred={item['pred']} "
            f"identity={item['identity']} "
            f"condition={item['condition']} severity={item['severity']} "
            f"path={item['path']}"
        )


def plot_confusion(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_condition_models(rows, feature_columns, feature_tag, args):
    joint_labels = [row["joint_label"] for row in rows]

    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=joint_labels,
    )

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    X_train = stack_features(train_rows, feature_columns)
    X_test = stack_features(test_rows, feature_columns)

    print("\n" + "=" * 60)
    print(f"CONDITION / SEVERITY CLASSIFICATION ({feature_tag.upper()})")
    print("=" * 60)
    print(
        f"Train size: {len(train_rows)} | Test size: {len(test_rows)} | "
        f"Joint label distribution (train): {dict(sorted(Counter(r['joint_label'] for r in train_rows).items()))}"
    )

    label_specs = [
        ("condition", "Condition"),
        ("severity", "Severity"),
        ("joint_label", "Condition+Severity"),
    ]

    for target_key, pretty_name in label_specs:
        y_train = [row[target_key] for row in train_rows]
        y_test = [row[target_key] for row in test_rows]

        model = train_multiclass_logreg(X_train, y_train, args.random_state)
        train_pred = model.predict(X_train)
        pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        acc = accuracy_score(y_test, pred)

        ordered_labels = sorted(set(y_train) | set(y_test))
        print(f"\n[{pretty_name}] Train Accuracy: {train_acc * 100:.2f}%")
        print(f"[{pretty_name}] Test Accuracy : {acc * 100:.2f}%")
        print(
            f"[{pretty_name}] Train-Test Gap: {(train_acc - acc) * 100:.2f} percentage points"
        )
        print(classification_report(y_test, pred, digits=3, zero_division=0))
        summarize_failure_cases(test_rows, y_test, pred, f"{feature_tag.upper()} {pretty_name}")

        output_path = OUTPUT_DIR / (
            f"{sanitize_label(feature_tag)}_{sanitize_label(target_key)}_confusion.png"
        )
        plot_confusion(
            y_true=y_test,
            y_pred=pred,
            labels=ordered_labels,
            title=f"{feature_tag.upper()} - {pretty_name} Classification",
            output_path=output_path,
        )


def known_probe_records(probe_records):
    return [row for row in probe_records if row["identity"] != "UnknownProbe"]


def unknown_probe_records(probe_records):
    return [row for row in probe_records if row["identity"] == "UnknownProbe"]


def evaluate_vgg_reidentification(gallery_rows, probe_rows, random_state, probe_test_size):
    known_probe = known_probe_records(probe_rows)
    unknown_probe = unknown_probe_records(probe_rows)

    probe_joint_labels = [f"{row['identity']}__{row['condition']}__{row['severity']}" for row in known_probe]
    train_probe_rows, test_probe_rows = train_test_split(
        known_probe,
        test_size=probe_test_size,
        random_state=random_state,
        stratify=probe_joint_labels,
    )

    train_rows = list(gallery_rows) + list(train_probe_rows)
    X_train = np.vstack([row["vgg"] for row in train_rows])
    y_train = np.array([row["identity"] for row in train_rows])

    X_test = np.vstack([row["vgg"] for row in test_probe_rows])
    y_test = np.array([row["identity"] for row in test_probe_rows])

    baseline = Pipeline([("scaler", StandardScaler()), ("clf", NearestCentroid())])
    baseline.fit(X_train, y_train)
    baseline_train_pred = baseline.predict(X_train)
    baseline_pred = baseline.predict(X_test)
    baseline_train_acc = accuracy_score(y_train, baseline_train_pred)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    supervised = train_multiclass_logreg(X_train, y_train, random_state)
    supervised_train_pred = supervised.predict(X_train)
    supervised_pred = supervised.predict(X_test)
    supervised_train_acc = accuracy_score(y_train, supervised_train_pred)
    supervised_acc = accuracy_score(y_test, supervised_pred)

    print("\n" + "=" * 60)
    print("VGG RE-IDENTIFICATION")
    print("=" * 60)
    print(f"Gallery train images        : {len(gallery_rows)}")
    print(f"Known probe train images    : {len(train_probe_rows)}")
    print(f"Held-out probe test images  : {len(test_probe_rows)}")
    print(
        "Unknown probe images "
        "(probe Identity10 relabeled as UnknownProbe and excluded from ID accuracy): "
        f"{len(unknown_probe)}"
    )
    print(f"Baseline nearest-centroid train accuracy: {baseline_train_acc * 100:.2f}%")
    print(f"Baseline nearest-centroid test accuracy : {baseline_acc * 100:.2f}%")
    print(
        "Baseline train-test gap                : "
        f"{(baseline_train_acc - baseline_acc) * 100:.2f} percentage points"
    )
    print(f"Supervised logistic-regression train accuracy: {supervised_train_acc * 100:.2f}%")
    print(f"Supervised logistic-regression test accuracy : {supervised_acc * 100:.2f}%")
    print(
        "Supervised train-test gap                : "
        f"{(supervised_train_acc - supervised_acc) * 100:.2f} percentage points"
    )
    print("\n[Supervised VGG Identity Classification Report]")
    ordered_identities = sorted(set(y_test) | set(supervised_pred), key=identity_sort_key)
    print(classification_report(y_test, supervised_pred, digits=3, zero_division=0))
    summarize_failure_cases(
        test_probe_rows,
        y_test,
        supervised_pred,
        "VGG Re-Identification",
    )

    plot_confusion(
        y_true=y_test,
        y_pred=baseline_pred,
        labels=ordered_identities,
        title="VGG Baseline Re-Identification (Nearest Centroid)",
        output_path=OUTPUT_DIR / "vgg_reid_baseline_confusion.png",
    )
    plot_confusion(
        y_true=y_test,
        y_pred=supervised_pred,
        labels=ordered_identities,
        title="VGG Supervised Re-Identification (Logistic Regression)",
        output_path=OUTPUT_DIR / "vgg_reid_supervised_confusion.png",
    )

    if unknown_probe:
        X_unknown = np.vstack([row["vgg"] for row in unknown_probe])
        unknown_probs = supervised.predict_proba(X_unknown)
        max_probs = unknown_probs.max(axis=1)
        print(
            "Unknown probe confidence summary "
            f"(max class probability): min={max_probs.min():.3f}, "
            f"median={np.median(max_probs):.3f}, max={max_probs.max():.3f}"
        )


def main():
    args = parse_args()
    load_stage2_dependencies()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nCollecting Stage 3 Part 1 datasets...")
    print(f"Gallery directory: {GALLERY_DIR}")
    print(f"Probe directory  : {PROBE_DIR}")
    condition_records = collect_probe_condition_dataset()
    gallery_records, probe_records = collect_identity_datasets()

    print(f"Condition dataset size: {len(condition_records)}")
    print(
        "Joint condition/severity distribution: "
        f"{dict(sorted(Counter(r['joint_label'] for r in condition_records).items()))}"
    )
    print(f"Gallery identity dataset size: {len(gallery_records)}")
    print(f"Probe identity dataset size: {len(probe_records)}")

    need_arcface = args.condition_feature_set in {"arcface", "both"}
    need_vgg_for_conditions = args.condition_feature_set in {"vgg", "both"}

    arcface_condition_rows = None
    vgg_condition_rows = None

    if need_arcface:
        print("\nExtracting ArcFace features for condition classification...")
        arcface_condition_rows = extract_feature_table(
            condition_records, ArcFaceExtractor(), "arcface"
        )

    if need_vgg_for_conditions:
        print("\nExtracting VGG features for condition classification...")
        vgg_condition_rows = extract_feature_table(
            condition_records, VGG19Extractor(), "vgg"
        )

    if args.condition_feature_set == "arcface":
        evaluate_condition_models(
            arcface_condition_rows,
            ["arcface"],
            "arcface",
            args,
        )
    elif args.condition_feature_set == "vgg":
        evaluate_condition_models(
            vgg_condition_rows,
            ["vgg"],
            "vgg",
            args,
        )
    else:
        paired_condition_rows = merge_feature_tables(
            primary_rows=arcface_condition_rows,
            secondary_rows=vgg_condition_rows,
            primary_name="arcface",
            secondary_name="vgg",
        )
        evaluate_condition_models(
            paired_condition_rows,
            ["arcface", "vgg"],
            "arcface_plus_vgg",
            args,
        )

    print("\nExtracting VGG features for supervised re-identification...")
    gallery_vgg_rows = extract_feature_table(gallery_records, VGG19Extractor(), "vgg")
    probe_vgg_rows = extract_feature_table(probe_records, VGG19Extractor(), "vgg")
    evaluate_vgg_reidentification(
        gallery_rows=gallery_vgg_rows,
        probe_rows=probe_vgg_rows,
        random_state=args.random_state,
        probe_test_size=args.reid_probe_test_size,
    )


if __name__ == "__main__":
    main()
