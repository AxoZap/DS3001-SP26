import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train supervised identity classifiers from labeled reference vectors."
    )
    parser.add_argument(
        "--model",
        choices=["arcface", "vgg", "vgg1000", "both", "fusion", "all", "ensemble"],
        default="both",
        help="Which feature space(s) to train on.",
    )
    parser.add_argument(
        "--reference-data",
        default=None,
        help="Optional path to one .npz file created by build_reference_vectors.py. "
        "Only valid for non-fusion single-model training.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .pkl output path. Defaults to trained_models/identity_model_<model>[_<classifier>].pkl",
    )
    parser.add_argument(
        "--classifier",
        choices=["auto", "logreg", "knn"],
        default="auto",
        help="Classifier head to train. Logistic regression matches the rubric's suggested VGG fine-tune; "
        "cosine k-NN is often stronger for re-identification. "
        "Use 'auto' to choose the best head from gallery leave-one-out accuracy.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Neighbor count when --classifier knn is used.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Held-out test fraction when enough samples exist per class.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test splitting.",
    )
    parser.add_argument(
        "--keep-identity10",
        action="store_true",
        help="Keep Identity10 in supervised training. By default it is excluded because probe Identity10 is treated as UnknownProbe.",
    )
    parser.add_argument(
        "--ensemble-weights",
        default="0.4,0.2,0.2,0.2",
        help="Comma-separated weights for arcface_knn,fusion_knn,vgg_logreg,arcface_logreg when --model ensemble is used.",
    )
    return parser.parse_args()


def load_reference_dataset(path):
    data = np.load(Path(path).expanduser().resolve(), allow_pickle=True)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    labels = np.asarray(data["labels"]).astype(str)
    image_paths = np.asarray(data["image_paths"]).astype(str)
    model_name = str(np.asarray(data["model"]).item())
    return vectors, labels, image_paths, model_name


def default_output_path(model_name, classifier_name):
    out_dir = SCRIPTS_ROOT / "trained_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if classifier_name == "logreg" else f"_{classifier_name}"
    return out_dir / f"identity_model_{model_name}{suffix}.pkl"


def build_classifier(args):
    if args.classifier == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=args.random_state,
                    ),
                ),
            ]
        )

    return Pipeline(
        [
            ("normalizer", Normalizer()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=max(1, args.knn_k),
                    metric="cosine",
                    weights="distance",
                ),
            ),
        ]
    )


def build_named_classifier(name, random_state=42):
    if name == "logreg":
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
        )

    if not name.startswith("knn"):
        raise ValueError(f"Unsupported classifier name: {name}")

    k = int(name.replace("knn", ""))
    return Pipeline(
        [
            ("normalizer", Normalizer()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=max(1, k),
                    metric="cosine",
                    weights="distance",
                ),
            ),
        ]
    )


def candidate_classifier_names(model_name):
    if model_name == "arcface":
        return ["logreg", "knn1", "knn3", "knn5", "knn9"]
    if model_name == "vgg1000":
        return ["knn1", "logreg", "knn3", "knn5"]
    return ["logreg", "knn1", "knn3", "knn5"]


def choose_classifier_name(model_name, X, y, requested_classifier, random_state, knn_k):
    if requested_classifier != "auto":
        if requested_classifier == "knn":
            return f"knn{max(1, knn_k)}"
        return requested_classifier

    if len(set(y)) < 2 or min(Counter(y).values()) < 2:
        defaults = {"arcface": "logreg", "vgg1000": "knn1", "vgg": "logreg", "fusion": "knn5"}
        return defaults.get(model_name, "logreg")

    loo = LeaveOneOut()
    best_name = None
    best_score = -1.0
    for name in candidate_classifier_names(model_name):
        clf = build_named_classifier(name, random_state=random_state)
        score = float(cross_val_score(clf, X, y, cv=loo).mean())
        print(f"LOO {name:<6} accuracy: {score * 100:.2f}%")
        if score > best_score:
            best_score = score
            best_name = name

    return best_name or "logreg"


def build_centroid_baseline(X_train, y_train):
    baseline = Pipeline([("scaler", StandardScaler()), ("clf", NearestCentroid())])
    baseline.fit(X_train, y_train)
    return baseline


def enough_for_split(labels, test_size):
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return False

    n_samples = len(labels)
    if isinstance(test_size, float):
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)

    return n_test >= len(counts)


def maybe_filter_identity10(vectors, labels, image_paths, keep_identity10):
    if keep_identity10:
        return vectors, labels, image_paths

    mask = labels != "Identity10"
    return vectors[mask], labels[mask], image_paths[mask]


def load_single_spec(model_name, keep_identity10):
    dataset_path = SCRIPTS_ROOT / "trained_models" / f"reference_{model_name}.npz"
    vectors, labels, image_paths, loaded_model_name = load_reference_dataset(dataset_path)
    vectors, labels, image_paths = maybe_filter_identity10(
        vectors, labels, image_paths, keep_identity10
    )
    return {
        "model_name": loaded_model_name,
        "vectors": vectors,
        "labels": labels,
        "image_paths": image_paths,
        "feature_spaces": [loaded_model_name],
        "vector_dims": {loaded_model_name: vectors.shape[1]},
    }


def load_fusion_spec(keep_identity10):
    arc_vectors, arc_labels, arc_paths, _ = load_reference_dataset(
        SCRIPTS_ROOT / "trained_models" / "reference_arcface.npz"
    )
    vgg_vectors, vgg_labels, vgg_paths, _ = load_reference_dataset(
        SCRIPTS_ROOT / "trained_models" / "reference_vgg.npz"
    )

    arc_map = {
        path: (label, vector)
        for path, label, vector in zip(arc_paths, arc_labels, arc_vectors)
    }
    vgg_map = {
        path: (label, vector)
        for path, label, vector in zip(vgg_paths, vgg_labels, vgg_vectors)
    }

    shared_paths = sorted(set(arc_map) & set(vgg_map))
    fused_vectors = []
    fused_labels = []
    fused_paths = []
    for path in shared_paths:
        arc_label, arc_vector = arc_map[path]
        vgg_label, vgg_vector = vgg_map[path]
        if arc_label != vgg_label:
            continue
        fused_vectors.append(np.concatenate([arc_vector, vgg_vector]).astype(np.float32))
        fused_labels.append(str(arc_label))
        fused_paths.append(path)

    vectors = np.vstack(fused_vectors).astype(np.float32)
    labels = np.asarray(fused_labels).astype(str)
    image_paths = np.asarray(fused_paths).astype(str)
    vectors, labels, image_paths = maybe_filter_identity10(
        vectors, labels, image_paths, keep_identity10
    )

    return {
        "model_name": "fusion",
        "vectors": vectors,
        "labels": labels,
        "image_paths": image_paths,
        "feature_spaces": ["arcface", "vgg"],
        "vector_dims": {"arcface": 512, "vgg": 4096},
    }


def resolve_specs(args):
    if args.reference_data:
        if args.model in {"fusion", "all", "both", "ensemble"}:
            raise SystemExit(
                "--reference-data can only be used with a single non-fusion model."
            )
        dataset_path = Path(args.reference_data).expanduser().resolve()
        vectors, labels, image_paths, model_name = load_reference_dataset(dataset_path)
        vectors, labels, image_paths = maybe_filter_identity10(
            vectors, labels, image_paths, args.keep_identity10
        )
        return [
            {
                "model_name": model_name,
                "vectors": vectors,
                "labels": labels,
                "image_paths": image_paths,
                "feature_spaces": [model_name],
                "vector_dims": {model_name: vectors.shape[1]},
            }
        ]

    if args.model == "both":
        return [
            load_single_spec("arcface", args.keep_identity10),
            load_single_spec("vgg", args.keep_identity10),
            load_single_spec("vgg1000", args.keep_identity10),
        ]
    if args.model == "all":
        return [
            load_single_spec("arcface", args.keep_identity10),
            load_single_spec("vgg", args.keep_identity10),
            load_single_spec("vgg1000", args.keep_identity10),
            load_fusion_spec(args.keep_identity10),
        ]
    if args.model == "ensemble":
        return [{"model_name": "ensemble"}]
    if args.model == "fusion":
        return [load_fusion_spec(args.keep_identity10)]
    return [load_single_spec(args.model, args.keep_identity10)]


def parse_ensemble_weights(raw_text):
    try:
        values = [float(part.strip()) for part in raw_text.split(",")]
    except ValueError as exc:
        raise SystemExit(
            "--ensemble-weights must be four comma-separated numbers like 0.4,0.2,0.2,0.2"
        ) from exc

    if len(values) != 4:
        raise SystemExit(
            "--ensemble-weights must contain exactly four values for arcface_knn,fusion_knn,vgg_logreg,arcface_logreg"
        )
    if any(value < 0 for value in values):
        raise SystemExit("--ensemble-weights cannot contain negative values.")

    total = sum(values)
    if total <= 0:
        raise SystemExit("--ensemble-weights must sum to a positive value.")

    return {
        "arcface_knn": values[0] / total,
        "fusion_knn": values[1] / total,
        "vgg_logreg": values[2] / total,
        "arcface_logreg": values[3] / total,
    }


def train_ensemble_artifact(args, output_path):
    weights = parse_ensemble_weights(args.ensemble_weights)
    component_paths = {
        "arcface_knn": SCRIPTS_ROOT / "trained_models" / "identity_model_arcface_knn.pkl",
        "fusion_knn": SCRIPTS_ROOT / "trained_models" / "identity_model_fusion_knn.pkl",
        "vgg_logreg": SCRIPTS_ROOT / "trained_models" / "identity_model_vgg.pkl",
        "arcface_logreg": SCRIPTS_ROOT / "trained_models" / "identity_model_arcface.pkl",
    }

    components = {}
    for name, path in component_paths.items():
        if not path.exists():
            raise SystemExit(
                f"Missing component model for ensemble: {path}. "
                "Train arcface knn, fusion knn, and vgg logreg first."
            )
        with path.open("rb") as f:
            components[name] = pickle.load(f)

    class_lists = [tuple(component["classifier"].classes_) for component in components.values()]
    if len(set(class_lists)) != 1:
        raise SystemExit("Ensemble components do not share the same class labels.")

    artifact = {
        "model_name": "ensemble",
        "classifier_name": "weighted_ensemble",
        "feature_spaces": ["arcface_knn", "fusion_knn", "vgg_logreg", "arcface_logreg"],
        "class_labels": list(class_lists[0]),
        "component_model_paths": {name: str(path) for name, path in component_paths.items()},
        "ensemble_weights": weights,
        "excluded_identity10": True,
        "reference_vectors": np.empty((0, 0), dtype=np.float32),
        "reference_labels": np.asarray([], dtype=str),
        "reference_image_paths": np.asarray([], dtype=str),
    }

    with output_path.open("wb") as f:
        pickle.dump(artifact, f)

    print("\nModel          : ensemble")
    print("Classifier     : weighted_ensemble")
    print(f"Weights        : {weights}")
    print("Excluded Identity10: True")
    print(f"Saved trained model: {output_path}")


def main():
    args = parse_args()
    specs = resolve_specs(args)

    for spec in specs:
        if spec["model_name"] == "ensemble":
            output_path = (
                Path(args.output).expanduser().resolve()
                if args.output and len(specs) == 1
                else SCRIPTS_ROOT / "trained_models" / "identity_model_ensemble.pkl"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            train_ensemble_artifact(args, output_path)
            continue

        X = spec["vectors"]
        y = spec["labels"]
        model_name = spec["model_name"]
        selected_classifier_name = choose_classifier_name(
            model_name,
            X,
            y,
            args.classifier,
            args.random_state,
            args.knn_k,
        )
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output and len(specs) == 1
            else default_output_path(
                model_name,
                selected_classifier_name if args.classifier != "auto" else "logreg",
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model_name": model_name,
            "classifier_name": selected_classifier_name,
            "feature_spaces": spec["feature_spaces"],
            "vector_dims": spec["vector_dims"],
            "class_labels": sorted(set(y)),
            "reference_vectors": X,
            "reference_labels": y,
            "reference_image_paths": spec["image_paths"],
            "excluded_identity10": not args.keep_identity10,
        }

        print(f"\nModel          : {model_name}")
        print(f"Classifier     : {selected_classifier_name}")
        print(f"Samples        : {len(y)}")
        print(f"Identity counts: {dict(sorted(Counter(y).items()))}")
        print(f"Excluded Identity10: {not args.keep_identity10}")

        if enough_for_split(y, args.test_size):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=y,
            )

            baseline = build_centroid_baseline(X_train, y_train)
            baseline_pred = baseline.predict(X_test)
            baseline_acc = accuracy_score(y_test, baseline_pred)

            if selected_classifier_name.startswith("knn"):
                selected_k = int(selected_classifier_name.replace("knn", ""))
                temp_args = argparse.Namespace(
                    classifier="knn",
                    knn_k=selected_k,
                    random_state=args.random_state,
                )
            else:
                temp_args = argparse.Namespace(
                    classifier="logreg",
                    knn_k=args.knn_k,
                    random_state=args.random_state,
                )
            classifier = build_classifier(temp_args)
            classifier.fit(X_train, y_train)
            pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, pred)

            print(f"Baseline nearest-centroid accuracy : {baseline_acc * 100:.2f}%")
            print(f"Supervised {selected_classifier_name} accuracy : {acc * 100:.2f}%")
            print("\n[Held-out Classification Report]")
            print(classification_report(y_test, pred, digits=3))

            artifact["baseline_accuracy"] = baseline_acc
            artifact["heldout_accuracy"] = acc
        else:
            print(
                "Not enough samples for a stratified train/test split with the current "
                "number of identities and test_size. "
                "Training on all reference vectors without a held-out evaluation."
            )
            X_train, y_train = X, y

        if selected_classifier_name.startswith("knn"):
            selected_k = int(selected_classifier_name.replace("knn", ""))
            final_args = argparse.Namespace(
                classifier="knn",
                knn_k=selected_k,
                random_state=args.random_state,
            )
        else:
            final_args = argparse.Namespace(
                classifier="logreg",
                knn_k=args.knn_k,
                random_state=args.random_state,
            )
        classifier = build_classifier(final_args)
        classifier.fit(X_train, y_train)
        artifact["classifier"] = classifier

        scaler = StandardScaler().fit(X)
        scaled_vectors = scaler.transform(X)
        centroids = {}
        for label in sorted(set(y)):
            centroids[label] = scaled_vectors[y == label].mean(axis=0)
        artifact["centroid_scaler"] = scaler
        artifact["centroids"] = centroids

        with output_path.open("wb") as f:
            pickle.dump(artifact, f)

        print(f"\nSaved trained model: {output_path}")


if __name__ == "__main__":
    main()
