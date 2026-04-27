import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict identities for unknown feature vectors using a saved trained model."
    )
    parser.add_argument(
        "--model",
        choices=["arcface", "vgg", "vgg1000", "both", "fusion", "all", "ensemble"],
        default="both",
        help="Which saved identity model(s) to use.",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="Optional path to one saved .pkl model created by train_identity_model.py",
    )
    parser.add_argument(
        "--unknown",
        help="Optional path to one .npy vector or a folder of .npy vectors. If omitted, all vectors in input_vectors are used.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(SCRIPTS_ROOT / "input_vectors"),
        help="Folder of .npy vectors to process when --unknown is omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many ranked predictions to print for each unknown vector.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="If filenames begin with a label like Identity5__..., print only aggregate accuracy counts instead of per-file predictions.",
    )
    parser.add_argument(
        "--per-identity",
        action="store_true",
        help="When used with --summary-only, also print per-identity accuracy breakdowns.",
    )
    parser.add_argument(
        "--prefer-centroids",
        action="store_true",
        help="Use cosine similarity to enrolled reference centroids as the primary ranking. "
        "This is usually better when you only have one enrollment image per identity.",
    )
    return parser.parse_args()


def load_artifact(path):
    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Trained model file not found: {model_path}")
    with model_path.open("rb") as f:
        return pickle.load(f), model_path


def resolve_unknown_source(args):
    if args.unknown:
        return Path(args.unknown).expanduser().resolve()
    return Path(args.input_dir).expanduser().resolve()


def infer_vector_type(name):
    lowered = name.lower()
    if lowered.endswith("_arcface_vector"):
        return "arcface"
    if lowered.endswith("_vgg1000_vector"):
        return "vgg1000"
    if lowered.endswith("_vgg_vector"):
        return "vgg"
    return None


def strip_vector_suffix(name):
    if name.lower().endswith("_arcface_vector"):
        return name[:-len("_arcface_vector")]
    if name.lower().endswith("_vgg1000_vector"):
        return name[:-len("_vgg1000_vector")]
    if name.lower().endswith("_vgg_vector"):
        return name[:-len("_vgg_vector")]
    return name


def normalize_loaded_vector(raw):
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    if isinstance(raw, dict):
        if "features" in raw:
            raw = raw["features"]
        else:
            raise SystemExit(
                "Unsupported vector dictionary format. Expected a 'features' key."
            )

    vector = np.asarray(raw, dtype=np.float32).reshape(-1)
    return vector


def load_unknown_vectors(source_path):
    if not source_path.exists():
        raise SystemExit(f"Unknown vector path not found: {source_path}")

    vector_items = []
    if source_path.is_file():
        vector_items.append((source_path.stem, normalize_loaded_vector(np.load(source_path, allow_pickle=True))))
    else:
        for file_path in sorted(source_path.glob("*.npy")):
            vector_items.append((file_path.stem, normalize_loaded_vector(np.load(file_path, allow_pickle=True))))

    if not vector_items:
        raise SystemExit(f"No .npy vectors found in: {source_path}")

    cleaned = []
    for name, vector in vector_items:
        cleaned.append(
            {
                "name": name,
                "base_name": strip_vector_suffix(name),
                "vector": vector,
                "vector_type": infer_vector_type(name),
            }
        )
    return cleaned


def cosine_similarity(a, b):
    a_norm = a / max(np.linalg.norm(a), 1e-12)
    b_norm = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(a_norm, b_norm))


def summarize_confidence(scored_items):
    if not scored_items:
        return "low", 0.0
    top1 = float(scored_items[0][1])
    top2 = float(scored_items[1][1]) if len(scored_items) > 1 else 0.0
    gap = top1 - top2

    # Use a simple gap-based rule so we can compare both cosine scores and probabilities.
    if gap >= 0.20:
        return "high", gap
    if gap >= 0.08:
        return "medium", gap
    return "low", gap


def probability_confidence(scored_items):
    return summarize_confidence(scored_items)


def reference_confidence(scored_items, model_name):
    if not scored_items:
        return "low", 0.0, 0.0

    top1 = float(scored_items[0][1])
    top2 = float(scored_items[1][1]) if len(scored_items) > 1 else 0.0
    gap = top1 - top2

    if model_name == "arcface":
        if top1 >= 0.18 and gap >= 0.06:
            return "high", gap, top1
        if top1 >= 0.12 and gap >= 0.03:
            return "medium", gap, top1
        return "low", gap, top1

    if top1 >= 0.55 and gap >= 0.18:
        return "high", gap, top1
    if top1 >= 0.35 and gap >= 0.08:
        return "medium", gap, top1
    return "low", gap, top1


def confidence_rank(level):
    return {"low": 0, "medium": 1, "high": 2}[level]


def rank_to_confidence(rank):
    return {0: "low", 1: "medium", 2: "high"}[rank]


def combined_confidence(model_name, classifier_scored, reference_scored):
    clf_level, clf_gap = probability_confidence(classifier_scored)
    ref_level, ref_gap, ref_top1 = reference_confidence(reference_scored, model_name)

    combined_rank = min(confidence_rank(clf_level), confidence_rank(ref_level))
    if classifier_scored and reference_scored:
        if classifier_scored[0][0] != reference_scored[0][0]:
            combined_rank = max(0, combined_rank - 1)

    return (
        rank_to_confidence(combined_rank),
        clf_gap,
        ref_gap,
        ref_top1,
        clf_level,
        ref_level,
    )


def choose_final_guess(classifier_scored, reference_scored, confidence, clf_level, ref_level):
    if not classifier_scored and not reference_scored:
        return None
    if not classifier_scored:
        return reference_scored[0][0]
    if not reference_scored:
        return classifier_scored[0][0]

    classifier_label = classifier_scored[0][0]
    reference_label = reference_scored[0][0]

    # If the classifier is shaky, trust the direct reference match more.
    if confidence == "low":
        return reference_label

    # When reference evidence is clearly stronger than the classifier signal,
    # prefer the enrolled identity nearest to the query embedding.
    if ref_level == "high" and clf_level != "high":
        return reference_label

    return classifier_label


def centroid_ranking(vector, artifact):
    scaler = artifact["centroid_scaler"]
    centroids = artifact["centroids"]
    scaled_vector = scaler.transform(vector.reshape(1, -1))[0]
    scored = []
    for label, centroid in centroids.items():
        scored.append((label, cosine_similarity(scaled_vector, centroid)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def reference_vote_ranking(vector, artifact):
    reference_vectors = np.asarray(artifact["reference_vectors"], dtype=np.float32)
    reference_labels = np.asarray(artifact["reference_labels"]).astype(str)
    if reference_vectors.size == 0:
        return []

    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    vec = vec / max(np.linalg.norm(vec), 1e-12)
    refs = reference_vectors / np.maximum(
        np.linalg.norm(reference_vectors, axis=1, keepdims=True), 1e-12
    )
    scores = refs @ vec

    per_label = defaultdict(list)
    for label, score in zip(reference_labels, scores):
        per_label[str(label)].append(float(score))

    ranked = []
    for label, label_scores in per_label.items():
        label_scores.sort(reverse=True)
        top_scores = label_scores[: min(2, len(label_scores))]
        aggregate = float(np.mean(top_scores))
        ranked.append((label, aggregate))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def best_single_reference_ranking(vector, artifact):
    reference_vectors = np.asarray(artifact["reference_vectors"], dtype=np.float32)
    reference_labels = np.asarray(artifact["reference_labels"]).astype(str)
    if reference_vectors.size == 0:
        return []

    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    vec = vec / max(np.linalg.norm(vec), 1e-12)
    refs = reference_vectors / np.maximum(
        np.linalg.norm(reference_vectors, axis=1, keepdims=True), 1e-12
    )
    scores = refs @ vec

    best_by_label = {}
    for label, score in zip(reference_labels, scores):
        label = str(label)
        best_by_label[label] = max(best_by_label.get(label, -1.0), float(score))

    ranked = sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)
    return ranked


def primary_reference_ranking(vector, artifact):
    if artifact.get("classifier_name") == "knn1" and artifact.get("model_name") in {"vgg", "vgg1000"}:
        ranked = best_single_reference_ranking(vector, artifact)
        if ranked:
            return ranked
    return reference_vote_ranking(vector, artifact)


def ranking_for_artifact(vector, artifact, args):
    if args.prefer_centroids:
        ranked = reference_vote_ranking(vector, artifact)
        if ranked:
            return ranked
    return centroid_ranking(vector, artifact)


def predict_proba_for_artifact(artifact, vector):
    classifier = artifact["classifier"]
    return classifier.predict_proba(vector.reshape(1, -1))[0]


def predict_label_for_artifact(artifact, vector):
    classifier = artifact["classifier"]
    return classifier.predict(vector.reshape(1, -1))[0]


def extract_expected_label(name):
    match = re.match(r"^(Identity_?\d+|UnknownProbe)__", name)
    if not match:
        return None
    return match.group(1).replace("Identity_", "Identity")


def select_single_feature_vectors(unknown_vectors, model_name, expected_dim):
    matching = []
    skipped_other_model = 0
    skipped_wrong_dim = 0

    for item in unknown_vectors:
        if item["vector_type"] is not None and item["vector_type"] != model_name:
            skipped_other_model += 1
            continue
        if item["vector"].shape[0] != expected_dim:
            skipped_wrong_dim += 1
            continue
        matching.append({"name": item["name"], "vector": item["vector"]})

    return matching, skipped_other_model, skipped_wrong_dim


def select_fusion_vectors(unknown_vectors, vector_dims):
    grouped = defaultdict(dict)
    skipped_wrong_dim = 0

    for item in unknown_vectors:
        vector_type = item["vector_type"]
        if vector_type not in {"arcface", "vgg"}:
            continue
        expected_dim = vector_dims.get(vector_type)
        if expected_dim is None or item["vector"].shape[0] != expected_dim:
            skipped_wrong_dim += 1
            continue
        grouped[item["base_name"]][vector_type] = item

    matching = []
    for base_name in sorted(grouped):
        parts = grouped[base_name]
        if "arcface" not in parts or "vgg" not in parts:
            continue
        matching.append(
            {
                "name": base_name,
                "vector": np.concatenate(
                    [parts["arcface"]["vector"], parts["vgg"]["vector"]]
                ).astype(np.float32),
            }
        )

    return matching, skipped_wrong_dim


def load_ensemble_components(artifact):
    loaded = {}
    for name, path_text in artifact["component_model_paths"].items():
        with Path(path_text).expanduser().resolve().open("rb") as f:
            loaded[name] = pickle.load(f)
    return loaded


def select_vectors_for_ensemble(unknown_vectors, components):
    arc_component = components.get("arcface_knn") or components.get("arcface")
    vgg_component = components.get("vgg_logreg") or components.get("vgg")

    fusion_vectors, skipped_wrong_dim = select_fusion_vectors(
        unknown_vectors,
        {"arcface": 512, "vgg": 4096},
    )

    arc_expected_dim = arc_component["vector_dims"]["arcface"]
    vgg_expected_dim = vgg_component["vector_dims"]["vgg"]
    arc_map = {}    
    vgg_map = {}
    arc_wrong_dim = 0
    vgg_wrong_dim = 0
    for item in unknown_vectors:
        if item["vector_type"] == "arcface":
            if item["vector"].shape[0] == arc_expected_dim:
                arc_map[item["base_name"]] = item["vector"]
            else:
                arc_wrong_dim += 1
        elif item["vector_type"] == "vgg":
            if item["vector"].shape[0] == vgg_expected_dim:
                vgg_map[item["base_name"]] = item["vector"]
            else:
                vgg_wrong_dim += 1

    fusion_map = {item["name"]: item["vector"] for item in fusion_vectors}

    shared_names = sorted(set(arc_map) & set(vgg_map) & set(fusion_map))
    matching = []
    for name in shared_names:
        matching.append(
            {
                "name": name,
                "vectors": {
                    "arcface": arc_map[name],
                    "fusion": fusion_map[name],
                    "vgg": vgg_map[name],
                },
            }
        )

    return matching, skipped_wrong_dim + arc_wrong_dim + vgg_wrong_dim


def ensemble_predict_proba(artifact, components, vectors):
    classes = artifact["class_labels"]
    weights = artifact["ensemble_weights"]
    combined = np.zeros(len(classes), dtype=np.float64)

    component_vectors = {
        "arcface_knn": vectors["arcface"],
        "fusion_knn": vectors["fusion"],
        "vgg_logreg": vectors["vgg"],
        "arcface_logreg": vectors["arcface"],
        "arcface": vectors["arcface"],
        "fusion": vectors["fusion"],
        "vgg": vectors["vgg"],
    }

    for component_name in ["arcface_knn", "fusion_knn", "vgg_logreg", "arcface_logreg"]:
        component = components.get(component_name)
        if component is None:
            continue
        probs = predict_proba_for_artifact(component, component_vectors[component_name])
        class_to_index = {label: idx for idx, label in enumerate(component["classifier"].classes_)}
        for idx, label in enumerate(classes):
            combined[idx] += weights.get(component_name, 0.0) * probs[class_to_index[label]]

    return combined


def print_summary_rows(matching_vectors, classifier, artifact, args):
    evaluated = 0
    correct = 0
    skipped_unknown = 0
    labeled_found = 0
    per_identity = defaultdict(lambda: {"correct": 0, "total": 0})
    summary_top_k = min(3, max(1, args.top_k))
    top_hits = {k: 0 for k in range(1, summary_top_k + 1)}

    for item in matching_vectors:
        name = item["name"]
        vector = item["vector"]
        expected_label = extract_expected_label(name)
        if expected_label is None:
            continue
        labeled_found += 1
        if expected_label == "UnknownProbe":
            skipped_unknown += 1
            continue

        if args.prefer_centroids:
            ranked_labels = [
                label for label, _ in ranking_for_artifact(vector, artifact, args)[:summary_top_k]
            ]
        else:
            probs = classifier.predict_proba(vector.reshape(1, -1))[0]
            ranked_idx = np.argsort(probs)[::-1][:summary_top_k]
            ranked_labels = [classifier.classes_[idx] for idx in ranked_idx]

        pred = ranked_labels[0]
        evaluated += 1
        per_identity[expected_label]["total"] += 1
        if pred == expected_label:
            correct += 1
            per_identity[expected_label]["correct"] += 1
        for k in range(1, summary_top_k + 1):
            if expected_label in ranked_labels[:k]:
                top_hits[k] += 1

    print(f"Labeled vectors found     : {labeled_found}")
    print(f"Evaluated known identities: {evaluated}")
    print(f"Skipped UnknownProbe      : {skipped_unknown}")
    if evaluated <= 0:
        print("No labeled known-identity vectors were available for evaluation.")
        return

    print(f"Correct predictions       : {correct}/{evaluated}")
    print(f"Accuracy                  : {correct / evaluated * 100:.2f}%")
    for k in range(2, summary_top_k + 1):
        print(
            f"Top-{k} contains correct   : {top_hits[k]}/{evaluated} "
            f"({top_hits[k] / evaluated * 100:.2f}%)"
        )
    if args.per_identity:
        print("Per-identity breakdown:")
        for label in sorted(
            per_identity,
            key=lambda text: int(re.search(r"\d+", text).group()),
        ):
            hits = per_identity[label]["correct"]
            total = per_identity[label]["total"]
            print(f"- {label:<12} {hits:>2}/{total:<2} ({hits / total * 100:6.2f}%)")


def main():
    args = parse_args()
    if args.model_file:
        model_specs = [load_artifact(args.model_file)]
    else:
        if args.model == "both":
            model_names = ["arcface", "vgg", "vgg1000"]
        elif args.model == "all":
            model_names = ["arcface", "vgg", "vgg1000", "fusion"]
        else:
            model_names = [args.model]

        model_specs = []
        for model_name in model_names:
            if model_name == "ensemble":
                candidate_paths = [
                    SCRIPTS_ROOT / "trained_models" / "identity_model_ensemble.pkl",
                ]
            elif model_name == "fusion":
                candidate_paths = [
                    SCRIPTS_ROOT / "trained_models" / "identity_model_fusion_knn.pkl",
                    SCRIPTS_ROOT / "trained_models" / "identity_model_fusion.pkl",
                ]
            else:
                candidate_paths = [
                    SCRIPTS_ROOT / "trained_models" / f"identity_model_{model_name}.pkl",
                    SCRIPTS_ROOT / "trained_models" / f"identity_model_{model_name}_knn.pkl",
                ]

            loaded = None
            for candidate in candidate_paths:
                if candidate.exists():
                    loaded = load_artifact(candidate)
                    break
            if loaded is None:
                raise SystemExit(
                    f"No saved model found for '{model_name}'. Checked: {candidate_paths}"
                )
            model_specs.append(loaded)

    unknown_vectors = load_unknown_vectors(resolve_unknown_source(args))

    for artifact, model_path in model_specs:
        model_name = artifact["model_name"]
        classifier_name = artifact.get("classifier_name", "logreg")

        if model_name == "ensemble":
            components = load_ensemble_components(artifact)
            matching_vectors, skipped_wrong_dim = select_vectors_for_ensemble(
                unknown_vectors, components
            )
            print(f"\nLoaded model           : {model_path}")
            print("Feature space          : weighted ensemble (arcface + fusion + vgg)")
            print(f"Classifier             : {classifier_name}")
            print(f"Excluded Identity10    : {artifact.get('excluded_identity10', False)}")
            print(f"Weights                : {artifact['ensemble_weights']}")
            print(f"Loaded vector files    : {len(unknown_vectors)}")
            print(f"Matching ensemble rows : {len(matching_vectors)}")
            print(f"Skipped wrong dimension: {skipped_wrong_dim}")
        elif model_name == "fusion":
            classifier = artifact["classifier"]
            feature_spaces = artifact.get("feature_spaces", [model_name])
            vector_dims = artifact.get("vector_dims")
            if vector_dims is None:
                vector_dims = {model_name: artifact["reference_vectors"].shape[1]}

            matching_vectors, skipped_wrong_dim = select_fusion_vectors(
                unknown_vectors, vector_dims
            )
            print(f"\nLoaded model           : {model_path}")
            print(f"Feature space          : fusion (arcface + vgg)")
            print(f"Classifier             : {classifier_name}")
            print(f"Excluded Identity10    : {artifact.get('excluded_identity10', False)}")
            print(f"Loaded vector files    : {len(unknown_vectors)}")
            print(f"Matching fused vectors : {len(matching_vectors)}")
            print(f"Skipped wrong dimension: {skipped_wrong_dim}")
        else:
            classifier = artifact["classifier"]
            feature_spaces = artifact.get("feature_spaces", [model_name])
            vector_dims = artifact.get("vector_dims")
            if vector_dims is None:
                vector_dims = {model_name: artifact["reference_vectors"].shape[1]}
            expected_dim = vector_dims[feature_spaces[0]]
            matching_vectors, skipped_other_model, skipped_wrong_dim = (
                select_single_feature_vectors(unknown_vectors, model_name, expected_dim)
            )
            print(f"\nLoaded model           : {model_path}")
            print(f"Feature space          : {model_name}")
            print(f"Classifier             : {classifier_name}")
            print(f"Excluded Identity10    : {artifact.get('excluded_identity10', False)}")
            print(f"Loaded vector files    : {len(unknown_vectors)}")
            print(f"Expected vector size   : {expected_dim}")
            print(f"Matching vectors       : {len(matching_vectors)}")
            print(f"Skipped other model    : {skipped_other_model}")
            print(f"Skipped wrong dimension: {skipped_wrong_dim}")

        if not matching_vectors:
            print("No unknown vectors matched this model's feature requirements.")
            continue

        if args.summary_only:
            if model_name == "ensemble":
                evaluated = 0
                correct = 0
                skipped_unknown = 0
                labeled_found = 0
                per_identity = defaultdict(lambda: {"correct": 0, "total": 0})
                summary_top_k = min(3, max(1, args.top_k))
                top_hits = {k: 0 for k in range(1, summary_top_k + 1)}

                for item in matching_vectors:
                    name = item["name"]
                    expected_label = extract_expected_label(name)
                    if expected_label is None:
                        continue
                    labeled_found += 1
                    if expected_label == "UnknownProbe":
                        skipped_unknown += 1
                        continue

                    probs = ensemble_predict_proba(artifact, components, item["vectors"])
                    ranked_idx = np.argsort(probs)[::-1][:summary_top_k]
                    ranked_labels = [artifact["class_labels"][idx] for idx in ranked_idx]
                    pred = ranked_labels[0]
                    evaluated += 1
                    per_identity[expected_label]["total"] += 1
                    if pred == expected_label:
                        correct += 1
                        per_identity[expected_label]["correct"] += 1
                    for k in range(1, summary_top_k + 1):
                        if expected_label in ranked_labels[:k]:
                            top_hits[k] += 1

                print(f"Labeled vectors found     : {labeled_found}")
                print(f"Evaluated known identities: {evaluated}")
                print(f"Skipped UnknownProbe      : {skipped_unknown}")
                if evaluated <= 0:
                    print("No labeled known-identity vectors were available for evaluation.")
                    continue
                print(f"Correct predictions       : {correct}/{evaluated}")
                print(f"Accuracy                  : {correct / evaluated * 100:.2f}%")
                for k in range(2, summary_top_k + 1):
                    print(
                        f"Top-{k} contains correct   : {top_hits[k]}/{evaluated} "
                        f"({top_hits[k] / evaluated * 100:.2f}%)"
                    )
                if args.per_identity:
                    print("Per-identity breakdown:")
                    for label in sorted(
                        per_identity,
                        key=lambda text: int(re.search(r"\d+", text).group()),
                    ):
                        hits = per_identity[label]["correct"]
                        total = per_identity[label]["total"]
                        print(f"- {label:<12} {hits:>2}/{total:<2} ({hits / total * 100:6.2f}%)")
            else:
                print_summary_rows(matching_vectors, classifier, artifact, args)
            continue

        if model_name == "ensemble":
            top_k = max(1, min(args.top_k, len(artifact["class_labels"])))
            for item in matching_vectors:
                name = item["name"]
                probs = ensemble_predict_proba(artifact, components, item["vectors"])
                ranked_idx = np.argsort(probs)[::-1][:top_k]

                print("\n" + "=" * 60)
                print(f"Unknown vector: {name}")
                print("=" * 60)
                print("Ensemble predictions:")
                for rank, idx in enumerate(ranked_idx, start=1):
                    print(f"{rank}. {artifact['class_labels'][idx]:<20} prob={probs[idx]:.6f}")
            continue

        top_k = max(1, min(args.top_k, len(classifier.classes_)))
        for item in matching_vectors:
            name = item["name"]
            vector = item["vector"]
            centroid_ranked = centroid_ranking(vector, artifact)
            reference_ranked = primary_reference_ranking(vector, artifact)
            best_single_ranked = best_single_reference_ranking(vector, artifact)
            if args.prefer_centroids:
                top_ranked = reference_ranked[:top_k] if reference_ranked else centroid_ranked[:top_k]
            else:
                probs = classifier.predict_proba(vector.reshape(1, -1))[0]
                ranked_idx = np.argsort(probs)[::-1][:top_k]
                top_ranked = centroid_ranked[:top_k]

            print("\n" + "=" * 60)
            print(f"Unknown vector: {name}")
            print("=" * 60)
            if args.prefer_centroids:
                confidence, gap = summarize_confidence(top_ranked)
                if top_ranked:
                    print(f"Recommended final guess: {top_ranked[0][0]}")
                print(f"Confidence: {confidence} (gap={gap:.6f})")
                print("Reference cosine ranking:")
                for rank, (label, score) in enumerate(top_ranked, start=1):
                    print(f"{rank}. {label:<20} cosine={score:.6f}")
            else:
                classifier_scored = [
                    (classifier.classes_[idx], probs[idx]) for idx in ranked_idx
                ]
                (
                    confidence,
                    clf_gap,
                    ref_gap,
                    ref_top1,
                    clf_level,
                    ref_level,
                ) = combined_confidence(model_name, classifier_scored, reference_ranked)

                final_guess = choose_final_guess(
                    classifier_scored,
                    reference_ranked,
                    confidence,
                    clf_level,
                    ref_level,
                )
                if final_guess is not None:
                    print(f"Recommended final guess: {final_guess}")
                print(
                    "Confidence: "
                    f"{confidence} "
                    f"(classifier={clf_level}, reference={ref_level}, "
                    f"classifier_gap={clf_gap:.6f}, reference_gap={ref_gap:.6f}, "
                    f"reference_top={ref_top1:.6f})"
                )
                print("Classifier predictions:")
                for rank, idx in enumerate(ranked_idx, start=1):
                    score_label = "vote" if artifact.get("classifier_name") == "knn1" else "prob"
                    print(f"{rank}. {classifier.classes_[idx]:<20} {score_label}={probs[idx]:.6f}")

            if (
                not args.prefer_centroids
                and artifact.get("classifier_name") == "knn1"
                and artifact.get("model_name") in {"vgg", "vgg1000"}
            ):
                print("Best single-image ranking:")
                for rank, (label, score) in enumerate(best_single_ranked[:top_k], start=1):
                    print(f"{rank}. {label:<20} cosine={score:.6f}")

            if not args.prefer_centroids and reference_ranked:
                print("Reference vote ranking:")
                for rank, (label, score) in enumerate(reference_ranked[:top_k], start=1):
                    print(f"{rank}. {label:<20} cosine={score:.6f}")

            if not args.prefer_centroids:
                print("Centroid backup ranking:")
                for rank, (label, score) in enumerate(centroid_ranked[:top_k], start=1):
                    print(f"{rank}. {label:<20} cosine={score:.6f}")


if __name__ == "__main__":
    main()
