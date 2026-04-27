import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
TRAINED_MODELS_DIR = SCRIPTS_ROOT / "trained_models"
GALLERY_DIR = SCRIPTS_ROOT / "input_gallery"
INPUT_VECTORS_DIR = SCRIPTS_ROOT / "input_vectors"
SUPPORTED_MODELS = ("arcface", "vgg1000")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Notebook-style identity prediction from saved probe vectors."
    )
    parser.add_argument(
        "--model",
        choices=("auto", "both", "arcface", "vgg1000"),
        default="both",
        help="Which template space(s) to use.",
    )
    parser.add_argument(
        "--gallery-dir",
        default=str(GALLERY_DIR),
        help="Folder containing one subfolder per identity.",
    )
    parser.add_argument(
        "--unknown",
        help="Optional path to one .npy vector or a folder of .npy vectors. If omitted, input_vectors is used.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(INPUT_VECTORS_DIR),
        help="Folder of .npy vectors to process when --unknown is omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many matches to print.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("auto", "centroid", "best-single"),
        default="auto",
        help="Identity scoring rule. ArcFace defaults to centroid to match the Colab notebook.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional cosine threshold. Below it, the final prediction becomes Unknown identity.",
    )
    parser.add_argument(
        "--update-cache",
        action="store_true",
        help="Rebuild gallery template caches instead of reusing them.",
    )
    parser.add_argument(
        "--arcface-det-size",
        type=int,
        default=640,
        help="Detector input size for ArcFace gallery extraction.",
    )
    return parser.parse_args()


def resolve_unknown_source(args):
    if args.unknown:
        return Path(args.unknown).expanduser().resolve()
    return Path(args.input_dir).expanduser().resolve()


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(np.linalg.norm(vector), 1e-12)


def infer_vector_type(name):
    lowered = name.lower()
    if lowered.endswith("_arcface_vector"):
        return "arcface"
    if lowered.endswith("_vgg1000_vector"):
        return "vgg1000"
    return None


def normalize_loaded_vector(raw):
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.shape == ():
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.reshape(()).item()

    if isinstance(raw, dict):
        for key in ("features", "embedding", "vector"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raise SystemExit(
                "Unsupported vector dictionary format. Expected one of: "
                "'features', 'embedding', or 'vector'."
            )

    vector = np.asarray(raw, dtype=np.float32)
    vector = np.squeeze(vector)
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    return vector


def load_unknown_vectors(source_path):
    if not source_path.exists():
        raise SystemExit(f"Unknown vector path not found: {source_path}")

    items = []
    if source_path.is_file():
        items.append(
            (
                source_path.stem,
                normalize_loaded_vector(np.load(source_path, allow_pickle=True)),
            )
        )
    else:
        for file_path in sorted(source_path.glob("*.npy")):
            items.append(
                (
                    file_path.stem,
                    normalize_loaded_vector(np.load(file_path, allow_pickle=True)),
                )
            )

    if not items:
        raise SystemExit(f"No .npy vectors found in: {source_path}")

    return [
        {
            "name": name,
            "vector": vector,
            "vector_type": infer_vector_type(name),
        }
        for name, vector in items
    ]


def detect_model_from_vector(item):
    if item["vector_type"] in SUPPORTED_MODELS:
        return item["vector_type"]

    dim = int(item["vector"].shape[0])
    if dim == 512:
        return "arcface"
    if dim == 1000:
        return "vgg1000"
    return None


def iter_gallery_images(gallery_dir):
    root = Path(gallery_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Gallery folder not found: {root}")

    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue
        label = identity_dir.name.strip()
        if not label:
            continue
        image_paths = [
            path
            for path in sorted(identity_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        if image_paths:
            yield label, image_paths


def load_rgb_with_exif(path):
    try:
        return ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    except Exception:
        return None


def load_bgr_with_exif(path):
    image = load_rgb_with_exif(path)
    if image is None:
        return None
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def build_arcface_extractor(det_size):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit(
            "ArcFace prediction requires insightface. Install insightface and onnxruntime."
        ) from exc

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def extract(image_path):
        img = load_bgr_with_exif(image_path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        biggest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
        )
        return l2_normalize(np.asarray(biggest_face.normed_embedding, dtype=np.float32))

    return extract


def build_vgg1000_extractor():
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise SystemExit(
            "VGG1000 prediction requires torch and torchvision."
        ) from exc

    weights = models.VGG19_Weights.IMAGENET1K_V1
    model = models.vgg19(weights=weights)
    model.eval()
    transform = weights.transforms()

    def extract(image_path):
        image = load_rgb_with_exif(image_path)
        if image is None:
            return None
        with torch.no_grad():
            tensor = transform(image).unsqueeze(0)
            vector = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        return l2_normalize(vector)

    return extract


def cache_path_for(model_name):
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return TRAINED_MODELS_DIR / f"colab_style_gallery_{model_name}.npz"


def build_gallery_cache(model_name, gallery_dir, args):
    if model_name == "arcface":
        extractor = build_arcface_extractor(args.arcface_det_size)
    elif model_name == "vgg1000":
        extractor = build_vgg1000_extractor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    identity_vectors = []
    identity_labels = []
    identity_counts = []
    image_vectors = []
    image_labels = []
    image_paths = []

    print(f"\nBuilding {model_name} gallery templates from {Path(gallery_dir).resolve()}...")
    for label, paths in iter_gallery_images(gallery_dir):
        vectors = []
        for image_path in paths:
            vector = extractor(str(image_path))
            if vector is None:
                continue
            vectors.append(l2_normalize(vector))
            image_vectors.append(l2_normalize(vector))
            image_labels.append(label)
            image_paths.append(str(image_path))

        if not vectors:
            print(f"  [skip] {label} -- no usable images")
            continue

        centroid = l2_normalize(np.mean(np.vstack(vectors), axis=0))
        identity_vectors.append(centroid)
        identity_labels.append(label)
        identity_counts.append(len(vectors))
        print(f"  {label:<28} usable={len(vectors)}")

    if not identity_vectors:
        raise SystemExit(f"No usable {model_name} gallery templates were built.")

    cache_path = cache_path_for(model_name)
    np.savez(
        cache_path,
        model=np.asarray(model_name, dtype=object),
        identity_vectors=np.vstack(identity_vectors).astype(np.float32),
        identity_labels=np.asarray(identity_labels, dtype=object),
        identity_counts=np.asarray(identity_counts, dtype=np.int32),
        image_vectors=np.vstack(image_vectors).astype(np.float32),
        image_labels=np.asarray(image_labels, dtype=object),
        image_paths=np.asarray(image_paths, dtype=object),
    )
    return load_gallery_cache(model_name)


def load_gallery_cache(model_name):
    path = cache_path_for(model_name)
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "model": str(np.asarray(data["model"]).item()),
        "identity_vectors": np.asarray(data["identity_vectors"], dtype=np.float32),
        "identity_labels": np.asarray(data["identity_labels"]).astype(str),
        "identity_counts": np.asarray(data["identity_counts"], dtype=np.int32),
        "image_vectors": np.asarray(data["image_vectors"], dtype=np.float32),
        "image_labels": np.asarray(data["image_labels"]).astype(str),
        "image_paths": np.asarray(data["image_paths"]).astype(str),
    }


def ensure_gallery_cache(model_name, gallery_dir, args):
    cache = None if args.update_cache else load_gallery_cache(model_name)
    if cache is not None:
        return cache
    return build_gallery_cache(model_name, gallery_dir, args)


def resolve_match_mode(model_name, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    if model_name == "arcface":
        return "centroid"
    return "best-single"


def score_centroids(query, cache):
    scores = cache["identity_vectors"] @ query
    ranked = sorted(
        (
            (str(label), float(score), int(count))
            for label, score, count in zip(
                cache["identity_labels"],
                scores,
                cache["identity_counts"],
            )
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def score_best_single(query, cache):
    best = {}
    for label, score in zip(cache["image_labels"], cache["image_vectors"] @ query):
        label = str(label)
        best[label] = max(best.get(label, -1.0), float(score))
    ranked = sorted(
        ((label, score, 1) for label, score in best.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def score_gallery_images(query, cache):
    scores = cache["image_vectors"] @ query
    ranked = sorted(
        (
            (str(label), float(score), str(path))
            for label, score, path in zip(
                cache["image_labels"],
                scores,
                cache["image_paths"],
            )
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def confidence_from_ranked(ranked):
    if not ranked:
        return "low", 0.0
    top1 = float(ranked[0][1])
    top2 = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    gap = top1 - top2
    if gap >= 0.08:
        return "high", gap
    if gap >= 0.03:
        return "medium", gap
    return "low", gap


def print_probe_report(probe_name, model_name, ranked, gallery_ranked, match_mode, top_k, threshold):
    confidence, gap = confidence_from_ranked(ranked)
    top_score = float(ranked[0][1]) if ranked else 0.0
    accepted = threshold is None or top_score >= threshold
    prediction = ranked[0][0] if ranked and accepted else "Unknown identity"

    print("\n" + "=" * 60)
    print(f"Probe vector: {probe_name}")
    print(f"Compared as : {model_name}")
    print("=" * 60)
    print(f"Recommended identity: {prediction}")
    if ranked:
        print(f"Confidence         : {confidence} (gap={gap:.6f}, top={top_score:.6f})")
    if threshold is not None:
        print(f"Threshold          : {threshold:.6f}")
    print(f"Match mode         : {match_mode}")

    if ranked:
        print("Identity ranking:")
        for rank, (label, score, count) in enumerate(ranked[:top_k], start=1):
            print(f"{rank}. {label:<24} cosine={score:.6f} gallery_count={count}")

    winner = ranked[0][0] if ranked else None
    if winner is not None:
        winner_images = [
            (label, score, path)
            for label, score, path in gallery_ranked
            if label == winner
        ][:top_k]
        print("Closest winner gallery images:")
        for rank, (_, score, path) in enumerate(winner_images, start=1):
            print(f"{rank}. cosine={score:.6f} path={path}")

    print("Closest gallery images overall:")
    for rank, (label, score, path) in enumerate(gallery_ranked[:top_k], start=1):
        print(f"{rank}. {label:<24} cosine={score:.6f} path={path}")


def main():
    args = parse_args()
    unknown_vectors = load_unknown_vectors(resolve_unknown_source(args))
    print(f"Loaded vector files    : {len(unknown_vectors)}")

    if args.model == "both":
        model_groups = defaultdict(list)
        for item in unknown_vectors:
            model_name = detect_model_from_vector(item)
            if model_name in SUPPORTED_MODELS:
                model_groups[model_name].append(item)
    else:
        model_groups = defaultdict(list)
        for item in unknown_vectors:
            detected = detect_model_from_vector(item)
            model_name = args.model if args.model != "auto" else detected
            if model_name in SUPPORTED_MODELS and (args.model == "auto" or detected in {None, args.model}):
                model_groups[model_name].append(item)

    if not model_groups:
        print("No vectors matched a supported feature space.")
        return

    for model_name in SUPPORTED_MODELS:
        probes = model_groups.get(model_name, [])
        if not probes:
            continue

        cache = ensure_gallery_cache(model_name, args.gallery_dir, args)
        expected_dim = int(cache["identity_vectors"].shape[1])
        match_mode = resolve_match_mode(model_name, args.match_mode)

        print(f"\nReference model        : {cache['model']}")
        print(f"Gallery cache          : {cache['path']}")
        print(f"Gallery identities     : {len(cache['identity_labels'])}")
        print(f"Gallery images         : {len(cache['image_paths'])}")
        print(f"Expected vector size   : {expected_dim}")
        print(f"Matching vectors       : {len(probes)}")
        print(f"Match mode             : {match_mode}")

        for item in probes:
            query = np.asarray(item["vector"], dtype=np.float32).reshape(-1)
            if query.shape[0] != expected_dim:
                print("\n" + "=" * 60)
                print(f"Probe vector: {item['name']}")
                print("=" * 60)
                print(
                    f"Skipped: vector size {query.shape[0]} does not match {model_name} size {expected_dim}"
                )
                continue

            query = l2_normalize(query)
            if match_mode == "centroid":
                ranked = score_centroids(query, cache)
            else:
                ranked = score_best_single(query, cache)
            gallery_ranked = score_gallery_images(query, cache)
            print_probe_report(
                item["name"],
                model_name,
                ranked,
                gallery_ranked,
                match_mode,
                args.top_k,
                args.threshold,
            )


if __name__ == "__main__":
    main()
