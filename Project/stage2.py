# Heavily assisted by Codex

import re
import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parent
GALLERY_DIR = PROJECT_ROOT / "Gallery Set"
PROBE_DIR = PROJECT_ROOT / "Probe Set"
OUTPUT_DIR = SCRIPTS_ROOT / "stage2_outputs"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                  ".JPG", ".JPEG", ".PNG"}

NOISY_KEYWORDS = ["blur", "covered", "tilted", "bb_", "uncovered", "orientation"]


def identity_sort_key(label):
    m = re.match(r"Identity(\d+)$", label)
    return int(m.group(1)) if m else 10**9


def normalize_identity_name(raw_name):
    m = re.search(r"identity\s*(\d+)", raw_name, flags=re.IGNORECASE)
    if not m:
        return None
    return f"Identity{int(m.group(1))}"


def load_rgb_image_with_exif(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return ImageOps.exif_transpose(img)
    except Exception:
        return None


def load_bgr_image_with_exif(img_path):
    img = load_rgb_image_with_exif(img_path)
    if img is None:
        return None
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_gallery(root_dir):
    root = Path(root_dir)
    paths, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        identity = normalize_identity_name(folder.name)
        if identity is None:
            continue

        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in SUPPORTED_EXTS:
                paths.append(str(f))
                labels.append(identity)

    print(f"[Gallery] Loaded {len(paths)} clean gallery images.")
    return paths, labels


def load_probe_split(probe_dir):
    root = Path(probe_dir)
    clean_p, clean_l = [], []
    noisy_p, noisy_l = [], []
    ood_p, ood_l = [], []

    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in SUPPORTED_EXTS:
            continue

        rel_parts = f.relative_to(root).parts
        probe_identity = normalize_identity_name(rel_parts[0]) if len(rel_parts) > 0 else None

        # Per project requirement: Probe Identity 10 is a wildcard/OOD class.
        if probe_identity == "Identity10" or probe_identity is None:
            label = "Unknown"
        else:
            label = probe_identity

        rel_lower = str(f.relative_to(root)).lower()
        stem_lower = f.stem.lower()
        is_noisy = any(kw in rel_lower or kw in stem_lower for kw in NOISY_KEYWORDS)

        path_str = str(f)
        if label == "Unknown":
            ood_p.append(path_str)
            ood_l.append(label)
        elif is_noisy:
            noisy_p.append(path_str)
            noisy_l.append(label)
        else:
            clean_p.append(path_str)
            clean_l.append(label)

    print(f"[Probe] Split: {len(clean_p)} Clean | {len(noisy_p)} Noisy | {len(ood_p)} Unknown")
    return (clean_p, clean_l), (noisy_p, noisy_l), (ood_p, ood_l)


class ArcFaceExtractor:
    def __init__(self, use_preprocessing=False):
        self.use_preprocessing = use_preprocessing
        self.app = None
        self.recognizer = None

        if self.use_preprocessing:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        else:
            from insightface.model_zoo import get_model
            model_path = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
            self.recognizer = get_model(str(model_path), providers=["CPUExecutionProvider"])
            self.recognizer.prepare(ctx_id=0)

    def preprocess_without_detection(self, img_bgr):
        # ArcFace expects a fixed-size face crop. For the assignment's "skip detection/alignment"
        # setting, we feed a simple resized image directly into the recognizer.
        return cv2.resize(img_bgr, self.recognizer.input_size)

    def extract(self, img_path):
        img = load_bgr_image_with_exif(img_path)
        if img is None:
            return None

        if not self.use_preprocessing:
            direct_img = self.preprocess_without_detection(img)
            return self.recognizer.get_feat(direct_img).flatten()

        faces = self.app.get(img)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return face.embedding

        print(f"  -> No face detected after ArcFace preprocessing: {Path(img_path).name}")
        return None

    def extract_batch(self, paths, labels):
        embs, valid = [], []
        for p, l in zip(paths, labels):
            e = self.extract(p)
            if e is not None:
                embs.append(e)
                valid.append(l)
            else:
                print(f"  -> Failed to read image: {Path(p).name}")

        if len(embs) == 0:
            return np.array([]), []
        return np.array(embs), valid


class VGG19Extractor:
    def __init__(self, output_dim=4096):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if output_dim not in {4096, 1000}:
            raise ValueError("VGG19Extractor output_dim must be 4096 or 1000")
        self.output_dim = output_dim
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        classifier_layers = list(vgg.classifier.children())
        if output_dim == 4096:
            classifier_layers = classifier_layers[:-1]
        self.model = torch.nn.Sequential(
            *list(vgg.features.children()), vgg.avgpool, torch.nn.Flatten(),
            *classifier_layers)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def extract(self, img_path):
        img = load_rgb_image_with_exif(img_path)
        if img is None:
            return None
        t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(t).squeeze().cpu().numpy()

    def extract_batch(self, paths, labels):
        embs, valid = [], []
        for p, l in zip(paths, labels):
            e = self.extract(p)
            if e is not None:
                embs.append(e)
                valid.append(l)
            else:
                print(f"  -> Processing error: {Path(p).name}")
        return np.array(embs), valid


class SimpleKMeans:
    def __init__(self, n_clusters, n_init=20, random_state=42, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.inertia_ = None

    def _assign_clusters(self, X, centers):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = np.sum(dists[np.arange(len(X)), labels])
        return labels, inertia

    def _update_centers(self, X, labels, rng):
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                centers[k] = X[rng.integers(len(X))]
            else:
                centers[k] = members.mean(axis=0)
        return centers

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        best_centers, best_inertia = None, None

        for _ in range(self.n_init):
            init_idx = rng.choice(len(X), size=self.n_clusters, replace=False)
            centers = X[init_idx].copy()

            for _ in range(self.max_iter):
                labels, inertia = self._assign_clusters(X, centers)
                new_centers = self._update_centers(X, labels, rng)
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift <= self.tol:
                    break

            labels, inertia = self._assign_clusters(X, centers)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()

        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X):
        self.fit(X)
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels

    def predict(self, X):
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels


def evaluate_clean(feats, labels, original_total, model_name):
    unique_labels = sorted(set(labels), key=identity_sort_key)
    num_identities = len(unique_labels)
    label2int = {l: i for i, l in enumerate(unique_labels)}
    true_int = np.array([label2int[l] for l in labels])
    feats_norm = normalize(feats)
    extracted_total = len(labels)

    kmeans = SimpleKMeans(n_clusters=num_identities, n_init=20, random_state=42)
    pred_clusters = kmeans.fit_predict(feats_norm)

    conf = np.zeros((num_identities, num_identities), dtype=int)
    for t, p in zip(true_int, pred_clusters):
        conf[p, t] += 1

    row_ind, col_ind = linear_sum_assignment(-conf)
    cluster2label = {r: c for r, c in zip(row_ind, col_ind)}
    mapped = np.array([cluster2label[p] for p in pred_clusters])

    correct_matches = sum(true_int == mapped)
    strict_acc = correct_matches / extracted_total if extracted_total > 0 else 0.0
    coverage = extracted_total / original_total if original_total > 0 else 0.0

    centroids = kmeans.cluster_centers_
    sims = np.sum(feats_norm * centroids[pred_clusters], axis=1)
    threshold = np.min(sims) * 0.90

    print(f"\n[{model_name} - Clean Data]")
    print(f"Accuracy : {strict_acc * 100:.2f}% ({correct_matches}/{extracted_total})")
    print(f"Coverage : {coverage * 100:.2f}% ({extracted_total}/{original_total})")
    print(f"Threshold: {threshold:.3f}")

    cm = confusion_matrix(true_int, mapped, labels=list(range(num_identities)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels,
                yticklabels=unique_labels, ax=ax)
    ax.set_title(f"{model_name} - Part 2: Clean Data Clustering (Matched Labels)")
    ax.set_xlabel("Predicted Identity")
    ax.set_ylabel("True Identity")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"Part2_{model_name}_clean.png"
    plt.savefig(out, dpi=150)
    plt.close()

    return kmeans, cluster2label, unique_labels, threshold, strict_acc


def evaluate_noisy(noisy_feats, noisy_labels, model_name):
    if len(noisy_feats) == 0:
        print(f"\n[{model_name} - Noisy Data] No valid features extracted.")
        return 0, 0

    noisy_true = np.array(noisy_labels)
    known_mask = noisy_true != "Unknown"
    unknown_mask = noisy_true == "Unknown"

    known_labels = sorted(set(noisy_true[known_mask]), key=identity_sort_key)
    if len(known_labels) == 0:
        print(f"\n[{model_name} - Noisy Data] No known noisy identities available for clustering.")
        return 0, 0

    feats_norm = normalize(noisy_feats)
    kmeans = SimpleKMeans(n_clusters=len(known_labels), n_init=20, random_state=42)
    pred_clusters = kmeans.fit_predict(feats_norm)

    label2int = {l: i for i, l in enumerate(known_labels)}
    known_true_int = np.array([label2int[l] for l in noisy_true[known_mask]])
    known_pred_clusters = pred_clusters[known_mask]

    conf = np.zeros((len(known_labels), len(known_labels)), dtype=int)
    for t, p in zip(known_true_int, known_pred_clusters):
        conf[p, t] += 1

    row_ind, col_ind = linear_sum_assignment(-conf)
    cluster2label = {r: c for r, c in zip(row_ind, col_ind)}

    mapped_known = np.array([cluster2label[p] for p in known_pred_clusters])
    correct_known = np.sum(mapped_known == known_true_int)
    known_acc = correct_known / len(known_true_int) if len(known_true_int) > 0 else 0.0

    wildcard_cluster_ids = pred_clusters[unknown_mask]
    wildcard_assignments = []
    for cluster_id in wildcard_cluster_ids:
        mapped_int = cluster2label.get(cluster_id)
        wildcard_assignments.append(known_labels[mapped_int] if mapped_int is not None else "Unmapped")

    wildcard_majority_share = 0.0
    if wildcard_assignments:
        counts = {}
        for label in wildcard_assignments:
            counts[label] = counts.get(label, 0) + 1
        wildcard_majority_share = max(counts.values()) / len(wildcard_assignments)

    print(f"\n[{model_name} - Noisy Data]")
    print(f"Known noisy clustering accuracy: {known_acc * 100:.2f}% ({correct_known}/{len(known_true_int)})")
    if np.sum(unknown_mask) > 0:
        print(f"Wildcard probe images: {np.sum(unknown_mask)}")
        print(f"Wildcard majority-cluster share: {wildcard_majority_share * 100:.2f}%")

    cm = confusion_matrix(known_true_int, mapped_known, labels=list(range(len(known_labels))))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=known_labels, yticklabels=known_labels, ax=ax)
    ax.set_title(f"{model_name} - Part 3: Noisy Probe Clustering")
    ax.set_xlabel("Predicted Identity")
    ax.set_ylabel("True Identity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"Part3_{model_name}_noisy.png"
    plt.savefig(out, dpi=150)
    plt.close()

    return known_acc, wildcard_majority_share


def plot_distributions(clean_feats, clean_labels, noisy_feats, noisy_labels, model_name):
    for identity in ["Identity5", "Identity8", "Identity9"]:
        c_idx = [i for i, l in enumerate(clean_labels) if l == identity]
        n_idx = [i for i, l in enumerate(noisy_labels) if l == identity]

        if not c_idx or not n_idx:
            continue

        clean_target = clean_feats[c_idx]
        noisy_target = noisy_feats[n_idx]
        all_e = normalize(np.vstack([clean_target, noisy_target]))

        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_e)
        c2d = all_2d[:len(clean_target)]
        n2d = all_2d[len(clean_target):]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(c2d[:, 0], c2d[:, 1], label="Clean Data", marker="o", s=100, alpha=0.8)
        ax.scatter(n2d[:, 0], n2d[:, 1], label="Noisy Data", marker="x", s=100, alpha=0.8)
        ax.set_title(f"{model_name} - {identity}: Clean vs Noisy")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        plt.tight_layout()
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / f"Part3_{model_name}_{identity}_dist.png"
        plt.savefig(out, dpi=150)
        plt.close()


if __name__ == "__main__":
    gal_paths, gal_labels = load_gallery(GALLERY_DIR)
    (clean_p, clean_l), (noisy_p, noisy_l), (ood_p, ood_l) = load_probe_split(PROBE_DIR)

    if len(clean_p) == 0:
        print("[Part 2] No separate clean probe images found; clustering uses the clean gallery identities.")

    part2_paths = gal_paths + clean_p
    part2_labels = gal_labels + clean_l

    part3_paths = noisy_p + ood_p
    part3_labels = noisy_l + ood_l

    print("\n" + "-" * 40 + "\nARCFACE EXTRACTION\n" + "-" * 40)
    arc = ArcFaceExtractor()
    arc_p2_f, arc_p2_l = arc.extract_batch(part2_paths, part2_labels)
    arc_p3_f, arc_p3_l = arc.extract_batch(part3_paths, part3_labels)

    _, _, _, _, arc_clean_acc = evaluate_clean(
        arc_p2_f, arc_p2_l, len(part2_labels), "ArcFace"
    )
    arc_noisy_acc, arc_wildcard_share = evaluate_noisy(
        arc_p3_f, arc_p3_l, "ArcFace"
    )
    plot_distributions(arc_p2_f, arc_p2_l, arc_p3_f, arc_p3_l, "ArcFace")

    print("\n" + "-" * 40 + "\nVGG19 EXTRACTION\n" + "-" * 40)
    vgg = VGG19Extractor()
    vgg_p2_f, vgg_p2_l = vgg.extract_batch(part2_paths, part2_labels)
    vgg_p3_f, vgg_p3_l = vgg.extract_batch(part3_paths, part3_labels)

    _, _, _, _, vgg_clean_acc = evaluate_clean(
        vgg_p2_f, vgg_p2_l, len(part2_labels), "VGG19"
    )
    vgg_noisy_acc, vgg_wildcard_share = evaluate_noisy(
        vgg_p3_f, vgg_p3_l, "VGG19"
    )
    plot_distributions(vgg_p2_f, vgg_p2_l, vgg_p3_f, vgg_p3_l, "VGG19")

    print("\n" + "=" * 40 + "\nRESULTS SUMMARY\n" + "=" * 40)
    print(
        f"ArcFace  | Clean: {arc_clean_acc * 100:6.2f}% | Noisy: {arc_noisy_acc * 100:6.2f}% | Wildcard Share: {arc_wildcard_share * 100:6.2f}%")
    print(
        f"VGG19    | Clean: {vgg_clean_acc * 100:6.2f}% | Noisy: {vgg_noisy_acc * 100:6.2f}% | Wildcard Share: {vgg_wildcard_share * 100:6.2f}%")
