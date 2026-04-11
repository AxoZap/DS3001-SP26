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
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
GALLERY_DIR = SCRIPT_DIR / "Gallery Set"
PROBE_DIR = SCRIPT_DIR / "Probe Set"
OUTPUT_DIR = SCRIPT_DIR / "stage2_outputs"
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
        img = cv2.imread(img_path)
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
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(
            *list(vgg.features.children()), vgg.avgpool, torch.nn.Flatten(),
            *list(vgg.classifier.children())[:-1])
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def extract(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
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


def evaluate_noisy(prb_feats, prb_labels, original_labels, kmeans, cluster2label, unique_labels, threshold, model_name):
    original_true = np.array(original_labels)
    total_original_known = sum(original_true != "Unknown")
    total_original_unknown = sum(original_true == "Unknown")

    if len(prb_feats) == 0:
        print(f"\n[{model_name} - Noisy Data] No valid features extracted.")
        return 0, 0

    prb_norm = normalize(prb_feats)
    prb_clusters = kmeans.predict(prb_norm)
    centroids = kmeans.cluster_centers_
    sims = np.sum(prb_norm * centroids[prb_clusters], axis=1)

    prb_pred = []
    for i in range(len(prb_labels)):
        if sims[i] < threshold:
            prb_pred.append("Unknown")
        else:
            int_label = cluster2label[prb_clusters[i]]
            str_label = unique_labels[int_label]
            prb_pred.append(str_label)

    prb_pred = np.array(prb_pred)
    prb_true = np.array(prb_labels)

    known_mask = prb_true != "Unknown"
    correct_known = sum(prb_pred[known_mask] == prb_true[known_mask])
    strict_known_acc = correct_known / sum(known_mask) if sum(known_mask) > 0 else 0
    known_coverage = sum(known_mask) / total_original_known if total_original_known > 0 else 0

    unknown_mask = prb_true == "Unknown"
    correct_rejections = sum(prb_pred[unknown_mask] == "Unknown")
    strict_rej_rate = correct_rejections / sum(unknown_mask) if sum(unknown_mask) > 0 else 0
    unknown_coverage = sum(unknown_mask) / total_original_unknown if total_original_unknown > 0 else 0

    print(f"\n[{model_name} - Noisy Data]")
    print(f"Known Acc: {strict_known_acc * 100:.2f}% ({correct_known}/{sum(known_mask)})")
    print(f"Known Cov: {known_coverage * 100:.2f}% ({sum(known_mask)}/{total_original_known})")
    print(f"Rej. Rate: {strict_rej_rate * 100:.2f}% ({correct_rejections}/{sum(unknown_mask)})")
    print(f"Ukn.  Cov: {unknown_coverage * 100:.2f}% ({sum(unknown_mask)}/{total_original_unknown})")

    all_labels = unique_labels + ["Unknown"]
    cm = confusion_matrix(prb_true, prb_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=all_labels, yticklabels=all_labels, ax=ax)
    ax.set_title(f"{model_name} - Part 3: Noisy Probe Clustering")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"Part3_{model_name}_noisy.png"
    plt.savefig(out, dpi=150)
    plt.close()

    return strict_known_acc, strict_rej_rate


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

    arc_km, arc_c2l, arc_uniq, arc_thresh, arc_clean_acc = evaluate_clean(
        arc_p2_f, arc_p2_l, len(part2_labels), "ArcFace"
    )
    arc_noisy_acc, arc_rej = evaluate_noisy(
        arc_p3_f, arc_p3_l, part3_labels, arc_km, arc_c2l, arc_uniq, arc_thresh, "ArcFace"
    )
    plot_distributions(arc_p2_f, arc_p2_l, arc_p3_f, arc_p3_l, "ArcFace")

    print("\n" + "-" * 40 + "\nVGG19 EXTRACTION\n" + "-" * 40)
    vgg = VGG19Extractor()
    vgg_p2_f, vgg_p2_l = vgg.extract_batch(part2_paths, part2_labels)
    vgg_p3_f, vgg_p3_l = vgg.extract_batch(part3_paths, part3_labels)

    vgg_km, vgg_c2l, vgg_uniq, vgg_thresh, vgg_clean_acc = evaluate_clean(
        vgg_p2_f, vgg_p2_l, len(part2_labels), "VGG19"
    )
    vgg_noisy_acc, vgg_rej = evaluate_noisy(
        vgg_p3_f, vgg_p3_l, part3_labels, vgg_km, vgg_c2l, vgg_uniq, vgg_thresh, "VGG19"
    )
    plot_distributions(vgg_p2_f, vgg_p2_l, vgg_p3_f, vgg_p3_l, "VGG19")

    print("\n" + "=" * 40 + "\nRESULTS SUMMARY\n" + "=" * 40)
    print(
        f"ArcFace  | Clean: {arc_clean_acc * 100:6.2f}% | Noisy: {arc_noisy_acc * 100:6.2f}% | Rejection: {arc_rej * 100:6.2f}%")
    print(
        f"VGG19    | Clean: {vgg_clean_acc * 100:6.2f}% | Noisy: {vgg_noisy_acc * 100:6.2f}% | Rejection: {vgg_rej * 100:6.2f}%")


