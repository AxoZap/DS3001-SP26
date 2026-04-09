import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
GALLERY_DIR = SCRIPT_DIR
PROBE_DIR = SCRIPT_DIR.parent / "Probe Set"
NUM_IDENTITIES = 10
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                  ".JPG", ".JPEG", ".PNG"}

PROBE_LABEL_MAP = {
    "BB_3": "Identity9", "BB_4": "Identity9", "BB_5": "Identity9", "BB_6": "Identity9",
    "Slight_Blur_3": "Identity9",
    "Covered_1": "Identity9", "Covered_2": "Identity9", "Covered_4": "Identity9",
    "Uncovered_1": "Identity9", "Uncovered_2": "Identity9", "Uncovered_3": "Identity9",
    "Very_Covered_1": "Identity9", "Very_Covered_2": "Identity9", "Very_Covered_3": "Identity9",
    "Tilted_3": "Identity9", "Tilted_4": "Identity9", "Very_Tilted_4": "Identity9",
    "Covered_3": "Identity8", "Covered_5": "Identity8",
    "Uncovered_4": "Identity8", "Uncovered_5": "Identity8",
    "Very_Covered_4": "Identity8", "Very_Covered_5": "Identity8", "Very_Covered_6": "Identity8",
    "Normal_1": "Identity8", "Normal_2": "Identity8", "Normal_3": "Identity8",
    "Normal_4": "Identity8", "Normal_5": "Identity8",
    "Tilted_5": "Identity8", "Very_Tilted_5": "Identity8",
    "Very_Tilted_1": "Identity5", "Very_Tilted_2": "Identity5", "Very_Tilted_3": "Identity5",
    "Tilted_1": "Identity5", "Tilted_2": "Identity5",
    "Very_Blurry_3": "Identity5", "Very_Blurry_4": "Identity5", "Very_Blurry_5": "Identity5",
    "Very_Blurry_6": "Identity5",
    "Slight_Blur_4": "Identity5", "Slight_Blur_5": "Identity5", "Slight_Blur_6": "Identity5"
}

NOISY_KEYWORDS = ["blur", "covered", "tilted", "bb_", "uncovered"]


def load_gallery(root_dir):
    root = Path(root_dir)
    paths, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("Identity"):
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in SUPPORTED_EXTS:
                paths.append(str(f))
                labels.append(folder.name)
    print(f"[Gallery] Loaded {len(paths)} clean images.")
    return paths, labels


def load_probe_split(probe_dir):
    root = Path(probe_dir)
    clean_p, clean_l = [], []
    noisy_p, noisy_l = [], []
    ood_p, ood_l = [], []

    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in SUPPORTED_EXTS:
            continue

        path_str = str(f)
        stem = f.stem

        if stem in PROBE_LABEL_MAP:
            label = PROBE_LABEL_MAP[stem]
            is_noisy = any(kw in stem.lower() for kw in NOISY_KEYWORDS)

            if is_noisy:
                noisy_p.append(path_str)
                noisy_l.append(label)
            else:
                clean_p.append(path_str)
                clean_l.append(label)
        else:
            ood_p.append(path_str)
            ood_l.append("Unknown")

    print(f"[Probe] Split: {len(clean_p)} Clean | {len(noisy_p)} Noisy | {len(ood_p)} Unknown")
    return (clean_p, clean_l), (noisy_p, noisy_l), (ood_p, ood_l)


class ArcFaceExtractor:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        faces = self.app.get(img)
        if not faces: return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.embedding

    def extract_batch(self, paths, labels):
        embs, valid = [], []
        for p, l in zip(paths, labels):
            e = self.extract(p)
            if e is not None:
                embs.append(e);
                valid.append(l)
            else:
                print(f"  -> Face not detected: {Path(p).name}")
        if len(embs) == 0: return np.array([]), []
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
        except:
            return None
        t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(t).squeeze().cpu().numpy()

    def extract_batch(self, paths, labels):
        embs, valid = [], []
        for p, l in zip(paths, labels):
            e = self.extract(p)
            if e is not None:
                embs.append(e);
                valid.append(l)
            else:
                print(f"  -> Processing error: {Path(p).name}")
        return np.array(embs), valid


def evaluate_clean(feats, labels, original_total, model_name):
    unique_labels = [f"Identity{i}" for i in range(1, NUM_IDENTITIES + 1)]
    label2int = {l: i for i, l in enumerate(unique_labels)}
    true_int = np.array([label2int[l] for l in labels])
    feats_norm = normalize(feats)
    extracted_total = len(labels)

    kmeans = KMeans(n_clusters=NUM_IDENTITIES, n_init=20, random_state=42)
    pred_clusters = kmeans.fit_predict(feats_norm)

    conf = np.zeros((NUM_IDENTITIES, NUM_IDENTITIES), dtype=int)
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

    cm = confusion_matrix(true_int, pred_clusters, labels=list(range(NUM_IDENTITIES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(NUM_IDENTITIES)],
                yticklabels=unique_labels, ax=ax)
    ax.set_title(f"{model_name} — Part 2: Clean Data Clustering")
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Identity")
    plt.tight_layout()
    out = SCRIPT_DIR / f"Part2_{model_name}_clean.png"
    plt.savefig(out, dpi=150)
    plt.close()

    return kmeans, cluster2label, unique_labels, threshold, strict_acc


def evaluate_noisy(prb_feats, prb_labels, original_labels, kmeans, cluster2label, unique_labels, threshold, model_name):
    original_true = np.array(original_labels)
    total_original_known = sum(original_true != "Unknown")
    total_original_unknown = sum(original_true == "Unknown")
    extracted_total = len(prb_labels)

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
    ax.set_title(f"{model_name} — Part 3: Noisy Probe Clustering")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = SCRIPT_DIR / f"Part3_{model_name}_noisy.png"
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
        ax.set_title(f"{model_name} — {identity}: Clean vs Noisy")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        plt.tight_layout()
        out = SCRIPT_DIR / f"Part3_{model_name}_{identity}_dist.png"
        plt.savefig(out, dpi=150)
        plt.close()


if __name__ == "__main__":
    gal_paths, gal_labels = load_gallery(GALLERY_DIR)
    (clean_p, clean_l), (noisy_p, noisy_l), (ood_p, ood_l) = load_probe_split(PROBE_DIR)

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
