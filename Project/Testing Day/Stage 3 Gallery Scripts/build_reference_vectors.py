import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
MAIN_SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
STAGE2_DIR = MAIN_SCRIPTS_ROOT
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ArcFaceExtractor = None
VGG19Extractor = None
SUPPORTED_EXTS = None


def load_stage2_dependencies():
    global ArcFaceExtractor
    global VGG19Extractor
    global SUPPORTED_EXTS

    if ArcFaceExtractor is not None:
        return

    try:
        from stage2 import ArcFaceExtractor as _ArcFaceExtractor
        from stage2 import SUPPORTED_EXTS as _SUPPORTED_EXTS
        from stage2 import VGG19Extractor as _VGG19Extractor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency while importing stage2.py. "
            f"Install the Stage 2 requirements first (current failure: {exc})."
        ) from exc

    ArcFaceExtractor = _ArcFaceExtractor
    VGG19Extractor = _VGG19Extractor
    SUPPORTED_EXTS = _SUPPORTED_EXTS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build labeled reference vectors from identity folders of images."
    )
    parser.add_argument(
        "--model",
        choices=["arcface", "vgg1000", "both"],
        default="both",
        help="Feature extractor to use.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(SCRIPTS_ROOT / "input_gallery"),
        help="Folder containing one subfolder per identity. Defaults to Testing Day/input_gallery.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .npz output path. Defaults to trained_models/reference_<model>.npz",
    )
    parser.add_argument(
        "--arcface-preprocessing",
        action="store_true",
        help="Use ArcFace face detection/alignment instead of the direct resized-image path.",
    )
    return parser.parse_args()


def build_extractor(model_name, args):
    if model_name == "arcface":
        return ArcFaceExtractor(use_preprocessing=args.arcface_preprocessing)
    if model_name == "vgg1000":
        return VGG19Extractor(output_dim=1000)
    return VGG19Extractor()


def collect_reference_records(reference_dir):
    root = Path(reference_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Reference image folder not found: {root}")

    records = []
    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue
        label = identity_dir.name.strip()
        if not label:
            continue
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTS:
                records.append({"path": image_path, "label": label})

    if not records:
        raise SystemExit(
            "No labeled reference images were found in the chosen reference folder."
        )
    return records


def default_output_path(model_name):
    out_dir = SCRIPTS_ROOT / "trained_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"reference_{model_name}.npz"


def main():
    args = parse_args()
    load_stage2_dependencies()

    records = collect_reference_records(args.reference_dir)
    if args.model == "both":
        model_names = ["arcface", "vgg1000"]
    else:
        model_names = [args.model]

    for model_name in model_names:
        extractor = build_extractor(model_name, args)
        vectors = []
        labels = []
        image_paths = []

        for record in records:
            vector = extractor.extract(str(record["path"]))
            if vector is None:
                print(f"Skipped    : {record['path']} ({model_name} extraction failed)")
                continue
            vectors.append(np.asarray(vector, dtype=np.float32))
            labels.append(record["label"])
            image_paths.append(str(record["path"]))

        if not vectors:
            print(f"No usable vectors were extracted for {model_name}.")
            continue

        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output and len(model_names) == 1
            else default_output_path(model_name)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            vectors=np.vstack(vectors).astype(np.float32),
            labels=np.asarray(labels, dtype=object),
            image_paths=np.asarray(image_paths, dtype=object),
            model=np.asarray(model_name, dtype=object),
        )

        unique_labels = sorted(set(labels))
        print(f"Model           : {model_name}")
        if model_name == "arcface":
            print(f"Preprocessing   : {args.arcface_preprocessing}")
        print(f"Reference images: {len(image_paths)}")
        print(f"Identities      : {len(unique_labels)}")
        print(f"Output          : {output_path}")


if __name__ == "__main__":
    main()
