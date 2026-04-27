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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ArcFaceExtractor = None
VGG19Extractor = None
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_stage2_dependencies():
    global ArcFaceExtractor
    global VGG19Extractor

    if ArcFaceExtractor is not None:
        return

    try:
        from stage2 import ArcFaceExtractor as _ArcFaceExtractor
        from stage2 import VGG19Extractor as _VGG19Extractor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency while importing stage2.py. "
            f"Install the Stage 2 requirements first (current failure: {exc})."
        ) from exc

    ArcFaceExtractor = _ArcFaceExtractor
    VGG19Extractor = _VGG19Extractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a single image into an ArcFace or VGG feature vector."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Optional path to one input image. If omitted, all images in input_probe are processed.",
    )
    parser.add_argument(
        "--model",
        choices=["arcface", "vgg", "vgg1000", "both", "all"],
        default="both",
        help="Feature extractor to use.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to <image_stem>_<model>_vector.npy",
    )
    parser.add_argument(
        "--print-vector",
        action="store_true",
        help="Print the full vector to the console in addition to saving it.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(SCRIPTS_ROOT / "input_probe"),
        help="Folder of images to process when image_path is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPTS_ROOT / "input_vectors"),
        help="Folder where generated .npy files are written.",
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


def default_output_path(image_path, model_name):
    return image_path.with_name(f"{image_path.stem}_{model_name}_vector.npy")


def collect_images(args):
    if args.image_path:
        image_path = Path(args.image_path).expanduser().resolve()
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")
        return [image_path]

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input image folder not found: {input_dir}")

    image_paths = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    if not image_paths:
        raise SystemExit(f"No supported images found in: {input_dir}")
    return image_paths


def build_output_path(image_path, args):
    if args.output and len(collect_images(args)) == 1:
        return Path(args.output).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{image_path.stem}_{args.model}_vector.npy"


def main():
    args = parse_args()
    load_stage2_dependencies()

    image_paths = collect_images(args)
    if args.model in {"both", "all"}:
        model_names = ["arcface", "vgg", "vgg1000"]
    else:
        model_names = [args.model]
    extractors = {
        model_name: build_extractor(model_name, args) for model_name in model_names
    }

    print(f"Model      : {args.model}")
    print(f"Images     : {len(image_paths)}")
    if "arcface" in model_names:
        print(f"ArcFace preprocessing: {args.arcface_preprocessing}")

    for image_path in image_paths:
        print(f"Image      : {image_path}")
        for model_name in model_names:
            extractor = extractors[model_name]
            vector = extractor.extract(str(image_path))
            if vector is None:
                print(f"Skipped    : {image_path} ({model_name} extraction failed)")
                continue

            vector = np.asarray(vector, dtype=np.float32)
            output_path = (
                Path(args.output).expanduser().resolve()
                if args.output and len(image_paths) == 1 and len(model_names) == 1
                else Path(args.output_dir).expanduser().resolve() / f"{image_path.stem}_{model_name}_vector.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, vector)

            print(f"Model used : {model_name}")
            print(f"Output     : {output_path}")
            print(f"Shape      : {vector.shape}")
            print(f"Dtype      : {vector.dtype}")

            if args.print_vector:
                np.set_printoptions(suppress=True, linewidth=200, threshold=np.inf)
                print("\nFeature vector:")
                print(vector)


if __name__ == "__main__":
    main()
