import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
GALLERY_SCRIPTS_DIR = SCRIPT_DIR / "Stage 3 Gallery Scripts"
PROBE_SCRIPTS_DIR = SCRIPT_DIR / "Stage 3 Probe Scripts"
INPUT_GALLERY_DIR = SCRIPT_DIR / "input_gallery"
TOP_K = 5
DEFAULT_MATCH_MODE = "auto"
ARCFACE_MATCH_MODE = "auto"
USE_ARCFACE_PREPROCESSING = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single entry point for the identity-matching pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser(
        "prepare",
        help="Build gallery reference vectors and train the identity models.",
    )
    prepare.add_argument(
        "--model",
        choices=["arcface", "vgg1000", "both"],
        default="both",
        help="Which feature spaces to build for the challenge-day enrollment gallery.",
    )

    train = subparsers.add_parser(
        "train",
        help="Train the main identity models used by the project.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Print top-k identity guesses for the current probe vectors.",
    )

    subparsers.add_parser(
        "all",
        help="Run the full challenge-day workflow: enroll gallery -> train -> predict from existing vectors.",
    )

    argv = sys.argv[1:]
    if not argv:
        argv = ["all"]
        print("No command provided. Defaulting to: all")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("Please choose one of: prepare, train, evaluate, all")

    return args


def run_step(args):
    print(f"\n>>> {' '.join(args)}")
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def build_reference_vectors(model_name="both"):
    args = [
        sys.executable,
        str(GALLERY_SCRIPTS_DIR / "build_reference_vectors.py"),
        "--model",
        model_name,
        "--reference-dir",
        str(INPUT_GALLERY_DIR),
    ]
    if USE_ARCFACE_PREPROCESSING and model_name in {"arcface", "both"}:
        args.append("--arcface-preprocessing")
    run_step(args)


def train_models():
    run_step(
        [
            sys.executable,
            str(GALLERY_SCRIPTS_DIR / "train_identity_model.py"),
            "--model",
            "vgg1000",
        ]
    )
    run_step(
        [
            sys.executable,
            str(GALLERY_SCRIPTS_DIR / "train_identity_model.py"),
            "--model",
            "arcface",
        ]
    )
    run_step(
        [
            sys.executable,
            str(GALLERY_SCRIPTS_DIR / "train_identity_model.py"),
            "--model",
            "arcface",
            "--classifier",
            "knn",
            "--knn-k",
            "9",
        ]
    )


def evaluate_models():
    vgg1000_args = [
        sys.executable,
        str(PROBE_SCRIPTS_DIR / "predict_unknown_identity.py"),
        "--model",
        "vgg1000",
        "--match-mode",
        DEFAULT_MATCH_MODE,
        "--top-k",
        str(TOP_K),
    ]
    arcface_args = [
        sys.executable,
        str(PROBE_SCRIPTS_DIR / "predict_unknown_identity.py"),
        "--model",
        "arcface",
        "--match-mode",
        ARCFACE_MATCH_MODE,
        "--top-k",
        str(TOP_K),
    ]

    run_step(vgg1000_args)
    run_step(arcface_args)


def main():
    args = parse_args()

    if args.command == "prepare":
        build_reference_vectors(model_name=args.model)
        train_models()
        return

    if args.command == "train":
        train_models()
        return

    if args.command == "evaluate":
        evaluate_models()
        return

    if args.command == "all":
        build_reference_vectors(model_name="both")
        train_models()
        evaluate_models()


if __name__ == "__main__":
    main()
