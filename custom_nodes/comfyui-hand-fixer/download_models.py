"""Fetch the third-party files the nodes need into ComfyUI/models/hand_fixer/ (or --models-dir).

    python custom_nodes/comfyui-hand-fixer/download_models.py            # hand detector + HADM code + HADM-L weights
    python custom_nodes/comfyui-hand-fixer/download_models.py --no-hadm  # only the hand detector (CLIP student backend)

Nothing here is redistributed by this package:
* hand_yolov8s.pt   - ADetailer hand detector, Bingsu/adetailer on Hugging Face (Apache-2.0)
* hadm/HADM/        - git clone of github.com/wangkaihong/HADM at a pinned commit (no LICENSE file in that repo)
* hadm/HADM-L_0249999.pth - 5.7 GB, from the HADM authors' own download link (no license file; research use)
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hand_fixer.detector import ensure_detector  # noqa: E402
from hand_fixer.hadm_backend import ensure_repo, ensure_weights  # noqa: E402


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("HAND_FIXER_MODELS", str(HERE.parent.parent / "models" / "hand_fixer")))
    ap.add_argument("--no-hadm", action="store_true")
    ap.add_argument("--sha256", action="store_true", help="print sha256 of downloaded files")
    a = ap.parse_args()
    root = Path(a.models_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = [ensure_detector(root / "hand_yolov8s.pt", allow_download=True)]
    if not a.no_hadm:
        ensure_repo(root / "hadm", allow_clone=True)
        files.append(ensure_weights(root / "hadm", allow_download=True))
    for f in files:
        print(f, f.stat().st_size, sha256(f) if a.sha256 else "")


if __name__ == "__main__":
    main()
