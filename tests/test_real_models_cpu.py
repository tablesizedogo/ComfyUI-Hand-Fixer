"""Optional CPU tests with the real model code (skipped when the files are not available).
HAND_FIXER_TEST_HADM_REPO=<HADM clone>  -> builds HADM-L with random weights (checks the torch-2.x shims, ~30 s)
HAND_FIXER_TEST_YOLO=<hand_yolov8s.pt> and HAND_FIXER_TEST_IMAGE=<jpg> -> runs the real hand detector"""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.skipif(not os.environ.get("HAND_FIXER_TEST_HADM_REPO"), reason="no HADM clone given")
def test_hadm_builds_and_runs_on_cpu_with_random_weights(tmp_path):
    from hand_fixer.hadm_backend import HADMDetector
    (tmp_path / "HADM").symlink_to(Path(os.environ["HAND_FIXER_TEST_HADM_REPO"]).resolve())
    det = HADMDetector(tmp_path, device="cpu", random_init=True)
    out = det(np.full((256, 192, 3), 127, np.uint8))
    assert isinstance(out, list) and all(set(d) == {"box", "score", "cls"} for d in out)
    assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is None


@pytest.mark.skipif(not (os.environ.get("HAND_FIXER_TEST_YOLO") and os.environ.get("HAND_FIXER_TEST_IMAGE")), reason="no detector/image")
def test_real_hand_detector():
    from hand_fixer.detector import YoloHandDetector
    det = YoloHandDetector(os.environ["HAND_FIXER_TEST_YOLO"])
    hands = det(np.asarray(Image.open(os.environ["HAND_FIXER_TEST_IMAGE"]).convert("RGB")), conf=0.3)
    assert hands and all(len(b) == 4 and 0 < c <= 1 for b, c in hands)
