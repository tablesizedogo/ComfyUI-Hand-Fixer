"""Hand-quality scorers. Each returns, for a list of square regions in one image, (raw scores, calibrated P(bad)).

* HADMScorer (recommended): HADM-L hand-artifact detections inside the region; gold-check AUC 0.76 on the independent
  round (8 BAD), 0.72 pooled (17 BAD). Needs a GPU with ~6+ GB free (EVA-02-L at 1024 px) and the 5.7 GB weights.
* ClipStudentScorer (lightweight fallback, clearly weaker): frozen OpenCLIP ViT-L/14 embedding of the crop + our linear
  probe (ONNX). Gold AUC 0.59 (independent round) / 0.61 pooled - barely better than chance on subtle defects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import HADMCalibration, StudentCalibration
from .geometry import region_artifact_score

MODELS = Path(__file__).resolve().parent.parent / "models"


class HADMScorer:
    name = "hadm"

    def __init__(self, detector, calibration=None, min_inside=0.5):
        self.detector = detector  # callable(rgb) -> detections
        self.cal = calibration or HADMCalibration()
        self.min_inside = min_inside

    def score(self, rgb, regions):
        if not regions:
            return [], []
        dets = self.detector(rgb)
        raw = [region_artifact_score(dets, r, "hand", self.min_inside) for r in regions]
        return raw, [self.cal(x) for x in raw]


class ClipStudentScorer:
    name = "clip_l14_student"

    def __init__(self, device="cpu", onnx_path=MODELS / "hand_ok_clip_l14_logreg.onnx", calibration=None):
        import onnxruntime as ort
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
        self.model.eval().to(device)
        self.device = device
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.cal = calibration or StudentCalibration()

    def score(self, rgb, regions):
        if not regions:
            return [], []
        from PIL import Image
        im = Image.fromarray(rgb)
        crops = []
        for r in regions:
            c = im.crop(tuple(int(v) for v in r))
            c.thumbnail((448, 448))  # same as the training crops
            crops.append(self.pre(c))
        with self.torch.no_grad():
            f = self.model.encode_image(self.torch.stack(crops).to(self.device)).float()
            f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: f})
        p_ok = [float(p[1]) for p in out[1]]
        return p_ok, [self.cal(p) for p in p_ok]
