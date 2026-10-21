"""ComfyUI nodes. IMAGE tensors are [B, H, W, 3] float in [0, 1]; MASK tensors are [B, H, W]."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import pipeline

_CACHE = {}


def models_dir() -> Path:
    env = os.environ.get("HAND_FIXER_MODELS")
    if env:
        return Path(env)
    try:
        import folder_paths
        return Path(folder_paths.models_dir) / "hand_fixer"
    except ImportError:
        return Path.home() / ".cache" / "hand_fixer"


def _device(choice):
    import torch
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"


class HandScorerBundle:
    """What travels on the HAND_SCORER socket: a hand detector + a quality scorer."""

    def __init__(self, hand_detector, scorer):
        self.hand_detector, self.scorer = hand_detector, scorer

    @property
    def name(self):
        return self.scorer.name


def build_bundle(backend, device, detector_file, allow_download):
    from .detector import YoloHandDetector, ensure_detector
    from .scorers import ClipStudentScorer, HADMScorer
    root = models_dir()
    dev = _device(device)
    det = YoloHandDetector(ensure_detector(root / detector_file, allow_download), device="cpu" if dev == "mps" else dev)
    if backend == "hadm":
        from .hadm_backend import HADMDetector
        scorer = HADMScorer(HADMDetector(root / "hadm", device=dev, allow_download=allow_download))
    else:
        scorer = ClipStudentScorer(device=dev)
    return HandScorerBundle(det, scorer)


class LoadHandDefectScorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "backend": (["hadm", "clip_l14_student (weaker, light)"], {"default": "hadm",
                        "tooltip": "hadm: HADM-L artifact detector (recommended, ~6+ GB VRAM, 5.7 GB weights). "
                                   "clip_l14_student: frozen CLIP ViT-L/14 + linear probe; much weaker on subtle defects."}),
            "device": (["auto", "cuda", "cuda:0", "cuda:1", "cpu"], {"default": "auto"}),
            "hand_detector": ("STRING", {"default": "hand_yolov8s.pt", "tooltip": "file in models/hand_fixer/"}),
            "allow_download": ("BOOLEAN", {"default": False,
                               "tooltip": "download missing files: HADM code (git), HADM-L weights (5.7 GB, authors' link, "
                                          "no license file - research use) and the ADetailer hand detector"}),
        }}

    RETURN_TYPES = ("HAND_SCORER",)
    FUNCTION = "load"
    CATEGORY = "image/hand fixer"

    def load(self, backend, device, hand_detector, allow_download):
        key = (backend.split()[0], device, hand_detector)
        if key not in _CACHE:
            _CACHE[key] = build_bundle(key[0], device, hand_detector, allow_download)
        return (_CACHE[key],)


def _to_u8(img_t):
    return (img_t.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


class HandDefectScorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "hand_scorer": ("HAND_SCORER",),
            "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "a hand is BAD if calibrated P(bad) >= threshold (corpus base rate ~0.10)"}),
            "det_conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.01,
                         "tooltip": "hand detector confidence; low values also pick up non-hands"}),
            "mask_grow": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "enlarge each BAD hand box by this fraction on every side"}),
        }}

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("bad_hands_mask", "preview", "scores_json", "n_bad")
    FUNCTION = "score"
    CATEGORY = "image/hand fixer"

    def score(self, image, hand_scorer, threshold, det_conf, mask_grow):
        import torch
        masks, previews, allres = [], [], []
        for b in range(image.shape[0]):
            rgb = _to_u8(image[b])
            res = pipeline.score_image(rgb, hand_scorer.hand_detector, hand_scorer.scorer, det_conf, threshold)
            allres.append(res)
            masks.append(pipeline.bad_mask(rgb.shape[0], rgb.shape[1], res, mask_grow))
            previews.append(pipeline.draw_preview(rgb, res).astype(np.float32) / 255.0)
        n_bad = int(sum(r.bad for res in allres for r in res))
        js = pipeline.results_json(allres, hand_scorer.name, threshold)
        print(f"[hand_fixer] scores: {js}", flush=True)
        return {"ui": {"text": [js]},
                "result": (torch.from_numpy(np.stack(masks)), torch.from_numpy(np.stack(previews)), js, n_bad)}


class HandFixBestOfN:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",),
            "candidates": ("IMAGE", {"tooltip": "inpainted candidates; with B originals, pass B*N images grouped per original"}),
            "hand_scorer": ("HAND_SCORER",),
            "candidates_per_image": ("INT", {"default": 0, "min": 0, "max": 64,
                                     "tooltip": "N; 0 = all candidates belong to each original"}),
            "include_original": ("BOOLEAN", {"default": True, "tooltip": "keep the original if no candidate scores better"}),
            "composite_with_original": ("BOOLEAN", {"default": True,
                                        "tooltip": "paste each candidate into the original inside the mask before scoring"}),
            "det_conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.01}),
        }, "optional": {"mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("best", "report_json", "best_index")
    FUNCTION = "pick"
    CATEGORY = "image/hand fixer"

    def pick(self, original, candidates, hand_scorer, candidates_per_image, include_original, composite_with_original,
             det_conf, mask=None):
        import json
        import torch
        B = original.shape[0]
        n = candidates_per_image or candidates.shape[0]
        if candidates_per_image == 0 and B > 1:
            raise ValueError("with several originals set candidates_per_image")
        if candidates.shape[0] < B * n:
            raise ValueError(f"need {B * n} candidates, got {candidates.shape[0]}")
        outs, report, first_idx = [], [], -1
        for b in range(B):
            orig = original[b].cpu().numpy().astype(np.float32)
            m = None
            if mask is not None:
                mm = mask[min(b, mask.shape[0] - 1)].cpu().numpy().astype(np.float32)
                m = mm if mm.shape == orig.shape[:2] else None
            cands = [candidates[b * n + i].cpu().numpy().astype(np.float32) for i in range(n)]
            best, idx, scores, regions = pipeline.best_of_n(orig, cands, hand_scorer.scorer, hand_scorer.hand_detector, m,
                                                            det_conf, include_original=include_original,
                                                            composite_with_original=composite_with_original)
            outs.append(best.astype(np.float32))
            report.append(dict(image=b, best_index=idx, scores=scores, regions=[[round(v, 1) for v in r] for r in regions]))
            if b == 0:
                first_idx = idx
        js = json.dumps(dict(scorer=hand_scorer.name, results=report))
        print(f"[hand_fixer] best-of-n: {js}", flush=True)
        return {"ui": {"text": [js]}, "result": (torch.from_numpy(np.stack(outs)), js, first_idx)}


NODE_CLASS_MAPPINGS = {
    "HandFixerLoadScorer": LoadHandDefectScorer,
    "HandFixerDefectScorer": HandDefectScorer,
    "HandFixerBestOfN": HandFixBestOfN,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HandFixerLoadScorer": "Load Hand Defect Scorer",
    "HandFixerDefectScorer": "Hand Defect Scorer",
    "HandFixerBestOfN": "Hand Fix: Best-of-N Selector",
}
