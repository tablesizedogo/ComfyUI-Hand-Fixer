"""Scoring an image and choosing the best inpainted candidate (pure numpy; used by the ComfyUI nodes and the tests)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import box_mask, fit_to, mask_overlaps, square_region


@dataclass
class HandResult:
    box: list
    det_conf: float
    region: list
    raw: float
    p_bad: float
    bad: bool


def score_image(rgb, hand_detector, scorer, det_conf=0.3, threshold=0.2, pad=1.3):
    """rgb: uint8 HxWx3. Returns a list of HandResult (one per detected hand)."""
    h, w = rgb.shape[:2]
    hands = hand_detector(rgb, conf=det_conf)
    regions = [square_region(b, w, h, pad) for b, _ in hands]
    raw, p = scorer.score(rgb, regions)
    return [HandResult(box=list(b), det_conf=c, region=r, raw=float(x), p_bad=float(q), bad=bool(q >= threshold))
            for (b, c), r, x, q in zip(hands, regions, raw, p)]


def results_json(results_per_image, scorer_name, threshold):
    return json.dumps(dict(scorer=scorer_name, threshold=threshold,
                           images=[[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(r).items()}
                                    for r in res] for res in results_per_image]))


def bad_mask(height, width, results, grow=0.15):
    return box_mask(height, width, [r.box for r in results if r.bad], grow)


def draw_preview(rgb, results):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 300))
    fs = max(12, round(max(im.size) / 45))
    try:
        font = ImageFont.load_default(size=fs)  # Pillow >= 10.1
    except TypeError:
        font, fs = ImageFont.load_default(), 11
    for r in results:
        col = (230, 40, 40) if r.bad else (40, 200, 70)
        d.rectangle(r.box, outline=col, width=lw)
        label = f"P(bad) {r.p_bad:.2f}"
        tw = d.textlength(label, font=font)
        top = max(0, r.box[1] - fs - 4)
        d.rectangle([r.box[0], top, r.box[0] + tw + 6, top + fs + 4], fill=col)
        d.text((r.box[0] + 3, top + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def composite(original, candidate, mask):
    """float images HxWx3 in [0,1]; mask HxW in [0,1]: take the candidate only inside the mask."""
    m = mask[..., None]
    return original * (1 - m) + candidate * m


def best_of_n(original, candidates, scorer, hand_detector, mask=None, det_conf=0.3, pad=1.3, include_original=True,
              composite_with_original=True):
    """original: float HxWx3 [0,1]; candidates: list of float hxwx3; mask: HxW or None.
    The scored regions are fixed from the ORIGINAL image (hands that overlap the mask, or all hands without a mask), so a
    candidate cannot win by erasing the hand. Candidate score = max P(bad) over those regions. Returns
    (best_image, best_index, scores) where index -1 = original."""
    H, W = original.shape[:2]
    orig_u8 = (np.clip(original, 0, 1) * 255).round().astype(np.uint8)
    hands = hand_detector(orig_u8, conf=det_conf)
    boxes = [b for b, _ in hands if mask is None or mask_overlaps(mask, b)]
    if not boxes and mask is not None and mask.max() > 0.5:  # mask present but no detected hand on it: use the mask bbox
        ys, xs = np.where(mask > 0.5)
        boxes = [[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]]
    regions = [square_region(b, W, H, pad) for b in boxes]
    pool = ([(-1, original)] if include_original else []) + list(enumerate(candidates))
    scores = []
    for idx, img in pool:
        img = fit_to(img, H, W, original)
        if composite_with_original and mask is not None and idx >= 0:
            img = composite(original, img, mask)
        if not regions:
            s = 0.0
        else:
            _, p = scorer.score((np.clip(img, 0, 1) * 255).round().astype(np.uint8), regions)
            s = max(p)
        scores.append(dict(index=idx, max_p_bad=float(s), image=img))
    best = min(scores, key=lambda d: (d["max_p_bad"], d["index"] >= 0))  # tie -> keep the original
    return best["image"], best["index"], [dict(index=d["index"], max_p_bad=round(d["max_p_bad"], 4)) for d in scores], regions
