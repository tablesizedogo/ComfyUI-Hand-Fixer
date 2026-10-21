# Changelog

## 0.1.0 (unreleased)

- `Load Hand Defect Scorer`: ADetailer `hand_yolov8s` hand boxes + HADM-L hand-artifact score (calibrated P(bad)),
  or a light CLIP-L/14 linear-probe student (weaker).
- `Hand Defect Scorer`: MASK of bad hands, preview with P(bad) per hand, scores JSON, number of bad hands.
- `Hand Fix: Best-of-N Selector`: rescore N inpainted candidates on regions fixed from the original image and keep the
  best one (the original wins ties).
- Example API workflow with stock ComfyUI inpaint nodes (SD1.5 inpainting checkpoint).
- `download_models.py` for the third-party files; HADM is cloned at a pinned commit and run on torch 2.x via runtime shims.
