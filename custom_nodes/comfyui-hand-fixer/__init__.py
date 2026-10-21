"""
ComfyUI Hand Fixer
------------------
Find broken hands in generated images and keep the best inpainting attempt.

* Load Hand Defect Scorer  -> HAND_SCORER (ADetailer YOLOv8 hand detector + HADM-L or a light CLIP-L student)
* Hand Defect Scorer       -> mask of BAD hands, preview with P(bad) per hand, scores JSON, number of bad hands
* Hand Fix: Best-of-N      -> rescore N inpainted candidates per image and return the one with the lowest max P(bad)

See the repository README for installation, models and honest accuracy numbers.
"""
from .hand_fixer.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
