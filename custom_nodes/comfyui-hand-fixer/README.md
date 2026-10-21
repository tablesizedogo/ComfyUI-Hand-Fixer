# comfyui-hand-fixer

This folder is the node pack itself: copy it into `ComfyUI/custom_nodes/`, then run
`pip install -r requirements.txt` and `python download_models.py`.

Contents:
- `hand_fixer/nodes.py`: the three ComfyUI nodes;
- `pipeline.py` / `geometry.py`: scoring, masks and Best-of-N logic, with no ComfyUI imports;
- `hadm_backend.py`: HADM-L on torch 2.x via runtime shims;
- `detector.py`: the YOLOv8 hand detector;
- `scorers.py` and `calibration.py`: the two backends and their Platt calibrations;
- `models/`: the small files we ship (calibration JSONs, our 8 KB CLIP-L logistic-regression ONNX).

Full documentation, honest accuracy numbers and licenses are in the [repository README](../../README.md).
