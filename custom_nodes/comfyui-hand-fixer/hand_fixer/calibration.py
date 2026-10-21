"""Score calibration fitted on the user-labelled gold check (see the main project's README)."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "models"


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class HADMCalibration:
    """p_bad = sigmoid(a * raw + b); raw = HADM-L hand-artifact score (0 if no detection)."""

    def __init__(self, path=HERE / "hadm_calibration.json"):
        c = json.load(open(path))
        self.a, self.b = c["a"], c["b"]

    def __call__(self, raw):
        return sigmoid(self.a * raw + self.b)


class StudentCalibration:
    """p_ok = sigmoid(a * logit(student_p_ok) + b) (Platt); returns p_bad = 1 - p_ok."""

    def __init__(self, path=HERE / "clip_l14_student_calibration.json"):
        c = json.load(open(path))["platt"]
        self.a, self.b = c["a"], c["b"]

    def __call__(self, p_ok_raw):
        return 1.0 - sigmoid(self.a * logit(p_ok_raw) + self.b)
