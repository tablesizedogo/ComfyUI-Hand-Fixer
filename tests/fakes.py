"""Small deterministic stand-ins for the hand detector and the HADM network."""
import numpy as np


class FakeHands:
    """Returns fixed hand boxes (box, conf)."""

    def __init__(self, hands):
        self.hands = hands

    def __call__(self, rgb, conf=0.3):
        return [(b, c) for b, c in self.hands if c >= conf]


class RedArtifactHADM:
    """Fake HADM: reports a 'hand' artifact over every hand box whose pixels are mostly red, score = red fraction."""

    def __init__(self, hands):
        self.hands = hands

    def __call__(self, rgb):
        out = []
        for b, _ in self.hands:
            x0, y0, x1, y1 = (int(v) for v in b)
            patch = rgb[y0:y1, x0:x1].astype(int)
            red = float(((patch[..., 0] > 200) & (patch[..., 1] < 80)).mean())
            if red > 0:
                out.append(dict(box=[x0 + 2, y0 + 2, x1 - 2, y1 - 2], score=red, cls="hand"))
        out.append(dict(box=[0, 0, 5, 5], score=0.99, cls="face"))  # ignored class
        return out


def image_with_hands(h=64, w=96, red_boxes=()):
    img = np.full((h, w, 3), 128, np.uint8)
    for x0, y0, x1, y1 in red_boxes:
        img[y0:y1, x0:x1] = (255, 0, 0)
    return img
