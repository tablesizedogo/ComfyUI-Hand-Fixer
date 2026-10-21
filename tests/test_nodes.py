import json

import numpy as np
import pytest
import torch

from fakes import FakeHands, RedArtifactHADM, image_with_hands
from hand_fixer import nodes
from hand_fixer.scorers import HADMScorer

HANDS = [([10, 10, 30, 30], 0.9), ([60, 20, 80, 40], 0.8)]


@pytest.fixture
def bundle():
    return nodes.HandScorerBundle(FakeHands(HANDS), HADMScorer(RedArtifactHADM(HANDS)))


def test_node_mappings():
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    for cls in nodes.NODE_CLASS_MAPPINGS.values():
        assert "required" in cls.INPUT_TYPES() and hasattr(cls, cls.FUNCTION)
        assert len(cls.RETURN_TYPES) == len(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES))


def test_scorer_node_batch(bundle):
    imgs = np.stack([image_with_hands(red_boxes=[(10, 10, 30, 30)]), image_with_hands()]).astype(np.float32) / 255
    out = nodes.HandDefectScorer().score(torch.from_numpy(imgs), bundle, 0.2, 0.3, 0.0)
    mask, prev, js, n_bad = out["result"]
    assert out["ui"]["text"] == [js]
    assert mask.shape == (2, 64, 96) and prev.shape == (2, 64, 96, 3) and n_bad == 1
    assert mask[0].sum() == 400 and mask[1].sum() == 0
    assert len(json.loads(js)["images"]) == 2


def test_best_of_n_node_groups_candidates(bundle):
    o1 = image_with_hands(red_boxes=[(10, 10, 30, 30)]).astype(np.float32) / 255
    o2 = image_with_hands(red_boxes=[(60, 20, 80, 40)]).astype(np.float32) / 255
    f1 = o1.copy(); f1[10:30, 10:30] = 0.5
    f2 = o2.copy(); f2[20:40, 60:80] = 0.5
    orig = torch.from_numpy(np.stack([o1, o2]))
    cands = torch.from_numpy(np.stack([o1, f1, f2, o2]))     # image 0: [bad, fixed]; image 1: [fixed, bad]
    best, rep, idx0 = nodes.HandFixBestOfN().pick(orig, cands, bundle, 2, True, True, 0.3, mask=None)["result"]
    r = json.loads(rep)["results"]
    assert [x["best_index"] for x in r] == [1, 0] and idx0 == 1
    assert torch.allclose(best[0], torch.from_numpy(f1)) and torch.allclose(best[1], torch.from_numpy(f2))
    with pytest.raises(ValueError):
        nodes.HandFixBestOfN().pick(orig, cands[:3], bundle, 2, True, True, 0.3)
