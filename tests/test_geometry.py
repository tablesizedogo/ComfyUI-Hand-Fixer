import numpy as np

from hand_fixer import geometry as g


def test_square_region_matches_training_crop_rule():
    # 100x50 box centred at (150, 125): side 130, clipped ints
    assert g.square_region([100, 100, 200, 150], 1000, 1000) == [85.0, 60.0, 215.0, 190.0]
    # clipping at the border
    r = g.square_region([0, 0, 40, 40], 30, 30)
    assert r[0] == 0 and r[1] == 0 and r[2] == 30 and r[3] == 30


def test_region_score_uses_inside_fraction_not_iou():
    region = [0, 0, 100, 100]
    dets = [dict(box=[10, 10, 20, 20], score=0.9, cls="hand"),     # small box fully inside -> counts
            dict(box=[60, 60, 160, 160], score=0.95, cls="hand"),  # only 16% inside -> ignored
            dict(box=[0, 0, 50, 50], score=0.99, cls="face")]      # other class -> ignored
    assert g.region_artifact_score(dets, region) == 0.9
    assert g.region_artifact_score(dets, region, cls=None) == 0.99
    assert g.region_artifact_score([], region) == 0.0


def test_box_mask_grow_and_clip():
    m = g.box_mask(50, 50, [[10, 10, 20, 20]], grow=0.5)
    assert m.dtype == np.float32 and m.shape == (50, 50)
    assert m[5:25, 5:25].min() == 1.0 and m.sum() == 400
    assert g.box_mask(10, 10, [[0, 0, 10, 10]], grow=1.0).sum() == 100


def test_mask_overlaps():
    m = np.zeros((20, 20), np.float32); m[:10, :10] = 1
    assert g.mask_overlaps(m, [0, 0, 5, 5]) and not g.mask_overlaps(m, [12, 12, 20, 20])


def test_fit_to_centres_multiple_of_8_crop_and_resizes_otherwise():
    bg = np.zeros((21, 19, 3), np.float32); cand = np.ones((16, 16, 3), np.float32)
    out = g.fit_to(cand, 21, 19, bg)
    assert out.shape == (21, 19, 3) and out[2:18, 1:17].min() == 1 and out[0, 0, 0] == 0
    out2 = g.fit_to(np.ones((8, 8, 3), np.float32), 21, 19, bg)
    assert out2.shape == (21, 19, 3) and out2.min() > 0.99
