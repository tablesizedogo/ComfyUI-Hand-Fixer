import json

import numpy as np

from fakes import FakeHands, RedArtifactHADM, image_with_hands
from hand_fixer import pipeline
from hand_fixer.calibration import HADMCalibration, StudentCalibration
from hand_fixer.scorers import HADMScorer

HANDS = [([10, 10, 30, 30], 0.9), ([60, 20, 80, 40], 0.8), ([40, 50, 50, 60], 0.1)]


def test_calibrations_are_monotone_and_bounded():
    h = HADMCalibration(); s = StudentCalibration()
    xs = [h(x) for x in (0, 0.3, 0.6, 0.9)]
    assert all(0 < a < b < 1 for a, b in zip(xs, xs[1:]))
    assert s(0.9) < s(0.5) < s(0.1)


def test_score_image_flags_only_the_artifact_hand_and_respects_det_conf():
    img = image_with_hands(red_boxes=[(10, 10, 30, 30)])
    scorer = HADMScorer(RedArtifactHADM(HANDS))
    res = pipeline.score_image(img, FakeHands(HANDS), scorer, det_conf=0.3, threshold=0.2)
    assert len(res) == 2                      # the 0.1-confidence box is dropped
    assert res[0].bad and not res[1].bad
    assert res[0].raw == 1.0 and res[1].raw == 0.0
    m = pipeline.bad_mask(64, 96, res, grow=0.0)
    assert m[10:30, 10:30].min() == 1 and m.sum() == 400
    js = json.loads(pipeline.results_json([res], "hadm", 0.2))
    assert js["images"][0][0]["bad"] is True and js["scorer"] == "hadm"
    prev = pipeline.draw_preview(img, res)
    assert prev.shape == img.shape and not np.array_equal(prev, img)


def test_best_of_n_picks_the_clean_candidate_and_never_rewards_erasing():
    orig_u8 = image_with_hands(red_boxes=[(10, 10, 30, 30)])
    orig = orig_u8.astype(np.float32) / 255
    still_bad = orig.copy()
    half = orig.copy(); half[10:20, 10:30] = 0.5          # half the red removed
    fixed = orig.copy(); fixed[10:30, 10:30] = 0.5         # all red removed
    mask = np.zeros((64, 96), np.float32); mask[5:35, 5:35] = 1
    scorer = HADMScorer(RedArtifactHADM(HANDS))
    best, idx, scores, regions = pipeline.best_of_n(orig, [still_bad, half, fixed], scorer, FakeHands(HANDS), mask=mask)
    assert idx == 2 and np.allclose(best, fixed)
    assert [s["index"] for s in scores] == [-1, 0, 1, 2]
    assert len(regions) == 1                                # only the hand under the mask is scored
    # all candidates equally bad -> the original is kept
    _, idx2, _, _ = pipeline.best_of_n(orig, [still_bad], scorer, FakeHands(HANDS), mask=mask)
    assert idx2 == -1


def test_best_of_n_composites_only_inside_mask():
    orig = image_with_hands().astype(np.float32) / 255
    cand = np.zeros_like(orig)
    mask = np.zeros((64, 96), np.float32); mask[0:8, 0:8] = 1
    scorer = HADMScorer(RedArtifactHADM([]))
    best, idx, _, _ = pipeline.best_of_n(orig, [cand], scorer, FakeHands([]), mask=mask, include_original=False)
    assert idx == 0 and best[:8, :8].max() == 0 and np.allclose(best[20:, 20:], orig[20:, 20:])
