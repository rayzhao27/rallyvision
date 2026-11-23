# Copied from local_archive/articrafts/bounce/analysis/nms_whatif.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""What-if: keep only the highest-probability detection within ±r frames (train OOF, val untouched)."""
import json, sys
from pathlib import Path
import numpy as np
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/bounce/analysis"   # inputs and results of this study (local archive)
REPO = REPO_ROOT
from rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce import VAL_GAMES, TOL, PROB_GRID, by_game, fit_gbdt, load_clips, match_events, prf, REPO_ROOT  # noqa
from rallyvision.common.cli import load_config  # noqa

clips = load_clips([REPO_ROOT / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv",
                    REPO_ROOT / "local_archive/articrafts/diag/eval_t018/val_t018_per_sample.csv"],
                   REPO_ROOT / "local_archive/articrafts/bounce_inputs/labels", load_config())
train = [c for c in clips if c["game"] not in VAL_GAMES]

def nms(frames, scores, r):
    keep = []
    for i in np.argsort(-scores):
        if all(abs(frames[i] - frames[k]) > r for k in keep):
            keep.append(i)
    return sorted(frames[k] for k in keep)

# out-of-fold probabilities for every train clip (model never saw that clip's game)
oof = {}
for _, rest, held in by_game(train):
    scorer, _ = fit_gbdt(rest, "pred")
    oof.update({c["clip"]: scorer(c) for c in held})

out = {}
for r in (0, 3, 5, 8):
    best = (0, None, None)
    for t in PROB_GRID:   # best threshold on pooled OOF: same optimism for every r, fine for a relative comparison
        tot = np.zeros(3, int)
        for c in train:
            s = oof[c["clip"]]; f = c["frames"][c["pred"]["idx"]]; m = s > t
            dets = nms(f[m], s[m], r) if r else f[m].tolist()
            tot += match_events(dets, c["bounces"], TOL)
        p = prf(*tot)
        if p[2] > best[0]:
            best = (p[2], float(t), [round(v, 3) for v in p])
    out[r] = {"threshold": best[1], "precision_recall_f1": best[2]}
    print(r, out[r], flush=True)
DATA.joinpath("nms_whatif.json").write_text(json.dumps(out, indent=2))
