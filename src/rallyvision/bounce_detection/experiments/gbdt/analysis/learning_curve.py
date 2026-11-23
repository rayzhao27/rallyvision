# Copied from local_archive/articrafts/bounce/analysis/learning_curve.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Learning curve for the bounce classifier, using train games only (val untouched).

For each held-out train game, fit on k of the other 6 games (several random subsets),
score the held-out game, and pool tp/fp/fn per k.
"""
import itertools, json, random, sys
from pathlib import Path
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/bounce/analysis"   # inputs and results of this study (local archive)
REPO = REPO_ROOT
from rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce import VAL_GAMES, counts, fit_gbdt, load_clips, prf, REPO_ROOT  # noqa: E402
from rallyvision.common.cli import load_config  # noqa: E402

clips = load_clips([REPO_ROOT / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv",
                    REPO_ROOT / "local_archive/articrafts/diag/eval_t018/val_t018_per_sample.csv"],
                   REPO_ROOT / "local_archive/articrafts/bounce_inputs/labels", load_config())
train = [c for c in clips if c["game"] not in VAL_GAMES]
games = sorted({c["game"] for c in train})
rng = random.Random(0)
out = {}
for k in range(2, 7):
    tot = [0, 0, 0]
    for held in games:
        others = [g for g in games if g != held]
        subsets = list(itertools.combinations(others, k))
        for sub in rng.sample(subsets, min(3, len(subsets))):
            scorer, t = fit_gbdt([c for c in train if c["game"] in sub], "pred")
            tp, fp, fn = counts([c for c in train if c["game"] == held], "pred", scorer, t)
            tot = [tot[0] + tp, tot[1] + fp, tot[2] + fn]
    out[k] = prf(*tot)
    print(k, [round(v, 3) for v in out[k]], flush=True)
DATA.joinpath("learning_curve.json").write_text(json.dumps(out, indent=2))
