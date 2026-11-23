# Copied from local_archive/articrafts/motion_comp/court_pass.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Court model + homography on every frame of a video (same steps as run_rally pass 1); saves H per frame.

    python local_archive/articrafts/motion_comp/court_pass.py VIDEO OUT.npz
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/motion_comp"   # inputs and results of this study (local archive)
REPO = REPO_ROOT
from rallyvision.bounce_detection.experiments.gbdt.rally_gbdt import pad_16x9
from rallyvision.court_detection import INPUT_H, INPUT_W, decode_keypoints, fit_homography, load_court_model

video, out = Path(sys.argv[1]), Path(sys.argv[2])
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, _ = load_court_model(REPO / "local_archive/articrafts/court/court_v2/best.pt", device)
cap = cv2.VideoCapture(str(video))
H, valid, reason, err = [], [], [], []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    padded, _ = pad_16x9(frame)
    rgb = cv2.cvtColor(cv2.resize(padded, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    with torch.no_grad():
        hm = model.predict_heatmap(x[None].to(device))[0].cpu().numpy()
    c = fit_homography(decode_keypoints(hm, 0.2)[0])
    H.append(c["H"] if c["H"] is not None else np.full((3, 3), np.nan))
    valid.append(c["valid"]); reason.append(c["reason"]); err.append(c["err_px"])
np.savez(out, H=np.array(H), valid=np.array(valid), reason=np.array(reason), err_px=np.array(err), fps=cap.get(cv2.CAP_PROP_FPS))
print(f"{len(H)} frames, valid {np.mean(valid):.1%}")
