# Copied from local_archive/articrafts/rally/oracle/sheets.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Review sheets for the oracle: every frame near a bounce, crop with a grid in 640x360 model pixels."""
import sys, json, numpy as np, cv2
video, ball_npz, out, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
wins = json.loads(sys.argv[5])          # {"name": [first, last]}
ball = np.load(ball_npz)["ball"]
cap = cv2.VideoCapture(video); W, H = int(cap.get(3)), int(cap.get(4))
padw = max(W, int(round(H * 16 / 9))); pad = (padw - W) // 2; s = 640 / padw
need = {f for a, b in wins.values() for f in range(a, b + 1, step)}
frames = {}
for i in range(max(need) + 1):
    ok, fr = cap.read()
    if i in need: frames[i] = fr
for name, (a, b) in wins.items():
    seg = ball[a:b + 1]
    cx, cy = np.nanmedian(seg[:, 0]), np.nanmedian(seg[:, 1])
    if not np.isfinite(cx): cx, cy = 320, 180
    x0, y0 = int(round(cx)) - 60, int(round(cy)) - 40      # 120 x 80 model px window
    tiles = []
    for f in range(a, b + 1, step):
        fr = frames[f]
        M = np.array([[1 / s, 0, -pad], [0, 1 / s, 0]])   # model px -> original px
        ox0, oy0 = int((x0) / s - pad), int(y0 / s)
        crop = fr[max(oy0, 0):oy0 + int(80 / s), max(ox0, 0):ox0 + int(120 / s)]
        crop = cv2.resize(crop, (480, 320), interpolation=cv2.INTER_CUBIC)
        k = 480 / 120
        for g in range(0, 121, 10):
            cv2.line(crop, (int(g * k), 0), (int(g * k), 319), (90, 90, 90), 1)
            cv2.putText(crop, str(x0 + g), (int(g * k) + 1, 10), 0, 0.3, (255, 255, 255), 1)
        for g in range(0, 81, 10):
            cv2.line(crop, (0, int(g * k)), (479, int(g * k)), (90, 90, 90), 1)
            cv2.putText(crop, str(y0 + g), (1, int(g * k) - 1), 0, 0.3, (255, 255, 255), 1)
        if np.isfinite(ball[f, 0]):
            cv2.circle(crop, (int((ball[f, 0] - x0) * k), int((ball[f, 1] - y0) * k)), 10, (0, 0, 255), 1)
        cv2.putText(crop, f"{f}", (400, 310), 0, 0.6, (0, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{out}/{name}.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(name, a, b, "window x0,y0", x0, y0)
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/rally/oracle"   # inputs and results of this study (local archive)
