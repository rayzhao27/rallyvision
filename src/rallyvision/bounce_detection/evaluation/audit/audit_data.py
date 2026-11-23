# Copied from local_archive/articrafts/event_transformer/audit/audit_data.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Independent data checks: split overlap, label alignment vs E2E-Spot's own frame labels, clip overlaps, event order."""
import json, re, collections
from pathlib import Path
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/event_transformer/audit"   # inputs and results of this study (local archive)
HF = DATA / "hf_labels"; E2E = REPO_ROOT / "local_archive/articrafts/e2e_tennis/labels"
def parse(cid):
    m = re.fullmatch(r"(?:Tennis_)?(.+)_(\d+)_(\d+)", cid); return m.group(1), int(m.group(2)), int(m.group(3))
hf = {s: json.loads((HF / f"annotations-localization-{s}.json").read_text())["data"] for s in ("train", "valid", "test")}
e2e = {s: {c["video"]: c for c in json.loads((E2E / f"{s}.json").read_text())} for s in ("train", "val", "test")}
out = {}
# 1. splits: matches and clip time ranges
src = {s: {parse(it["id"])[0] for it in v} for s, v in hf.items()}
out["matches"] = {s: len(v) for s, v in src.items()}
out["match_overlap"] = {"train&valid": len(src["train"] & src["valid"]), "train&test": len(src["train"] & src["test"]),
                        "valid&test": len(src["valid"] & src["test"])}
ranges = collections.defaultdict(list)
for s, v in hf.items():
    for it in v:
        m, a, b = parse(it["id"]); ranges[m].append((a, b, s))
cross = 0
for m, r in ranges.items():
    r.sort()
    for (a1, b1, s1), (a2, b2, s2) in zip(r, r[1:]):
        if a2 < b1 and s1 != s2: cross += 1
out["time_overlap_between_splits"] = cross
within = {s: 0 for s in hf}
for m, r in ranges.items():
    for (a1, b1, s1), (a2, b2, s2) in zip(r, r[1:]):
        if a2 < b1 and s1 == s2: within[s1] += 1
out["overlapping_clip_pairs_within_split"] = within
# 2. label alignment: HF position_ms -> frame vs E2E-Spot frame labels for the same clip
split_map = {"train": "train", "valid": "val", "test": "test"}
mism, total, missing = 0, 0, 0
for s, v in hf.items():
    for it in v:
        name = it["id"].removeprefix("Tennis_"); fps = float(it["inputs"][0]["fps"])
        ref = e2e[split_map[s]].get(name)
        if ref is None: missing += 1; continue
        a = sorted((int(round(int(e["position_ms"]) / 1000 * fps)), e["label"]) for e in it["events"])
        b = sorted((e["frame"], e["label"]) for e in ref["events"])
        total += len(b); mism += len(set(b) ^ set(a))
out["label_alignment"] = {"clips_not_in_e2e_same_split": missing, "e2e_events": total, "mismatched_events": mism}
# 3. backbone: E2E-Spot model trained on E2E train (validated on val); does any test match appear in E2E train/val?
e2e_trainval = {parse(k)[0] for s in ("train", "val") for k in e2e[s]}
out["test_matches_in_e2e_trainval"] = len(src["test"] & e2e_trainval)
# 4. event order in test labels: consecutive bounces without a swing/serve between (second bounces)
bb = tot = 0
for it in hf["test"]:
    ev = sorted(it["events"], key=lambda e: int(e["position_ms"]))
    for x, y in zip(ev, ev[1:]):
        tot += 1; bb += x["label"].endswith("bounce") and y["label"].endswith("bounce")
out["test_bounce_followed_by_bounce"] = f"{bb} of {tot} consecutive event pairs"
out["fps_values"] = sorted({round(float(it["inputs"][0]["fps"]), 3) for v in hf.values() for it in v})
print(json.dumps(out, indent=1)); DATA.joinpath("audit_data.json").write_text(json.dumps(out, indent=1))
