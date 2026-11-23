"""RallyVision temporal event model (our own, trained here): Transformer over per-frame backbone features.

Classes (E2E-Spot class.txt order, 0 = background): far_court_bounce, far_court_swing, far_court_serve,
near_court_bounce, near_court_swing, near_court_serve.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

CLASSES = ["far_court_bounce", "far_court_swing", "far_court_serve",
           "near_court_bounce", "near_court_swing", "near_court_serve"]


class EventTransformer(nn.Module):
    def __init__(self, in_dim=768, d_model=128, layers=3, heads=4, ff=256, window=100, n_cls=7, dropout=0.1,
                 feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)              # drops input feature values (training only)
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))

    def forward(self, x):                                   # [B, T, in_dim] -> logits [B, T, n_cls]
        return self.head(self.encoder(self.inp(self.feat_drop(x)) + self.pos[:, :x.shape[1]]))


def load_event_transformer(checkpoint: Path, device="cpu"):
    """The saved model (eval mode), its training config and its val-chosen decoding {group: {thr, sep}}."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    c = ck["config"]
    model = EventTransformer(ck["in_dim"], c["d_model"], c["layers"], c["heads"], c["ff"], c["window"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, c, ck.get("decoding")
