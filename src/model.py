import torch.nn as nn


class LaneChangeModel(nn.Module):
    """Feedforward ANN for binary lane-change classification.

    Input:  75 features (normalised)
    Output: raw logit — apply torch.sigmoid() externally for inference,
            or pass directly to BCEWithLogitsLoss during training.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(75, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)
