"""Reference causal chunk inference. Carries raw feature history between calls."""

import torch


class EnvelopeStream:
    def __init__(self, model):
        self.model = model.eval()
        self.history = None

    def reset(self):
        self.history = None

    @torch.no_grad()
    def predict(self, features):
        if features.ndim != 3 or features.shape[1] != 5 or features.shape[2] == 0:
            raise ValueError("Expected nonempty [batch, 5, time] feature tensor")
        if self.history is not None and features.shape[:2] != self.history.shape[:2]:
            raise ValueError("Reset the stream before changing batch dimensions")
        joined = features if self.history is None else torch.cat((self.history, features), dim=-1)
        result = self.model(joined)[:, -features.shape[-1] :]
        self.history = joined[:, :, -28:].detach().clone()
        return result
