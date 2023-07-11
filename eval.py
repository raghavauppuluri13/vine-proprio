from typing import Dict, Any, List
import torch.nn.functional as F
import numpy as np
import cv2
import torch
from model import ProprioNet

class EvalProprioNet:
    """
    Uses trained NN to classify object faces
    execute() -> int (class)
    Params:
      - model_path (Required)
      - compute_type (default: 'cpu')
      - axes (Required)
    """

    # Parameter defaults
    _compute_type: str = "cuda"

    def __init__(self, model_path, state_dim, input_dim, compute_type):

        self._compute_type =  compute_type
        self._input_dim: List[str] = input_dim
        self._state_dim: int = state_dim

        self._device = torch.device(self._compute_type)
        self._model = ProprioNet(self._input_dim, state_dim)
        self._model.to(self._device)
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()
        torch.no_grad()

    def run(self, frame):
        frame = torch.from_numpy(np.float32(frame)).to(self._device)
        frame = torch.unsqueeze(frame, 0)
        frame = torch.permute(frame,(0,3,1,2))
        out: torch.Tensor = self._model(frame)
        out = torch.flatten(out)
        out = out.detach().cpu().numpy()
        assert out.shape[0] == self._state_dim, print(out.shape)
        return out
