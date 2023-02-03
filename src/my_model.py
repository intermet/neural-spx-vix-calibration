import torch
from vol import V_and_MuY_rho_tanh

MODEL = {
    "nets": {
        "phi": torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 4),
        ).to("cuda:0"),
    },
    "name": "spx_vix_small",
    "optimizer": None,
}

V_AND_MUY = V_and_MuY_rho_tanh
