import torch
from vol import V_and_MuY_rho_tanh

MODEL = {
    "nets" : {
        "phi" : torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 4)).to("cuda:0"),
    },
    "name" : "spx_vix",
    "optimizer" : None
}

V_AND_MUY = V_and_MuY_rho_tanh
