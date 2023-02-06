import torch
from utils import compute_smile_spx, compute_smile_vix
from smile import Smile
from vix import compute_VIX2


def run_model_ta_tb(gen, spot, t, xy0, spx_smiles, bm):
    TAU = torch.tensor(30 / 365, device="cuda:0")
    _, XY, R, _ = gen(t, xy0, bm=bm)
    R = 2 * R / TAU
    return XY, R


def run_model01(gen01, xy0, data):
    T01, XY01, _, BM01 = gen01(data.T0, data.T1, xy0, trajectories=False)
    smile = compute_smile_spx(data.T1, XY01, data)
    X1 = XY01[-1, :, :1]
    BM1 = BM01[-1]
    return T01, X1, BM1, smile


def run_model12(gen12, F, X1, BM1, data):
    Y1 = F(BM1)
    XY1 = torch.cat([X1, Y1], axis=1)
    T12, XY12, R, BM12 = gen12(data.T1, data.T2, XY1, trajectories=False)
    R = 2 * R / data.TAU
    VIX2 = compute_VIX2(XY12, R, degree=8, k="M")
    smile_spx = compute_smile_spx(data.T2, XY12, data)
    VIX, smile_vix = compute_smile_vix(VIX2, data)
    return T12, XY12, R, VIX, smile_spx, smile_vix
