import torch
import numpy as np
from smile import Smile
from bs import implied_vol
from scipy.interpolate import interp1d
from vix import compute_VIX2

def compute_smile_spx(maturity, xy, spot, spx_smiles):
    strikes = spx_smiles[maturity].strikes
    t0 = torch.tensor(0.)
    s = spot * xy[:, :1].exp()
    f = s.mean().detach()
    c = (s - strikes).relu().mean(axis=0)
    p = (strikes - s).relu().mean(axis=0)
    t = torch.tensor(maturity.t, device="cuda:0")
    i = implied_vol(f, t0, t, strikes, c, p).cuda()
    smile = Smile("SPX model", maturity, {
        "strikes" : strikes,
        "fwd" : f,
        "mids" : i,
        "asks" : i,
        "bids" : i,
        "calls" : c,
        "puts" : p,
    })
    return smile

def compute_smile_vix(maturity, XY, R, vix_smiles):
    VIX2 = compute_VIX2(XY, R, degree=8, k="M")
    strikes = vix_smiles[maturity].strikes
    VIX = 100 * VIX2.sqrt()
    fVIX = VIX.mean()
    t0 = torch.tensor(0.)
    CVIX = (VIX - strikes).relu().mean(axis=0)
    PVIX = (strikes - VIX).relu().mean(axis=0)
    t = torch.tensor(maturity.t, device="cuda:0")
    IVIX = implied_vol(fVIX, t0, t, strikes, CVIX, PVIX).cuda()
    smile = Smile("VIX model", maturity, {
        "strikes" : strikes,
        "fwd" : fVIX,
        "mids" : IVIX,
        "asks" : IVIX,
        "bids" : IVIX,
        "calls" : CVIX,
        "puts" : PVIX
    })
    return smile

def compute_loss_smile_spx(smile1, smile2):
    return ((smile1.mids / smile2.mids - 1) ** 2).mean()

def compute_loss_smile_vix(smile1, smile2):
    loss_future = (smile1.fwd / smile2.fwd - 1) ** 2
    strikes = smile1.strikes
    loss_calls = ((smile1.calls / smile2.calls - 1) ** 2).mean()
    return loss_future, loss_calls


def tensorize_smile(data, maturity, instrument, nb_points, alpha1, alpha2, device="cuda:0"):
    data = data[maturity]["smile"]
    ts = lambda array : torch.tensor(array, device=device, dtype=torch.float32)
    fwd = ts(data["fwd"])
    strikes = ts(data["strikes"])
    mids = ts(data["mids"])
    asks = ts(data["asks"])
    bids = ts(data["bids"])
    
    mask = torch.logical_and(alpha1 * fwd <= strikes, strikes <= alpha2 * fwd)
    strikes, mids, asks, bids = strikes[mask], mids[mask], asks[mask], bids[mask]
    
    m = strikes.shape[0]
    k = max(1, m // nb_points + 1)
    strikes, mids, asks, bids = strikes[::k], mids[::k], asks[::k], bids[::k]
    data = {
        "fwd" : fwd,
        "strikes" : strikes,
        "mids" : mids,
        "bids" : bids,
        "asks" : asks,
        "calls" : None,
        "puts" : None
    }
    return Smile(instrument, maturity, data)

def fwd_curve(data):
    spx_maturities = data["spx_maturities"]
    t = [maturity.t for maturity in spx_maturities]
    fwd = [data["spx_smiles"][maturity]["smile"]["fwd"] for maturity in spx_maturities]
    fSPX_func = interp1d(t, fwd, kind="cubic")
    fSPX = lambda t : float(fSPX_func(np.maximum(t.cpu(), 0.)))
    return fSPX

def yield_curve(data):
    t = data["yield_curve"]["days"] / 365
    rate = data["yield_curve"]["rate"]
    func = interp1d(t, rate, kind="cubic", fill_value="extrapolate")
    R = lambda t : float(func(np.maximum(t.cpu(), 0.)))
    return R
