import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import erf
from math import sqrt


def callbs(S, t, T, K, sigma):
    sigma = torch.abs(sigma)
    tT = T - t
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    d2 = d1 - sigma * stT
    N1 = 0.5 * (1 + torch.erf(d1 / sqrt(2)))
    N2 = 0.5 * (1 + torch.erf(d2 / sqrt(2)))
    c = N1 * S - N2 * K
    return c


def putbs(S, t, T, K, sigma):
    sigma = torch.abs(sigma)
    tT = T - t
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    d2 = d1 - sigma * stT
    N1 = 0.5 * (1 + torch.erf(-d1 / sqrt(2)))
    N2 = 0.5 * (1 + torch.erf(-d2 / sqrt(2)))
    c = -N1 * S + N2 * K
    return c


def gammabs(S, t, T, K, sigma):
    tT = T - t
    tT = torch.reshape(tT, [t.size()[0], 1, 1])
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    g = torch.exp(-0.5 * d1 * d1) / sqrt(2 * np.pi) / S / sigma / stT
    return g


def deltabs(S, t, T, K, sigma):
    tT = T - t
    tT = torch.reshape(tT, [t.size()[0], 1, 1])
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    N = 0.5 * (1 + torch.erf(d1 / sqrt(2.0)))
    return N


def vegabs(S, t, T, K, sigma):
    tT = T - t
    stT = torch.sqrt(tT)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    v = torch.exp(-0.5 * d1 * d1) / sqrt(2 * np.pi) * S * stT
    return v


def callbs_np(S, t, T, K, sigma):
    tT = T - t
    stT = np.sqrt(tT)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    d2 = d1 - sigma * stT
    N1, N2 = 0.5 * (1 + erf([d1 / sqrt(2), d2 / sqrt(2)]))
    c = N1 * S - N2 * K
    return c


def putbs_np(S, t, T, K, sigma):
    tT = T - t
    stT = np.sqrt(tT)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tT) / sigma / stT
    d2 = d1 - sigma * stT
    N1, N2 = 0.5 * (1 + erf([-d1 / sqrt(2), -d2 / sqrt(2)]))
    c = -N1 * S + N2 * K
    return c


def implied_vol_np(S, t, T, K, C, cp_flag, nan=False):
    if cp_flag == "C":
        f = lambda sigma: callbs_np(S, t, T, K, sigma) - C
    else:
        f = lambda sigma: putbs_np(S, t, T, K, sigma) - C
    try:
        i = brentq(f, a=0.001, b=20)
        if nan and i == 0.001:
            i = np.nan
    except:
        i = np.nan
    return i


class ImpliedVolTorch_C(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, t, T, K, C):
        S_np, C_np = S.cpu().detach().numpy(), C.cpu().detach().numpy()
        K_np = K.cpu().detach().numpy()
        T_np = T.cpu().detach().numpy()
        t_np = t.cpu().detach().numpy()
        sigma = torch.tensor(implied_vol_np(S_np, t_np, T_np, K_np, C_np, cp_flag="C"))
        ctx.save_for_backward(sigma, S, t, T, K, C)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        sigma, S, t, T, K, C = ctx.saved_tensors
        grad_input = grad_output.clone()
        v = vegabs(S, t, T, K, sigma)
        g = grad_input / v
        if g.isnan():
            return (
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
            )
        else:
            return 0 * grad_input, 0 * grad_input, 0 * grad_input, 0 * grad_input, g


class ImpliedVolTorch_P(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, t, T, K, C):
        S_np, C_np = S.cpu().detach().numpy(), C.cpu().detach().numpy()
        K_np = K.cpu().detach().numpy()
        T_np = T.cpu().detach().numpy()
        t_np = t.cpu().detach().numpy()
        sigma = torch.tensor(implied_vol_np(S_np, t_np, T_np, K_np, C_np, cp_flag="P"))
        ctx.save_for_backward(sigma, S, t, T, K, C)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        sigma, S, t, T, K, C = ctx.saved_tensors
        grad_input = grad_output.clone()
        v = vegabs(S, t, T, K, sigma)
        g = grad_input / v
        if g.isnan():
            return (
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
                0 * grad_input,
            )
        else:
            return 0 * grad_input, 0 * grad_input, 0 * grad_input, 0 * grad_input, g


def implied_vol(S, t, T, K, C, P, device=None):
    IC = lambda k, c: ImpliedVolTorch_C.apply(S.detach().cpu(), t, T, k, c)
    IP = lambda k, p: ImpliedVolTorch_P.apply(S.detach().cpu(), t, T, k, p)
    I = [(IC(k, c) if k > S else IP(k, p)) for k, c, p in zip(K, C.cpu(), P.cpu())]
    I = torch.stack(I)
    return I.to(device)
