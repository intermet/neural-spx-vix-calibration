import torch
import orthnet

KINDS = {
    "C": orthnet.Chebyshev,
    "C2": orthnet.Chebyshev2,
    "H": orthnet.Hermite,
    "H2": orthnet.Hermite2,
    "LA": orthnet.Laguerre,
    "LE": orthnet.Legendre,
    "LEN": orthnet.Legendre_Normalized,
    "M": None,
}


def Monomial(XY, degree):
    X, Y = XY[:, 0], XY[:, 1]
    P = []
    for k in range(degree + 1):
        P.append(torch.pow(X, k))
    for k in range(1, degree + 1):
        P.append(torch.pow(Y, k))
    for n in range(degree + 1):
        for k in range(1, n):
            P.append(P[k] * P[degree + n - k])
    P = torch.stack(P, axis=-1)
    return P


def poly(XY, degree, k="M"):
    if k == "M":
        return Monomial(XY, degree)
    elif k in KINDS:
        return KINDS[k](XY, degree).tensor
    else:
        raise Exception("Unkown kind")
