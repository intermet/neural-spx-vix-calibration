import torch
from polynomial import poly


def tXYfy(t, XY, device):
    N = XY.shape[1]
    t = t[:, None, None]
    t = t * torch.ones([t.shape[0], N, 1])
    tXY = torch.cat([t.to(device), XY.to(device)], axis=-1)
    return tXY


def do_lstsq(A, R, method="approx"):
    if method == "approx":
        B = A.T @ A + 1e-6 * torch.eye(A.shape[1]).cuda()
        lstsq = torch.inverse(B) @ A.T @ R
    elif method == "qr":
        q, r = torch.linalg.qr(A)
        lstsq = torch.linalg.solve_triangular(r, q.T @ R, upper=True)
    elif method == "solve":
        B = A.T @ A
        try:
            lstsq = torch.linalg.solve(B, A.T @ R)
        except torch.linalg.LinAlgError:
            print("LinAlgError")
            B = A.T @ A + 1e-3 * torch.eye(A.shape[1]).cuda()
            lstsq = torch.inverse(B) @ A.T @ R
    else:
        lstsq = torch.linalg.lstsq(A, R).solution
    return lstsq


def compute_VIX2(XY12, R, degree, k="M"):
    if len(XY12.shape) == 3:
        X, Y = XY12[0, :, 0], XY12[0, :, 1]
        XY12 = XY12[0]
    elif len(XY12.shape) == 2:
        X, Y = XY12[:, 0], XY12[:, 1]
    else:
        raise Exception("Shape of XY12 should be of length 2 or 3")
    PSI = poly(XY12, degree, k)
    lstsq = do_lstsq(PSI, R, "qr")
    VIX2_lstsq = PSI @ lstsq
    VIX2_lstsq = VIX2_lstsq[VIX2_lstsq > 0]
    return VIX2_lstsq.unsqueeze(-1)


def compute_VIX2_nested(gen12, T1, T2, TAU, XY1, sub_batch_size, batch_size_nested, k):
    R_nested = torch.zeros([k, sub_batch_size, 1], device="cuda:0")
    n = batch_size_nested // k
    XY1_nested = XY1[:sub_batch_size].repeat(k, 1)
    with torch.no_grad():
        for j in range(n):
            T12, XY12, R, _ = gen12([T1.t, T2.t], XY1_nested, bm=None)
            R = R[-1]
            RR = 2 * R / TAU
            RR = RR.reshape([k, sub_batch_size, 1])
            R_nested += RR
            print(f"{j + 1}/{n}", end="\r")
        R_nested /= n
    VIX2 = R_nested.mean(axis=0)
    return VIX2
