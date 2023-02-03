import torch
import torchsde
from math import log


class BrownianMotion(torch.nn.Module):
    sde_type = "ito"
    noise_type = "general"

    def __init__(self):
        super(BrownianMotion, self).__init__()

    def f(self, t, x):
        b = torch.zeros_like(x, device="cuda:0")
        return b

    def g(self, t, x):
        N = x.shape[0]
        ones = torch.ones([N, 1], device="cuda:0")
        zeros = torch.zeros([N, 1], device="cuda:0")
        V = torch.concat([ones, zeros, ones, zeros], axis=-1)
        V = V.reshape([N, 2, 2])
        return V

    def forward(self, t, bm):
        xy0 = torch.zeros(bm.shape, device="cuda:0")
        dt = bm.dt
        with torch.no_grad():
            return torchsde.sdeint(self, xy0, t, dt=dt, bm=bm)


class GeneratorIto(torch.nn.Module):
    sde_type = "ito"
    noise_type = "general"

    def __init__(self, dt, V_and_Mu, fSPX):
        super(GeneratorIto, self).__init__()
        self.V_and_Mu = V_and_Mu
        self.fSPX = fSPX
        self.dt = dt

    def f_and_g(self, t, XY):
        N = XY.shape[0]
        XY = XY[:, :2]
        tXY = torch.cat([t * torch.ones([N, 1], device="cuda:0"), XY], axis=1)
        V, MuY = self.V_and_Mu(tXY)
        Sigma2 = V[:, [0]] ** 2 + V[:, [1]] ** 2
        zeros = torch.zeros([N, 1], device="cuda:0")
        W = torch.concat([V, zeros, zeros], axis=-1)
        V = W.reshape([N, 3, 2])
        fwd = log(self.fSPX(t) / self.fSPX(t - self.dt)) / self.dt
        Mu = torch.concat([fwd - 0.5 * Sigma2, MuY, 0.5 * Sigma2], axis=-1)
        return Mu, V

    def forward(self, t, xy0, bm=None):
        # print(f"t_start = {t_start} t_end = {t_end}")
        batch_size = xy0.shape[0]
        xyr0 = torch.concat(
            [xy0, torch.zeros([batch_size, 1], device="cuda:0")], axis=-1
        )
        if bm is None:
            t_start = t[0]
            t_end = t[-1]
            # print(f"t_start = {t_start} t_end = {t_end + 2 * self.dt,}")
            bm = torchsde.BrownianInterval(
                t0=t_start, t1=t_end, dt=self.dt, size=[batch_size, 2], device="cuda:0"
            )

        xyr = torchsde.sdeint(self, xyr0, t, dt=self.dt, bm=bm)
        BM = BrownianMotion()(t, bm)
        xy = xyr[:, :, :2]
        r = xyr[:, :, [2]]
        assert torch.equal(xy[0], xy0)
        return t, xy, r, BM
