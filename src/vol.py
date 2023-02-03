import torch

Rho_min = torch.tensor(-0.99999, device="cuda:0")
Rho_max = torch.tensor(+0.99999, device="cuda:0")


def V_and_MuY_rho_tanh(model, alpha=1):
    def V_and_MuY(tXY):
        """
        SigmaX = 1 + tanh
        SigmaY = 1 + tanh
        rho = tanh
        muY = muY
        """
        batch_size = tXY.shape[0]
        zeros = torch.zeros([batch_size, 1], device="cuda:0")
        U = model["nets"]["phi"](tXY)
        SigmaX = 1 + torch.tanh(U[:, [0]])
        SigmaY = alpha * (1 + torch.tanh(U[:, [1]]))
        Rho = U[:, [2]]
        Rho = torch.tanh(U[:, [2]])
        Rho = torch.minimum(Rho, Rho_max)
        Rho = torch.maximum(Rho, Rho_min)
        RhoP = torch.sqrt(1 - Rho**2)
        MuY = U[:, [3]]
        V = torch.concat([SigmaX, zeros, Rho * SigmaY, RhoP * SigmaY], axis=-1)
        return V, MuY

    return V_and_MuY


def V_and_MuY_rho_exp(model):
    def V_and_MuY(tXY):
        """
        SigmaX = exp
        SigmaY = exp
        rho = tanh
        muY = muY
        """
        batch_size = tXY.shape[0]
        zeros = torch.zeros([batch_size, 1], device="cuda:0")
        U = model["nets"]["phi"](tXY)
        SigmaX = torch.exp(U[:, [0]])
        SigmaY = torch.exp(U[:, [1]])
        Rho = U[:, [2]]
        Rho = torch.tanh(U[:, [2]])
        Rho = torch.minimum(Rho, Rho_max)
        Rho = torch.maximum(Rho, Rho_min)
        RhoP = torch.sqrt(1 - Rho**2)
        MuY = U[:, [3]]
        V = torch.concat([SigmaX, zeros, Rho * SigmaY, RhoP * SigmaY], axis=-1)
        return V, MuY

    return V_and_MuY


def V_and_MuY_rho_softplus(model):
    def V_and_MuY(tXY):
        """
        SigmaX = softplus
        SigmaY = softplus
        rho = tanh
        muY = muY
        """
        batch_size = tXY.shape[0]
        zeros = torch.zeros([batch_size, 1], device="cuda:0")
        U = model["nets"]["phi"](tXY)
        SigmaX = torch.nn.functional.softplus(U[:, [0]])
        SigmaY = torch.nn.functional.softplus(U[:, [1]])
        Rho = torch.tanh(U[:, [2]])
        Rho = torch.minimum(Rho, Rho_max)
        Rho = torch.maximum(Rho, Rho_min)
        RhoP = torch.sqrt(1 - Rho**2)
        MuY = U[:, [3]]
        V = torch.concat([SigmaX, zeros, Rho * SigmaY, RhoP * SigmaY], axis=-1)
        return V, MuY

    return V_and_MuY
