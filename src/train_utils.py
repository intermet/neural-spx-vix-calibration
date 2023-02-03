import torch
import time
import checkpoint
import torchsde
import json
import os
from smile import Datetime
from collections import OrderedDict
from vix import compute_VIX2
from run import run_model_ta_tb
from utils import compute_smile_spx
from utils import compute_smile_vix
from utils import compute_loss_smile_spx
from utils import compute_loss_smile_vix
from utils import tensorize_smile
from utils import fwd_curve
from smile import Datetime


def train(model, epoch, loss_func, checkpoint_func, save_step):
    opt = model["optimizer"]
    while True:
        opt.zero_grad(set_to_none=True)
        # with torch.cuda.amp.autocast():
        loss, output = loss_func(epoch)
        loss.backward()

        if epoch % save_step == 1:
            checkpoint_func(model, epoch, loss, output)

        opt.step()
        print(f"{epoch} loss={loss.item()}")
        epoch += 1


def wrap_loss_func(date, params, smiles, maturities, gen, dt, fSPX):
    batch_size = params["batch_size"]
    xy0 = torch.zeros([batch_size, 2], device="cuda:0")
    spx_maturities = maturities["spx"]
    vix_maturities = maturities["vix"]
    all_maturities = maturities["all"]
    spx_vix_maturities = maturities["spx_vix"]

    spx_smiles = smiles["spx"]
    vix_smiles = smiles["vix"]

    T0 = Datetime(date)
    T_last = max(all_maturities)
    bm = torchsde.BrownianInterval(
        t0=T0.t,
        t1=T_last.t + 2 * dt,
        dt=dt,
        size=[batch_size, 2],
        device="cuda:0",
        entropy=42,
    )

    spot = fSPX(torch.tensor(0.0))
    t = [0] + [maturity.t for maturity in all_maturities]

    w_fVIX, w_CVIX, w_SPX = params["w_fVIX"], params["w_CVIX"], params["w_SPX"]
    n_vix = len(vix_maturities)
    n_spx = len(spx_maturities)

    def loss_func(epoch, trajectories=False):
        XY, R = run_model_ta_tb(gen, spot, t, xy0, spx_smiles, bm)

        loss_dict = {
            "SPX": 0,
            "fVIX": 0,
            "CVIX": 0,
        }

        model_smiles = {maturity.string: {} for maturity in spx_vix_maturities}

        for i, maturity in enumerate(all_maturities):
            if maturity in spx_maturities:
                XYi = XY[i + 1]
                smile_spx = compute_smile_spx(maturity, XYi, spot, spx_smiles)
                model_smiles[maturity]["spx"] = {
                    "model": smile_spx,
                    "market": spx_smiles[maturity],
                }
                if maturity in spx_maturities[:2]:
                    loss_spx = compute_loss_smile_spx(smile_spx, spx_smiles[maturity], use_weights=True)
                else:
                    loss_spx = compute_loss_smile_spx(smile_spx, spx_smiles[maturity], use_weights=False)
                loss_dict["SPX"] += loss_spx
            if maturity in vix_maturities:
                XYi = XY[i + 1]
                j = all_maturities.index(maturity.plus_30d())
                Rij = R[j + 1] - R[i + 1]
                smile_vix = compute_smile_vix(maturity, XYi, Rij, vix_smiles)
                model_smiles[maturity]["vix"] = {
                    "model": smile_vix,
                    "market": vix_smiles[maturity],
                }
                loss_fVIX, loss_CVIX = compute_loss_smile_vix(
                    smile_vix, vix_smiles[maturity],
                    use_weights=False
                )
                loss_dict["fVIX"] += loss_fVIX
                loss_dict["CVIX"] += loss_CVIX

        loss = (
            w_fVIX * loss_dict["fVIX"] / n_vix
            + w_SPX * loss_dict["SPX"] / n_spx
            + w_CVIX * loss_dict["CVIX"] / n_vix
        )

        output = {
            "loss": float(loss.detach().cpu().numpy()),
            "time": time.time(),
            "epoch": epoch,
            "smiles": model_smiles,
            "XY": XY if trajectories else None,
            "R": R if trajectories else None,
        }

        return loss, output

    return t, T0, loss_func


def prepare_data(data, date, maturities):
    fSPX = fwd_curve(data)

    spx_maturities = sorted(
        [Datetime(maturity, T0=date) for maturity in maturities["spx"]]
    )
    vix_maturities = sorted(
        [Datetime(maturity, T0=date) for maturity in maturities["vix"]]
    )

    spx_smiles = OrderedDict(
        {
            maturity: tensorize_smile(
                #data["spx_smiles"], maturity, "SPX market", 40, 0.90, 1.05
                data["spx_smiles"], maturity, "SPX market", 40, None,  None
            )
            for maturity in spx_maturities
        }
    )

    vix_smiles = OrderedDict(
        {
            maturity: tensorize_smile(
                data["vix_smiles"], maturity, "VIX market", 40, 0.85, 1.85
            )
            for maturity in vix_maturities
        }
    )

    vix_maturities_plus30d = [maturity.plus_30d() for maturity in vix_maturities]
    spx_vix_maturities = sorted(set(spx_maturities + vix_maturities))
    all_maturities = sorted(
        set(spx_maturities + vix_maturities + vix_maturities_plus30d)
    )

    smiles = {"vix": vix_smiles, "spx": spx_smiles}

    maturities = {
        "spx": spx_maturities,
        "vix": vix_maturities,
        "spx_vix": spx_vix_maturities,
        "all": all_maturities,
    }
    return smiles, maturities, fSPX


def checkpoint_func(model, epoch, loss, output):
    name = model["name"]
    js = json.dumps(output, indent=4, cls=checkpoint.Encoder)
    path = f"checkpoints/{name}/{epoch}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(js)
    checkpoint.save_checkpoint(epoch, loss, model)
