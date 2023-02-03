import os
import datetime
import glob
import torch
import json
from smile import Smile


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Smile):
            return obj.to_json()
        else:
            return super().default(obj)


def save_checkpoint(epoch, loss, model):
    name = model["name"]
    state = {
        "loss": loss.item(),
        "epoch": epoch,
        **{k: m.state_dict() for k, m in model["nets"].items()},
        "optimizer": model["optimizer"].state_dict(),
    }
    time = datetime.datetime.now().strftime("%m%d-%H:%M:%S")
    path = f"checkpoints/{name}/{state['epoch']}-{time}-{state['loss']:.5f}.ckpt"
    torch.save(state, path)
    print(f"{path} saved. epoch : {epoch}, l : {state['loss']:.4f}")


def load(path, model):
    print(f"Loading {path}")
    state = torch.load(path, map_location="cuda:0")
    for k, m in state.items():
        if k not in ["loss", "epoch", "optimizer"]:
            model["nets"][k].load_state_dict(m)
    if model["optimizer"] is None:
        params = params_of_model(model)
        model["optimizer"] = torch.optim.Adam(params, 0.001)
    model["optimizer"].load_state_dict(state["optimizer"])
    loss = state["loss"]
    epoch = state["epoch"] + 1
    print(f"{path} loaded. epoch : {epoch}, l : {loss:.4f}")
    return model, epoch


def params_of_model(model):
    params = []
    for _, net in model["nets"].items():
        params += list(net.parameters())
    return params


def load_last_checkpoint(model):
    name = model["name"]
    checkpoints = sorted(glob.glob(f"checkpoints/{name}/*.ckpt"), key=os.path.getmtime)
    params = params_of_model(model)
    model["optimizer"] = torch.optim.Adam(params, 0.001)

    if len(checkpoints) == 0:
        return model, 1

    last = checkpoints[-1]
    return load(last, model)


# def load_model(model, device, load_last=None, path=None):
#     assert not (load_last and path)
#     params = []
#     for _, net in model.items():
#         params += list(net.parameters())
#     opt = torch.optim.Adam(params, 0.001)
#     if load_last:
#         load_last_checkpoint(model, opt, device, load_last)
#     else:
#         load(path, model, opt, device)
#     return opt
