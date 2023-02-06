import argparse
import json
import os
import torch
import time
import checkpoint
import importlib
from smile import Datetime
from data import load_dataset
from generator import GeneratorIto
from train_utils import train
from train_utils import wrap_loss_func
from train_utils import prepare_data
from train_utils import checkpoint_func

parser = argparse.ArgumentParser(prog="SPX-VIX joint neural calibration")
parser.add_argument("--date", type=str)
parser.add_argument("--maturities", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--params", type=str)

args = parser.parse_args()
date = args.date

with open(args.maturities) as f:
    maturities = json.load(f)

with open(args.params) as f:
    params = json.load(f)

my_model = importlib.import_module(args.model)

data = load_dataset(date)
smiles, maturities, fSPX = prepare_data(data, date, maturities)

dt = torch.tensor(params["dt"] / 365)
GEN = GeneratorIto(dt, my_model.V_AND_MUY(my_model.MODEL), fSPX)

model, epoch = checkpoint.load_last_checkpoint(my_model.MODEL)
_, T0, loss_func = wrap_loss_func(date, params, smiles, maturities, GEN, dt, fSPX)
train(my_model.MODEL, epoch, loss_func, checkpoint_func, save_step=100)
