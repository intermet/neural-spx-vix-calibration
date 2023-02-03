import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
from bs import callbs
from bs import putbs

plt.rcParams["figure.figsize"] = (9, 6)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Datetime:
    def __init__(self, string, T0=None):
        self.string = string
        self.datetime = pd.to_datetime(string)
        self.T0 = T0
        if T0:
            self.t = (self.datetime - pd.to_datetime(T0)).days / 365
            self.t = float(self.t)
        else:
            self.t = 0.0

    def __hash__(self):
        return self.string.__hash__()

    def plus_30d(self):
        tau = datetime.timedelta(days=30)
        return self.__class__((self.datetime + tau).strftime("%Y/%m/%d"), T0=self.T0)

    def __str__(self):
        return self.string

    def __repr__(self):
        return self.string.__repr__()

    def __lt__(self, other):
        if type(other) == Datetime:
            return self.datetime < other.datetime
        elif type(other) == str:
            return self < Datetime(other)
        else:
            raise Exception("Impossible comparaison.")

    def __le__(self, other):
        if type(other) == Datetime:
            return self.datetime <= other.datetime
        elif type(other) == str:
            return self <= Datetime(other)
        else:
            raise Exception("Impossible comparaison.")

    def __gt__(self, other):
        if type(other) == Datetime:
            return self.datetime > other.datetime
        elif type(other) == str:
            return self > Datetime(other)
        else:
            raise Exception("Impossible comparaison.")

    def __ge__(self, other):
        if type(other) == Datetime:
            return self.datetime >= other.datetime
        elif type(other) == str:
            return self >= Datetime(other)
        else:
            raise Exception("Impossible comparaison.")

    def __eq__(self, other):
        if type(other) == Datetime:
            return self.datetime == other.datetime
        elif type(other) == str:
            return self.string == other
        else:
            raise Exception("Impossible comparaison.")


class Smile:
    def __init__(self, instrument, maturity, data):
        self.name = f"{instrument}"
        self.maturity = maturity
        self.strikes = data["strikes"].clone()
        self.fwd = data["fwd"].clone()
        self.fwd_ask = data["fwd_ask"].clone()
        self.fwd_bid = data["fwd_bid"].clone()
        self.mids = data["mids"].clone()
        self.asks = data["asks"].clone()
        self.bids = data["bids"].clone()
        if data["calls"] is None:
            t = torch.tensor(maturity.t, device=self.fwd.device)
            self.calls = callbs(self.fwd, 0.0, t, self.strikes, self.mids)
            self.calls = callbs(self.fwd, 0.0, t, self.strikes, self.mids)
        else:
            self.calls = data["calls"]

        if data["puts"] is None:
            t = torch.tensor(maturity.t, device=self.fwd.device)
            self.puts = putbs(self.fwd, 0.0, t, self.strikes, self.mids)
        else:
            self.puts = data["puts"]

    def plot(self, color=None, legend=True):
        with torch.no_grad():
            plt.plot(
                self.strikes.cpu(),
                self.mids.cpu(),
                color=color,
                label=f"{self.name} mid implied vol.",
            )
            plt.scatter(
                self.strikes.cpu(), self.mids.cpu(), marker="x", s=20, color=color
            )

            plt.fill_between(
                self.strikes.cpu(),
                self.bids.cpu(),
                self.asks.cpu(),
                color=color,
                alpha=0.5,
                label="bid-ask",
            )

            plt.axvline(
                self.fwd.cpu(), color=color, linestyle="--", label=f"{self.name} future"
            )
            plt.axvline(
                self.fwd_ask.cpu(), color=color, linestyle="--", label=f"{self.name} future"
            )
            plt.axvline(
                self.fwd_bid.cpu(), color=color, linestyle="--", label=f"{self.name} future"
            )
            plt.xlabel("Strike")
            plt.ylabel("Implied volatility")
            if legend:
                plt.legend()
            plt.title(self.name)

    def to_json(self):
        data = {
            "name": self.name,
            "maturity": (self.maturity.T0, self.maturity.string),
            "strikes": self.strikes.detach().cpu().tolist(),
            "fwd": float(self.fwd.detach().cpu().numpy()),
            "fwd_bid": float(self.fwd_bid.detach().cpu().numpy()),
            "fwd_ask": float(self.fwd_ask.detach().cpu().numpy()),
            "mids": self.mids.cpu().detach().tolist(),
            "asks": self.asks.detach().cpu().tolist(),
            "bids": self.bids.detach().cpu().tolist(),
            "calls": self.bids.detach().cpu().tolist(),
            "puts": self.bids.detach().cpu().tolist(),
        }
        return data

    def from_json(js):
        name = js["name"]
        T0, maturity = js["maturity"]
        maturity = Datetime(maturity, T0)
        strikes = torch.tensor(
            js["strikes"],
        )
        fwd = torch.tensor(js["fwd"], device=device)
        fwd_bid = torch.tensor(js["fwd_bid"], device=device)
        fwd_ask = torch.tensor(js["fwd_ask"], device=device)
        mids = torch.tensor(js["mids"], device=device)
        asks = torch.tensor(js["asks"], device=device)
        bids = torch.tensor(js["bids"], device=device)
        calls = torch.tensor(js["calls"], device=device)
        puts = torch.tensor(js["puts"], device=device)
        return Smile(
            name,
            maturity,
            {
                "name": name,
                "strikes": strikes,
                "fwd": fwd,
                "fwd_bid": fwd_bid,
                "fwd_ask": fwd_ask,
                "mids": mids,
                "asks": asks,
                "bids": bids,
                "calls": calls,
                "puts": puts,
            },
        )
