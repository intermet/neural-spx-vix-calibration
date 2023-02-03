import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch
import json
import datetime
from bs import implied_vol_np
from bs import callbs_np
from smile import Datetime
from scipy.optimize import minimize

plt.style.use("seaborn")


tau = datetime.timedelta(days=30)


SPX = pd.read_csv("./data/csv/spx_bid_ask.csv").query("am_settlement == 0")
SPX.strike_price = SPX.strike_price / 1000

VIX = pd.read_csv("./data/csv/vix_bid_ask.csv")
VIX.strike_price = VIX.strike_price / 1000

YIELD_CURVE = pd.read_csv("./data/csv/yield-curve.csv")

def choose_fwd(asks_fwd, bids_fwd):
    def f(fwd):
        d = np.abs(asks_fwd - fwd) + np.abs(fwd - bids_fwd)
        return d.mean()
    
    fwd0 = 0.5 * (bids_fwd[0] + asks_fwd[0])

    fwd = minimize(f, x0=fwd0).x[0]

    bid = bids_fwd[bids_fwd < fwd].max()
    ask = asks_fwd[asks_fwd > fwd].min()

    return fwd, bid, ask 


def maturities_df(df):
    T0 = df[["impl_volatility", "date"]].dropna().date.drop_duplicates().values
    assert len(T0) == 1
    date_0 = str(T0[0])
    dates = df.exdate.drop_duplicates().values
    return [Datetime(d, T0=date_0) for d in dates]


def compute_vol(fwd, maturity, strikes, prices, cp_flag):
    asks_vol = np.array(
        [
            implied_vol_np(fwd, 0.0, maturity.t, k, c, cp_flag, nan=True)
            for k, c in zip(strikes, prices["asks"])
        ]
    )
    bids_vol = np.array(
        [
            implied_vol_np(fwd, 0.0, maturity.t, k, c, cp_flag, nan=True)
            for k, c in zip(strikes, prices["bids"])
        ]
    )
    mask = np.logical_or(np.isnan(asks_vol), np.isnan(bids_vol))
    strikes, asks_vol, bids_vol = strikes[~mask], asks_vol[~mask], bids_vol[~mask]
    return strikes, asks_vol, bids_vol


def extract_spx_smile(spx, maturity):
    df = spx.query(f"exdate == '{maturity}'")
    calls = df.query("cp_flag == 'C'")
    puts = df.query("cp_flag == 'P'")
    assert np.equal(calls.strike_price.values, puts.strike_price.values).all()
    strikes = calls.strike_price.values
    argsort_strikes = np.argsort(strikes)
    bid_ask = {
        "strikes": strikes[argsort_strikes],
        "calls": {
            "bids": calls.best_bid.values[argsort_strikes],
            "asks": calls.best_offer.values[argsort_strikes],
        },
        "puts": {
            "bids": puts.best_bid.values[argsort_strikes],
            "asks": puts.best_offer.values[argsort_strikes],
        },
    }
    ask_fwd = bid_ask["calls"]["asks"] - bid_ask["puts"]["bids"] + bid_ask["strikes"]
    bid_fwd = bid_ask["calls"]["bids"] - bid_ask["puts"]["asks"] + bid_ask["strikes"]
    # i = np.argmin(ask_fwd - bid_fwd)
    # fwd = 0.5 * (ask_fwd[i] + bid_fwd[i])
    fwd, fwd_bid, fwd_ask = choose_fwd(ask_fwd, bid_fwd)


    strikes_c, vol_c_asks, vol_c_bids = compute_vol(
        fwd, maturity, bid_ask["strikes"], bid_ask["calls"], "C"
    )
    strikes_p, vol_p_asks, vol_p_bids = compute_vol(
        fwd, maturity, bid_ask["strikes"], bid_ask["puts"], "P"
    )

    bid_ask["fwd"] = {"fwd": fwd, "bid" : fwd_bid, "ask" : fwd_ask, "asks": ask_fwd, "bids": bid_fwd}
    bid_ask["vol"] = {
        "c": {"strikes": strikes_c, "asks": vol_c_asks, "bids": vol_c_bids},
        "p": {"strikes": strikes_p, "asks": vol_p_asks, "bids": vol_p_bids},
    }

    smile_strikes = np.concatenate(
        [strikes_p[strikes_p < fwd], strikes_c[strikes_c >= fwd]]
    )
    smile_asks = np.concatenate(
        [vol_p_asks[strikes_p < fwd], vol_c_asks[strikes_c >= fwd]]
    )
    smile_bids = np.concatenate(
        [vol_p_bids[strikes_p < fwd], vol_c_bids[strikes_c >= fwd]]
    )

    smile = {
        "fwd": fwd,
        "fwd_bid" : fwd_bid,
        "fwd_ask" : fwd_ask,
        "strikes": smile_strikes,
        "asks": smile_asks,
        "bids": smile_bids,
        "mids": 0.5 * (smile_asks + smile_bids),
    }
    return smile, bid_ask


def extract_vix_smile(vix, maturity):
    df = vix.query(f"exdate == '{maturity}'")
    calls = df.query("cp_flag == 'C'")
    puts = df.query("cp_flag == 'P'")
    assert np.equal(calls.strike_price.values, puts.strike_price.values).all()
    strikes = calls.strike_price.values
    argsort_strikes = np.argsort(strikes)
    bid_ask = {
        "strikes": strikes[argsort_strikes],
        "calls": {
            "bids": calls.best_bid.values[argsort_strikes],
            "asks": calls.best_offer.values[argsort_strikes],
        },
        "puts": {
            "bids": puts.best_bid.values[argsort_strikes],
            "asks": puts.best_offer.values[argsort_strikes],
        },
    }
    ask_fwd = bid_ask["calls"]["asks"] - bid_ask["puts"]["bids"] + bid_ask["strikes"]
    bid_fwd = bid_ask["calls"]["bids"] - bid_ask["puts"]["asks"] + bid_ask["strikes"]
    # i = np.argmin(ask_fwd - bid_fwd)
    # fwd = 0.5 * (ask_fwd[i] + bid_fwd[i])
    fwd, fwd_bid, fwd_ask = choose_fwd(ask_fwd, bid_fwd)


    strikes_c, vol_c_asks, vol_c_bids = compute_vol(
        fwd, maturity, bid_ask["strikes"], bid_ask["calls"], "C"
    )
    strikes_p, vol_p_asks, vol_p_bids = compute_vol(
        fwd, maturity, bid_ask["strikes"], bid_ask["puts"], "P"
    )

    bid_ask["fwd"] = {"fwd": fwd, "bid" : fwd_bid, "ask" : fwd_ask, "asks": ask_fwd, "bids": bid_fwd}
    bid_ask["vol"] = {
        "c": {"strikes": strikes_c, "asks": vol_c_asks, "bids": vol_c_bids},
        "p": {"strikes": strikes_p, "asks": vol_p_asks, "bids": vol_p_bids},
    }

    smile_strikes = np.concatenate(
        [strikes_p[strikes_p < fwd], strikes_c[strikes_c >= fwd]]
    )
    smile_asks = np.concatenate(
        [vol_p_asks[strikes_p < fwd], vol_c_asks[strikes_c >= fwd]]
    )
    smile_bids = np.concatenate(
        [vol_p_bids[strikes_p < fwd], vol_c_bids[strikes_c >= fwd]]
    )

    smile = {
        "fwd": fwd,
        "fwd_bid" : fwd_bid,
        "fwd_ask" : fwd_ask,
        "strikes": smile_strikes,
        "asks": smile_asks,
        "bids": smile_bids,
        "mids": 0.5 * (smile_asks + smile_bids),
    }
    return smile, bid_ask


def extract_yield_curve(yield_curve):
    return {"days": yield_curve.days.values, "rate": yield_curve.rate.values}


def load_dataset(date, full=False):
    spx = SPX.query(f"date == '{date}'")
    vix = VIX.query(f"date == '{date}'")
    yield_curve = YIELD_CURVE.query(f"date == '{date}'")
    T0 = spx.date.drop_duplicates().values
    assert len(T0) == 1
    T0 = pd.to_datetime(str(T0[0]))

    vix_maturities = maturities_df(vix)
    spx_maturities = maturities_df(spx)

    # if spx_maturities[0] == date:
    #    spx_maturities = spx_maturities[1:]

    yield_curve = extract_yield_curve(yield_curve)

    data = {
        "vix_smiles": {},
        "spx_smiles": {},
        "vix_maturities": vix_maturities,
        "spx_maturities": spx_maturities,
        "yield_curve": yield_curve,
    }

    for maturity in vix_maturities:
        smile, bid_ask = extract_spx_smile(vix, maturity)
        data["vix_smiles"][maturity] = {
            "smile": smile,
            "bid_ask": bid_ask if full else None,
        }

    for maturity in spx_maturities:
        smile, bid_ask = extract_spx_smile(spx, maturity)
        data["spx_smiles"][maturity] = {
            "smile": smile,
            "bid_ask": bid_ask if full else None,
        }
    return data


def plot(smiles, maturities):
    n_maturities = len(maturities)
    plt.figure(figsize=(27, 6 * n_maturities))
    for i, maturity in enumerate(maturities):
        bid_ask = smiles[maturity]["bid_ask"]
        smile = smiles[maturity]["smile"]

        fwd = bid_ask["fwd"]["fwd"]
        strikes = bid_ask["strikes"]

        plt.subplot(n_spx_maturities, 3, i * 3 + 1)

        plt.scatter(
            strikes, bid_ask["fwd"]["asks"], color="blue", label="forward ask", s=5
        )
        plt.axhline(fwd, color="black", label="chosen forward")
        plt.scatter(
            strikes, bid_ask["fwd"]["bids"], color="green", label="forward bid", s=5
        )
        plt.xlabel("strike")
        plt.ylabel("C - P + K")
        plt.legend()
        plt.title(maturity)

        plt.subplot(n_spx_maturities, 3, i * 3 + 2)
        plt.scatter(
            bid_ask["vol"]["c"]["strikes"],
            bid_ask["vol"]["c"]["asks"],
            color="blue",
            s=1,
            label="calls implied vol. ask",
        )
        plt.scatter(
            bid_ask["vol"]["c"]["strikes"],
            bid_ask["vol"]["c"]["bids"],
            color="blue",
            s=1,
            label="calls implied vol. bid",
        )
        plt.fill_between(
            bid_ask["vol"]["c"]["strikes"],
            bid_ask["vol"]["c"]["bids"],
            bid_ask["vol"]["c"]["asks"],
            color="blue",
            alpha=0.3,
        )

        plt.scatter(
            bid_ask["vol"]["p"]["strikes"],
            bid_ask["vol"]["p"]["asks"],
            color="green",
            s=1,
            label="puts implied vol. ask",
        )
        plt.scatter(
            bid_ask["vol"]["p"]["strikes"],
            bid_ask["vol"]["p"]["bids"],
            color="green",
            s=1,
            label="puts implied vol. ask",
        )
        plt.fill_between(
            bid_ask["vol"]["p"]["strikes"],
            bid_ask["vol"]["p"]["bids"],
            bid_ask["vol"]["p"]["asks"],
            color="green",
            alpha=0.3,
        )
        plt.axvline(fwd, color="black", label="forward")

        plt.xlabel("strike")
        plt.ylabel("implied vol.")
        plt.legend()
        plt.title(maturity)

        plt.subplot(n_spx_maturities, 3, i * 3 + 3)

        plt.scatter(smile["strikes"], smile["asks"], color="red", s=5)
        plt.fill_between(
            smile["strikes"],
            smile["bids"],
            smile["asks"],
            color="red",
            alpha=0.3,
            label="bid-ask",
        )
        plt.scatter(smile["strikes"], smile["bids"], color="red", s=5)
        plt.plot(
            smile["strikes"], smile["mids"], color="red", label="mid", linestyle="--"
        )
        plt.axvline(fwd, color="black", label="forward")

        plt.xlabel("strike")
        plt.ylabel("implied vol.")
        plt.legend()
        plt.title(maturity)
