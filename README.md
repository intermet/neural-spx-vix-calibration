# Neural Joint SPX-VIX Calibration

<p align="center">
  <img width="600" height="450" src="./training.gif">
</p>

## Requirements
python >= 3.6, torch>=1.13.0 and torchsde>=0.2.5

## Data

Data should be downloaded from the [[Wharton Research Data Service Optionmetrics database]](https://wrds-www.wharton.upenn.edu/login/?next=/pages/get-data/optionmetrics/ivy-db-us/options/option-prices) and placed in `src/data/csv` under the csv format:
```
$ ls src/data/csv
spx_bid_ask.csv
vix_bid_ask.csv
```

The header of the data set `spx_bid_ask.csv` and `vix_bid_ask_csv` are respectively
```bash
$ cat src/data/csv/spx_bix_ask.csv | head -n 1
secid,date,exdate,cp_flag,strike_price,best_bid,best_offer,volume,impl_volatility,optionid,am_settlement,forward_price,index_flag,issuer,exercise_style

$ cat src/data/csv/vix_bix_ask.csv | head -n 1
date,exdate,cp_flag,strike_price,best_bid,best_offer,volume,impl_volatility,optionid,am_settlement,forward_price,index_flag,issuer,exercise_style
```

## Example

To start the training of a model defined in `my_model.py`, for the SPX and VIX maturities defined in `maturities.json` and with the hyper-parameters defined in `params.json`, use the following command in directory `src`
```bash
python train.py --model my_model.py \
                --maturities maturities.json \
                --params params.json 
```


We give below a typical example of configuration files.
### `my_model.py`
```python
import torch
from vol import V_and_MuY_rho_tanh

MODEL = {
    "nets" : {
        "phi" : torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 4)).to("cuda:0"),
    },
    "name" : "spx_vix",
    "optimizer" : None
}

def V_AND_MUY(model, alpha=1):
    Rho_min = torch.tensor(-0.99999, device="cuda:0")
    Rho_max = torch.tensor(+0.99999, device="cuda:0")
    def V_and_MuY(tXY):
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

```
A model is given by a dict `MODEL` including a family of neural networks `MODEL["nets"]` and a function `V_AND_MUY` that computes the volatility and the drift of the calibration model as outputs of the neural networks. In the example above, the volatility and the drift are given by
```math
\displaylines{\sigma_X(t, x, y) = 1 + \tanh(\Phi_1(txy))\\
\sigma_Y(t, x, y) = 1 + \tanh(\Phi_2(txy))\\
\rho(t, x, y) = \tanh(\Phi_3(txy))\\
\mu_Y(t, x, y) = \Phi_4(txy)}
```
where $\Phi = (\Phi_1, \Phi_2, \Phi_3, \Phi_4)$ is the output of the neural network `MODEL["nets"]["phi"]`.
### `maturities.json`
```json
{
    "spx" : [
        "2021/10/08",
        "2021/10/22",
        "2021/11/19",
        "2021/12/17",
        "2022/01/21",
        "2022/02/18",
        "2022/03/31",
        "2022/06/30"
    ],

    "vix" : [
        "2021/10/06",
        "2021/10/13",
        "2021/10/20",
        "2021/11/17",
        "2021/12/22",
        "2022/01/19",
        "2022/02/16",
        "2022/03/15"
    ]
    
}
```
### `params.json`
```json
{
    "batch_size" : 150000,
    "dt" : 0.5,
    "w_fVIX" : 30,
    "w_CVIX" : 2,
    "w_SPX" : 3
}
```


## References
\[1\] Julien Guyon, Scander Mustapha. "Neural Joint S&P 500/VIX smile calibration" (2022) [[SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4309576)

