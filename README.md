# Neural Joint SPX-VIX Calibration

<p align="center">
  <img width="600" height="450" src="./training.gif">
</p>

## Requirements
python >= 3.6, torch>=1.13.0 and torchsde>=0.2.5

## Data

Data should be download from the [[Wharton Research Data Service]](https://wrds-www.wharton.upenn.edu/login/?next=/pages/get-data/optionmetrics/ivy-db-us/options/option-prices) and placed in `src/data/csv` under the csv format:
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

To start the training of a model defined in `my_model.py`, for the SPX and VIX maturities defined in `maturities.json` and with the hyper-parameters defined in `params.json`, use the following command in directoy `src`
```bash
python train.py --model my_model.py \
                --maturities maturities.json \
                --params params.json 
```


We give below typical example of configuration files.
### `my_model.py`
```python
import torch
from vol import V_and_MuY_rho_tanh

MODEL = {
    "nets" : {
        "phi" : torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 4)).to("cuda:0"),
    },
    "name" : "spx_vix",
    "optimizer" : None
}

V_AND_MUY = V_and_MuY_rho_tanh
```
### `maturities.json`
```json
{
    "spx" : [
        "2021/10/08",
        "2021/10/15",
        "2021/10/22",
        "2021/10/29",
        "2021/11/05",
        "2021/11/12",
        "2021/11/19",
        "2021/11/26",
        "2021/12/17"
    ],

    "vix" : [
        "2021/10/13",
        "2021/10/20",
        "2021/11/17"
    ]
    
}
```
### `params.json`
```json
{
    "batch_size" : 200000,
    "dt" : 0.5,
    "w_fVIX" : 10,
    "w_CVIX" : 1,
    "w_SPX" : 3
}
```


## References
\[1\] Julien Guyon, Scander Mustapha. "Neural Joint SPX-VIX calibration" (2022) [[SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4309576)

