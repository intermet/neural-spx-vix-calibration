# Neural Joint SPX-VIX Calibration

<p align="center">
  <img width="600" height="450" src="./training.gif">
</p>

## Requirements
python >= 3.6, torch>=1.13.0 and torchsde>=0.2.5

## Example

Configuration file:
```json
{
    "batch_size" : 200000,
    "dt" : 0.5,
    "w_fVIX" : 10,
    "w_CVIX" : 1,
    "w_SPX" : 3
}
```
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

```bash
python src/train.py --maturities maturities.json \
                --params params.json \
                --model my_model.py
```


## References
\[1\] Julien Guyon, Scander Mustapha. "Neural Joint SPX-VIX calibration" (2022) [[SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4309576)

