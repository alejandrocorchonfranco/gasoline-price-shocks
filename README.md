# Do Voters Punish Gasoline Price Shocks?
### Evidence from U.S. House Elections, 1992–2022
**Alejandro Corchón Franco** — Universitat Pompeu Fabra

---

Does retrospective accountability for gasoline prices operate differently in midterm and presidential elections? Using a state-by-election-year panel covering all sixteen U.S. federal election cycles from 1992 to 2022, this paper finds that it does not: the interaction between pre-election gasoline prices and state-level automobile dependence yields **β ≈ −0.030** in midterms and **β ≈ −0.027** in presidential elections — economically indistinguishable magnitudes (F-test of equality: F = 0.12, p = 0.73). If voters punish the incumbent party for high gasoline prices, they do so with equal force regardless of whether the presidential race is also on the ballot.

---

## Repository structure

```
code/    corchon_gasoline_elections_CODE.py   ← full replication script (v3.0)
data/    panel_main.csv                        ← state × election-year panel
         weekly_energy.csv                     ← weekly gas prices & SPR stocks
output/  .tex                                  ← paper (LaTeX)
         regression_results.csv               ← all regression coefficients
         key_numbers.txt                       ← key statistics for paper
         corchon_gasoline_elections_data.xlsx  ← complete dataset (Excel)
Figures/ Figure1.png / Figure1.pdf             ← gas price + SPR timeline
         Figure2.png / Figure2.pdf             ← marginal effects by election type
```

## Data sources

| Series | Source | Period |
|---|---|---|
| National gas price (GASREGCOVW) | FRED / EIA | Aug 1990 – present |
| PADD 1–5 regional gas prices | EIA bulk XLS | May 1992 – present |
| SPR crude stocks (WCSSTUS1) | EIA bulk XLS | Aug 1982 – present |
| House election returns | MIT Election Data and Science Lab | 1976–2022 |
| Automobile-dependence index | ACS, NHTS 2017 | Time-invariant |

## Replication

```bash
pip install pandas numpy scipy matplotlib linearmodels statsmodels openpyxl xlrd requests
python code/corchon_gasoline_elections_CODE.py
```

All data are downloaded automatically at runtime. No API keys required.
Running time: approximately 2–3 minutes.
