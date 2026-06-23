# Do Voters Punish Gasoline Price Shocks?
### Evidence from U.S. House Elections, 1992–2022
**Alejandro Corchón Franco** — Universitat Pompeu Fabra

---

Does the retail price of gasoline affect how voters reward or punish the president's party at the polls? This paper identifies the effect using cross-state differences in automobile dependence: a national price shock should hurt the incumbent party most in states where commuting by car is most prevalent. State and election-year fixed effects absorb aggregate conditions; identification comes from whether more car-dependent states respond more strongly to the same national price movement.

**Main finding:** a one-dollar increase in the decay-weighted pre-election gasoline price is associated with a **3.1 percentage-point larger incumbent-party swing loss** for each standard deviation of automobile dependence — and this holds equally in midterm and presidential elections (β = −0.031 and −0.030, both p < 0.001).

---

## Repository structure

```
code/   corchon_gasoline_elections_CODE.py   # full replication script
data/   20260623.tex                         # paper (LaTeX)
        panel_main.csv                       # state × election-year panel
        weekly_energy.csv                    # weekly gas prices & SPR stocks
figures/                                     # Figures 1–4 (PDF + PNG)
output/ regression_results.csv              # all regression coefficients
        corchon_gasoline_elections_data.xlsx # Excel workbook
```

## Data sources

| Series | Source |
|---|---|
| Weekly retail gasoline price (GASREGCOVW) | FRED / EIA, from Aug 1990 |
| Weekly SPR crude stocks (WCSSTUS1) | EIA bulk download, from Aug 1982 |
| House election returns | MIT Election Data and Science Lab (1976–2022) |
| Automobile-dependence index | ACS, NHTS 2017 |

## Replication

```bash
pip install pandas numpy scipy matplotlib linearmodels statsmodels openpyxl xlrd requests
python code/corchon_gasoline_elections_CODE.py
```

All data are downloaded automatically at runtime. No API keys required.
