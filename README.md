

# *A Comparative Evaluation of ML and DL Models for Brent Crude Oil Price Forecasting. UWE Bristol Dissertation Project.

> MSc Data Science · UFCF9Y-60-M CSCT Masters Project · UWE Bristol · 2024–25

## Live Dashboard
🌐 **[View the interactive dashboard] https://uchenlemadim-ml-vs-dl-models-for-brent-crude-oil-price-forecas.streamlit.app/**

---

## Research Question
Which model architecture — statistical (ARIMA), machine learning
(Random Forest, XGBoost), or deep learning (LSTM, GRU) — most
accurately forecasts Brent crude oil daily closing prices on the
same 25-year dataset under identical evaluation conditions?

---

## Dataset
| Property | Value |
|----------|-------|
| Source | Investing.com (2024) Brent Crude Oil Futures Historical Data |
| Period | 4 January 2000 to 31 December 2024 |
| Total trading days | 6,422 |
| Feature used | Daily closing Price (USD/barrel) — univariate only |
| Dropped columns | Open, High, Low, Vol., Change % |
| Train split | 5,389 rows (2000-01-04 to 2020-12-31) |
| Val split | 517 rows (2021-01-04 to 2022-12-30) |
| Test split | 516 rows (2023-01-03 to 2024-12-31) |

---

## Models Evaluated and Key References
| Model | Type | Key Reference |
|-------|------|---------------|
| ARIMA | Statistical baseline | Mohammad and Panigrahi (2023) |
| Random Forest | Machine learning | Toppo et al. (2024) |
| XGBoost | Machine learning | Paul, Chacko and Bhattacharjee (2025) |
| LSTM | Deep learning | Daneshvar et al. (2022) |
| GRU | Deep learning | Zhang and Lahmiri (2025) |

---

## Results
| Rank | Model | RMSE ($/barrel) | MAE ($/barrel) | MAPE (%) |
|------|-------|----------------|----------------|----------|
| 1 | Random Forest | 1.3744 | 1.0458 | 1.30% |
| 2 | XGBoost | 1.4477 | 1.1303 | 1.40% |
| 3 | ARIMA | 1.4584 | 1.1167 | 1.39% |
| 4 | GRU | 1.4730 | 1.1287 | 1.41% |
| 5 | LSTM | 1.4838 | 1.1431 | 1.42% |

**Key finding:** Random Forest ranked first across all three metrics.
The Diebold-Mariano test confirmed Random Forest significantly
outperforms the ARIMA baseline (DM=3.13, p=0.0018, α=0.05).
Contrary to expectation, both deep learning models ranked below
Random Forest and XGBoost on the 2023-2024 test period.


## Project Structure

project-root/

 ├── data/
 │   ├── raw/                         ← place CSV here
 │   └── processed/                   ← created by Notebook 01
 ├── notebooks/
 │   ├── 01_data_preprocessing.ipynb
 │   ├── 02_eda_analysis.ipynb
 │   ├── 03_arima_baseline.ipynb
 │   ├── 04_ml_models.ipynb
 │   ├── 05_dl_models.ipynb
 │   └── 06_evaluation_comparison.ipynb
 ├── results/
 │   ├── arima/
 │   ├── random_forest/
 │   ├── xgboost/
 │   ├── lstm/
 │   └── gru/
 ├── artifacts/
 │   └── plots/
 ├── src/
 │   ├── preprocessing.py
 │   ├── evaluation.py
 │   └── walk_forward_cv.py
 ├── configs/
 │   └── config.yaml
 ├── dashboard.py
 ├── requirements.txt
 └── README.md

## Notebooks
| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_preprocessing.ipynb` | Load, clean, split, scale, save |
| 02 | `02_eda_analysis.ipynb` | Time series, ACF/PACF, ADF stationarity test |
| 03 | `03_arima_baseline.ipynb` | auto_arima order selection, walk-forward validation |
| 04 | `04_ml_models.ipynb` | Random Forest and XGBoost with RandomizedSearchCV |
| 05 | `05_dl_models.ipynb` | LSTM and GRU with EarlyStopping |
| 06 | `06_evaluation_comparison.ipynb` | Master metrics table, DM test, all plots |

---

## Dashboard
Run the interactive Streamlit dashboard locally:
```bash
streamlit run dashboard.py
```
Or view the live version: 🌐 **https://uchenlemadim-ml-vs-dl-models-for-brent-crude-oil-price-forecas.streamlit.app/**

Features:
- Model selector (ARIMA, Random Forest, XGBoost, LSTM, GRU)
- Date range filter for the test period
- Actual vs Predicted interactive chart with residuals
- All models comparison chart with RMSE in legend
- Future forecast (7, 30, or 90 days) using best performing model
- 95% confidence band on future forecasts

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/UcheNlemadim/ml-vs-dl-models-for-brent-crude-oil-price-forecasting.git
cd ml-vs-dl-models-for-brent-crude-oil-price-forecasting
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Register Jupyter kernel
```bash
python -m ipykernel install --user \
  --name=brent_oil_venv \
  --display-name "Python (Brent Oil .venv)"
```

### 5. Place the dataset
Download Brent Crude Oil Futures Historical Data from
https://www.investing.com and save to: data/raw/Brent_Oil_Futures_Historical_Data_20002024.csv

### 6. Run notebooks in order
Open each notebook in VSCode, select the
**"Python (Brent Oil .venv)"** kernel, and click Run All.
Run strictly in order: 01 → 02 → 03 → 04 → 05 → 06

### 7. Launch the dashboard
```bash
streamlit run dashboard.py
```

---

## Reproducibility
| Setting | Value |
|---------|-------|
| random_state | 42 throughout all models and searches |
| Scaler | MinMaxScaler fitted once in NB01 on train only |
| NB06 | Reads only saved files — never re-trains |
| Config | All constants in configs/config.yaml |
| Splits | Train 5,389 / Val 517 / Test 516 rows |
| ARIMA order | ARIMA(0,1,1) selected by auto_arima on AIC |

---

## Ethical Considerations
- Publicly available historical price data only
- No human participants or personal data involved
- Full codebase open-source for academic transparency
- Ethics review submitted to UWE Bristol
- Raw data excluded from repository (terms of use)

---

## References

Ajowho Adenusi, C., Rebecca Vincent, O., Abayomi-Alli, A., Mathew Olayiwola, O., Olawunmi Shamsudeen, B. and Titilayo Mary, S. (2022) Predicting the Upshot of Covid-19 On Crude-Oil Prices In Nigeria Using MLPARIMA Model. Proceedings of the 5th International Conference on Information Technology for Education and Development: Changing the Narratives Through Building a Secure Society with Disruptive Technologies, ITED 2022 [online]. Institute of Electrical and Electronics Engineers Inc. pp. 1–8. Available from: https://ieeexplore.ieee.org/document/10051492 [Accessed 16 May 2026].

Daneshvar, A., Ebrahimi, M., Salahi, F., Rahmaty, M. and Homayounfar, M. (2022) Brent Crude Oil Price Forecast Utilizing Deep Neural Network Architectures. Computational Intelligence and Neuroscience [online]. 2022, pp. 1–13.

Gulati, K., Gupta, J., Rani, L. and Sarangi, P.K. (2022) Crude Oil Prices Predictions in India Using Machine Learning based Hybrid Model. 2022 10th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions), ICRITO 2022 [online]. Institute of Electrical and Electronics Engineers Inc. pp. 1–6. Available from: https://ieeexplore.ieee.org/document/9964577. [Accessed 16 May 2026].

Gumus, M. and Kiran, M.S. (2017) Crude oil price forecasting using XGBoost IEEE Xplore. 1 October 2017 [online]. pp. 1100–1103. Available from: https://ieeexplore.ieee.org/abstract/document/8093500?casa_token=umdVFyuLzuMAAAAA:Pbo5_ML4ZceYGyRE2-6ULBImKs0c1gTYzdxSdcewXmzDrRS51ElFaFM6U84WQUlN63vOdnFOWgqjA0M [Accessed 16 May 2026].

Hyndman, R.J. and Athanasopoulos, G. (2018) Forecasting: Principles and Practice (3rd ed) otexts.com [online]. Available from: https://otexts.com/fpp3.

Habelalmateen, M.I., Singh, M.P., Chaithra, K.N., V Bala Dhandayuthapani and Irfan, M.M. (2024) Improved Hierarchical Bayesian Optimization Algorithm with LSTM for Crude Oil Price Forecasting. 3rd IEEE International Conference on Distributed Computing and Electrical Circuits and Electronics, ICDCECE 2024 [online]. Institute of Electrical and Electronics Engineers Inc.
pp. 1–4. Available from: https://ieeexplore.ieee.org/document/10548309/. [Accessed 16 May 2026].

Awijen, H., ben Ameur, H., Ftiti, Z. and Louhichi, W. (2025) Forecasting Oil Price In Times Of Crisis: A New Evidence From Machine Learning Versus Deep Learning Models. Annals of Operations Research [online]. 345 (2), pp. 979–1002. [Accessed 15 May 2026].

Mohammad, F.T. and Shrikant Krupasindhu Panigrahi (2023) Forecasting Crude Oil Price Using SARIMAX Machine Learning Approach. 2023 International Conference on Sustainable Islamic Business and Finance, SIBF 2023 [online]. pp. 131–135. [Accessed 16 May 2026].

Nagendra Kumar, Y.J., Preetham, P., Kiran Varma, P., Rohith, P. and Dilip Kumar, P. (2020) Crude Oil Price Prediction Using Deep Learning IEEE Xplore. 1 July 2020 [online]. pp. 118–123. Available from: https://ieeexplore.ieee.org/document/9183258 [Accessed 16 May 2026].

Obite, C.P., Bartholomew, D.C., Nwosu, U.I., Esiaba, G.E. and Kiwu, L.C. (2021) The Optimal Machine Learning Modeling of Brent Crude Oil Price. Quarterly Journal of Econometrics Research [online]. 7 (1), pp. 31–43. [Accessed 16 May 2026].

Paul, S.N., Chaco, A.M. and Bhattacharjee, B. (2025) Crude Oil Price Forecasting using Multi-Feed Data: A Comparative Study. ETIS International Conference on Emerging Technologies for Intelligent Systems, ETIS 2025 [online]. Institute of Electrical and Electronics Engineers Inc. pp. 1–6. [Accessed 19 May 2026].

Toppo, A., Mahajan, J., Singh, V.P., Paswan, A.S. and Saxena, A. (2024) Machine Learning Insights into Predicting Crude Oil Prices. 2024 International Conference on Trends in Quantum Computing and Emerging Business Technologies [online]. pp. 1–5. [Accessed 19 May 2026].

Yang, H., Zhang, Y. and Jiang, F. (2019) Crude Oil Prices Forecast Based on EMD and BP Neural Network. 2019 Chinese Control Conference (CCC) [online]. pp. 8944–8949. [Accessed 19 May 2026].

Zhang, Y. and Lahmiri, S. (2025) A Deep Learning-Based Ensemble System for Brent and WTI Crude Oil Price Analysis and Prediction. Entropy [online]. 27 (11), p. 1122. [Accessed 16 May 2026].

---

## Acknowledgements
My Family and Friends,
MSc Data Science, School of Computing and Creative Technologies,
UWE Bristol. 


