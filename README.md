# Bangalore Urban Heat Island Analyser

An interactive Streamlit app that analyses ten years of Bangalore weather data (2014–2023) to show how heat has changed, how it varies by season and festival period, and what it could mean for different groups of people.

Built as an AI/ML mini project at PES University. It supports **SDG 13 (Climate Action)** and **SDG 12 (Responsible Consumption and Production)**.

**Live app:** https://bangalore-uhi.streamlit.app/

## What the app does

| Page | What you get |
|---|---|
| **Home** | Headline numbers (average Heat Index, 2030 projection) and a quick health-risk check for a chosen month, user profile and festival setting |
| **Forecast** | A 30-day Heat Index outlook from a chosen start date, with a range band and Safe / Caution / Danger / Extreme labels |
| **Insights** | Climate trend with a 2030 projection, festival vs non-festival heat comparison, and model results (feature importance, heat-zone clusters) |
| **Health Risk** | A personalised risk score and advice for five profiles: healthy adult, child, elderly, outdoor worker or athlete, and person with a respiratory condition |

## How it works

| Component | Method | Notes |
|---|---|---|
| 30-day outlook | Historical monthly average | Mean Heat Index for each calendar month, with a ±1 standard deviation band. This is a climatological baseline, not a time-series forecast |
| Climate trend and 2030 projection | Linear and degree-2 polynomial regression | Fitted on yearly mean temperature. The polynomial fit scored R² 0.832 vs 0.491 for linear (in-sample, about ten yearly points) |
| Heat Index predictor | Random Forest Regressor | 100 trees, max depth 10, 80/20 split, MinMax scaling, 5-fold cross-validation and feature importance. Reported R² 0.891 |
| Heat zones | K-Means (4 clusters) | Clusters on temperature, Heat Index, humidity and month, labelled Cool / Warm / Hot / Extreme by mean temperature |
| Festival impact | Group comparison | Mean Heat Index on festival vs non-festival days, overall and by month |
| Health risk score | Rule-based weighted score | Heat Index 40%, humidity 25%, visibility 15%, wind 10%, festival 10%, scaled by a per-profile multiplier (1.0 to 1.3) and capped at 100 |
| Rain predictor | Decision Tree (GridSearchCV), Random Forest, Logistic Regression | Target is precipitation above 0.5 mm. SMOTE is applied on the training split only, and the best model by F1 is saved. Trained in `train.py` but not currently used by the app |

Risk labels use Heat Index thresholds of below 32 °C (Safe), 32–35 °C (Caution), 35–38 °C (Danger) and 38 °C or above (Extreme).

## Dataset

- **Source:** [Bangalore Weather Data (Visual Crossing) on Kaggle](https://www.kaggle.com/datasets/rishi1903/bangalore-weather-data-visual-crossing-weather)
- **Raw size:** 6,695 daily records × 18 columns, covering 2014 to 2024 (2024 has only 3 rows, so it is dropped)

### Cleaning (`eda.ipynb`)
- Dropped 9 columns that were constant, redundant or free text (address, latitude, longitude, minimum and maximum temperature, precipitation cover, sea level pressure, weather type, conditions)
- Converted temperature, dew point and Heat Index from Fahrenheit to Celsius
- Filled 1,538 missing Heat Index values (23%) with the Rothfusz regression formula, using temperature and humidity
- Filled 7 missing Cloud Cover values with the median
- Removed 121 duplicate rows, leaving 6,574 rows × 9 columns
- Capped Wind Speed at the 99th percentile (23.0 km/h; the raw maximum was 121.2 km/h)

### Feature engineering (`feature_engineering.ipynb`)
- `month`, `year`, and `season` (Winter, Post-Monsoon, Monsoon, Summer)
- `is_festival`: 476 days marked across major Bangalore and Karnataka festivals and events
- `is_extreme_heat`: Heat Index above 35 °C (663 days, 10.1% of the real data)
- `temp_anomaly`: temperature minus the monthly historical average

### Synthetic data
To reach 10,000 rows, 3,426 synthetic rows were added. Each one copies a randomly chosen real row, adds Gaussian noise (5% of each column's standard deviation) to the numeric features, and recomputes Heat Index with the Rothfusz formula. They are flagged with `is_synthetic`. After dropping 2024, the training set has 9,997 rows.

## Limitations

Please read these before relying on any number in the app.

- **Synthetic rows are used in training and evaluation.** About a third of the rows are noisy copies of real rows. `train.py` drops the `is_synthetic` flag but keeps the rows, so near-duplicates can land in both the training and test sets. The reported scores are likely optimistic. Retraining and testing on real rows only is the first planned fix.
- **The Heat Index target is largely a formula of its inputs.** Heat Index is computed from temperature and humidity, and those are model inputs. The missing real values and all synthetic values were filled with that same formula. A high R² partly reflects the formula, not learned weather behaviour.
- **The 30-day outlook is a monthly average, not a forecast.** It does not use recent weather or a time-series model.
- **The 2030 projection is a rough extrapolation.** It is fitted on about ten yearly averages and scored on the same data, so treat it as indicative only.
- **Festival impact is an association.** Many festivals fall in hotter months, so the festival effect is confounded with season.
- **Date handling needs checking.** The raw date column mixes formats and the cleaned data has more rows than calendar days. Dates were parsed with `format='mixed'`, so month and season results should be treated as approximate until this is verified.
- **It does not measure the urban heat island effect directly.** The data is weather for one city, with no urban vs rural comparison, so heat trends are used as a proxy.
- **The health risk score is a heuristic.** The weights and profile multipliers are hand-chosen and not clinically validated. It is not medical advice.

## Project structure

```
├── app.py                      # Streamlit app
├── train.py                    # trains models and saves them to data/
├── eda.ipynb                   # cleaning and exploration
├── feature_engineering.ipynb   # features, festival flags, synthetic rows
├── requirements.txt
└── data/                       # datasets and saved models (not all committed)
```

## Run it locally

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Hrithi89/<repo-name>.git
   cd <repo-name>
   pip install -r requirements.txt
   ```
2. Download the dataset from Kaggle and place the CSV in `data/` as `Bangalore Weather Data (Visual Crossing Weather).csv`.
3. Run `eda.ipynb` to create `data/bangalore_weather_cleaned.csv`.
4. Run `feature_engineering.ipynb` to create `data/bangalore_weather_final.csv`.
5. Train and save the models:
   ```bash
   python train.py
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```

## Tech stack

Python, Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn, Streamlit, Joblib

## Planned improvements

- Retrain and evaluate on real rows only, with a time-based train/test split
- Replace the monthly-average outlook with a proper time-series forecast
- Add an extreme-heat classifier to the app, and connect the rain predictor
- Verify and fix date parsing
- Compare with a rural reference point to measure the urban heat island effect directly
