# train.py — Bangalore Urban Heat Island Analyser
# Models: Historical Averaging, Linear + Polynomial Regression,Random Forest Regressor, Decision Tree Classifier, K-Means, Rule-based Health Risk, Rain Predictor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.preprocessing   import MinMaxScaler, PolynomialFeatures
from sklearn.cluster         import KMeans
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics         import (silhouette_score, r2_score,
                                      mean_squared_error, accuracy_score,
                                      f1_score, classification_report)
from imblearn.over_sampling  import SMOTE

#STEP 1 — Load Dataset

print("\n")
print("STEP 1 — Loading Dataset")
print("\n")

df = pd.read_csv('data/bangalore_weather_final.csv')

if 'is_synthetic' in df.columns:
    df.drop(columns=['is_synthetic'], inplace=True)

print(f"Loaded: {df.shape}")
print(f"Years: {sorted(df['year'].unique())}")

# Drop 2024 — only 3 rows (January only), incomplete year ruins trend
df = df[df['year'] != 2024]
print(f"After dropping 2024: {df.shape}")
print(f"Years remaining: {sorted(df['year'].unique())}")

# STEP 2 — Historical Average Model (30-day Forecasting)

print("\n" )
print("STEP 2 — Building Historical Average Forecast Model")

monthly_stats = df.groupby('month').agg(
    heat_index_mean = ('Heat Index',        'mean'),
    heat_index_std  = ('Heat Index',        'std'),
    temp_mean       = ('Temperature',       'mean'),
    temp_std        = ('Temperature',       'std'),
    humidity_mean   = ('Relative Humidity', 'mean'),
    wind_mean       = ('Wind Speed',        'mean'),
    precip_mean     = ('Precipitation',     'mean'),
    cloud_mean      = ('Cloud Cover',       'mean'),
    visibility_mean = ('Visibility',        'mean'),
).round(3)

monthly_stats['hi_upper'] = (monthly_stats['heat_index_mean'] +
                              monthly_stats['heat_index_std']).round(3)
monthly_stats['hi_lower'] = (monthly_stats['heat_index_mean'] -
                              monthly_stats['heat_index_std']).round(3)

def get_risk_label(hi):
    if hi < 32:   return 'Safe'
    elif hi < 35: return 'Caution'
    elif hi < 38: return 'Danger'
    else:         return 'Extreme'

monthly_stats['risk_label'] = monthly_stats['heat_index_mean'].apply(get_risk_label)

print("Monthly forecast stats:")
print(monthly_stats[['heat_index_mean', 'hi_upper', 'hi_lower', 'risk_label']])

joblib.dump(monthly_stats, 'data/monthly_stats.pkl')
print("\nMonthly stats saved!")


# STEP 3 — 30-Day Forecast Lookup Table

print("\n")
print("STEP 3 — Building 30-Day Forecast Lookup Table")
print("\n")

month_days   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
daily_lookup = []
day_num      = 1

for m_idx, days in enumerate(month_days):
    month_num = m_idx + 1
    stats     = monthly_stats.loc[month_num]
    for d in range(1, days + 1):
        daily_lookup.append({
            'day_of_year'    : day_num,
            'month'          : month_num,
            'day'            : d,
            'heat_index_mean': stats['heat_index_mean'],
            'hi_upper'       : stats['hi_upper'],
            'hi_lower'       : stats['hi_lower'],
            'temp_mean'      : stats['temp_mean'],
            'humidity_mean'  : stats['humidity_mean'],
            'risk_label'     : stats['risk_label'],
        })
        day_num += 1

daily_df = pd.DataFrame(daily_lookup)
joblib.dump(daily_df, 'data/daily_lookup.pkl')
print(f"Daily lookup table: {daily_df.shape}")
print(daily_df.head(5))

# STEP 4 — Linear Regression: Trend + 2030 Projection

print("\n" + "\n")
print("STEP 4 — Linear Regression: Climate Trend & 2030 Projection")
print("\n")

yearly_temp = df.groupby('year')['Temperature'].mean().reset_index()
yearly_hi   = df.groupby('year')['Heat Index'].mean().reset_index()

X_years = yearly_temp['year'].values.reshape(-1, 1)
y_temp  = yearly_temp['Temperature'].values
y_hi    = yearly_hi['Heat Index'].values

lr_temp = LinearRegression()
lr_temp.fit(X_years, y_temp)

lr_hi = LinearRegression()
lr_hi.fit(X_years, y_hi)

future_years    = np.array(range(2014, 2031)).reshape(-1, 1)
temp_projection = lr_temp.predict(future_years)
hi_projection   = lr_hi.predict(future_years)

warming_per_year = round(float(lr_temp.coef_[0]), 4)
temp_2024        = round(float(lr_temp.predict([[2024]])[0]), 2)
temp_2030        = round(float(lr_temp.predict([[2030]])[0]), 2)
hi_2030          = round(float(lr_hi.predict([[2030]])[0]), 2)
total_warming    = round(temp_2030 - float(lr_temp.predict([[2014]])[0]), 2)

lr_r2 = round(r2_score(y_temp, lr_temp.predict(X_years)), 3)

print(f"Linear Regression R²   : {lr_r2}")
print(f"Warming per year       : +{warming_per_year} C")
print(f"Projected temp 2030    : {temp_2030} C")
print(f"Total warming 2014->2030: +{total_warming} C")

trend_data = {
    'yearly_temp'     : yearly_temp,
    'yearly_hi'       : yearly_hi,
    'future_years'    : future_years.flatten().tolist(),
    'temp_projection' : temp_projection.tolist(),
    'hi_projection'   : hi_projection.tolist(),
    'warming_per_year': warming_per_year,
    'temp_2024'       : temp_2024,
    'temp_2030'       : temp_2030,
    'hi_2030'         : hi_2030,
    'total_warming'   : total_warming,
    'lr_r2'           : lr_r2,
}
joblib.dump(trend_data, 'data/trend_data.pkl')
print("Linear trend data saved!")

# STEP 4B — Polynomial Regression: Compare with Linear

print("\n" + "\n")
print("STEP 4B — Polynomial Regression (degree=2) vs Linear")
print("\n")

# Fit Polynomial Regression (degree 2) on temperature trend
poly        = PolynomialFeatures(degree=2)
X_poly      = poly.fit_transform(X_years)
X_poly_fut  = poly.transform(future_years)

poly_temp = LinearRegression()
poly_temp.fit(X_poly, y_temp)

poly_projection = poly_temp.predict(X_poly_fut)
poly_r2         = round(r2_score(y_temp, poly_temp.predict(X_poly)), 3)
poly_temp_2030  = round(float(poly_temp.predict(
                    poly.transform([[2030]]))[0]), 2)

print(f"Linear Regression R²    : {lr_r2}")
print(f"Polynomial Regression R²: {poly_r2}")
print(f"Better model            : {'Polynomial' if poly_r2 > lr_r2 else 'Linear'}")
print(f"Poly projected temp 2030: {poly_temp_2030} C")

# Save polynomial data
poly_trend_data = {
    'poly'              : poly,
    'poly_temp'         : poly_temp,
    'poly_projection'   : poly_projection.tolist(),
    'poly_r2'           : poly_r2,
    'lr_r2'             : lr_r2,
    'poly_temp_2030'    : poly_temp_2030,
    'future_years'      : future_years.flatten().tolist(),
    'yearly_temp'       : yearly_temp,
}
joblib.dump(poly_trend_data, 'data/poly_trend_data.pkl')
print("Polynomial trend data saved!")

# STEP 5 — Festival Heat Impact Analysis

print("\n" + "\n")
print("STEP 5 — Festival Heat Impact Analysis")
print("\n")

fest_days     = df[df['is_festival'] == 1]
non_fest_days = df[df['is_festival'] == 0]

fest_avg_hi     = round(fest_days['Heat Index'].mean(), 2)
non_fest_avg_hi = round(non_fest_days['Heat Index'].mean(), 2)
fest_impact     = round(fest_avg_hi - non_fest_avg_hi, 2)

print(f"Festival days avg Heat Index     : {fest_avg_hi} C")
print(f"Non-festival days avg Heat Index : {non_fest_avg_hi} C")
print(f"Festival heat impact             : +{fest_impact} C")

monthly_fest         = df.groupby(
    ['month', 'is_festival'])['Heat Index'].mean().unstack()
monthly_fest.columns = ['Non-Festival', 'Festival']
monthly_fest['Impact'] = (monthly_fest['Festival'] -
                           monthly_fest['Non-Festival']).round(2)

print("\nPer-month festival impact:")
print(monthly_fest.sort_values('Impact', ascending=False))

festival_data = {
    'fest_avg_hi'    : fest_avg_hi,
    'non_fest_avg_hi': non_fest_avg_hi,
    'fest_impact'    : fest_impact,
    'monthly_fest'   : monthly_fest,
    'fest_count'     : len(fest_days),
    'non_fest_count' : len(non_fest_days),
}
joblib.dump(festival_data, 'data/festival_data.pkl')
print("Festival data saved!")

# STEP 6 — K-Means Clustering (Heat Zones)

print("\n" + "\n")
print("STEP 6 — K-Means Heat Zone Clustering")
print("\n")

FEATURES_C = ['Temperature', 'Heat Index', 'Relative Humidity', 'month']
scaler_C   = MinMaxScaler()
X_C        = scaler_C.fit_transform(df[FEATURES_C])

kmeans    = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_C)
sil       = silhouette_score(X_C, kmeans.labels_)

df['cluster'] = kmeans.labels_
cluster_temps = df.groupby('cluster')['Temperature'].mean().sort_values()
cluster_map   = {
    int(cluster_temps.index[0]): 'Cool',
    int(cluster_temps.index[1]): 'Warm',
    int(cluster_temps.index[2]): 'Hot',
    int(cluster_temps.index[3]): 'Extreme',
}

print(f"Silhouette Score: {sil:.3f}")
print(f"Cluster mapping : {cluster_map}")

joblib.dump(kmeans,      'data/model_kmeans.pkl')
joblib.dump(scaler_C,    'data/scaler_c.pkl')
joblib.dump(cluster_map, 'data/cluster_map.pkl')
print("K-Means saved!")

# STEP 7 — Health Risk Score Calibration

print("\n" + "\n")
print("STEP 7 — Health Risk Score Calibration")
print("\n")

RISK_WEIGHTS = {
    'heat_index': 0.40,
    'humidity'  : 0.25,
    'visibility': 0.15,
    'wind_speed': 0.10,
    'festival'  : 0.10,
}

RISK_RANGES = {
    'heat_index': (float(df['Heat Index'].min()),        float(df['Heat Index'].max())),
    'humidity'  : (float(df['Relative Humidity'].min()), float(df['Relative Humidity'].max())),
    'visibility': (float(df['Visibility'].min()),        float(df['Visibility'].max())),
    'wind_speed': (float(df['Wind Speed'].min()),        float(df['Wind Speed'].max())),
}

PROFILE_MULTIPLIERS = {
    'Healthy Adult'                    : 1.0,
    'Child (under 12)'                 : 1.2,
    'Elderly (60+)'                    : 1.3,
    'Outdoor Worker / Athlete'         : 1.15,
    'Person with respiratory condition': 1.25,
}

PROFILE_ADVICE = {
    'Healthy Adult': {
        'Safe'   : 'Normal day. Stay hydrated if outdoors.',
        'Caution': 'Limit outdoor activity between 12-3 PM.',
        'Danger' : 'Avoid prolonged sun exposure. Drink water every hour.',
        'Extreme': 'Stay indoors. Avoid all outdoor activity between 10 AM-5 PM.',
    },
    'Child (under 12)': {
        'Safe'   : 'Safe for outdoor play. Keep water handy.',
        'Caution': 'Limit outdoor play to morning and evening hours only.',
        'Danger' : 'No outdoor play between 10 AM-4 PM. Keep indoors.',
        'Extreme': 'Keep children indoors all day. Risk of heat exhaustion.',
    },
    'Elderly (60+)': {
        'Safe'   : 'Comfortable day. Short outdoor walks are fine.',
        'Caution': 'Avoid midday sun. Rest frequently if outdoors.',
        'Danger' : 'Stay indoors. High risk of heat stroke for elderly.',
        'Extreme': 'Do not go outside. Keep home cool. Check on neighbours.',
    },
    'Outdoor Worker / Athlete': {
        'Safe'   : 'Good conditions. Stay hydrated during exercise.',
        'Caution': 'Take breaks every 30 min. Drink 500ml water per hour.',
        'Danger' : 'Reschedule outdoor work to early morning or evening.',
        'Extreme': 'Postpone all outdoor work. Risk of heat stroke.',
    },
    'Person with respiratory condition': {
        'Safe'   : 'Air quality acceptable. Carry inhaler as precaution.',
        'Caution': 'Limit time outdoors. Humidity may worsen symptoms.',
        'Danger' : 'Stay indoors. High humidity + heat worsens breathing.',
        'Extreme': 'Do not go outside. Seek medical attention if worsens.',
    },
}

risk_config = {
    'weights'            : RISK_WEIGHTS,
    'ranges'             : RISK_RANGES,
    'profile_multipliers': PROFILE_MULTIPLIERS,
    'profile_advice'     : PROFILE_ADVICE,
}
joblib.dump(risk_config, 'data/risk_config.pkl')
print("Health risk config saved!")

# STEP 8 — Save Summary

print("\n")
print("STEP 8 — Saving Summary")
print("\n")

summary = {
    'total_rows'      : len(df),
    'year_range'      : f"{int(df['year'].min())}–{int(df['year'].max())}",
    'avg_heat_index'  : round(float(df['Heat Index'].mean()), 2),
    'max_heat_index'  : round(float(df['Heat Index'].max()), 2),
    'extreme_days_pct': round(float((df['Heat Index'] > 35).mean() * 100), 1),
    'warming_per_year': warming_per_year,
    'temp_2024'       : temp_2024,
    'temp_2030'       : temp_2030,
    'hi_2030'         : hi_2030,
    'total_warming'   : total_warming,
    'fest_impact'     : fest_impact,
    'silhouette_score': round(sil, 3),
}
joblib.dump(summary, 'data/summary.pkl')
print("Summary saved!")

# STEP 9 — Random Forest Regressor: Heat Index Prediction + K-Fold Cross Validation + Feature Importance

print("\n")
print("STEP 9 — Random Forest Regressor for Heat Index")
print("\n")

FEATURES_RF = [
    'Temperature', 'Dew Point', 'Relative Humidity',
    'Wind Speed', 'Precipitation', 'Visibility',
    'Cloud Cover', 'month', 'season', 'is_festival'
]
TARGET_RF = 'Heat Index'

X_rf = df[FEATURES_RF]
y_rf = df[TARGET_RF]

scaler_rf  = MinMaxScaler()
X_rf_scaled = scaler_rf.fit_transform(X_rf)

X_train, X_test, y_train, y_test = train_test_split(
    X_rf_scaled, y_rf, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)
rf_preds = rf_reg.predict(X_test)

rf_r2   = round(r2_score(y_test, rf_preds), 3)
rf_rmse = round(float(np.sqrt(mean_squared_error(y_test, rf_preds))), 3)

print(f"Random Forest Regressor R²  : {rf_r2}")
print(f"Random Forest Regressor RMSE: {rf_rmse}")

# K-Fold Cross Validation (5 folds)
print("\n--- K-Fold Cross Validation (5 folds) ---")
kf          = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores   = cross_val_score(rf_reg, X_rf_scaled, y_rf,
                               cv=kf, scoring='r2')
cv_mean     = round(float(cv_scores.mean()), 3)
cv_std      = round(float(cv_scores.std()), 3)

print(f"CV R² scores  : {[round(s, 3) for s in cv_scores]}")
print(f"CV Mean R²    : {cv_mean} ± {cv_std}")

# Feature Importance
importances = pd.Series(
    rf_reg.feature_importances_,
    index=FEATURES_RF
).sort_values(ascending=False).round(4)

print("\nFeature Importance:")
print(importances)

# Save
joblib.dump(rf_reg,    'data/model_rf_heatindex.pkl')
joblib.dump(scaler_rf, 'data/scaler_rf.pkl')

feature_importance_data = {
    'features'      : FEATURES_RF,
    'importances'   : importances.to_dict(),
    'rf_r2'         : rf_r2,
    'rf_rmse'       : rf_rmse,
    'cv_mean'       : cv_mean,
    'cv_std'        : cv_std,
    'cv_scores'     : [round(float(s), 3) for s in cv_scores],
    'lr_r2_baseline': lr_r2,
}
joblib.dump(feature_importance_data, 'data/feature_importance.pkl')
print("\nRandom Forest Regressor + Feature Importance saved!")

# STEP 10 — Rain Predictor Decision Tree + Random Forest + Logistic Regression with GridSearchCV tuning + SMOTE

print("\n" + "\n")
print("STEP 10 — Rain Predictor (Should I carry an umbrella?)")
print("\n")

# Create target: will_rain = 1 if precipitation > 0.5mm
df['will_rain'] = (df['Precipitation'] > 0.5).astype(int)

print(f"Rain days    : {df['will_rain'].sum()} "
      f"({df['will_rain'].mean()*100:.1f}%)")
print(f"No rain days : {(df['will_rain']==0).sum()} "
      f"({(df['will_rain']==0).mean()*100:.1f}%)")

FEATURES_RAIN = [
    'Relative Humidity', 'Cloud Cover', 'Dew Point',
    'Wind Speed', 'Visibility', 'month', 'season'
]
TARGET_RAIN = 'will_rain'

X_rain = df[FEATURES_RAIN]
y_rain = df[TARGET_RAIN]

scaler_rain   = MinMaxScaler()
X_rain_scaled = scaler_rain.fit_transform(X_rain)

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_rain_scaled, y_rain, test_size=0.2,
    random_state=42, stratify=y_rain)

# Apply SMOTE to handle class imbalance
print("\n--- Applying SMOTE ---")
print(f"Before SMOTE: {dict(pd.Series(y_r_train).value_counts())}")

smote = SMOTE(random_state=42)
X_r_sm, y_r_sm = smote.fit_resample(X_r_train, y_r_train)
print(f"After  SMOTE: {dict(pd.Series(y_r_sm).value_counts())}")

# Model 1 — Decision Tree with GridSearchCV tuning
print("\n--- Decision Tree + GridSearchCV ---")
dt_params = {
    'max_depth'        : [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params, cv=5, scoring='f1', n_jobs=-1)
dt_grid.fit(X_r_sm, y_r_sm)

dt_best      = dt_grid.best_estimator_
dt_pred      = dt_best.predict(X_r_test)
dt_acc       = round(accuracy_score(y_r_test, dt_pred), 3)
dt_f1        = round(f1_score(y_r_test, dt_pred), 3)
dt_params_best = dt_grid.best_params_

print(f"Best params : {dt_params_best}")
print(f"Accuracy    : {dt_acc}")
print(f"F1 Score    : {dt_f1}")

# Model 2 — Random Forest Classifier
print("\n--- Random Forest Classifier ---")
rf_cls = RandomForestClassifier(
    n_estimators=100, random_state=42)
rf_cls.fit(X_r_sm, y_r_sm)
rf_pred    = rf_cls.predict(X_r_test)
rf_cls_acc = round(accuracy_score(y_r_test, rf_pred), 3)
rf_cls_f1  = round(f1_score(y_r_test, rf_pred), 3)

print(f"Accuracy    : {rf_cls_acc}")
print(f"F1 Score    : {rf_cls_f1}")

# Model 3 — Logistic Regression
print("\n--- Logistic Regression ---")
log_cls = LogisticRegression(random_state=42, max_iter=1000)
log_cls.fit(X_r_sm, y_r_sm)
log_pred    = log_cls.predict(X_r_test)
log_cls_acc = round(accuracy_score(y_r_test, log_pred), 3)
log_cls_f1  = round(f1_score(y_r_test, log_pred), 3)

print(f"Accuracy    : {log_cls_acc}")
print(f"F1 Score    : {log_cls_f1}")

# Pick best rain model
rain_results = {
    'Decision Tree'     : {'model': dt_best,  'acc': dt_acc,     'f1': dt_f1},
    'Random Forest'     : {'model': rf_cls,   'acc': rf_cls_acc, 'f1': rf_cls_f1},
    'Logistic Regression': {'model': log_cls, 'acc': log_cls_acc,'f1': log_cls_f1},
}
best_rain_name  = max(rain_results, key=lambda k: rain_results[k]['f1'])
best_rain_model = rain_results[best_rain_name]['model']
best_rain_f1    = rain_results[best_rain_name]['f1']
best_rain_acc   = rain_results[best_rain_name]['acc']

print(f"\nBest rain model : {best_rain_name}")
print(f"Best F1 Score   : {best_rain_f1}")
print(f"Best Accuracy   : {best_rain_acc}")

print("\nClassification Report (best model):")
print(classification_report(
    y_r_test, best_rain_model.predict(X_r_test),
    target_names=['No Rain', 'Rain']))

# Save rain models
joblib.dump(best_rain_model, 'data/model_rain.pkl')
joblib.dump(scaler_rain,     'data/scaler_rain.pkl')

rain_config = {
    'features'       : FEATURES_RAIN,
    'best_model_name': best_rain_name,
    'best_f1'        : best_rain_f1,
    'best_acc'       : best_rain_acc,
    'best_params'    : dt_params_best,
    'all_results'    : {
        k: {'acc': v['acc'], 'f1': v['f1']}
        for k, v in rain_results.items()
    },
}
joblib.dump(rain_config, 'data/rain_config.pkl')
print("\nRain predictor saved!")


# FINAL SUMMARY

print("\n" + "\n")
print("TRAINING COMPLETE!")
print("\n")
print(f"\nStep 2  — Historical Avg Forecast  : monthly stats built")
print(f"Step 3  — Daily Lookup Table       : 365 rows")
print(f"Step 4  — Linear Regression R²    : {lr_r2}")
print(f"Step 4B — Polynomial Regression R²: {poly_r2}")
print(f"          Better model             : {'Polynomial' if poly_r2 > lr_r2 else 'Linear'}")
print(f"Step 5  — Festival Impact          : +{fest_impact} C")
print(f"Step 6  — K-Means Silhouette       : {sil:.3f}")
print(f"Step 9  — RF Regressor R²          : {rf_r2}")
print(f"          CV Mean R²               : {cv_mean} ± {cv_std}")
print(f"Step 10 — Rain Predictor (best)    : {best_rain_name}")
print(f"          F1 Score                 : {best_rain_f1}")
print(f"          Accuracy                 : {best_rain_acc}")

print("\nSaved files:")
print("  data/monthly_stats.pkl       <-forecast baseline")
print("  data/daily_lookup.pkl        <-day-level forecast")
print("  data/trend_data.pkl          <-linear trend + 2030")
print("  data/poly_trend_data.pkl     <-polynomial trend")
print("  data/festival_data.pkl       <-festival impact")
print("  data/model_kmeans.pkl        <-K-Means clustering")
print("  data/scaler_c.pkl            <-K-Means scaler")
print("  data/cluster_map.pkl         <-cluster labels")
print("  data/risk_config.pkl         <-health risk formula")
print("  data/summary.pkl             <-overall summary")
print("  data/model_rf_heatindex.pkl  <-RF Heat Index regressor")
print("  data/scaler_rf.pkl           <-RF scaler")
print("  data/feature_importance.pkl  <-feature importance data")
print("  data/model_rain.pkl          <-rain predictor")
print("  data/scaler_rain.pkl         <-rain scaler")
print("  data/rain_config.pkl         <-rain model metadata")
