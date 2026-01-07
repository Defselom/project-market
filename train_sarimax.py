import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
import os
import pickle

warnings.filterwarnings("ignore")

# Create output directory
os.makedirs('trained_models', exist_ok=True)

# 1. Loading and Preprocessing
def load_data():
    # Use the same data as other models for consistency in the new daily-aggregated pipeline
    train_X_path = 'donnees_pretraitees/X_train_7_jours.csv'
    train_y_path = 'donnees_pretraitees/y_train_7_jours.csv'
    test_X_path = 'donnees_pretraitees/X_test_7_jours.csv'
    test_y_path = 'donnees_pretraitees/y_test_7_jours.csv'

    X_train = pd.read_csv(train_X_path)
    y_train = pd.read_csv(train_y_path)
    X_test = pd.read_csv(test_X_path)
    y_test = pd.read_csv(test_y_path)

    if isinstance(y_train, pd.DataFrame): y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame): y_test = y_test.iloc[:, 0]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

# 2. Grid Search Function
def optimize_sarima(series, exog, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [s]))
    
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    print("Starting Grid Search for SARIMAX...")
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    series,
                    exog=exog,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = mod.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
                    best_seasonal_order = param_seasonal
                    best_model = results
            except:
                continue
                
    if best_model:
        print(f"Best model found: SARIMAX{best_order}x{best_seasonal_order} - AIC:{best_aic}")
    return best_model, best_order, best_seasonal_order

# 3. Training
print("\n=== Training Global SARIMAX Model ===")

# Exogenous variables: use a subset of features from X_train
exog_cols = ['holiday_promotion', 'year', 'month', 'dayofweek']
exog_train = X_train[[col for col in exog_cols if col in X_train.columns]]
exog_test = X_test[[col for col in exog_cols if col in X_test.columns]]

# Grid Search ranges (reduced for pipeline speed)
p = d = q = range(0, 2)
P = D = Q = range(0, 1) # Seasonal simplified
s = 7 # Weekly seasonality for daily data

best_model, best_order, best_seasonal_order = optimize_sarima(
    y_train, exog_train, p, d, q, P, D, Q, s
)

# Evaluation
if best_model:
    predictions = best_model.forecast(steps=len(y_test), exog=exog_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print(f"SARIMAX Results - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Saving
    model_path = 'trained_models/sarimax_model_optimized.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    with open('trained_models/sarimax_metrics.txt', 'w') as f:
        f.write(f"RMSE: {rmse}\nMAE: {mae}\nOrder: {best_order}\nSeasonal Order: {best_seasonal_order}")

    print(f"SARIMAX model saved: {model_path}")
else:
    print("No SARIMAX model could be trained.")
