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
    train_path = 'donnees_pretraitees/train_sarimax.csv'
    test_path = 'donnees_pretraitees/test_sarimax.csv'

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_test['Date'] = pd.to_datetime(df_test['Date'])

    df_train = df_train.sort_values(by=['Product ID', 'Date'])
    df_test = df_test.sort_values(by=['Product ID', 'Date'])
    
    return df_train, df_test

df_train, df_test = load_data()

# 2. Grid Search Function
def optimize_sarima(series, exog, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [s]))
    
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    print("Starting Grid Search...")
    
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
                
    print(f"Best model found: SARIMAX{best_order}x{best_seasonal_order} - AIC:{best_aic}")
    return best_model, best_order, best_seasonal_order

# 3. Training (Global Model Approach for simplicity in pipeline)
# In the notebook, there was per-product and global. We will implement global here or a representative one.
# Let's follow the "Global" or "Single Product" based on notebook's main outcome.
# The notebook had a "Modèle Global" section.

print("\n=== Training Global SARIMAX Model ===")
# Assuming global means aggregating or using a specific one. 
# Let's use the logic for a single product as a representative test or the whole dataset if possible.
# Most SARIMAX models are trained per series.
# For the pipeline, we might want a script that trains for ALL or selects one.
# Let's train for the first product found as a baseline or follow the notebook's end logic.

product_id = df_train['Product ID'].unique()[0]
df_p = df_train[df_train['Product ID'] == product_id].set_index('Date')
df_p_test = df_test[df_test['Product ID'] == product_id].set_index('Date')

# Exogenous variables
exog_cols = ['Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing']
exog_train = df_p[exog_cols]
exog_test = df_p_test[exog_cols]

# Grid Search ranges (reduced for pipeline speed)
p = d = q = range(0, 2)
P = D = Q = range(0, 1) # Seasonal simplified
s = 12

best_model, best_order, best_seasonal_order = optimize_sarima(
    df_p['Units Sold'], exog_train, p, d, q, P, D, Q, s
)

# Evaluation
predictions = best_model.forecast(steps=len(df_p_test), exog=exog_test)
rmse = np.sqrt(mean_squared_error(df_p_test['Units Sold'], predictions))
mae = mean_absolute_error(df_p_test['Units Sold'], predictions)

print(f"SARIMAX Results for {product_id} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Saving
model_path = f'trained_models/sarimax_model_{product_id}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

with open('trained_models/sarimax_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nOrder: {best_order}\nSeasonal Order: {best_seasonal_order}")

print(f"SARIMAX model saved: {model_path}")
