import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import os
import pickle
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# Create output directory
os.makedirs('trained_models', exist_ok=True)

# 1. Loading and Preparation
def load_data():
    train_X_path = 'donnees_pretraitees/X_train_7_jours.csv'
    train_y_path = 'donnees_pretraitees/y_train_7_jours.csv'
    test_X_path = 'donnees_pretraitees/X_test_7_jours.csv'
    test_y_path = 'donnees_pretraitees/y_test_7_jours.csv'

    X_train = pd.read_csv(train_X_path)
    y_train = pd.read_csv(train_y_path)
    X_test = pd.read_csv(test_X_path)
    y_test = pd.read_csv(test_y_path)

    if y_train.shape[1] == 1: y_train = y_train.iloc[:, 0]
    if y_test.shape[1] == 1: y_test = y_test.iloc[:, 0]

    cat_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

# 2. Optimization
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'enable_categorical': True,
        'n_jobs': -1,
        'random_state': 42,
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        d_tr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        d_val = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        model = xgb.train(
            params,
            d_tr,
            num_boost_round=500,
            evals=[(d_val, 'Val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        p_val = model.predict(d_val)
        rmse = np.sqrt(mean_squared_error(y_val, p_val))
        scores.append(rmse)

    return np.mean(scores)

print("Starting Hyperparameter Optimization for XGBoost...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\nBest Hyperparameters:")
print(study.best_params)

# 3. Final Training and Evaluation
best_params = study.best_params
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'
best_params['enable_categorical'] = True
best_params['n_jobs'] = -1
best_params['random_state'] = 42

print("Training final XGBoost model...")
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'Test')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Predictions
predictions = final_model.predict(dtest)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f"\nFinal Results on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Saving
model_path = 'trained_models/xgboost_model_optimized.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)

with open('trained_models/xgboost_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nBest Params: {best_params}")

print(f"Optimized XGBoost model saved: {model_path}")
