import pandas as pd
import numpy as np
import lightgbm as lgb
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

    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1: y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1: y_test = y_test.iloc[:, 0]

    # Data Leakage Prevention
    features_to_drop = ['moyenne_ventes_produit', 'moyenne_ventes_magasin']
    X_train = X_train.drop(columns=[col for col in features_to_drop if col in X_train.columns])
    X_test = X_test.drop(columns=[col for col in features_to_drop if col in X_test.columns])

    # Convert categorical columns if they exist
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
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    tscv = TimeSeriesSplit(n_splits=3)
    rmse_scores = []

    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            param, 
            dtrain, 
            valid_sets=[dval], 
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

print("Starting Hyperparameter Optimization for LightGBM...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

print("\nBest Hyperparameters:")
print(study.best_params)

# 3. Final Training and Evaluation
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['random_state'] = 42
best_params['verbose'] = -1

print("Training final LightGBM model...")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

final_model = lgb.train(
    best_params,
    train_data,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)

# Predictions
predictions = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f"\nFinal Results on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Saving
model_path = 'trained_models/lightgbm_model_optimized.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)

with open('trained_models/lightgbm_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nBest Params: {best_params}")

print(f"Optimized LightGBM model saved: {model_path}")
