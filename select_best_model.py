import os
import re

def parse_metrics(file_path):
    metrics = {}
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        content = f.read()
        rmse_match = re.search(r'RMSE:\s*([\d.]+)', content)
        if rmse_match:
            metrics['RMSE'] = float(rmse_match.group(1))
    return metrics

def main():
    metrics_dir = 'trained_models'
    models_metrics = {
        'xgboost': os.path.join(metrics_dir, 'xgboost_metrics.txt'),
        'lightgbm': os.path.join(metrics_dir, 'lightgbm_metrics.txt'),
        'sarimax': os.path.join(metrics_dir, 'sarimax_metrics.txt')
    }

    best_model_name = None
    best_rmse = float('inf')

    print("Comparing models based on RMSE...")
    for model_name, path in models_metrics.items():
        metrics = parse_metrics(path)
        if metrics:
            rmse = metrics['RMSE']
            print(f"- {model_name}: RMSE = {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name
        else:
            print(f"- {model_name}: No metrics found.")

    if best_model_name:
        print(f"\nBest model selected: {best_model_name} with RMSE {best_rmse}")
        with open('best_model.txt', 'w') as f:
            f.write(best_model_name)
    else:
        print("\nError: No valid models found to select.")
        exit(1)

if __name__ == "__main__":
    main()
