import os
import shutil

def main():
    if not os.path.exists('best_model.txt'):
        print("Error: best_model.txt not found. Run selection script first.")
        exit(1)

    with open('best_model.txt', 'r') as f:
        best_model_name = f.read().strip()

    model_files = {
        'xgboost': 'trained_models/xgboost_model_optimized.pkl',
        'lightgbm': 'trained_models/lightgbm_model_optimized.pkl',
        'sarimax': None # We'll find it below
    }

    if best_model_name == 'sarimax':
        # Find the first pkl file starting with sarimax_model_
        files = os.listdir('trained_models')
        for f in files:
            if f.startswith('sarimax_model_') and f.endswith('.pkl'):
                model_files['sarimax'] = os.path.join('trained_models', f)
                break

    source_path = model_files.get(best_model_name)
    
    if not source_path or not os.path.exists(source_path):
        print(f"Error: Model file for {best_model_name} not found at {source_path}")
        exit(1)

    os.makedirs('deploy', exist_ok=True)
    destination_path = os.path.join('deploy', 'best_model.pkl')
    
    print(f"Deploying {best_model_name} from {source_path} to {destination_path}...")
    shutil.copy(source_path, destination_path)
    
    # Also copy the metrics
    metrics_source = f'trained_models/{best_model_name}_metrics.txt'
    if os.path.exists(metrics_source):
        shutil.copy(metrics_source, os.path.join('deploy', 'best_model_metrics.txt'))

    print("Deployment successful.")

if __name__ == "__main__":
    main()
