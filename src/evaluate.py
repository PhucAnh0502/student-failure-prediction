import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from src import utils

def compare_models():
    # 1. Load dataset
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    models_to_test = {
        'Random Forest': 'models/random_forest_model.pkl',
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl'
    }

    comparison_results = []

    for name, path in models_to_test.items():
        model = utils.load_model(path)
        y_pred = model.predict(X_test)

        # Tính toán các chỉ số đánh giá
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }
        comparison_results.append(metrics)

    # 2. In kết quả so sánh
    df_results = pd.DataFrame(comparison_results)
    print("\n=== KẾT QUẢ SO SÁNH CÁC MÔ HÌNH ===")
    print(df_results)

    # 3. Lưu kết quả so sánh
    utils.save_results(df_results, 'results/metrics.csv')

if __name__ == "__main__":
    compare_models()