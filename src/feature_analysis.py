import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import utils
import os

def run_feature_analysis():
    # 1. Chuẩn bị dữ liệu và thư mục
    os.makedirs('results', exist_ok=True)
    # Lấy tên các cột từ file đã xử lý đặc trưng
    X_train = pd.read_csv('data/processed/X_train.csv')
    feature_names = X_train.columns
    
    # 2. Phân tích Random Forest
    rf_model = utils.load_model('models/random_forest_model.pkl')
    rf_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # 3. Phân tích Decision Tree
    dt_model = utils.load_model('models/decision_tree_model.pkl')
    dt_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # 4. Phân tích Logistic Regression
    log_model = utils.load_model('models/logistic_regression_model.pkl')
    log_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': log_model.coef_[0]
    })
    log_coefs['Abs_Effect'] = log_coefs['Coefficient'].abs()
    log_coefs = log_coefs.sort_values(by='Abs_Effect', ascending=False)

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(12, 18))

    # Biểu đồ 1: Random Forest
    plt.subplot(3, 1, 1)
    sns.barplot(x='Importance', y='Feature', data=rf_importances, palette='viridis')
    plt.title('Mức độ ảnh hưởng của các yếu tố (Random Forest)')

    # Biểu đồ 2: Decision Tree
    plt.subplot(3, 1, 2)
    sns.barplot(x='Importance', y='Feature', data=dt_importances, palette='magma')
    plt.title('Mức độ ảnh hưởng của các yếu tố (Decision Tree)')

    # Biểu đồ 3: Logistic Regression
    plt.subplot(3, 1, 3)
    # Tô màu: Xanh cho yếu tố giúp Pass, Đỏ cho yếu tố dễ gây Fail
    colors = ['green' if x > 0 else 'red' for x in log_coefs['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=log_coefs, palette=colors)
    plt.title('Trọng số các yếu tố (Logistic Regression: Xanh=Tốt, Đỏ=Xấu)')

    plt.tight_layout()
    plt.savefig('results/feature_importance_comparison.png')
    print("Đã tạo biểu đồ so sánh tại results/feature_importance_comparison.png")
    
    # Lưu bảng dữ liệu ra CSV để xem chi tiết
    rf_importances.to_csv('results/rf_importance.csv', index=False)

if __name__ == "__main__":
    run_feature_analysis()