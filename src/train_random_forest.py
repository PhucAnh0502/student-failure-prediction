import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src import utils

def train_random_forest():
    # 1. Load dữ liệu đã xử lý
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    # 2. Khởi tạo mô hình
    # n_estimators : Số lượng cây
    # random_state : Đảm bảo kết quả giống nhau mỗi lần chạy
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

    # 3. Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình Random Forest...")
    model.fit(X_train, y_train)

    # 4. Lưu mô hình đã huấn luyện
    utils.save_model(model, 'models/random_forest_model.pkl')
    print("Hoàn thành huấn luyện mô hình Random Forest.")

if __name__ == "__main__":
    train_random_forest()