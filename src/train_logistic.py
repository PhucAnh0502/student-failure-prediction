import pandas as pd
from sklearn.linear_model import LogisticRegression
from src import utils

def train_logistic():
    # 1. Load dữ liệu đã xử lý
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    # 2. Khởi tạo mô hình
    # max_iter : Số vòng lặp tối đa để hội tụ
    # random_state : Đảm bảo kết quả giống nhau mỗi lần chạy
    model = LogisticRegression(max_iter=1000, random_state=42)

    # 3. Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình Logistic Regression...")
    model.fit(X_train, y_train)

    # 4. Lưu mô hình đã huấn luyện
    utils.save_model(model, 'models/logistic_regression_model.pkl')
    print("Hoàn thành huấn luyện mô hình Logistic Regression.")

if __name__ == "__main__":
    train_logistic()