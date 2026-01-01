import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src import utils

def train_decision_tree():
    # 1. Load dữ liệu đã xử lý
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    # 2. Khởi tạo mô hình
    # max_depth : Độ sâu tối đa của cây
    # random_state : Đảm bảo kết quả giống nhau mỗi lần chạy
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # 3. Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình Decision Tree...")
    model.fit(X_train, y_train)

    # 4. Lưu mô hình đã huấn luyện
    utils.save_model(model, 'models/decision_tree_model.pkl')
    print("Hoàn thành huấn luyện mô hình Decision Tree.")

if __name__ == "__main__":
    train_decision_tree()