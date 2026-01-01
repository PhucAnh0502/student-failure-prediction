import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE 
from pathlib import Path

def run_feature_engineering():
    # 1. Đọc dữ liệu 
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # 2. Loại bỏ các cột không cần thiết
    drop_cols = ['Student_ID', 'Pass_Fail', 'Performance_Level', 'Final_Percentage',
                'Math_Score', 'Science_Score', 'English_Score', 'Previous_Year_Score']
    target_col = 'Target'
    
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns] + [target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns] + [target_col])
    y_test = test_df[target_col]

    # 3. Mã hóa Label Encoding
    cat_cols = X_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # 4. Chuẩn hóa dữ liệu (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. THỰC HIỆN OVERSAMPLING VỚI SMOTE
    print(f"Trước khi SMOTE - Nhãn 0 (Trượt): {sum(y_train == 0)}, Nhãn 1 (Đậu): {sum(y_train == 1)}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Sau khi SMOTE - Nhãn 0 (Trượt): {sum(y_train_resampled == 0)}, Nhãn 1 (Đậu): {sum(y_train_resampled == 1)}")

    # 6. Lưu dữ liệu đã cân bằng
    output_path = Path('data/processed')
    X_train_final = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    X_train_final.to_csv(output_path / 'X_train.csv', index=False)
    X_test_final.to_csv(output_path / 'X_test.csv', index=False)
    y_train_resampled.to_csv(output_path / 'y_train.csv', index=False)
    y_test.to_csv(output_path / 'y_test.csv', index=False)
    
    joblib.dump(scaler, 'models/scaler.joblib')

if __name__ == "__main__":
    run_feature_engineering()