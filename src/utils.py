import pickle
import joblib
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Cấu hình logging mặc định
def setup_logging(log_file='logs/pipeline.log', level=logging.INFO):
    """
    Thiết lập logging cho dự án
    
    Args:
        log_file: Đường dẫn file log
        level: Mức độ logging
    """
    # Tạo thư mục logs nếu chưa tồn tại
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Cấu hình logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Hàm lưu mô hình
def save_model(model, model_path, method='joblib'):
    """
    Lưu mô hình đã huấn luyện
    
    Args:
        model: Mô hình cần lưu
        model_path: Đường dẫn lưu
        method: Phương pháp lưu ('joblib' hoặc 'pickle')
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == 'joblib':
        joblib.dump(model, model_path)
    elif method == 'pickle':
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Method không hợp lệ: {method}")
    
    logging.info(f"Đã lưu mô hình: {model_path}")
    return model_path

# Hàm load mô hình
def load_model(model_path, method='joblib'):
    """
    Load mô hình đã lưu
    
    Args:
        model_path: Đường dẫn file model
        method: Phương pháp load ('joblib' hoặc 'pickle')
    
    Returns:
        model: Mô hình đã load
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    
    if method == 'joblib':
        model = joblib.load(model_path)
    elif method == 'pickle':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Method không hợp lệ: {method}")
    
    logging.info(f"Đã load mô hình: {model_path}")
    return model

# Hàm load dữ liệu
def load_data(data_path):
    """
    Load dữ liệu từ file CSV
    
    Args:
        data_path: Đường dẫn file CSV
    
    Returns:
        df: DataFrame
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")
    
    df = pd.read_csv(data_path)
    logging.info(f"Đã load dữ liệu: {data_path} ({len(df)} dòng)")
    
    return df

# Hàm lưu kết quả đánh giá
def save_results(results, output_path):
    """
    Lưu kết quả đánh giá mô hình
    
    Args:
        results: Dictionary hoặc DataFrame chứa kết quả
        output_path: Đường dẫn file output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        # Lưu dạng JSON
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        # Lưu dạng CSV
        elif output_path.suffix == '.csv':
            df = pd.DataFrame([results])
            df.to_csv(output_path, index=False)
    
    elif isinstance(results, pd.DataFrame):
        results.to_csv(output_path, index=False)
    
    logging.info(f"Đã lưu kết quả: {output_path}")
    return output_path

# Hàm kiểm tra tính hợp lệ của dữ liệu
def validate_data(df, required_columns):
    """
    Kiểm tra tính hợp lệ của dữ liệu
    
    Args:
        df: DataFrame cần kiểm tra
        required_columns: List các cột bắt buộc
    
    Returns:
        bool: True nếu hợp lệ
    """
    # Kiểm tra các cột bắt buộc
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logging.error(f"Thiếu các cột: {missing_cols}")
        return False
    
    # Kiểm tra missing values
    missing_count = df[required_columns].isnull().sum().sum()
    if missing_count > 0:
        logging.warning(f"Phát hiện {missing_count} missing values")
    
    # Kiểm tra duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logging.warning(f"Phát hiện {dup_count} dòng trùng lặp")
    
    logging.info("Validation hoàn tất")
    return True

# Hàm lấy danh sách cột đặc trưng
def get_feature_columns(df, exclude_cols=None):
    """
    Lấy danh sách cột đặc trưng (loại bỏ target và các cột không cần)
    
    Args:
        df: DataFrame
        exclude_cols: List các cột cần loại bỏ
    
    Returns:
        list: Danh sách cột đặc trưng
    """
    if exclude_cols is None:
        exclude_cols = ['Student_ID', 'Target', 'Pass_Fail', 'Performance_Level']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logging.info(f"Số lượng đặc trưng: {len(feature_cols)}")
    return feature_cols

# Hàm tạo metadata cho mô hình
def create_model_metadata(model_name, model, train_score, test_score, 
                          training_time, hyperparameters=None):
    """
    Tạo metadata cho mô hình
    
    Args:
        model_name: Tên mô hình
        model: Đối tượng mô hình
        train_score: Điểm trên tập train
        test_score: Điểm trên tập test
        training_time: Thời gian huấn luyện (giây)
        hyperparameters: Dict các tham số
    
    Returns:
        dict: Metadata
    """
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'train_score': float(train_score),
        'test_score': float(test_score),
        'training_time_seconds': float(training_time),
        'created_at': datetime.now().isoformat(),
        'hyperparameters': hyperparameters or {}
    }
    
    return metadata

# Hàm in phân cách đẹp
def print_separator(title="", width=60, char="="):
    """
    In dòng phân cách đẹp
    
    Args:
        title: Tiêu đề (optional)
        width: Độ rộng
        char: Ký tự phân cách
    """
    if title:
        side = (width - len(title) - 2) // 2
        print(f"{char * side} {title} {char * side}")
    else:
        print(char * width)

# Hàm kiểm tra class imbalance trong target variable mục đích phân loại
def check_class_imbalance(y, threshold=0.3):
    """
    Kiểm tra class imbalance
    
    Args:
        y: Target variable
        threshold: Ngưỡng cảnh báo (nếu class nhỏ < threshold)
    
    Returns:
        dict: Thông tin về class balance
    """
    value_counts = pd.Series(y).value_counts()
    proportions = value_counts / len(y)
    
    min_proportion = proportions.min()
    is_imbalanced = min_proportion < threshold
    
    result = {
        'is_imbalanced': is_imbalanced,
        'class_counts': value_counts.to_dict(),
        'class_proportions': proportions.to_dict(),
        'min_proportion': float(min_proportion),
        'imbalance_ratio': float(proportions.max() / min_proportion)
    }
    
    if is_imbalanced:
        logging.warning(f"Phát hiện class imbalance! Tỷ lệ class nhỏ: {min_proportion:.2%}")
    
    return result

# Hàm tạo thư mục experiment mới với timestamp mục đích lưu trữ kết quả
def create_experiment_folder(base_dir='experiments'):
    """
    Tạo thư mục cho experiment mới với timestamp
    
    Args:
        base_dir: Thư mục gốc
    
    Returns:
        Path: Đường dẫn thư mục experiment
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f'exp_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Tạo experiment folder: {exp_dir}")
    return exp_dir


# Hàm lấy các cột số
def get_numeric_columns(df):
    """Lấy các cột số"""
    return df.select_dtypes(include=[np.number]).columns.tolist()


# Hàm lấy các cột phân loại
def get_categorical_columns(df):
    """Lấy các cột phân loại"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


# Hàm tính memory usage của DataFrame
def memory_usage_mb(df):
    """Tính memory usage của DataFrame (MB)"""
    return df.memory_usage(deep=True).sum() / 1024**2

# Hàm in tóm tắt nhanh về DataFrame
def quick_summary(df):
    """
    In tóm tắt nhanh về DataFrame
    
    Args:
        df: DataFrame
    """
    print_separator("DATA SUMMARY")
    print(f"Shape: {df.shape}")
    print(f"Memory: {memory_usage_mb(df):.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print_separator()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Utils module loaded successfully")