import logging
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import run_feature_engineering
from src.train_random_forest import train_random_forest
from src.train_decision_tree import train_decision_tree
from src.train_logistic import train_logistic
from src.evaluate import compare_models
from src.feature_analysis import run_feature_analysis
from src.utils import setup_logging

def run_pipeline():
    logger = setup_logging()
    logger.info("=== BẮT ĐẦU QUÁ TRÌNH DỰ ĐOÁN KẾT QUẢ HỌC TẬP CỦA SINH VIÊN ===")

    try:
        # Bước 1: Tiền xử lý dữ liệu thô
        logger.info("Bước 1: Đang tiền xử lý dữ liệu...")
        preprocessor = DataPreprocessor(
            raw_data_path='data/raw/Student_Performance_Dataset.csv',
            processed_data_dir='data/processed'
        )
        preprocessor.run_pipeline()

        # Bước 2: Kỹ thuật đặc trưng (Tập trung vào hành vi)
        logger.info("Bước 2: Đang xử lý đặc trưng hành vi...")
        run_feature_engineering()

        # Bước 3: Huấn luyện đồng thời 3 mô hình
        logger.info("Bước 3: Đang huấn luyện các mô hình...")
        train_logistic()
        train_decision_tree()
        train_random_forest()

        # Bước 4: Đánh giá và so sánh
        logger.info("Bước 4: Đang đánh giá hiệu suất...")
        compare_models()

        # Bước 5: Phân tích mức độ ảnh hưởng của các yếu tố
        logger.info("Bước 5: Đang phân tích đặc trưng (Feature Analysis)...")
        run_feature_analysis()

        logger.info("=== QUY TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG. ===")
        print("\n>>> Kiểm tra thư mục 'results/' để xem biểu đồ và kết quả so sánh.")

    except Exception as e:
        logger.error(f"Lỗi hệ thống: {str(e)}")

if __name__ == "__main__":
    run_pipeline()