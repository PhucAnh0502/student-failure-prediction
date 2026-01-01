import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class xử lý tiền xử lý dữ liệu"""
    
    def __init__(self, raw_data_path='data/raw/student-mat.csv', 
                 processed_data_dir='data/processed'):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.df = None
        
    def load_data(self):
        """Đọc dữ liệu từ file CSV"""
        try:
            logger.info(f"Đang đọc dữ liệu từ {self.raw_data_path}")
            self.df = pd.read_csv(self.raw_data_path)
            logger.info(f"Đọc thành công {len(self.df)} dòng dữ liệu")
            logger.info(f"Số cột: {len(self.df.columns)}")
            logger.info(f"Các cột: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu: {e}")
            raise
    
    def explore_data(self):
        """Khám phá dữ liệu cơ bản"""
        if self.df is None:
            logger.error("Chưa load dữ liệu. Gọi load_data() trước.")
            return
        
        logger.info("\n=== THÔNG TIN DỮ LIỆU ===")
        logger.info(f"Kích thước: {self.df.shape}")
        logger.info(f"\nKiểu dữ liệu:\n{self.df.dtypes}")
        logger.info(f"\nThống kê mô tả:\n{self.df.describe()}")
        logger.info(f"\nMissing values:\n{self.df.isnull().sum()}")
        logger.info(f"\nDuplicate rows: {self.df.duplicated().sum()}")
        
        # Phân bố nhãn Pass/Fail
        if 'Pass_Fail' in self.df.columns:
            logger.info(f"\nPhân bố nhãn Pass/Fail:\n{self.df['Pass_Fail'].value_counts()}")
            logger.info(f"Tỷ lệ Pass: {(self.df['Pass_Fail']=='Pass').mean()*100:.2f}%")
    
    def clean_data(self):
        """Làm sạch dữ liệu"""
        if self.df is None:
            logger.error("Chưa load dữ liệu")
            return
        
        logger.info("\n=== BẮT ĐẦU LÀM SẠCH DỮ LIỆU ===")
        initial_rows = len(self.df)
        
        # 1. Xử lý duplicate
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Tìm thấy {duplicates} dòng trùng lặp, đang xóa...")
            self.df = self.df.drop_duplicates()
        
        # 2. Xử lý missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Tìm thấy missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Điền missing cho các cột số bằng median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.info(f"Điền {col} bằng median: {median_val}")
            
            # Điền missing cho các cột phân loại bằng mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    logger.info(f"Điền {col} bằng mode: {mode_val}")
        
        # 3. Xử lý outliers (optional - có thể bật nếu cần)
        # self._handle_outliers()
        
        # 4. Chuẩn hóa giá trị
        self._standardize_values()
        
        logger.info(f"Làm sạch xong. Số dòng: {initial_rows} -> {len(self.df)}")
        return self.df
    
    def _standardize_values(self):
        """Chuẩn hóa giá trị các cột"""
        # Chuẩn hóa Gender
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].str.strip().str.capitalize()
        
        # Chuẩn hóa Pass_Fail
        if 'Pass_Fail' in self.df.columns:
            self.df['Pass_Fail'] = self.df['Pass_Fail'].str.strip().str.capitalize()
        
        # Chuẩn hóa Internet_Access, Extracurricular_Activities
        for col in ['Internet_Access', 'Extracurricular_Activities']:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip().str.capitalize()
        
        logger.info("Chuẩn hóa giá trị hoàn tất")
    
    def create_target_variable(self, threshold=50.0):
        """
        Tạo biến mục tiêu Pass/Fail dựa trên Final_Percentage
        
        Args:
            threshold: Ngưỡng điểm đậu (mặc định 50.0)
        """
        if 'Final_Percentage' not in self.df.columns:
            logger.error("Không tìm thấy cột Final_Percentage")
            return
        
        # Tạo nhãn nhị phân: 0 = Fail, 1 = Pass
        self.df['Target'] = (self.df['Final_Percentage'] >= threshold).astype(int)
        
        logger.info(f"\nTạo biến mục tiêu với ngưỡng {threshold}:")
        logger.info(f"Pass (1): {(self.df['Target']==1).sum()}")
        logger.info(f"Fail (0): {(self.df['Target']==0).sum()}")
        logger.info(f"Tỷ lệ đậu: {self.df['Target'].mean()*100:.2f}%")
        
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành train và test
        
        Args:
            test_size: Tỷ lệ test set (mặc định 0.2 = 20%)
            random_state: Random seed để tái lập kết quả
        """
        if self.df is None or 'Target' not in self.df.columns:
            logger.error("Cần load và tạo target variable trước")
            return None, None
        
        logger.info(f"\n=== CHIA DỮ LIỆU (Train/Test = {1-test_size}/{test_size}) ===")
        
        # Chia dữ liệu với stratify để giữ tỷ lệ Pass/Fail
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['Target']
        )
        
        logger.info(f"Train set: {len(train_df)} mẫu")
        logger.info(f"Test set: {len(test_df)} mẫu")
        logger.info(f"Train - Tỷ lệ Pass: {train_df['Target'].mean()*100:.2f}%")
        logger.info(f"Test - Tỷ lệ Pass: {test_df['Target'].mean()*100:.2f}%")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df, test_df):
        """Lưu dữ liệu đã xử lý"""
        # Tạo thư mục nếu chưa tồn tại
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu train và test
        train_path = self.processed_data_dir / 'train.csv'
        test_path = self.processed_data_dir / 'test.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"\n=== LƯU DỮ LIỆU ===")
        logger.info(f"Train data: {train_path}")
        logger.info(f"Test data: {test_path}")
        
        # Lưu thêm metadata
        metadata = {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(test_df),
            'n_features': len(self.df.columns) - 1,  # Trừ target
            'train_pass_rate': float(train_df['Target'].mean()),
            'test_pass_rate': float(test_df['Target'].mean())
        }
        
        metadata_path = self.processed_data_dir / 'metadata.txt'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Metadata: {metadata_path}")
        logger.info("Lưu dữ liệu hoàn tất!")
    
    def run_pipeline(self, test_size=0.2, random_state=42, threshold=50.0):
        """
        Chạy toàn bộ pipeline tiền xử lý
        
        Args:
            test_size: Tỷ lệ test set
            random_state: Random seed
            threshold: Ngưỡng điểm đậu
        """
        logger.info("="*60)
        logger.info("BẮT ĐẦU DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Bước 1: Load data
        self.load_data()
        
        # Bước 2: Explore data
        self.explore_data()
        
        # Bước 3: Clean data
        self.clean_data()
        
        # Bước 4: Create target
        self.create_target_variable(threshold=threshold)
        
        # Bước 5: Split data
        train_df, test_df = self.split_data(test_size=test_size, random_state=random_state)
        
        # Bước 6: Save data
        if train_df is not None and test_df is not None:
            self.save_processed_data(train_df, test_df)
        
        logger.info("="*60)
        logger.info("HOÀN THÀNH DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        return train_df, test_df


def main():
    """Hàm main để chạy preprocessing"""
    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/Student_Performance_Dataset.csv',
        processed_data_dir='data/processed'
    )
    
    # Chạy pipeline
    train_df, test_df = preprocessor.run_pipeline(
        test_size=0.2,
        random_state=42,
        threshold=50.0
    )
    
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = main()
