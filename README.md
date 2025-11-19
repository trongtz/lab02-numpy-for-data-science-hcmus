# BankChurners Prediction Project

## 1. Mô tả ngắn gọn
Dự án này nhằm **dự đoán khả năng rời bỏ của khách hàng thẻ tín dụng** dựa trên dữ liệu khách hàng BankChurners.  
Mục tiêu là áp dụng **NumPy** để xử lý dữ liệu, **one-hot encoding**, chuẩn hóa dữ liệu numeric và xây dựng mô hình **Logistic Regression** và **Random Forest** để dự đoán khách hàng có rời bỏ hay không.

---

## 2. Mục lục
1. [Giới thiệu](#3-giới-thiệu)
2. [Dataset](#4-dataset)
3. [Method](#5-method)
4. [Installation & Setup](#6-installation--setup)
5. [Usage](#7-usage)
6. [Results](#8-results)
7. [Project Structure](#9-project-structure)
8. [Challenges & Solutions](#10-challenges--solutions)
9. [Future Improvements](#11-future-improvements)
10. [Contributors](#12-contributors)
11. [License](#13-license)

---

## 3. Giới thiệu
- **Bài toán:** Dự đoán khả năng khách hàng rời bỏ thẻ tín dụng dựa trên các thông tin cá nhân và tài chính.  
- **Động lực và ứng dụng thực tế:** Giúp ngân hàng **dự đoán churn** và thực hiện các biện pháp giữ khách hàng.  
- **Mục tiêu:** Xây dựng pipeline dữ liệu bằng NumPy, trực quan hóa bằng Matplotlib/Seaborn, và huấn luyện mô hình học máy.

---

## 4. Dataset
- **Nguồn:** [BankChurners.csv]  
- **Số lượng mẫu:** 10127 khách hàng  
- **Features:** Thông tin cá nhân và tài chính (Age, Gender, Income_Category, Credit_Limit, Card_Category, v.v.)  
- **Target:** `Attrition_Flag` (0 = Existing Customer, 1 = Attrited Customer)

---

## 5. Method

### 5.1 Xử lý dữ liệu (NumPy)
- Load dữ liệu CSV bằng NumPy.
- Thay thế missing value (`Unknown`) bằng **mode** của cột.
- Loại bỏ các cột không cần thiết (`CLIENTNUM`, cột dự đoán Naive Bayes).
- Tách target `Attrition_Flag`.
- One-hot encoding các cột categorical (`Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category`).
- Chuẩn hóa numeric bằng **z-score** cho Logistic Regression; Random Forest không cần scale.

### 5.2 Mô hình
- **Logistic Regression**:
  - Sử dụng StandardScaler trên feature numeric.
  - max_iter=2000.
- **Random Forest**:
  - n_estimators=200, random_state=42.
  - Không cần chuẩn hóa dữ liệu.

### 5.3 Đánh giá
- Sử dụng metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
- Phân tích phân bố label để kiểm tra imbalance.

---

## 6. Installation & Setup

### Clone repository
git clone <repo-url>

### Tạo môi trường ảo (khuyến nghị)

python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

### Cài đặt dependencies
pip install -r requirements.txt

## 7. Usage

- Preprocessing: src/data_processing.py

- Visualizations: src/visualization.py

- Modeling: src/models.py

- Notebook: notebooks/01_data_exploration.ipynb, 02_preprocessing.ipynb, 03_modeling.ipynb

Chạy các notebook theo thứ tự để:

- Khám phá dữ liệu

- Xử lý missing value, one-hot, scale numeric

- Huấn luyện Logistic Regression và Random Forest

## 8. Results

### Logistic Regression

- Accuracy: ~88–90%

- Confusion Matrix và classification report hiển thị f1-score của class 1 (Attrited Customer) hơi thấp do imbalance.

### Random Forest

- Accuracy: ~95%

- Recall của class 1 tốt hơn Logistic Regression.

- Không cần chuẩn hóa dữ liệu, hiệu năng ổn định.

### Phân bố label

- Full dataset: Existing ~84%, Attrited ~16%

- Train split giữ tỷ lệ tương tự nhờ stratify.

Nhận xét: Random Forest mạnh hơn Logistic Regression trong dự đoán khách hàng rời bỏ, đặc biệt với class ít.

## 9. Project Structure

project-name/ <br>
├── README.md <br>
├── requirements.txt<br>
├── data/
│   ├── raw/             # dữ liệu gốc<br>
│   └── processed/       # dữ liệu đã xử lý<br>
├── notebooks/<br>
│   ├── 01_data_exploration.ipynb<br>
│   ├── 02_preprocessing.ipynb<br>
│   └── 03_modeling.ipynb<br>
├── src/<br>
│   ├── __init__.py<br>
│   ├── data_processing.py<br>
│   ├── visualization.py<br>
│   └── models.py<br>

## 10. Challenges & Solutions

Challenge: Xử lý dữ liệu missing value và categorical mà không dùng Pandas.

Solution: Sử dụng NumPy masking, np.unique, np.argmax để fill mode.

Challenge: Scale numeric chỉ cho Logistic Regression.

Solution: Tách X_numeric, chuẩn hóa riêng, concat với one-hot.

Challenge: Class imbalance.

Solution: Stratify khi train/test split, có thể dùng class_weight='balanced' cho Logistic Regression.

## 11. Future Improvements

Implement Logistic Regression từ đầu bằng NumPy, không dùng scikit-learn.

Hyperparameter tuning Random Forest (max_depth, min_samples_leaf, max_features).

Thêm cross-validation để kiểm tra hiệu năng ổn định hơn.

Thêm trực quan hóa feature importance.

## 12. Contributors

Trần Minh Trọng – 23120100

Email: your-email@example.com

## 13. License

This project is licensed under MIT License.