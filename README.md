# BankChurners Prediction Project

## 1. Mô tả ngắn gọn
Project này nhằm **dự đoán khả năng rời bỏ của khách hàng thẻ tín dụng** dựa trên dữ liệu khách hàng BankChurners.  
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

- Bài toán: Dự đoán khả năng khách hàng rời bỏ thẻ tín dụng (Customer Churn) dựa trên hành vi tiêu dùng và dữ liệu khách hàng.

- Động lực: Việc giữ chân khách hàng cũ chi phí rẻ hơn nhiều so với tìm kiếm khách hàng mới. Dự đoán sớm sẽ giúp ngân hàng có chiến lược CSKH phù hợp.

- Mục tiêu kỹ thuật:

  - Loại bỏ hoàn toàn vòng lặp (for loops) trong xử lý dữ liệu, thay thế bằng NumPy Vectorization.

  - Tự cài đặt thuật toán Logistic Regression bằng Gradient Descent.

---

## 4. Dataset
- **Nguồn:** [BankChurners.csv]  
- **Số lượng mẫu:** 10127 khách hàng, 21 đặc trưng.  
- **Features:** Thông tin cá nhân và tài chính (Age, Gender, Income_Category, Credit_Limit, Card_Category, v.v.)  
- **Target:** `Attrition_Flag` (0 = Existing Customer, 1 = Attrited Customer)

---

## 5. Method

### 5.1 Xử lý dữ liệu (NumPy Only)
- **Data Loading:** Sử dụng `np.genfromtxt` và `np.loadtxt` thay vì Pandas.
- **Data Cleaning:**
    - Xử lý Missing Value (`Unknown`) bằng **Mode Imputation** (vectorized masking).
    - Xử lý Outlier (Giá trị ngoại lai) cho các cột số bằng kỹ thuật **Winsorization** (kẹp giá trị trong khoảng percentile 1% - 99%).
- **Feature Engineering (Tạo đặc trưng mới):**
    - `Avg_Trans_Amt`: Tổng tiền / Tổng số lần giao dịch.
    - `Credit_Util_Rate`: Số dư nợ / Hạn mức tín dụng.
- **Encoding:** One-hot encoding sử dụng **Broadcasting** thay vì vòng lặp.
- **Scaling:** Tự viết class `NumpyScaler` để chuẩn hóa Z-score ($z = \frac{x - \mu}{\sigma}$).

### 5.2 Mô hình hóa (Algorithm Implementation)

#### a. Custom Logistic Regression (NumPy)
Tự cài đặt thuật toán dựa trên **Gradient Descent**:
- **Hàm kích hoạt:** Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ (có xử lý chống tràn số `np.clip`).
- **Hàm mất mát:** Binary Cross-Entropy.
- **Cập nhật trọng số:**
  $$w = w - \alpha \cdot \frac{1}{m} X^T (\hat{y} - y)$$
  *(Sử dụng `np.dot` để tính toán ma trận, tăng tốc độ gấp nhiều lần so với vòng lặp)*.

#### b. Random Forest (Scikit-learn Baseline)
- Sử dụng làm mô hình cơ sở để so sánh hiệu năng.
- Cấu hình: `n_estimators=200`, `random_state=42`.

---

## 6. Installation & Setup

### 1. Clone repository

git clone <https://github.com/trongtz/lab02-numpy-for-data-science-hcmus>

### 2. Tạo môi trường ảo (khuyến nghị)

python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

### 3. Cài đặt dependencies
pip install -r requirements.txt

## 7. Usage

Dự án được chia thành các module trong thư mục src và chạy thông qua Jupyter Notebook:

- Preprocessing: src/data_processing.py

- Visualizations: src/visualization.py

- Modeling: src/models.py

- Notebook: notebooks/01_data_exploration.ipynb, 02_preprocessing.ipynb, 03_modeling.ipynb

Các bước thực hiện:

- Khám phá dữ liệu: Chạy notebooks/01_data_exploration.ipynb

- Tiền xử lý dữ liệu: Chạy notebooks/02_preprocessing.ipynb

- Huấn luyện mô hình: Chạy notebooks/03_modeling.ipynb

## 8. Results

### Logistic Regression

- Accuracy: ~90%

- Confusion Matrix và classification report hiển thị f1-score của class 1 (Attrited Customer) hơi thấp do imbalance.

### Random Forest

- Accuracy: ~96%

- Recall của class 1 tốt hơn Logistic Regression.

- Không cần chuẩn hóa dữ liệu, hiệu năng ổn định.

### Phân bố label

- Full dataset: Existing ~84%, Attrited ~16%

- Train split giữ tỷ lệ tương tự nhờ stratify.

`Nhận xét:` Random Forest mạnh hơn Logistic Regression trong dự đoán khách hàng rời bỏ, đặc biệt với class ít.

`Insight quan trọng:`

- Khách hàng rời bỏ thường có tổng số lần giao dịch (Total_Trans_Ct) thấp đột biến, feature này tương quan thuận với Total_Trans_Amt (tổng tiền giao dịch), chi tiêu giảm mạnh là tín hiệu báo động đỏ.

- Số dư nợ quay vòng (Revolving_Bal) của nhóm rời bỏ thường rất thấp (trả hết nợ trước khi hủy thẻ).
## 9. Project Structure

23120100/ <br>
├── data/ <br>
│   ├── raw/             # chứa dữ liệu gốc<br>
│   └── processed/       # chứa dữ liệu đã xử lý<br>
├── notebooks/<br>
│   ├── 01_data_exploration.ipynb<br>
│   ├── 02_preprocessing.ipynb<br>
│   └── 03_modeling.ipynb<br>
├── src/<br>
│   ├── _init__.py<br>
│   ├── data_processing.py<br>
│   ├── visualization.py<br>
│   └── models.py<br>
├── README.md <br>
├── requirements.txt<br>

## 10. Challenges & Solutions

1. Thách thức: No For-Loops

- Vấn đề: Việc chuyển đổi logic từ xử lý từng dòng (row-by-row) sang xử lý nguyên mảng (matrix operations) rất khó, đặc biệt là với One-Hot Encoding và tính toán Gradient.

- Giải pháp: Sử dụng Broadcasting ([:, None] == unique) và phép nhân ma trận (np.dot) để loại bỏ hoàn toàn vòng lặp Python.

2. Thách thức: Ổn định số học 

- Vấn đề: Hàm exp(-z) trong Sigmoid bị tràn số (overflow) khi $z$ quá lớn hoặc quá nhỏ.

- Giải pháp: Sử dụng np.clip để giới hạn giá trị đầu vào và thêm epsilon (1e-9) vào hàm Log Loss để tránh lỗi chia cho 0.

3. Thách thức: Dữ liệu mất cân bằng (Imbalance Data)

- Vấn đề: Chỉ 16% khách hàng rời bỏ, khiến mô hình dễ bị thiên kiến về nhóm khách hàng hiện tại.

- Giải pháp: Sử dụng tham số stratify khi chia tập dữ liệu và tập trung phân tích chỉ số Recall/F1-score thay vì chỉ nhìn Accuracy.

## 11. Future Improvements

- Cài đặt thuật toán SMOTE (Synthetic Minority Over-sampling Technique) bằng NumPy để cân bằng lại dữ liệu.

- Tự cài đặt Grid Search để tìm learning rate tối ưu tự động.

## 12. Contributors

Trần Minh Trọng – 23120100

Email: 23120100@student.hcmus.edu.vn

## 13. License

This project is licensed under MIT License.