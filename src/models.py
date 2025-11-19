import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 1. Train/Test Split (NumPy)
def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    # random seed để mỗi lần chạy ra kết quả giống nhau
    np.random.seed(random_state)    
    # Tạo mảng index rồi xáo trộn nó lên
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    # Tính vị trí cắt dựa trên test_size
    test_count = int(len(indices) * test_size)
    # Cắt lát để chia index thành 2 nhóm
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    # Trả về dữ liệu tương ứng với các index đã chia
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# 2. Numpy Standard Scaler
class NumpyScaler:
    def fit_transform(self, X):
        # Tính trung bình (mean) và độ lệch chuẩn (std) theo cột (axis=0)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Nếu std = 0 (cột toàn giá trị giống nhau) thì gán = 1, để tránh lỗi chia cho 0 
        self.std[self.std == 0] = 1.0       
        # Công thức Z-score: (x - mean) / std
        return (X - self.mean) / self.std
    def transform(self, X):
        # Dùng lại mean/std của tập train để scale cho tập test/new data
        return (X - self.mean) / self.std

# 3. Logistic Regression (NumPy)
class NumpyLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate      # Tốc độ học (bước nhảy)
        self.n_iter = n_iterations   # Số vòng lặp training
        self.weights = None          # Trọng số (w)
        self.bias = None             # Hệ số tự do (b)
        self.costs = []            

    def _sigmoid(self, z):
        # Hàm kích hoạt Sigmoid: đưa giá trị về khoảng (0, 1)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape        
        # Khởi tạo tham số w toàn số 0, b = 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Bắt đầu vòng lặp Gradient Descent
        for i in range(self.n_iter):
            # 1. Forward: Tính y_hat = sigmoid(w*x + b)
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            # 2. Backward: Tính đạo hàm (Gradient) theo w và b
            # Vectorization: nhân ma trận X.T với sai số (pred - y) thay vì dùng vòng lặp
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            # 3. Cập nhật tham số theo chiều ngược gradient
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            # Tính loss mỗi 100 vòng để theo dõi
            # Thêm 1e-9 vào log để tránh lỗi log(0) = -inf
            if i % 100 == 0:
                cost = -(1/n_samples) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
                self.costs.append(cost)

    def predict(self, X):
        # Tính xác suất
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        # Ngưỡng 0.5: Lớn hơn thì là 1 (Attrited), nhỏ hơn là 0 (Existing)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])


# 4. Evaluate Models
def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))