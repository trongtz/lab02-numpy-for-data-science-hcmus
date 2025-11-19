import numpy as np

# 1. Load dữ liệu
def load_csv(file_path):
    """
    Đọc dữ liệu sử dụng np.genfromtxt.
    Trả về header (danh sách tên cột) và data_raw (mảng 2 chiều kiểu chuỗi).
    """
    # Đọc hàng đầu tiên làm header
    header_arr = np.genfromtxt(file_path, delimiter=',', max_rows=1, dtype='U100', encoding='utf-8')   
    # Xóa dấu ngoặc kép thừa trong header nếu có
    header = [h.replace('"', '').strip() for h in header_arr] 
    # Đọc dữ liệu (bỏ qua header), kiểu dữ liệu là Unicode String để chứa cả số và chữ
    data_raw = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype='U100', encoding='utf-8', filling_values='')  
    # Xử lý xóa dấu ngoặc kép trong dữ liệu
    data_raw = np.char.replace(data_raw, '"', '')  
    return header, data_raw

# 2. xử lí cột
def detect_numeric_columns(data_raw, header):
    """
    Tự động xác định cột số bằng cách thử ép kiểu float cho toàn bộ cột.
    """
    numeric_idx = [] 
    # Duyệt qua các cột (số lượng cột ít nên dùng vòng lặp được, NHƯNG không duyệt dòng)
    num_cols = data_raw.shape[1]
    for i in range(num_cols):
        col_data = data_raw[:, i]
        try:
            # Thử ép kiểu float cho cả cột, nếu thành công thì là cột số
            col_data.astype(float)
            numeric_idx.append(i)
        except ValueError:
            # Nếu có lỗi (do chứa chữ cái) thì là cột Categorical
            continue           
    numeric_cols = [header[i] for i in numeric_idx]
    return numeric_idx, numeric_cols

def convert_data(data_raw, numeric_idx):
    """
    Chuyển đổi các cột được chỉ định sang kiểu float.
    Input: data_raw (mảng string)
    Output: data_numeric (mảng float chỉ chứa các cột số)
    """
    # Lấy tất cả dòng, chỉ lấy các cột numeric
    selected_cols = data_raw[:, numeric_idx]  
    # Chuyển string rỗng thành 'nan' 
    selected_cols[selected_cols == ''] = 'nan'
    return selected_cols.astype(float)

# 3. thống kê mô tả (Vectorized Statistics)
def compute_statistics(num_data):
    """
    Tính toán các chỉ số thống kê trên mảng số.
    """
    mean = np.nanmean(num_data, axis=0)
    median = np.nanmedian(num_data, axis=0)
    std = np.nanstd(num_data, axis=0)
    min_val = np.nanmin(num_data, axis=0)
    max_val = np.nanmax(num_data, axis=0)
    return mean, median, std, min_val, max_val

# 4. Xử lí missing
def count_missing(num_data):
    """Đếm số lượng NaN trong mảng số"""
    return np.sum(np.isnan(num_data), axis=0)

def count_missing_categorical(data_raw, header, cat_cols_to_check):
    """Đếm giá trị thiếu cho cột phân loại (chuỗi rỗng, None, 'Unknown')"""
    missing_counts = {}    
    for col in cat_cols_to_check:
        if col not in header:
            missing_counts[col] = "Not Found"
            continue           
        col_idx = header.index(col)
        col_data = data_raw[:, col_idx]       
        # Kiểm tra điều kiện trên toàn bộ mảng, điều kiện: Rỗng hoặc "Unknown"
        is_missing = (col_data == '') | (col_data == 'Unknown') | (col_data == 'NaN')
        # Đếm số lượng True
        count = np.sum(is_missing)
        missing_counts[col] = count
    return missing_counts

# 5. trích xuất dữ liệu 
def extract_attrition_flag(data_raw, header):
    """Trích xuất cột Attrition_Flag"""
    idx = header.index('Attrition_Flag')
    return data_raw[:, idx]

def extract_categorical_column(data_raw, header, col_name):
    """Trích xuất bất kỳ cột phân loại nào"""
    idx = header.index(col_name)
    return data_raw[:, idx]

# 6. Rút gọn
def shorten(text_arr):
    if isinstance(text_arr, np.ndarray):
        return np.char.replace(text_arr, " ", "\n")
    else:
        return text_arr.replace(" ", "\n")

def get_grouped_counts(data, target):
    """
    Tính số lượng target theo nhóm.
    Tối ưu hóa bằng Boolean Masking của NumPy.
    """
    unique_data = np.unique(data)
    unique_target = np.unique(target)
    counts = {t_val: [] for t_val in unique_target}

    for d_val in unique_data:
        # Tạo mask cho nhóm hiện tại
        mask = (data == d_val)  
        # Lọc target tương ứng
        filtered_target = target[mask]        
        # Đếm số lượng
        t_unique, t_counts = np.unique(filtered_target, return_counts=True)        
        # Map vào dictionary
        temp_counts = dict(zip(t_unique, t_counts))        
        for t_val in unique_target:
            counts[t_val].append(temp_counts.get(t_val, 0))           
    return unique_data, unique_target, counts