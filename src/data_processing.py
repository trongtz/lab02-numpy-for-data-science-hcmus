import numpy as np

# Load CSV và tách header + dữ liệu
def load_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = [h.strip().replace('"', '') for h in lines[0].split(',')]
    data_raw = [line.strip().split(',') for line in lines[1:]]
    return header, data_raw

# Xác định các cột số tự động
def detect_numeric_columns(data_raw, header):
    numeric_idx = []
    for i in range(len(header)):
        is_num = True
        for row in data_raw:
            try:
                float(row[i])
            except:
                is_num = False
                break
        if is_num:
            numeric_idx.append(i)

    numeric_cols = [header[i] for i in numeric_idx]
    return numeric_idx, numeric_cols

# Chuyển toàn bộ dữ liệu: số → float, chữ → string
def convert_data(data_raw, numeric_idx):
    converted = []
    for row in data_raw:
        new_row = []
        for i, val in enumerate(row):
            if i in numeric_idx:
                try:
                    new_row.append(float(val))
                except:
                    new_row.append(np.nan)
            else:
                new_row.append(val.replace('"',''))
        converted.append(new_row)
    return np.array(converted, dtype=object)

# Tính thống kê mô tả cơ bản
def compute_statistics(num_data):
    mean = np.nanmean(num_data, axis=0)
    median = np.nanmedian(num_data, axis=0)
    std = np.nanstd(num_data, axis=0)
    min_val = np.nanmin(num_data, axis=0)
    max_val = np.nanmax(num_data, axis=0)
    return mean, median, std, min_val, max_val

# Missing values
def count_missing(num_data):
    return np.sum(np.isnan(num_data), axis=0)

# Trích cột Attrition_Flag
def extract_attrition_flag(data_raw, header):
    attr_idx = header.index('Attrition_Flag')
    return np.array([row[attr_idx].replace('"', '') for row in data_raw])

# Trích dữ liệu categorical
def extract_categorical_column(data_raw, header, col_name):
    idx = header.index(col_name)
    return np.array([row[idx].replace('"', '') for row in data_raw])
