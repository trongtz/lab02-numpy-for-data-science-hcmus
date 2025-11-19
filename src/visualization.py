import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Hàm tính số lượng từng giá trị của cột phân loại theo nhóm target
def get_grouped_counts(col_data, target):
    """
    Trả về:
        cat_vals: các giá trị duy nhất trong col_data
        target_vals: các giá trị duy nhất trong target
        counts: dict lưu số lượng mỗi giá trị của col_data theo target
    Ở bài này target là Attrition_Flag
    """
    cat_vals = np.unique(col_data)
    target_vals = np.unique(target)
    counts = {t_val: [] for t_val in target_vals}
    
    # Duyệt từng giá trị của col_data
    for c in cat_vals:
        mask = col_data == c  # chọn những dòng bằng giá trị c
        for t_val in target_vals:
            counts[t_val].append(np.sum(target[mask] == t_val))  # đếm số dòng trong nhóm target
    return cat_vals, target_vals, counts

# 2. Vẽ biểu đồ tròn Attrition_Flag
def plot_attrition_distribution(attr_flag):
    unique, counts = np.unique(attr_flag, return_counts=True)
    plt.figure(figsize=(5,5))
    plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=90)
    plt.title("Attrition_Flag distribution")
    plt.show()

# 3. Vẽ biểu đồ phân phối cột phân loại và so sánh theo Attrition_Flag
def plot_categorical_distribution(data_raw, header, cat_data, attr_flag, shorten=lambda x:x):
    for col in cat_data:
        col_idx = header.index(col)
        col_data = data_raw[:, col_idx]
        unique_vals, counts_single = np.unique(col_data, return_counts=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5)) 

        # Biểu đồ phân bố chung
        axes[0].bar([shorten(u) for u in unique_vals], counts_single, color='skyblue', edgecolor='black')
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_ylabel("Count")
        axes[0].set_xlabel(shorten(col))

        # Biểu đồ phân bố theo Attrition_Flag 
        cat_vals, target_vals, counts = get_grouped_counts(col_data, attr_flag)
        x_labels = [shorten(val) for val in cat_vals]
        width = 0.35
        x = np.arange(len(cat_vals))
        ax1 = axes[1]
        target_order = ['Attrited Customer', 'Existing Customer'] if len(target_vals) == 2 else target_vals

        for i, t_val in enumerate(target_order):
            offset = (i - (len(target_order)-1)/2) * width
            ax1.bar(x + offset, counts.get(t_val, [0]*len(cat_vals)), width, label=t_val)

        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels)
        ax1.set_ylabel('Count')
        ax1.set_xlabel(shorten(col))
        ax1.set_title(f"Attrition Comparison by {col}")
        ax1.legend(title='Attrition_Flag')

        plt.tight_layout()
        plt.show()

# 4. Vẽ histogram và boxplot cho các cột số
def plot_numeric_distribution(num_data, numeric_cols, attr_flag, shorten_col=lambda x:x):
    for i, col in enumerate(numeric_cols):
        fig, axes = plt.subplots(1, 2, figsize=(10,3))
        
        # Histogram: phân bố dữ liệu
        axes[0].hist(num_data[:, i], bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title(f"Hist: {shorten_col(col)}")
        axes[0].set_xlabel(shorten_col(col))
        axes[0].set_ylabel("Count")
        
        # Boxplot so sánh giá trị theo Attrition_Flag
        existing = num_data[np.array(attr_flag)=='Existing Customer', i]
        attrited = num_data[np.array(attr_flag)=='Attrited Customer', i]
        axes[1].boxplot([existing, attrited], labels=['Existing','Attrited'])
        axes[1].set_title(f"Boxplot: {shorten_col(col)}")
        
        plt.tight_layout()
        plt.show()

# 5. Vẽ heatmap ma trận tương quan giữa các cột số
def plot_correlation_heatmap(num_data_full, num_cols_full):
    corr_matrix = np.corrcoef(num_data_full.T)  # tính ma trận tương quan
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, xticklabels=num_cols_full, yticklabels=num_cols_full, 
                cmap='coolwarm', annot=True, fmt=".2f")
    plt.title("Correlation giữa các cột số")
    plt.show()
