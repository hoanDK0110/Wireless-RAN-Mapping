import matplotlib.pyplot as plt
import numpy as np
import os

# Đường dẫn tới thư mục kết quả
results_dir = "results_running_mode_0"

# Hàm để tải dữ liệu từ file .npy trong thư mục kết quả
def load_data_from_npy(filename):
    file_path = os.path.join(results_dir, filename)
    return np.load(file_path)

def chart1(results_dir):
    # Đọc dữ liệu sử dụng năng lượng từ các file .npy
    used_power = load_data_from_npy("used_power.npy")
    unused_power = load_data_from_npy("unused_power.npy")
    
    # Đọc dữ liệu sử dụng RB từ file .npy
    used_rb = load_data_from_npy("used_rb.npy")
    unused_rb = load_data_from_npy("unused_rb.npy")

    categories_power = ['RU1', 'RU2', 'RU3', 'RU4']
    categories_rb = ['RB']

    plt.figure(figsize=(12, 5))

    # Vẽ biểu đồ Sử dụng Năng lượng
    plt.subplot(1, 2, 1)
    bar_width = 0.5
    plt.bar(categories_power, used_power, bar_width, label="Used power")
    plt.bar(categories_power, unused_power, bar_width, bottom=used_power, label="Unused power", color='lightgrey')
    plt.xlabel("Categories")
    plt.ylabel("Power usage (%)")
    plt.ylim(0, 110)
    plt.title("Power Usage")
    plt.legend(["Used power", "Unused power"], loc="upper left")

    # Vẽ biểu đồ Sử dụng RB
    plt.subplot(1, 2, 2)
    bar_width = 0.15  # Đặt độ rộng cột nhỏ
    x_positions_rb = [0]  # Vị trí cột

    plt.bar(x_positions_rb, [used_rb], bar_width, label="Used RB")
    plt.bar(x_positions_rb, [unused_rb], bar_width, bottom=[used_rb], label="Unused RB", color='lightgrey')
    plt.xticks(x_positions_rb, categories_rb)  # Đặt nhãn trục x
    plt.xlim(-0.5, 0.5)  # Thu nhỏ giới hạn trục x
    plt.xlabel("Categories")
    plt.ylabel("RB usage (%)")
    plt.ylim(0, 110)
    plt.title("RB Usage")
    plt.legend(["Used RB", "Unused RB"], loc="upper left")

    plt.tight_layout()
    plt.show()
    
def chart2(results_dir):
    # Đọc dữ liệu sử dụng DU và CU từ các file .npy
    used_du = load_data_from_npy("used_du.npy")
    unused_du = load_data_from_npy("unused_du.npy")
    used_cu = load_data_from_npy("used_cu.npy")
    unused_cu = load_data_from_npy("unused_cu.npy")

    categories_du = ['DU1', 'DU2']
    categories_cu = ['CU1', 'CU2']

    plt.figure(figsize=(12, 5))

    # Vẽ biểu đồ Sử dụng DU
    plt.subplot(1, 2, 1)
    bar_width = 0.2
    plt.bar(categories_du, used_du, bar_width, label="Used DU")
    plt.bar(categories_du, unused_du, bar_width, bottom=used_du, label="Unused DU", color='lightgrey')
    plt.xlabel("Categories")
    plt.ylabel("DU usage (%)")
    plt.ylim(0, 110)
    plt.title("DU Usage")
    plt.legend(["Used DU", "Unused DU"], loc="upper left")

    # Vẽ biểu đồ Sử dụng CU
    plt.subplot(1, 2, 2)
    bar_width = 0.2
    plt.bar(categories_cu, used_cu, bar_width, label="Used CU")
    plt.bar(categories_cu, unused_cu, bar_width, bottom=used_cu, label="Unused CU", color='lightgrey')
    plt.xlabel("Categories")
    plt.ylabel("CU usage (%)")
    plt.ylim(0, 110)
    plt.title("CU Usage")
    plt.legend(["Used CU", "Unused CU"], loc="upper left")

    plt.tight_layout()
    plt.show()

def plot_charts(running_mode):
    if running_mode == 0:
        results_dir = "results_running_mode_0"
    elif running_mode == 1:
        results_dir = "results_running_mode_1"
    else:
        raise ValueError("Invalid running_mode value")

# Gọi hàm chart1 và chart2 để vẽ
chart1(results_dir)
chart2(results_dir)
