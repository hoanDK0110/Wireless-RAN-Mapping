import matplotlib.pyplot as plt
import numpy as np

def chart1():
    # Đọc dữ liệu từ tệp
    used_power = np.load("used_power.npy")
    unused_power = np.load("unused_power.npy")
    used_rb = np.load("used_rb.npy")
    unused_rb = np.load("unused_rb.npy")

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
    bar_width = 0.2
    plt.bar(categories_rb, [used_rb], bar_width, label="Used RB")
    plt.bar(categories_rb, [unused_rb], bar_width, bottom=[used_rb], label="Unused RB", color='lightgrey')
    plt.xlabel("Categories")
    plt.ylabel("RB usage (%)")
    plt.ylim(0, 110)
    plt.title("RB Usage")
    plt.legend(["Used RB", "Unused RB"], loc="upper left")

    plt.tight_layout()
    plt.show()

def chart2():
    # Đọc dữ liệu từ tệp
    used_du = np.load("used_du.npy")
    unused_du = np.load("unused_du.npy")
    used_cu = np.load("used_cu.npy")
    unused_cu = np.load("unused_cu.npy")

    categories_du = ['DU1', 'DU2']
    categories_cu = ['CU1', 'CU2']

    plt.figure(figsize=(12, 5))

    # Vẽ biểu đồ Sử dụng Năng lượng
    plt.subplot(1, 2, 1)
    bar_width = 0.3
    plt.bar(categories_du, used_du, bar_width, label="Used DU")
    plt.bar(categories_du, unused_du, bar_width, bottom=used_du, label="Unused DU", color='lightgrey')
    plt.xlabel("Categories")
    plt.ylabel("DU usage (%)")
    plt.ylim(0, 110)
    plt.title("DU Usage")
    plt.legend(["Used DU", "Unused DU"], loc="upper left")

    # Vẽ biểu đồ Sử dụng RB
    plt.subplot(1, 2, 2)
    bar_width = 0.3
    plt.bar(categories_cu, [used_cu], bar_width, label="Used CU")
    plt.bar(categories_cu, [unused_cu], bar_width, bottom=[used_cu], label="Unused CU", color='lightgrey')
    plt.xlabel("Categories")
    plt.ylabel("CU usage (%)")
    plt.ylim(0, 110)
    plt.title("CU Usage")
    plt.legend(["Used CU", "Unused CU"], loc="upper left")

    plt.tight_layout()
    plt.show()

# Gọi hàm chart1 để vẽ
chart1()
chart2()
