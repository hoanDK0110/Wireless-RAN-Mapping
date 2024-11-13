import matplotlib.pyplot as plt
import numpy as np

def plot_power_usage(used_power, unused_power):
    categories = ['RU1', 'RU2', 'RU3', 'RU4']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    bar_width = 0.5
    bar1 = plt.bar(categories, used_power, bar_width, label="Used power")
    bar2 = plt.bar(categories, unused_power, bar_width, bottom=used_power, label="Unused power", color='lightgrey')
    plt.xlabel("categories")
    plt.ylabel("Power usage (%)")
    plt.ylim = [0,105]
    plt.title("Power usage")
    plt.legend(["Used power", "Unused power"], loc="upper left")
    plt.tight_layout()
    plt.show()
def plot_rb_usage(used_rb, unused_rb):
    categories = ['RB']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    bar_width = 0.5
    bar1 = plt.bar(categories, used_rb, bar_width, label="Used RB")
    bar2 = plt.bar(categories, unused_rb, bar_width, bottom=used_rb, label="Unused RB", color='lightgrey')
    plt.xlabel("categories")
    plt.ylabel("RB usage (%)")
    plt.ylim = [0,105]
    plt.title("RB usage")
    plt.legend(["Used RB", "Unused RB"], loc="upper left")
    plt.tight_layout()
    plt.show()












    # # Kiểm tra kích thước của used_power và unused_power
    # if used_power.shape != unused_power.shape:
    #     print(f"Shape mismatch: used_power shape = {used_power.shape}, unused_power shape = {unused_power.shape}")
    #     return

    # # Nếu các mảng có nhiều chiều (ví dụ, (4, 3)), bạn cần tính tổng theo chiều thứ hai (axis=1)
    # if used_power.ndim > 1:
    #     used_power = np.sum(used_power, axis=1)  # Tính tổng theo chiều thứ hai
    #     unused_power = np.sum(unused_power, axis=1)  # Tính tổng theo chiều thứ hai

    # # In kích thước sau khi tính tổng để đảm bảo đúng
    # print(f"Used power shape after summing: {used_power.shape}")  # Nên in (4,)
    # print(f"Unused power shape after summing: {unused_power.shape}")  # Nên in (4,)

    # # Kiểm tra lại nếu kích thước đã là 1D sau khi tính tổng
    # if used_power.shape != (len(categories),) or unused_power.shape != (len(categories),):
    #     print("Final shape mismatch after summing. Ensure used_power and unused_power have shape matching categories.")
    #     return

    # # Chiều rộng của cột
    # bar_width = 0.35

    # # Vị trí của các cột trên trục x
    # r = np.arange(len(categories))

    # # Tạo biểu đồ
    # fig, ax = plt.subplots()

    # # Biểu đồ dạng cột stacked (chồng lên nhau)
    # ax.bar(r, used_power, color='#d2a03d', edgecolor='black', label='used_power')
    # ax.bar(r, unused_power, bottom=used_power, color='#e6e6af', edgecolor='black', label='unused_power')

    # # Thêm nhãn, tiêu đề, và chú thích
    # ax.set_ylabel('Power usage (%)')
    # ax.set_xticks(r)
    # ax.set_xticklabels(categories)
    # ax.legend(loc='center right')

    # # Hiển thị biểu đồ
    # plt.title("(a) Power usage.")
    # plt.show()
