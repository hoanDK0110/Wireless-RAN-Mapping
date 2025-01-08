import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_coordinates_RU(num_RUs, radius):
    circle_RU_out = radius * 0.65
    angles = np.linspace(0, 2 * np.pi, num_RUs - 1, endpoint=False) 
    x = np.concatenate(([0], circle_RU_out * np.cos(angles)))  
    y = np.concatenate(([0], circle_RU_out * np.sin(angles)))  
    coordinates_RU = list(zip(x, y)) 
    return coordinates_RU

def gen_coordinates_UE(num_UEs, radius_in, radius_out):
    np.random.seed(1)
    angles = np.random.uniform(0, 2 * np.pi, num_UEs)
    r = np.random.uniform(radius_in, radius_out, num_UEs)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    coordinates_UE = list(zip(x, y))  
    return coordinates_UE

def calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs):
    distances_RU_UE = np.zeros((num_RUs, num_UEs))
    for i in range(num_RUs):
        for j in range(num_UEs):
            x_RU, y_RU = coordinates_RU[i]
            x_UE, y_UE = coordinates_UE[j]
            distances_RU_UE[i, j] = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
    return distances_RU_UE

def create_and_assign_slices(num_UEs, slices, D_j, D_m, R_min):  
    # Khởi tạo các danh sách để lưu trữ từng thuộc tính
    names = []
    R_min_values = []
    D_j_values = []
    D_m_values = []

    # Tạo và gán slice ngẫu nhiên cho từng UE
    for _ in range(num_UEs):
        # Chọn ngẫu nhiên loại slice
        slice_type = np.random.choice(slices)
        
        if slice_type == "eMBB":
            # Cấu hình slice eMBB
            slice_config = {
                "name": "eMBB",
                "R_min": np.random.choice(R_min), 
                "D_j": np.random.choice(D_j),      
                "D_m": np.random.choice(D_m)       
            }
        
        elif slice_type == "ULLRC":
            # Cấu hình slice ULLRC
            slice_config = {
                "name": "ULLRC",
                "R_min": np.random.choice(R_min) * 2,   
                "D_j": np.random.choice(D_j),         
                "D_m": np.random.choice(D_m)        
            }

        else: 
            # Cấu hình slice mMTC
            slice_config = {
                "name": "mMTC",
                "R_min": np.random.choice(R_min) * 0.5,   
                "D_j": np.random.choice(D_j),        
                "D_m": np.random.choice(D_m)             
            }
        
        # Thêm từng thuộc tính vào danh sách tương ứng
        names.append(slice_config["name"])
        R_min_values.append(slice_config["R_min"])
        D_j_values.append(slice_config["D_j"])
        D_m_values.append(slice_config["D_m"])
    
    # Trả về các mảng riêng biệt
    return names, R_min_values, D_j_values, D_m_values


def plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out, output_folder_time):
    # Vẽ các vòng tròn cho Inner và Outer Radius
    circle_in = plt.Circle((0, 0), radius_in, color='gray', fill=False, linestyle='--', label='Inner Radius')
    circle_out = plt.Circle((0, 0), radius_out, color='black', fill=False, linestyle='--', label='Outer Radius')
    
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    # Vẽ các điểm RU
    for (x, y) in coordinates_RU:
        plt.scatter(x, y, color='green', marker='^', s=100, label='RU' if (x, y) == (0, 0) else "")
    
    # Vẽ các điểm UE
    for index, (x, y) in enumerate(coordinates_UE):
        plt.scatter(x, y, color='blue', marker='o')
        if index == 0:  # Chỉ chú thích cho UE đầu tiên
            plt.scatter(x, y, color='blue', marker='o', label='UE')

    # Cài đặt giới hạn trục và các đặc tính đồ họa
    plt.xlim(-radius_out * 1.2, radius_out * 1.2)
    plt.ylim(-radius_out * 1.2, radius_out * 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()

    # Thêm tiêu đề và nhãn cho các trục
    plt.title("Network with RU and UE")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(output_folder_time, exist_ok=True)
    
    # Đặt tên file lưu với thời gian hiện tại
    fig_name = os.path.join(output_folder_time, f"network_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    
    # Lưu hình vẽ dưới định dạng PDF
    plt.savefig(fig_name, format="PDF")
    plt.close()  # Đóng để tránh hiển thị ảnh thêm nữa
    print(f"Network saved in {fig_name}")


def gen_mapping_and_requirements(num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list):
    # Khởi tạo ma trận ánh xạ với tất cả giá trị là 0
    slice_mapping = np.zeros((num_slices, num_UEs), dtype=int)
    
    # Danh sách để lưu yêu cầu tài nguyên cho mỗi UE
    D_j_list = []
    D_m_list = []
    R_min_list = []
    
    if num_slices == 1:
        # Nếu chỉ có một slice, tất cả UE được ánh xạ vào slice 0
        slice_mapping[0, :] = 1
    else:
        for ue in range(num_UEs):
            # Mỗi UE chỉ được yêu cầu một loại slice, chọn ngẫu nhiên một slice
            chosen_slice = np.random.randint(0, num_slices)
            slice_mapping[chosen_slice][ue] = 1
            
    # Tạo ngẫu nhiên các giá trị D_j, D_m, R_min cho mỗi UE
    D_j_list = np.random.choice(D_j_random_list, size=num_UEs).tolist()
    D_m_list = np.random.choice(D_m_random_list, size=num_UEs).tolist()
    R_min_list = np.random.choice(R_min_random_list, size=num_UEs).tolist()
    
    return slice_mapping, D_j_list, D_m_list, R_min_list
