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
    angles = np.random.uniform(0, 2 * np.pi, num_UEs)
    r = np.random.uniform(radius_in, radius_out, num_UEs)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    coordinates_UE = list(zip(x, y))  
    return coordinates_UE

def adjust_coordinates_UE(coordinates_UE, delta_coordinate):
    # Khởi tạo seed cho ngẫu nhiên để kết quả có thể tái tạo
    new_coordinates_UE = []
    
    for x, y in coordinates_UE:
        # Tạo độ lệch ngẫu nhiên trong khoảng [-delta_coordinate, delta_coordinate] cho cả x và y
        delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
        delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)
        
        # Tọa độ mới sau khi thêm độ lệch
        new_x = x + delta_x
        new_y = y + delta_y
        
        # Thêm tọa độ mới vào danh sách
        new_coordinates_UE.append((new_x, new_y))
    
    return new_coordinates_UE

def calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs):
    distances_RU_UE = np.zeros((num_RUs, num_UEs))
    for i in range(num_RUs):
        for j in range(num_UEs):
            x_RU, y_RU = coordinates_RU[i]
            x_UE, y_UE = coordinates_UE[j]
            distances_RU_UE[i, j] = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
    return distances_RU_UE

def create_and_assign_slices(num_UEs, num_slices, D_j, D_m, R_min):  
    # Khởi tạo các danh sách để lưu trữ từng thuộc tính
    names = []
    R_min_values = []
    D_j_values = []
    D_m_values = []

    # Tạo và gán slice ngẫu nhiên cho từng UE
    for _ in range(num_UEs):
        # Chọn ngẫu nhiên loại slice
        slice_type = np.random.choice(num_slices)
        
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



def plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out):
    circle_in = plt.Circle((0, 0), radius_in, color='gray', fill=False, linestyle='--', label='Inner Radius')
    circle_out = plt.Circle((0, 0), radius_out, color='black', fill=False, linestyle='--', label='Outer Radius')
    
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    for (x, y) in coordinates_RU:
        plt.scatter(x, y, color='green', marker='^', s=100, label='RU' if (x, y) == (0, 0) else "")
    
    for index, (x, y) in enumerate(coordinates_UE):
        plt.scatter(x, y, color='blue', marker='o')
        if index == 0:  # Chỉ chú thích cho UE đầu tiên
            plt.scatter(x, y, color='blue', marker='o', label='UE')

    plt.xlim(-radius_out * 1.2, radius_out * 1.2)
    plt.ylim(-radius_out * 1.2, radius_out * 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.title("Network with RU and UE")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    fig_name = os.path.join(result_dir, f"network_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    plt.savefig(fig_name)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


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
