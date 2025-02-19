import numpy as np
import os

def extract_optimization_results(long_pi_sk, long_z_ib_sk, long_phi_i_sk):
    def extract_values(array, dtype):
        shape = array.shape
        flat_array = np.array([x.value for x in array.flatten()], dtype=dtype)
        return flat_array.reshape(shape)

    arr_long_pi_sk = extract_values(long_pi_sk, int)
    arr_long_z_ib_sk = extract_values(long_z_ib_sk, int)
    arr_long_phi_i_sk = extract_values(long_phi_i_sk, int)

    return arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk

def generate_new_num_UEs(num_UEs, delta_num_UE):
    # Tính sai số ngẫu nhiên trong khoảng [-delta_num_UE, delta_num_UE]
    delta = np.random.randint(-delta_num_UE, delta_num_UE)

    # Tính số lượng UE mới
    new_num_UEs = num_UEs + delta
    # Đảm bảo số lượng UE không âm
    return max(new_num_UEs, 0)

def mapping_nearest_RU_UE(distances_RU_UE, slice_mapping, num_RUs, num_UEs, num_slices):
    # Khởi tạo biến nhị phân nearest_phi_i_sk với tất cả giá trị là 0
    nearest_phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=int)
    
    for k in range(num_UEs):
        # Tìm RU gần nhất cho UE k (tức là tìm i có khoảng cách nhỏ nhất với UE k)
        i_nearest = np.argmin(distances_RU_UE[:, k])
        
        # Xác định slice mà UE k đang chọn
        for s in range(num_slices):
            if slice_mapping[s, k] == 1:  # UE k thuộc slice s
                nearest_phi_i_sk[i_nearest, s, k] = 1  # Đánh dấu RU gần nhất phục vụ UE này

    return nearest_phi_i_sk

def mapping_random_RU_UE(num_RUs, num_UEs, num_slices, slice_mapping):

    # Khởi tạo ma trận ánh xạ ngẫu nhiên với tất cả giá trị là 0
    random_phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=int)

    for k in range(num_UEs):
        # Chọn RU ngẫu nhiên từ danh sách các RU có sẵn
        chosen_RU = np.random.randint(0, num_RUs)

        # Xác định slice mà UE k thuộc về
        for s in range(num_slices):
            if slice_mapping[s, k] == 1:  # UE k thuộc slice s
                random_phi_i_sk[chosen_RU, s, k] = 1  # Đánh dấu ánh xạ ngẫu nhiên

    return random_phi_i_sk


def save_simulation_parameters(output_folder_time, **parameters):
    # Tạo đường dẫn đến file .txt để lưu
    output_file = os.path.join(output_folder_time, "simulation_parameters.txt")
    
    # Mở file ở chế độ ghi (write) và lưu các tham số dưới dạng văn bản
    with open(output_file, 'w') as txt_file:
        # Lặp qua từng tham số trong dictionary `parameters` và ghi vào file
        for key, value in parameters.items():
            txt_file.write(f"{key}: {value}\n")
    
    print(f"Simulation parameters saved to {output_file}")

def save_results(result_prefix, execution_time, total_pi_sk, objective, output_folder):
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"{execution_time:.4f}\n")
    with open(pi_sk_file, "a") as f:
        f.write(f"{total_pi_sk}\n")
    with open(objective_file, "a") as f:
        f.write(f"{objective.value}\n")

    #print(f"Results saved for {result_prefix} in {result_folder}.")


def convert_to_array(num_RUs, num_slices, num_UEs, long_phi_i_sk):
    # Tạo mảng NumPy 3 chiều rỗng
    array_long_phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs), dtype=float)

    # Ánh xạ giá trị từ long_phi_i_sk sang mảng NumPy
    for i, ru in enumerate(long_phi_i_sk):
        for s, slice_ in enumerate(ru):
            for k, variable in enumerate(slice_):
                array_long_phi_i_sk[i, s, k] = 1.0 if variable.value else 0.0

    return array_long_phi_i_sk


def print_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    # In kết quả cho từng biến
    print("Kết quả của pi_sk:")
    for s in range(pi_sk.shape[0]):
        for k in range(pi_sk.shape[1]):
            print(f"pi_sk[{s}, {k}] = {pi_sk[s, k].value}")

    print("\nKết quả của z_ib_sk:")
    for i in range(z_ib_sk.shape[0]):
        for b in range(z_ib_sk.shape[1]):
            for s in range(z_ib_sk.shape[2]):
                for k in range(z_ib_sk.shape[3]):
                    print(f"z_ib_sk[{i}, {b}, {s}, {k}] = {z_ib_sk[i, b, s, k].value}")
                    
    print("\nKết quả của P_ib_sk:")
    for i in range(p_ib_sk.shape[0]):
        for b in range(p_ib_sk.shape[1]):
            for s in range(p_ib_sk.shape[2]):
                for k in range(p_ib_sk.shape[3]):
                    print(f"P_ib_sk[{i}, {b}, {s}, {k}] = {p_ib_sk[i, b, s, k].value}")

    print("\nKết quả của phi_i_sk:")
    for i in range(phi_i_sk.shape[0]):
        for s in range(phi_i_sk.shape[1]):
            for k in range(phi_i_sk.shape[2]):
                print(f"phi_i_sk[{i}, {s}, {k}] = {phi_i_sk[i, s, k].value}")

    """print("\nKết quả của phi_j_sk:")
    for j in range(phi_j_sk.shape[0]):
        for s in range(phi_j_sk.shape[1]):
            for k in range(phi_j_sk.shape[2]):
                print(f"phi_j_sk[{j}, {s}, {k}] = {phi_j_sk[j, s, k].value}")

    print("\nKết quả của phi_m_sk:")
    for m in range(phi_m_sk.shape[0]):
        for s in range(phi_m_sk.shape[1]):
            for k in range(phi_m_sk.shape[2]):
                print(f"phi_m_sk[{m}, {s}, {k}] = {phi_m_sk[m, s, k].value}")

    print("\nKết quả của P_ib_sk:")
    for i in range(p_ib_sk.shape[0]):
        for b in range(p_ib_sk.shape[1]):
            for s in range(p_ib_sk.shape[2]):
                for k in range(p_ib_sk.shape[3]):
                    print(f"P_ib_sk[{i}, {b}, {s}, {k}] = {p_ib_sk[i, b, s, k].value}")

    print("\nKết quả của mu_ib_sk:")
    for i in range(mu_ib_sk.shape[0]):
        for b in range(mu_ib_sk.shape[1]):
            for s in range(mu_ib_sk.shape[2]):
                for k in range(mu_ib_sk.shape[3]):
                    print(f"mu_ib_sk[{i}, {b}, {s}, {k}] = {mu_ib_sk[i, b, s, k].value})"""