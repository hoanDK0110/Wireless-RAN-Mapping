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

def mapping_RU_UE(slice_mapping, distances_RU_UE):
    num_RU, num_UEs = distances_RU_UE.shape
    num_slices, _ = slice_mapping.shape
    
    # Khởi tạo ma trận nearest_phi_i_sk
    nearest_phi_i_sk = np.zeros((num_RU, num_slices, num_UEs), dtype=int)
    
    for j in range(num_UEs):  # Duyệt qua từng UE
        for k in range(num_slices):  # Duyệt qua từng slice
            if slice_mapping[k, j] == 1:  # Kiểm tra UE thuộc slice nào
                # Tìm RU gần nhất với UE j
                nearest_RU = np.argmin(distances_RU_UE[:, j])
                # Đánh dấu ánh xạ trong nearest_phi_i_sk
                nearest_phi_i_sk[nearest_RU, k, j] = 1
    
    return nearest_phi_i_sk

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

