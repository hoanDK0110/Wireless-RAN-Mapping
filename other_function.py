import numpy as np
import os
def extract_optimization_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    def extract_values(array, dtype):
        shape = array.shape
        flat_array = np.array([x.value for x in array.flatten()], dtype=dtype)
        return flat_array.reshape(shape)

    arr_pi_sk = extract_values(pi_sk, int)
    arr_z_ib_sk = extract_values(z_ib_sk, int)
    arr_p_ib_sk = extract_values(p_ib_sk, float)
    arr_mu_ib_sk = extract_values(mu_ib_sk, float)
    arr_phi_i_sk = extract_values(phi_i_sk, int)
    arr_phi_j_sk = extract_values(phi_j_sk, int)
    arr_phi_m_sk = extract_values(phi_m_sk, int)

    return arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk


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

import os
import numpy as np

def save_variable_results(output_folder, result_prefix, **results):
    """
    Lưu giá trị của các biến vào các file riêng biệt trong thư mục con,
    bao gồm cả các biến mảng nhiều chiều và các giá trị đơn lẻ.

    :param output_folder: Thư mục nơi lưu các thư mục và file kết quả.
    :param result_prefix: Tiền tố cho tên file kết quả.
    :param results: Các tham số được truyền vào dưới dạng key-value, như pi_sk, z_ib_sk, ...
    """
    # Tạo thư mục chứa kết quả với tên theo dạng: output_folder/result_prefix
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Duyệt qua các biến trong results
    for var_name, var_value in results.items():
        # Tạo tên file với tên biến
        file_name = f"{var_name}.txt"
        file_path = os.path.join(result_folder, file_name)

        # Mở file với encoding utf-8 để xử lý ký tự Unicode
        with open(file_path, 'w', encoding='utf-8') as file:
            if isinstance(var_value, np.ndarray):
                # Nếu là mảng numpy, duyệt qua các chỉ số để lưu kết quả
                shape = var_value.shape

                # Đối với mảng 2 chiều (ví dụ: pi_sk)
                if len(shape) == 2:
                    file.write(f"Kết quả của {var_name}:\n")
                    for s in range(shape[0]):
                        for k in range(shape[1]):
                            file.write(f"{var_name}[{s}, {k}] = {var_value[s, k]}\n")

                # Đối với mảng 4 chiều (ví dụ: z_ib_sk, p_ib_sk)
                elif len(shape) == 4:
                    file.write(f"Kết quả của {var_name}:\n")
                    for i in range(shape[0]):
                        for b in range(shape[1]):
                            for s in range(shape[2]):
                                for k in range(shape[3]):
                                    file.write(f"{var_name}[{i}, {b}, {s}, {k}] = {var_value[i, b, s, k]}\n")

                # Đối với mảng 3 chiều (ví dụ: phi_i_sk)
                elif len(shape) == 3:
                    file.write(f"Kết quả của {var_name}:\n")
                    for i in range(shape[0]):
                        for s in range(shape[1]):
                            for k in range(shape[2]):
                                file.write(f"{var_name}[{i}, {s}, {k}] = {var_value[i, s, k]}\n")
                
            else:
                # Nếu không phải là mảng (ví dụ: các giá trị đơn)
                file.write(f"{var_name} = {var_value}\n")

    print(f"Results saved in {result_folder}")
