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
def save_variable_results(output_folder, result_prefix, pi_sk, z_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, objective, execution_time):

    # Tạo thư mục chứa kết quả với tên theo dạng: output_folder/result_prefix
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)
    
    # Lưu kết quả cho từng biến vào các file riêng biệt trong thư mục result_folder
    with open(os.path.join(result_folder, 'pi_sk.txt'), 'w') as f:
        for s in range(pi_sk.shape[0]):
            for k in range(pi_sk.shape[1]):
                f.write(f"pi_sk[{s}, {k}] = {pi_sk[s, k].value}\n")

    with open(os.path.join(result_folder, 'z_ib_sk.txt'), 'w') as f:
        for i in range(z_ib_sk.shape[0]):
            for b in range(z_ib_sk.shape[1]):
                for s in range(z_ib_sk.shape[2]):
                    for k in range(z_ib_sk.shape[3]):
                        f.write(f"z_ib_sk[{i}, {b}, {s}, {k}] = {z_ib_sk[i, b, s, k].value}\n")

    """with open(os.path.join(result_folder, 'p_ib_sk.txt'), 'w') as f:
        for i in range(z_ib_sk.shape[0]):
            for b in range(z_ib_sk.shape[1]):
                for s in range(z_ib_sk.shape[2]):
                    for k in range(z_ib_sk.shape[3]):
                        f.write(f"p_ib_sk[{i}, {b}, {s}, {k}] = {p_ib_sk[i, b, s, k].value}\n")"""

    """with open(os.path.join(result_folder, 'mu_ib_sk.txt'), 'w') as f:
        for i in range(mu_ib_sk.shape[0]):
            for b in range(mu_ib_sk.shape[1]):
                for s in range(mu_ib_sk.shape[2]):
                    for k in range(mu_ib_sk.shape[3]):
                        f.write(f"mu_ib_sk[{i}, {b}, {s}, {k}] = {mu_ib_sk[i, b, s, k].value}\n")"""

    with open(os.path.join(result_folder, 'phi_i_sk.txt'), 'w') as f:
        for i in range(phi_i_sk.shape[0]):
            for s in range(phi_i_sk.shape[1]):
                for k in range(phi_i_sk.shape[2]):
                    f.write(f"phi_i_sk[{i}, {s}, {k}] = {phi_i_sk[i, s, k].value}\n")

    with open(os.path.join(result_folder, 'phi_j_sk.txt'), 'w') as f:
        for j in range(phi_j_sk.shape[0]):
            for s in range(phi_j_sk.shape[1]):
                for k in range(phi_j_sk.shape[2]):
                    f.write(f"phi_j_sk[{j}, {s}, {k}] = {phi_j_sk[j, s, k].value}\n")

    with open(os.path.join(result_folder, 'phi_m_sk.txt'), 'w') as f:
        for m in range(phi_m_sk.shape[0]):
            for s in range(phi_m_sk.shape[1]):
                for k in range(phi_m_sk.shape[2]):
                    f.write(f"phi_m_sk[{m}, {s}, {k}] = {phi_m_sk[m, s, k].value}\n")

    # Lưu kết quả của total_R_sk vào file 'total_R_sk.txt'
    with open(os.path.join(result_folder, 'total_R_sk.txt'), 'w') as f:
        f.write(f"total_R_sk = {total_R_sk.value}\n")

    # Lưu kết quả của objective vào file 'objective.txt'
    with open(os.path.join(result_folder, 'objective.txt'), 'w') as f:
        f.write(f"objective = {objective.value}\n")

    # Lưu thời gian thực thi thuật toán vào file 'execution_time.txt'
    with open(os.path.join(result_folder, 'execution_time.txt'), 'w') as f:
        f.write(f"Execution Time: {execution_time:.4f} seconds\n")

    print(f"Results {result_prefix} saved in {result_folder}")



def save_variable_results_1(output_folder, result_prefix, pi_sk, z_ib_k, phi_i_k, phi_j_k, phi_m_k, total_R_k, objective, execution_time):
    """
    Lưu kết quả của các thuật toán vào thư mục kết quả, bao gồm cả thời gian thực thi.

    Parameters:
    - output_folder: Thư mục đầu ra lưu kết quả.
    - result_prefix: Tiền tố của tên thư mục kết quả.
    - pi_sk, z_ib_k, phi_i_k, phi_j_k, phi_m_k, total_R_k, objective: Các kết quả cần lưu vào các file.
    - execution_time: Thời gian thực thi thuật toán.
    """
    # Tạo thư mục chứa kết quả với tên theo dạng: output_folder/result_prefix
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả cho từng biến vào các file riêng biệt trong thư mục result_folder
    with open(os.path.join(result_folder, 'pi_sk.txt'), 'w') as f:
        for k in range(pi_sk.shape[0]):
            f.write(f"pi_sk[{k}] = {pi_sk[k].value}\n")

    with open(os.path.join(result_folder, 'z_ib_k.txt'), 'w') as f:
        for i in range(z_ib_k.shape[0]):
            for b in range(z_ib_k.shape[1]):
                for k in range(z_ib_k.shape[2]):
                    f.write(f"z_ib_k[{i}, {b}, {k}] = {z_ib_k[i, b, k].value}\n")

    with open(os.path.join(result_folder, 'phi_i_k.txt'), 'w') as f:
        for i in range(phi_i_k.shape[0]):
            for k in range(phi_i_k.shape[1]):
                f.write(f"phi_i_k[{i}, {k}] = {phi_i_k[i, k].value}\n")

    with open(os.path.join(result_folder, 'phi_j_k.txt'), 'w') as f:
        for j in range(phi_j_k.shape[0]):
            for k in range(phi_j_k.shape[1]):
                f.write(f"phi_j_k[{j}, {k}] = {phi_j_k[j, k].value}\n")

    with open(os.path.join(result_folder, 'phi_m_k.txt'), 'w') as f:
        for m in range(phi_m_k.shape[0]):
            for k in range(phi_m_k.shape[1]):
                f.write(f"phi_m_k[{m}, {k}] = {phi_m_k[m, k].value}\n")

    # Lưu kết quả của total_R_k vào file 'total_R_k.txt'
    with open(os.path.join(result_folder, 'total_R_k.txt'), 'w') as f:
        f.write(f"total_R_k = {total_R_k.value}\n")

    # Lưu kết quả của objective vào file 'objective.txt'
    with open(os.path.join(result_folder, 'objective.txt'), 'w') as f:
        f.write(f"objective = {objective.value}\n")

    # Lưu thời gian thực thi thuật toán vào file 'execution_time.txt'
    with open(os.path.join(result_folder, 'execution_time.txt'), 'w') as f:
        f.write(f"Execution Time: {execution_time:.4f} seconds\n")

    print(f"Results {result_prefix} saved in {result_folder}")


def save_results(step, algo_type, execution_time, pi_sk, objective, output_folder, result_prefix):
    """
    Tạo thư mục chứa kết quả và lưu kết quả của thuật toán vào các file tương ứng.
    
    Parameters:
        step (int): Bước hiện tại.
        algo_type (str): Loại thuật toán ('long_term' hoặc 'nearest_mapping').
        execution_time (float): Thời gian thực thi của thuật toán.
        pi_sk (np.ndarray): Giá trị pi_sk của thuật toán.
        objective (float): Giá trị mục tiêu (objective) của thuật toán.
        output_folder (str): Đường dẫn tới thư mục lưu kết quả chính.
        result_prefix (str): Tiền tố để đặt tên thư mục con của thuật toán.
    """
    # Tạo thư mục chứa kết quả cho thuật toán
    result_folder = os.path.join(output_folder, result_prefix)
    os.makedirs(result_folder, exist_ok=True)

    # Lưu kết quả vào các file
    time_file = os.path.join(result_folder, "times.txt")
    pi_sk_file = os.path.join(result_folder, "pi_sk.txt")
    objective_file = os.path.join(result_folder, "objective.txt")

    # Append kết quả vào các file
    with open(time_file, "a") as f:
        f.write(f"Step {step + 1}: {algo_type} time = {execution_time:.4f} seconds\n")
    with open(pi_sk_file, "a") as f:
        f.write(f"Step {step + 1}: {algo_type}_pi_sk = {pi_sk.value}\n")
    with open(objective_file, "a") as f:
        f.write(f"Step {step + 1}: {algo_type} Objective = {objective.value}\n")

    print(f"Results saved for {algo_type} in {result_folder}.")
