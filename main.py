import gen_RU_UE
import wireless
import RAN_topo
import solving
import benchmark
import other_function
import numpy as np
import datetime
import os
from datetime import datetime
import time

# =======================================================
# ============== Tham số mô phỏng =======================
# =======================================================
num_RUs = 5                                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 4                                             # Số lượng DU
num_CUs = 4                                             # Số lượng CU

num_RBs = 25                                            # Số lượng của RBs
num_antennas = 8                                        # Số lượng anntenas

radius_in = 100                                         # Bán kính vòng tròn trong (m)
radius_out = 1000                                       # Bán kính vòng tròn ngoài (m)

rb_bandwidth = 180e3                                    # Băng thông của mỗi RBs (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                                   # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10)       # Công suất tại mỗi RU (mW)
noise_power_watts = 1e-10                               # Công suất nhiễu (mW) 
p_ib_sk = max_tx_power_mwatts / num_RBs                 # Phân bổ công suất đều cho các resource block

epsilon = 1e-5                                         # Giá trị nhỏ ~ 0

P_j_random_list = [max_tx_power_mwatts]

path_loss_ref = 128.1
path_loss_exp = 37.6

num_slices = 1                                          # Số lượng loại dịch vụ
if num_slices == 1:
    slices = ["eMBB"]                                   # Tập các loại slice
else:
    slices = ["eMBB", "ULLRC", "mMTC"]                  # Tập các loại slice

D_j_random_list = [10]                                   # Các loại yêu cầu tài nguyên của node DU j 
D_m_random_list = [10]                                     # Các loại yêu cầu tài nguyên của node CU m 

A_j_random_list = [100]                                  # Các loại tài nguyên của node DU j
A_m_random_list = [100]                                  # Các loại tài nguyên của node CU m

R_min_random_list = [1e6]                               # Các loại yêu cầu Data rate ngưỡng

delta_coordinate = 5                                    # Sai số toạ độ của UE
delta_num_UE = 5                                        # Sai số số lượng UE

time_slot = 5                                           # Số lượng time slot trong 1 frame
num_frame = 5

gamma = 0.8                                             # Hệ số tối ưu

num_step = 2                                          # Số lần chạy mô phỏng
# ===========================================
# ============== Main =======================
# ===========================================

# Đường dẫn thư mục output
SAVE_PATH  = "./result/"

def main():
    num_UEs = 25                                            # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
    seed = 2
    np.random.seed(seed)

    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Tạo thư mục chính theo cấu trúc result/{current_time}
    output_folder_time = os.path.join(SAVE_PATH, current_time)
    os.makedirs(output_folder_time, exist_ok=True)
    print(f"Output folder created: {output_folder_time}")

    # Tạo mạng RAN
    print("Creating RAN topology...")
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

    # Gọi hàm lưu các tham số mô phỏng
    for step in range(num_step):
        print(f"\n===== Step {step + 1}/{num_step} =====")
        step_start_time = time.time()

        # Tạo tọa độ RU và UE
        print(f"Step {step + 1}: Generating coordinates...")
        coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
        coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)

        # Danh sách các liên kết trong mạng và các capacity của các node DU, CU và công suất tại RU
        print(f"Step {step + 1}: Retrieving network links and capacities...")
        l_ru_du, l_du_cu = RAN_topo.get_links(G)
        _, A_j, A_m = RAN_topo.get_node_cap(G)

        # Tạo yêu cầu của từng UE
        print(f"Step {step + 1}: Generating UE requirements...")
        slice_mapping, D_j, D_m, R_min = gen_RU_UE.gen_mapping_and_requirements(num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list)

        # Tính ma trận khoảng cách của UE - RU
        print(f"Step {step + 1}: Calculating RU-UE distances...")
        distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

        # Tính matrix channel
        print(f"Step {step + 1}: Computing channel power, SNR, and data rate...")
        channel_power, SNR, data_rate = wireless.channel_matrix(
            distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas,
            path_loss_ref, path_loss_exp, noise_power_watts, rb_bandwidth, p_ib_sk
        )

        # Bắt đầu chạy các thuật toán
        print(f"Step {step + 1}: Running long-term algorithm...")
        long_term_start_time = time.time()
        pi_sk, z_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, objective = solving.long_term_2(
            num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m,
            R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate
        )
        long_term_end_time = time.time()
        long_term_time = long_term_end_time - long_term_start_time
        other_function.save_results(step, "long_term", long_term_time, pi_sk, objective, output_folder_time, "long_term")
        print(f"Step {step + 1}: Long-term algorithm completed in {long_term_time:.4f} seconds.")

        # Mapping UE cho RU gần nhất
        print(f"Step {step + 1}: Mapping UEs to the nearest RU...")
        arr_phi_i_sk = other_function.mapping_RU_UE(slice_mapping, distances_RU_UE)
        print(f"Step {step + 1}: Running nearest mapping algorithm...")
        nearest_start_time = time.time()
        nearest_pi_sk, nearest_z_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_objective = solving.mapping_RU_nearest_UE(
            num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m,
            R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
            arr_phi_i_sk, data_rate
        )
        nearest_end_time = time.time()
        nearest_time = nearest_end_time - nearest_start_time
        other_function.save_results(step, "nearest_mapping", nearest_time, nearest_pi_sk, nearest_objective, output_folder_time, "nearest_mapping")
        print(f"Step {step + 1}: Nearest mapping algorithm completed in {nearest_time:.4f} seconds.")

        step_end_time = time.time()
        print(f"Step {step + 1}: Completed in {step_end_time - step_start_time:.4f} seconds.")
        
        # Tạo số lượng UE mới
        num_UEs = other_function.generate_new_num_UEs(num_UEs, delta_num_UE)
    
    print("\nAll steps completed.")


# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()




