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
num_DUs = 5                                             # Số lượng DU
num_CUs = 5                                             # Số lượng CU
num_UEs = 4                                             # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
num_RBs = 4                                                # Số lượng của RBs
num_antennas = 8                                        # Số lượng anntenas

radius_in = 100                                         # Bán kính vòng tròn trong (km)
radius_out = 1000                                       # Bán kính vòng tròn ngoài (km)

rb_bandwidth = 180e3                                    # Băng thông của mỗi RBs (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                                   # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10)       # Công suất tại mỗi RU (mW)
noise_power_watts = 1e-10                               # Công suất nhiễu (mW) 
p_ib_sk = max_tx_power_mwatts / num_RBs                 # Phân bổ công suất đều cho các resource block

epsilon = 1e-10                                         # Giá trị nhỏ ~ 0

P_j_random_list = [max_tx_power_mwatts]

path_loss_ref = 128.1
path_loss_exp = 37.6

num_slices = 1                                          # Số lượng loại dịch vụ
if num_slices == 1:
    slices = ["eMBB"]                                   # Tập các loại slice
else:
    slices = ["eMBB", "ULLRC", "mMTC"]                  # Tập các loại slice

D_j_random_list = [20]                                   # Các loại yêu cầu tài nguyên của node DU j 
D_m_random_list = [20]                                   # Các loại yêu cầu tài nguyên của node CU m 

A_j_random_list = [100]                                  # Các loại tài nguyên của node DU j
A_m_random_list = [100]                                  # Các loại tài nguyên của node CU m

R_min_random_list = [2.5e6]                               # Các loại yêu cầu Data rate ngưỡng

delta_coordinate = 5                                    # Sai số toạ độ của UE
delta_num_UE = 5                                        # Sai số số lượng UE

time_slot = 5                                           # Số lượng time slot trong 1 frame
num_frame = 5

gamma = 0.5                                             # Hệ số tối ưu

# ===========================================
# ============== Main =======================
# ===========================================

# Đường dẫn thư mục output
SAVE_PATH  = "./result/"

def main():
    seed = 1
    np.random.seed(seed)

    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # Tạo thư mục con theo cấu trúc result/{current_time}/{folder_name}
    output_folder_time = os.path.join(SAVE_PATH, current_time)
    os.makedirs(output_folder_time, exist_ok=True)
    print(f"Output folder created: {output_folder_time}")

    # Gọi hàm lưu các tham số mô phỏng
    print("Saving simulation parameters...")
    other_function.save_simulation_parameters(
        output_folder_time,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        num_UEs=num_UEs,
        num_RBs=num_RBs,
        num_antennas=num_antennas,
        radius_in=radius_in,
        radius_out=radius_out,
        rb_bandwidth=rb_bandwidth,
        max_tx_power_dbm=max_tx_power_dbm,
        max_tx_power_mwatts=max_tx_power_mwatts,
        noise_power_watts=noise_power_watts,
        epsilon=epsilon,
        P_j_random_list=P_j_random_list,
        path_loss_ref=path_loss_ref,
        path_loss_exp=path_loss_exp,
        num_slices=num_slices,
        slices=slices,
        D_j_random_list=D_j_random_list,
        D_m_random_list=D_m_random_list,
        A_j_random_list=A_j_random_list,
        A_m_random_list=A_m_random_list,
        R_min_random_list=R_min_random_list,
        delta_coordinate=delta_coordinate,
        delta_num_UE=delta_num_UE,
        time_slot=time_slot,
        num_frame=num_frame,
        gamma=gamma
    )
    print("Simulation parameters saved.")

    # Toạ tọa độ RU và UE
    print("Generating RU and UE coordinates...")
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
    coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
    print("Coordinates generated.")

    # Vẽ đồ thị và lưu hình Topo_Wireless
    print("Plotting and saving network topology...")
    gen_RU_UE.plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out, output_folder_time)
    print("Network topology saved.")

    # Tạo mạng RAN
    print("Creating RAN topology...")
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)
    print("RAN topology created.")

    # Vẽ đồ thị và lưu hình ảnh Topo_RAN
    print("Plotting and saving RAN topology...")
    RAN_topo.draw_topo(G, output_folder_time)
    print("RAN topology saved.")

    # Danh sách các liên kết trong mạng và các capacity của các node DU, CU và công suất tại RU
    print("Getting links and node capacities...")
    l_ru_du, l_du_cu = RAN_topo.get_links(G)
    P_i, A_j, A_m = RAN_topo.get_node_cap(G)
    print("Links and node capacities obtained.")

    # Tạo yêu cầu của từng UE
    print("Generating UE requirements and slice mapping...")
    slice_mapping, D_j, D_m, R_min = gen_RU_UE.gen_mapping_and_requirements(
        num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list
    )
    print("UE requirements and slice mapping generated.")

    # Tính ma trận khoảng cách của UE - RU
    print("Calculating distances between RU and UE...")
    distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
    print("Distances calculated.")

    # Tính gain
    print("Calculating channel gain...")
    gain = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
    print("Channel gain calculated.")

    # Bắt đầu chạy các thuật toán
    print("Starting long-term algorithm...")
    long_term_start_time = time.time()
    
    # Gọi hàm long-term: giải bài toán toàn cục
    pi_sk, z_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, objective = solving.long_term(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, p_ib_sk)
    long_term_end_time = time.time()

    long_term_time = long_term_end_time - long_term_start_time
    print(f"Long-term algorithm completed in {long_term_time:.4f} seconds.")

    # Ánh xạ UE cho RU gần nhất
    print("Mapping UE to nearest RU...")
    arr_phi_i_sk = other_function.mapping_RU_UE(slice_mapping, distances_RU_UE)

    nearest_start_time = time.time()

    # Gọi hàm mapping_RU_nearest_UE: giải bài toán ánh xạ RU gần nhất
    nearest_pi_sk, nearest_z_ib_sk, nearest_mu_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_objective = solving.mapping_RU_nearest_UE(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, arr_phi_i_sk, p_ib_sk)

    nearest_end_time = time.time()
    nearest_time = nearest_end_time - nearest_start_time
    print(f"Mapping UE to nearest RU completed in {nearest_time:.4f} seconds.")

    # Lưu các kết quả
    print("Saving long-term results...")
    other_function.save_variable_results(output_folder_time, "long_term", pi_sk, z_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, objective, long_term_time)

    print("Saving mapping RU nearest UE results...")
    other_function.save_variable_results(output_folder_time, "mapping_RU_nearest_UE", nearest_pi_sk, nearest_z_ib_sk, nearest_mu_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_objective, nearest_time)

    print("All results saved.")


# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()




