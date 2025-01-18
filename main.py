import gen_RU_UE
import wireless
import RAN_topo
import solving
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
num_UEs = 30                                            # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
num_RBs = 30                                            # Số lượng của RBs
num_antennas = 8                                        # Số lượng anntenas

radius_in = 100                                         # Bán kính vòng tròn trong (m)
radius_out = 1000                                       # Bán kính vòng tròn ngoài (m)

bandwidth_per_RB = 180e3                                # Băng thông của mỗi RBs (Hz)
total_bandwidth = bandwidth_per_RB * num_RBs            # Tổng băng thông (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                                   # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10)       # Công suất tại mỗi RU (mW)
noise_power_density = 1e-10                             # Mật độ công suất nhiễu (W/Hz) 
P_ib_sk = max_tx_power_mwatts * num_RUs / num_RBs       # Phân bổ công suất đều cho các resource block

epsilon = 1e-5                                          # Giá trị nhỏ ~ 0

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
R_min = 1e6

delta_coordinate = 5                                    # Sai số toạ độ của UE (met)
delta_num_UE = 2                                        # Sai số số lượng UE (UE)

time_slots = 2                                           # Số lượng time slot trong 1 frame
num_frames = 2    

gamma = 0.5                                             # Hệ số tối ưu

num_step = 5                                          # Số lần chạy mô phỏng
# ===========================================
# ============== Main =======================
# ===========================================


# Đường dẫn lưu kết quả
SAVE_PATH = "./result/"

def main():
    seed = 1
    np.random.seed(seed)
    # Lấy thời gian hiện tại để tạo thư mục lưu trữ
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_folder_time = os.path.join(SAVE_PATH, current_time)
    
    # Tạo thư mục lưu kết quả
    try:
        os.makedirs(output_folder_time, exist_ok=True)
        print(f"Output folder created: {output_folder_time}")
    except Exception as e:
        print(f"Error creating output folder: {e}")
        return

    # ==== TẠO TOPOLOGY MẠNG RAN ====
    print("Creating RAN topology...")
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

    # ===========================================
    # ==== BẮT ĐẦU CÁC BƯỚC MÔ PHỎNG ====
    # ===========================================
    for step in range(num_step):
        
        print(f"\n===== Step {step + 1}/{num_step} =====")
        step_start_time = time.time()

        # 1. Tạo tọa độ RU và UE
        print("Generating RU and UE coordinates...")
        coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
        coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
        
        # 2. Lấy liên kết và thông số mạng
        print("Retrieving network links and capacities...")
        l_ru_du, l_du_cu = RAN_topo.get_links(G)
        _, A_j, A_m = RAN_topo.get_node_cap(G)

        # 3. Tạo yêu cầu của từng UE
        print("Generating UE requirements...")
        slice_mapping, D_j, D_m = gen_RU_UE.gen_mapping_and_requirements(num_UEs, num_slices, D_j_random_list, D_m_random_list)

        # 4. Tính toán khoảng cách và ma trận kênh
        print("Calculating distances and channel matrix...")
        distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

        gain, _, data_rate = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)
        
        # ===========================================
        # ==== CHẠY THUẬT TOÁN DÀI HẠN ====
        # ===========================================
        print("Running long-term algorithm...")
        long_term_start_time = time.time()
        # Chạy thuật toán dài hạn
        long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk, long_total_R_sk, long_total_pi_sk, long_objective = solving.long_term_2(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts)
        long_term_end_time = time.time()
        long_term_time = long_term_end_time - long_term_start_time
        # Lưu kết quả
        other_function.save_results("long_term", long_term_time, long_total_pi_sk, long_objective, output_folder_time)
        print(f"Long-term algorithm completed in {long_term_time:.4f} seconds.")

        # ===========================================
        # ==== Thuật toán ÁNH XẠ UE VÀO RU GẦN NHẤT ====
        # ===========================================
        # Chọn RU cho UE gần nhất
        print("Mapping UEs to the nearest RU...")
        arr_phi_i_sk = other_function.mapping_RU_UE(slice_mapping, distances_RU_UE)
        # Chạy thuật toán ánh xạ gần nhất
        print("Running nearest mapping algorithm...")
        nearest_start_time = time.time()
        nearest_pi_sk, nearest_z_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_total_pi_sk, nearest_objective = solving.mapping_RU_nearest_UE(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, arr_phi_i_sk, data_rate, P_ib_sk, max_tx_power_mwatts)
        nearest_end_time = time.time()
        nearest_time = nearest_end_time - nearest_start_time
        # Lưu kết quả ánh xạ gần nhất
        other_function.save_results("nearest_mapping", nearest_time, nearest_total_pi_sk, nearest_objective, output_folder_time)
        print(f"Nearest mapping algorithm completed in {nearest_time:.4f} seconds.")
        
        # ===========================================
        # ==== Chạy Long-Short term ====
        # ===========================================
        long_short_num_UEs = num_UEs  # Số lượng UE ban đầu
        for frame in range(num_frames):
            long_short_coordinates_UE = gen_RU_UE.gen_coordinates_UE(long_short_num_UEs, radius_in, radius_out)
            # Tính lại khoảng cách từ UE  RU
            print("Calculating distances and channel matrix...")
            long_short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, long_short_coordinates_UE, num_RUs, long_short_num_UEs)

            # Tính lại gain, data rate
            long_short_gain, _, long_short_data_rate = wireless.channel_gain(long_short_distances_RU_UE, num_slices, num_RUs, long_short_num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)

            # 3. Tạo yêu cầu của từng UE
            long_short_slice_mapping, long_short_D_j, long_short_D_m = gen_RU_UE.gen_mapping_and_requirements(long_short_num_UEs, num_slices, D_j_random_list, D_m_random_list)

            # Chạy thuật toán dài hạn
            long_short_start_time = time.time()
            long_short_pi_sk, long_short_z_ib_sk, long_short_phi_i_sk, long_short_phi_j_sk, long_short_phi_m_sk, long_short_total_R_sk, long_short_total_pi_sk, long_short_objective = solving.long_term_2(num_slices, long_short_num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, long_short_D_j, long_short_D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, long_short_slice_mapping, long_short_data_rate, P_ib_sk, max_tx_power_mwatts)
            long_short_end_time = time.time()
            long_short_time = long_short_end_time - long_short_start_time

            # Lưu kết quả long_short
            other_function.save_results("long_short", long_short_time, long_short_total_pi_sk, long_short_objective, output_folder_time)
            print(f"Long_short algorithm completed in {long_short_time:.4f} seconds.")

            arr_long_short_pi_sk, arr_long_short_phi_i_sk = other_function.extract_optimization_results(long_short_pi_sk, long_short_phi_i_sk)
            for slot in range(time_slots):
                short_start_time = time.time()
                short_pi_sk, short_total_pi_sk, short_objective = solving.short_term_1(num_slices, long_short_num_UEs, num_RUs, num_RBs, bandwidth_per_RB, long_short_gain, R_min, epsilon, arr_long_short_pi_sk, arr_long_short_phi_i_sk, max_tx_power_mwatts)
                short_end_time = time.time()
                short_time = short_end_time - short_start_time

                # Lưu kết quả long_short
                other_function.save_results("long_short", short_time, short_total_pi_sk, short_objective, output_folder_time)
                
                # UE di chuyển (có toạ độ mới)
                long_short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(long_short_coordinates_UE, delta_coordinate)

                # Khoảng cách mới từ UE - RU sau khi di chuyển
                long_short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, long_short_coordinates_UE, num_RUs, long_short_num_UEs)

                # Tính gain cho kênh truyền thay đổi
                long_short_gain, _, _ = wireless.channel_gain(long_short_distances_RU_UE, num_slices, num_RUs, long_short_num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)
                
            # Sinh số lượng UE mới cho vòng tiếp theo
            long_short_num_UEs = other_function.generate_new_num_UEs(long_short_num_UEs, delta_num_UE)
            print("long_short_num_UEs = ", long_short_num_UEs)
            
            
        # Hết step
        step_end_time = time.time()
        print(f"Step {step + 1} completed in {step_end_time - step_start_time:.4f} seconds.")

    # ==== KẾT THÚC ====
    print("\nAll steps completed.")


def main_1():
    # ==== CẤU HÌNH BAN ĐẦU ====
    seed = 2
    np.random.seed(seed)

    # Lấy thời gian hiện tại để tạo thư mục lưu trữ
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_folder_time = os.path.join(SAVE_PATH, current_time)

    # Tạo thư mục lưu kết quả
    try:
        os.makedirs(output_folder_time, exist_ok=True)
        print(f"Output folder created: {output_folder_time}")
    except Exception as e:
        print(f"Error creating output folder: {e}")
        return

    # ==== TẠO TOPOLOGY MẠNG RAN ====
    print("Creating RAN topology...")
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

    # Tạo tọa độ RU và UE
    print("Generating RU and UE coordinates...")
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
    coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)

    # Lấy liên kết và thông số mạng
    print("Retrieving network links and capacities...")
    l_ru_du, l_du_cu = RAN_topo.get_links(G)
    _, A_j, A_m = RAN_topo.get_node_cap(G)

    # Tạo yêu cầu của từng UE
    print("Generating UE requirements...")
    slice_mapping, D_j, D_m, _ = gen_RU_UE.gen_mapping_and_requirements(
        num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list
    )

    # Tính toán khoảng cách và ma trận kênh
    print("Calculating distances and channel matrix...")
    distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
    channel_power, SNR, data_rate = wireless.channel_matrix(
        distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk
    )


    # Chạy thuật toán dài hạn
    print("Running long-term algorithm...")
    long_term_start_time = time.time()

    # Chạy thuật toán dài hạn
    long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk, long_total_R_sk, long_total_pi_sk, long_objective = solving.long_term_2(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts)
    long_term_end_time = time.time()
    long_term_time = long_term_end_time - long_term_start_time

    print("long_total_pi_sk = ", long_total_pi_sk)
    print("long_objective = ", long_objective.value)
    # Lưu kết quả
    other_function.save_results(
        0, "long_term", long_term_time, long_total_pi_sk, long_objective, output_folder_time, "long_term"
    )
    print(f"Long-term algorithm completed in {long_term_time:.4f} seconds.")

    # ==== KẾT THÚC ====
    print("Long-term algorithm execution completed.\n")


    # ==== ÁNH XẠ UE VÀO RU GẦN NHẤT ====
    print("Mapping UEs to the nearest RU...")
    arr_phi_i_sk = other_function.mapping_RU_UE(slice_mapping, distances_RU_UE)

    # Chạy thuật toán ánh xạ gần nhất
    print("Running nearest mapping algorithm...")
    nearest_start_time = time.time()
    nearest_pi_sk, nearest_z_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_total_pi_sk, nearest_objective = solving.mapping_RU_nearest_UE(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, arr_phi_i_sk, data_rate, P_ib_sk, max_tx_power_mwatts)
    print("nearest_total_pi_sk = ", nearest_total_pi_sk)
    print("nearest_objective = ", nearest_objective.value)
    nearest_end_time = time.time()
    nearest_time = nearest_end_time - nearest_start_time

        # Lưu kết quả ánh xạ gần nhất
    other_function.save_results(0, "nearest_mapping", nearest_time, nearest_total_pi_sk, nearest_objective, output_folder_time, "nearest_mapping")
    print(f"Nearest mapping algorithm completed in {nearest_time:.4f} seconds.")

def main_2():
    # ==== CẤU HÌNH BAN ĐẦU ====
    seed = 2
    np.random.seed(seed)
    print("\n[INFO] Random seed set to:", seed)

    # Lấy thời gian hiện tại để tạo thư mục lưu trữ
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_folder_time = os.path.join(SAVE_PATH, current_time)

    # Tạo thư mục lưu kết quả
    try:
        os.makedirs(output_folder_time, exist_ok=True)
        print(f"[INFO] Output folder created at: {output_folder_time}")
    except Exception as e:
        print(f"[ERROR] Failed to create output folder: {e}")
        return

    # ==== TẠO TOPOLOGY MẠNG RAN ====
    print("\n[INFO] Creating RAN topology...")
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)
    print("[INFO] RAN topology created successfully.")

    # ==== TẠO TỌA ĐỘ RU VÀ UE ====
    print("\n[INFO] Generating RU and UE coordinates...")
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
    coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
    print("[INFO] RU and UE coordinates generated successfully.")

    # ==== LẤY LIÊN KẾT VÀ THÔNG SỐ MẠNG ====
    print("\n[INFO] Retrieving network links and capacities...")
    l_ru_du, l_du_cu = RAN_topo.get_links(G)
    _, A_j, A_m = RAN_topo.get_node_cap(G)
    print("[INFO] Network links and capacities retrieved successfully.")

    # ==== TẠO YÊU CẦU CỦA UE ====
    print("\n[INFO] Generating UE requirements...")
    slice_mapping, D_j, D_m = gen_RU_UE.gen_mapping_and_requirements(
        num_UEs, num_slices, D_j_random_list, D_m_random_list
    )
    print("[INFO] UE requirements generated successfully.")

    # ==== TÍNH TOÁN KHOẢNG CÁCH VÀ MA TRẬN KÊNH ====
    print("\n[INFO] Calculating distances and channel matrix...")
    distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
    gain, SNR, data_rate = wireless.channel_matrix(
        distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas,
        path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk
    )
    print("[INFO] Channel matrix calculated successfully.")

    # ==== THUẬT TOÁN DÀI HẠN ====
    
    long_short_num_UEs = num_UEs  # Số lượng UE ban đầu

    for frame in range(num_frames):
        print(f"\n[INFO] Frame {frame + 1}/{num_frames}")
        long_term_start_time = time.time()

        print("\n[INFO] Running long-term algorithm...")
        # Chạy thuật toán dài hạn
        (
            long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk,
            long_phi_m_sk, long_total_R_sk, long_total_pi_sk, long_objective
        ) = solving.long_term_2(
            num_slices, long_short_num_UEs, num_RUs, num_DUs, num_CUs, num_RBs,
            D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma,
            slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts
        )

        print(f"    [RESULT] Long-term total power (pi_sk): {long_total_pi_sk}")
        print(f"    [RESULT] Long-term objective value: {long_objective.value:.4f}")

        long_term_end_time = time.time()
        long_term_time = long_term_end_time - long_term_start_time
        print(f"    [INFO] Long-term algorithm completed in {long_term_time:.4f} seconds.")
        
        print(f"    [INFO] Long-term results saved successfully.")

        arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk = other_function.extract_optimization_results(long_pi_sk, long_z_ib_sk, long_phi_i_sk)
        # ==== THUẬT TOÁN NGẮN HẠN (MỖI SLOT THỜI GIAN) ====
        for slot in range(time_slots):
            print(f"        [INFO] Time slot {slot + 1}/{time_slots}")
            print(f"        [DEBUG] long_pi_sk: {long_pi_sk.value}")
            print(f"        [DEBUG] total_ long_pi_sk: {long_total_pi_sk}")
            print(f"        [DEBUG] long_phi_i_sk: {arr_long_phi_i_sk}")

            # UE di chuyển (có toạ độ mới)
            short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(coordinates_UE, delta_coordinate)

            # Khoảng cách mới từ UE - RU sau khi di chuyển
            short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, short_coordinates_UE, num_RUs, long_short_num_UEs)

            # Tính gain cho kênh truyền thay đổi
            short_gain, _, _ = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)

            # Chạy thuật toán ngắn hạn
            print("\n[INFO] Running Short-term algorithm...")
            short_pi_sk, short_objective = solving.short_term_1(
                num_slices, long_short_num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain,
                R_min, epsilon, arr_long_pi_sk, arr_long_phi_i_sk, max_tx_power_mwatts
            )

            print(f"        [RESULT] Short-term total power (pi_sk): {short_pi_sk.value}")
            #print(f"        [DEBUG] total_ short_pi_sk: {short_total_pi_sk}")
            print(f"        [RESULT] Short-term objective value: {short_objective.value:.4f}")

            # Sinh số lượng UE mới cho vòng tiếp theo
        long_short_num_UEs = other_function.generate_new_num_UEs(long_short_num_UEs, delta_num_UE)
        print(f"        [INFO] Updated number of UEs: {long_short_num_UEs}")

    # ==== KẾT THÚC ====
    print("\n[INFO] Simulation completed successfully.")


def main_2_short_term():
    # ==== CẤU HÌNH BAN ĐẦU ====
    seed = 2
    np.random.seed(seed)
    print("\n[INFO] Random seed set to:", seed)

    # ==== TẠO TOPOLOGY MẠNG RAN ====
    print("\n[INFO] Creating RAN topology...")
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)
    print("[INFO] RAN topology created successfully.")

    # ==== TẠO TỌA ĐỘ RU VÀ UE ====
    print("\n[INFO] Generating RU and UE coordinates...")
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
    coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
    print("[INFO] RU and UE coordinates generated successfully.")

    # ==== TẠO YÊU CẦU CỦA UE ====
    print("\n[INFO] Generating UE requirements...")
    slice_mapping, D_j, D_m = gen_RU_UE.gen_mapping_and_requirements(
        num_UEs, num_slices, D_j_random_list, D_m_random_list
    )
    print("[INFO] UE requirements generated successfully.")

    # ==== THUẬT TOÁN NGẮN HẠN ====
    print("\n[INFO] Starting short-term algorithm...")

    # UE di chuyển (có toạ độ mới)
    short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(coordinates_UE, delta_coordinate)

    # Khoảng cách mới từ UE - RU sau khi di chuyển
    short_distances_RU_UE = gen_RU_UE.calculate_distances(
        coordinates_RU, short_coordinates_UE, num_RUs, num_UEs
    )

    # Tính gain cho kênh truyền thay đổi
    short_gain, _, _ = wireless.channel_gain(short_distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)
    #print("short_gain: ", short_gain)

    # Giá trị long_pi_sk và long_phi_i_sk
    arr_long_pi_sk = np.array([[1, 1, 0, 1, 1]])
    arr_long_phi_i_sk = np.array([
        [[0, 0, 0, 0, 1]],
        [[0, 1, 0, 0, 0]],
        [[1, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0]],
        [[0, 0, 0, 1, 0]],
    ])

    # Chạy thuật toán ngắn hạn
    print("\n[INFO] Running short-term algorithm...")
    try:
        short_pi_sk, short_total_pi_sk, short_objective = solving.short_term_1(
            num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain, R_min, epsilon, arr_long_pi_sk, arr_long_phi_i_sk, max_tx_power_mwatts)

        if short_pi_sk.value is not None:
            print(f"[RESULT] Short-term total power (pi_sk): {short_pi_sk.value}")
            print(f"[RESULT] Short-term total pi_sk: {short_total_pi_sk}")
            print(f"[RESULT] Short-term objective value: {short_objective.value:.4f}")
        else:
            print("[WARNING] Short-term optimization failed. No solution found.")

    except Exception as e:
        print(f"[ERROR] Short-term algorithm failed: {e}")

    # ==== KẾT THÚC ====
    print("\n[INFO] Short-term simulation completed successfully.")

def random():
    # ==== CẤU HÌNH BAN ĐẦU ====
    seed = 2
    np.random.seed(seed)

    # ==== TẠO TOPOLOGY MẠNG RAN ====
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

    # Tạo tọa độ RU và UE
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)
    coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)

    # Tính toán khoảng cách và ma trận kênh
    distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
        
    # Lấy liên kết và thông số mạng
    l_ru_du, l_du_cu = RAN_topo.get_links(G)
    _, A_j, A_m = RAN_topo.get_node_cap(G)


    # ==== TẠO YÊU CẦU CỦA UE ====
    print("\n[INFO] Generating UE requirements...")
    slice_mapping, D_j, D_m = gen_RU_UE.gen_mapping_and_requirements(
        num_UEs, num_slices, D_j_random_list, D_m_random_list
    )

    # Tính gain cho kênh truyền thay đổi
    _, _, data_rate = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)
    #print("short_gain: ", short_gain)

    # Chạy thuật toán ngắn hạn
    print("\n[INFO] Running short-term algorithm...")
    pi_sk, z_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, total_pi_sk, status = solving.random_choice(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts)

    if pi_sk.value is not None:
        print(f"[RESULT] Short-term total power (pi_sk): {pi_sk.value}")
        print(f"[RESULT] Short-term total pi_sk: {total_pi_sk}")
    else:
        print("[WARNING] Short-term optimization failed. No solution found.")

    # ==== KẾT THÚC ====
    print("\n[INFO] Short-term simulation completed successfully.")
# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()




