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
num_UEs = 30                                           # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
num_RBs = 40                                           # Số lượng của RBs
num_antennas = 16                                        # Số lượng anntenas

radius_in = 100                                         # Bán kính vòng tròn trong (m)
radius_out = 1000                                       # Bán kính vòng tròn ngoài (m)

bandwidth_per_RB = 180e3                                # Băng thông của mỗi RBs (Hz)
total_bandwidth = bandwidth_per_RB * num_RBs            # Tổng băng thông (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                                   # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10)       # Công suất tại mỗi RU (mW)
noise_power_density = 1e-10                             # Mật độ công suất nhiễu (W/Hz) 
P_ib_sk = max_tx_power_mwatts / num_RBs                 # Phân bổ công suất đều cho các resource block

epsilon = 1e-5                                          # Giá trị nhỏ ~ 0

P_j_random_list = [max_tx_power_mwatts]

path_loss_ref = 128.1
path_loss_exp = 37.6

num_slices = 1                                          # Số lượng loại dịch vụ
if num_slices == 1:
    slices = ["eMBB"]                                   # Tập các loại slice
else:
    slices = ["eMBB", "ULLRC", "mMTC"]                  # Tập các loại slice

D_j_random_list = [10]                                  # Các loại yêu cầu tài nguyên của node DU j 
D_m_random_list = [10]                                  # Các loại yêu cầu tài nguyên của node CU m 

A_j_random_list = [100]                                 # Các loại tài nguyên của node DU j
A_m_random_list = [100]                                 # Các loại tài nguyên của node CU m                              
R_min = 2e6                                             # Các loại yêu cầu Data rate ngưỡng

delta_coordinate = 10                                   # Sai số toạ độ của UE (met)
delta_num_UE = 3                                        # Sai số số lượng UE (UE)

time_slots = 3                                          # Số lượng time slot trong 1 frame
num_frames = 2   

num_step = 20                                           # Số lần chạy mô phỏng 
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
    # Chạy với các gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for gamma in [0.9]:
        output_folder_gamma = os.path.join(output_folder_time, f"gamma_{gamma}")
        # Tạo thư mục lưu kết quả
        try:
            os.makedirs(output_folder_time, exist_ok=True)
            print(f"Output folder created: {output_folder_time}")
        except Exception as e:
            print(f"Error creating output folder: {e}")
            return

        # TẠO TOPOLOGY MẠNG RAN
        G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

        # Tạo tọa độ RU
        coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)

        # Lấy liên kết và thông số mạng
        l_ru_du, l_du_cu = RAN_topo.get_links(G)
        _, A_j, A_m = RAN_topo.get_node_cap(G)

        # ===========================================
        # ==== BẮT ĐẦU CÁC BƯỚC MÔ PHỎNG ====
        # ===========================================
        for step in range(num_step):
        
            print(f"\n===== Step {step + 1}/{num_step} =====")
            step_start_time = time.time()
            # Tạo toạ độ cho UE
            coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)

            # Tính khoảng cách từ UE - RU
            distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

            # Tính gain, data rate
            gain, _, data_rate = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)

            # Tạo yêu cầu của từng UE
            slice_mapping, D_j, D_m = gen_RU_UE.gen_mapping_and_requirements(num_UEs, num_slices, D_j_random_list, D_m_random_list)
            # ===========================================
            # ========= Chạy thuật toán random mapping ==========
            # ===========================================
            # Chọn random RU cho UE 
            random_phi_i_sk = other_function.mapping_random_RU_UE(num_RUs, num_UEs, num_slices, slice_mapping)
            print("Running Random mapping algorithm...")
            random_start_time = time.time()
            random_pi_sk, random_z_ib_sk, random_phi_i_sk, random_phi_j_sk, random_phi_m_sk, random_total_R_sk, random_total_pi_sk, random_objective, random_total_z_ib_sk, random_total_R_sk, random_total_p_ib_sk = solving.Random_RU(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, random_phi_i_sk, data_rate, P_ib_sk, max_tx_power_mwatts)
            random_end_time = time.time()
            random_time = random_end_time - random_start_time
            # Lưu kết quả Random_RU
            other_function.save_results("Random_RU", random_time, random_total_pi_sk, random_total_R_sk, random_total_z_ib_sk, random_total_p_ib_sk, random_objective, output_folder_gamma)
            print(f"Random algorithm completed in {random_time:.4f} seconds.")
            # ===========================================
            # ==== Thuật toán ÁNH XẠ UE VÀO RU GẦN NHẤT ====
            # ===========================================
            # Chọn RU cho UE gần nhất
            arr_nearest_phi_i_sk = other_function.mapping_nearest_RU_UE(distances_RU_UE, slice_mapping, num_RUs, num_UEs, num_slices)
            print("Running nearest mapping algorithm...")
            nearest_start_time = time.time()
            nearest_pi_sk, nearest_z_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_total_pi_sk, nearest_objective, nearest_total_z_ib_sk, nearest_total_R_sk, nearest_total_p_ib_sk = solving.Nearest_RU(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, arr_nearest_phi_i_sk, data_rate, P_ib_sk, max_tx_power_mwatts)
            nearest_end_time = time.time() 
            nearest_time = nearest_end_time - nearest_start_time
        
            # Lưu kết quả ánh xạ gần nhất
            other_function.save_results("Nearest_RU", nearest_time, nearest_total_pi_sk, nearest_total_R_sk, nearest_total_z_ib_sk, nearest_total_p_ib_sk, nearest_objective, output_folder_gamma)
            print(f"Nearest mapping algorithm completed in {nearest_time:.4f} seconds.")
   
            # ===========================================
            # ======== Chạy Doraemon =============
            # ===========================================
            # Chạy thuật toán dài hạn
            print("Running Doraemon mapping algorithm...")
            Doraemon_start_time = time.time()
            long_start_time = time.time()
            long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk, long_total_pi_sk, long_objective, long_total_z_ib_sk, long_total_R_sk, long_total_p_ib_sk = solving.Long_Doraemon(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts)
            long_end_time = time.time()
            long_time = long_end_time - long_start_time
            # Làm tròn long_pi_sk
            # Làm tròn long_pi_sk về 0 hoặc 1
            """print("long_pi_sk = ", np.rint(long_pi_sk.value).astype(int))

            # Làm tròn long_z_ib_sk về 0 hoặc 1
            long_z_values = np.rint(np.array([[[[long_z_ib_sk[i, b, s, k].value 
                                      for k in range(num_UEs)] 
                                      for s in range(num_slices)] 
                                      for b in range(num_RBs)] 
                                      for i in range(num_RUs)]))

            print("long_z_ib_sk = ", long_z_values.astype(int))"""


            # Lưu kết quả ánh xạ Long_Doraemon
            other_function.save_results("Doraemon", long_time, long_total_pi_sk, long_total_R_sk, long_total_z_ib_sk, long_total_p_ib_sk, long_objective, output_folder_gamma)
            #other_function.save_results("Long_Doraemon", long_time, total_pi_sk, total_R_sk, total_z_ib_sk, total_p_ib_sk, total_objective, output_folder_gamma)
            #print(f"Long_Doraemon algorithm completed in {long_time:.4f}.")
            
            arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk = other_function.extract_optimization_results(long_pi_sk, long_z_ib_sk, long_phi_i_sk)
            
            short_pi_sk, short_z_ib_sk, short_phi_i_sk, short_total_R_sk, short_total_pi_sk, short_objective, short_total_z_ib_sk, short_total_R_sk, short_total_p_ib_sk = solving.Short_Doraemon(num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, gain, R_min, epsilon, arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk, max_tx_power_mwatts, gamma)
            Doraemon_end_time = time.time()
            Doraemon_time = Doraemon_end_time - Doraemon_start_time
            print(f"Doraemon mapping algorithm completed in {Doraemon_time:.4f} seconds.")
            # Lưu kết quả long_short
            other_function.save_results("Doraemon", Doraemon_time, short_total_pi_sk, short_total_R_sk, short_total_z_ib_sk, short_total_p_ib_sk, short_objective, output_folder_gamma)
            # Hết step
            step_end_time = time.time()
            print(f"Step {step + 1} completed in {step_end_time - step_start_time:.4f} seconds.")

    # ==== KẾT THÚC ====
    print("\nAll steps completed.")

def main_2():
    seed = 1
    np.random.seed(seed)
    # Lấy thời gian hiện tại để tạo thư mục lưu trữ
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_folder_time = os.path.join(SAVE_PATH, current_time)
    # Chạy với các gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for gamma in [0.5]:
        output_folder_gamma = os.path.join(output_folder_time, f"gamma_{gamma}")
        # Tạo thư mục lưu kết quả
        try:
            os.makedirs(output_folder_time, exist_ok=True)
            print(f"Output folder created: {output_folder_time}")
        except Exception as e:
            print(f"Error creating output folder: {e}")
            return

        # TẠO TOPOLOGY MẠNG RAN
        G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

        # Tạo tọa độ RU
        coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)

        # Lấy liên kết và thông số mạng
        l_ru_du, l_du_cu = RAN_topo.get_links(G)
        _, A_j, A_m = RAN_topo.get_node_cap(G)

        # ===========================================
        # ========= Chạy Doraemon term ==============
        # ===========================================
        for step in range(num_step):
            print(f"\n===== Step {step + 1}/{num_step} =====")
            step_start_time = time.time()
            doraemon_num_UEs = num_UEs  # Số lượng UE ban đầu
            for frame in range(num_frames):
                # Tạo toạ độ cho UE
                coordinates_UE = gen_RU_UE.gen_coordinates_UE(doraemon_num_UEs, radius_in, radius_out)

                # Tính lại khoảng cách từ UE - RU
                distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, doraemon_num_UEs)

                # Tính lại gain, data rate
                short_gain, _, data_rate = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, doraemon_num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)

                # 3. Tạo yêu cầu của từng UE
                slice_mapping, D_j, D_m = gen_RU_UE.gen_mapping_and_requirements(doraemon_num_UEs, num_slices, D_j_random_list, D_m_random_list)

                # Chạy thuật toán dài hạn
                long_start_time = time.time()
                long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk, long_pi_sk, long_z_ib_sk, long_phi_i_sk, long_phi_j_sk, long_phi_m_sk, long_total_pi_sk, long_objective, long_total_z_ib_sk, long_total_R_sk, long_total_p_ib_sk = solving.Long_Doraemon(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts)
                long_end_time = time.time()
                long_time = long_end_time - long_start_time
            
                # Lưu kết quả ánh xạ Long_Doraemon
                other_function.save_results("Doraemon", long_time, long_total_pi_sk, long_total_R_sk, long_total_z_ib_sk, long_total_p_ib_sk, long_objective, output_folder_gamma)
                print(f"Long_Doraemon algorithm completed in {long_time:.4f} seconds at frame {frame}.")

                arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk = other_function.extract_optimization_results(long_pi_sk, long_z_ib_sk, long_phi_i_sk)
                for slot in range(time_slots):
                    if slot == 0:
                        short_coordinates_UE = coordinates_UE
                    short_start_time = time.time()
                    short_pi_sk, short_z_ib_sk, short_phi_i_sk, short_total_R_sk, short_total_pi_sk, short_objective, short_total_z_ib_sk, short_total_R_sk, short_total_p_ib_sk = solving.Short_Doraemon(num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain, R_min, epsilon, arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk, max_tx_power_mwatts, gamma)
                    short_end_time = time.time()
                    short_time = short_end_time - short_start_time
                    print(f"Short_Doraemon mapping algorithm completed in {short_time:.4f} seconds at slot {slot}.")

                    # Lưu kết quả long_short
                    other_function.save_results("Doraemon", short_time, short_total_pi_sk, short_total_R_sk, short_total_z_ib_sk, short_total_p_ib_sk, short_objective, output_folder_gamma)
                
                    # UE di chuyển (có toạ độ mới)
                    short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(short_coordinates_UE, delta_coordinate)

                    # Khoảng cách mới từ UE - RU sau khi di chuyển
                    short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, short_coordinates_UE, num_RUs, doraemon_num_UEs)

                    # Tính gain cho kênh truyền thay đổi
                    short_gain, _, _ = wireless.channel_gain(short_distances_RU_UE, num_slices, num_RUs, doraemon_num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_density, bandwidth_per_RB, P_ib_sk)
                
                # Sinh số lượng UE mới cho vòng tiếp theo
                doraemon_num_UEs = other_function.generate_new_num_UEs(doraemon_num_UEs, delta_num_UE)
            
            # Hết step
            step_end_time = time.time()
            print(f"Step {step + 1} completed in {step_end_time - step_start_time:.4f} seconds.")

    # ==== KẾT THÚC ====
    print("\nAll steps completed.")

# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()




