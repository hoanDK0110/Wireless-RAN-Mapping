import time
import gen_RU_UE
import wireless
import RAN_topo
import solving
import benchmark
import other_function

SAVE_PATH = "./result" # SAVE PATH

num_RUs = 4                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 2                             # Số lượng DU
num_CUs = 2                             # Số lượng CU
num_UEs = 5                             # Số lượng user
num_RBs = 3                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas
num_slices = 1                           # Số lượng loại dịch vụ

radius_in = 100                         # Bán kính vòng tròn trong (km)
radius_out = 1000                       # Bán kính vòng tròn ngoài (km)
capacity_node = 100                     # Tài nguyên node

rb_bandwidth = 180e3                                # Băng thông của mỗi RBs (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                               # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10)   # Công suất tại mỗi RU (mW)
noise_power_watts = 1e-10                           # Công suất nhiễu (mW)   

epsilon = 1e-10                                     # Giá trị nhỏ ~ 0

P_j_random_list = [max_tx_power_mwatts]

path_loss_ref = 128.1
path_loss_exp = 37.6
if num_slices == 1:
    slices = ["eMBB"]                               # Tập các loại slice
else:
    slices = ["eMBB", "ULLRC", "mMTC"]              # Tập các loại slice

D_j_random_list = [5]                                  # Các loại yêu cầu tài nguyên của node DU j 
D_m_random_list = [5]                                  # Các loại yêu cầu tài nguyên của node CU m 

A_j_random_list = [50]                       # Các loại tài nguyên của node DU j
A_m_random_list = [50]                       # Các loại tài nguyên của node CU m

R_min_random_list = [1e6]                               # Các loại yêu cầu Data rate ngưỡng

delta_coordinate = 20                               # Sai số toạ độ của UE
delta_num_UE = 2                                    # Sai số số lượng UE

time_slot = 1                                       # Số lượng time slot trong 1 frame
num_frame = 1

gamma = 0.8                                         # Hệ số tối ưu

# ===========================================
# ===========================================
# ===========================================
print("==== ORAN_MAPPING ====")
logger = other_function.Stopwatch("ORAN_MAPPING", silent=False)
logger.start()
logger.add("[instance] Generate instance")

# Toạ toạ độ RU
coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)                  
 
# Tạo mạng RAN
G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

# Danh sách tập các liên kết trong mạng
l_ru_du, l_du_cu = RAN_topo.get_links(G)

# Tập các capacity của các node DU, CU
P_i, A_j, A_m = RAN_topo.get_node_cap(G)

logger.add("[instance] Save instance")
# Save parameters
import os
os.makedirs(SAVE_PATH, exist_ok=True)
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix = f"{SAVE_PATH}/{timestamp}"
filename_problem = f"{filename_prefix}_problem.pkl.gz"
filename_solution = f"{filename_prefix}_solution"
other_function.save_object(
    f"{filename_prefix}_physicalnet.pkl.gz",
    G
)
other_function.save_object(
    f"{filename_prefix}_coords_RUs.pkl.gz",
    coordinates_RU
)
# End save region

logger.add("[solve] Start solving...")
for f in range(num_frame):
    logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Preparation")
    # Tạo yêu cầu từng UE
    names, R_min, D_j, D_m = gen_RU_UE.create_and_assign_slices(num_UEs, slices, D_j_random_list, D_m_random_list, R_min_random_list)

    # Tạo toạ độ UE
    coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
    other_function.save_object(
        f"{filename_prefix}_coords_UEs_f{f}.pkl.gz",
        coordinates_UE
    )

    # Ma trận khoảng cách của UE - RU
    distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

    # Tính gain cho mạng
    gain = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)

    # Gọi làm long-term: giải bài toán toàn cục
    logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Long-term solve")
    pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk = solving.long_term(
        num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, 
        logger
    )
    if (pi_sk is None):
        continue
    # In ra nghiệm
    # benchmark.print_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk)
    logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Long-term save")
    other_function.save_object(
        f"{filename_solution}_longterm_f{f}.pkl.gz",
        (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk)
    )
    logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term")
    for t in range(time_slot):
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term time_slot={t+1} of {time_slot}: Preparation")
        # UE di chuyển (có toạ độ mới)
        short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(coordinates_UE, delta_coordinate)
        #print("new_coordinates_UE: ", new_coordinates_UE)
        other_function.save_object(
            f"{filename_prefix}_coords_UEs_f{f}_t{t}.pkl.gz",
            short_coordinates_UE
        )

        # Khoảng cách mới từ UE - RU sau khi di chuyển
        short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, short_coordinates_UE, num_RUs, num_UEs)
        #print("new_distances_RU_UE: ", new_distances_RU_UE)

        # Tính gain cho kênh truyền thay đổi
        short_gain = wireless.channel_gain(short_distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)

        # Chuyển kết quả thành mảng
        arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk = other_function.extract_optimization_results(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs,pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk)

        # Tối ưu Short-term
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term time_slot={t+1} of {time_slot}: Solve")
        short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk = solving.short_term(
            num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, arr_pi_sk, arr_phi_i_sk,
            logger
        )

        if short_pi_sk is None:
            continue
        # In kết quả short-term
        # benchmark.print_result_short_term(short_z_ib_sk, short_p_ib_sk)
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term time_slot={t+1} of {time_slot}: Save")
        other_function.save_object(
            f"{filename_solution}_shortterm_f{f}_t{t}.pkl.gz",
            (short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk)
        )
logger.stop()
logger.write_to_file(f"{filename_prefix}.log")
