import gen_RU_UE
import wireless
import RAN_topo
import solving
import benchmark

num_RUs = 2                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 2                             # Số lượng DU
num_CUs = 2                             # Số lượng CU
num_UEs = 7                             # Số lượng user
num_RBs = 3                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas

radius_in = 100                         # Bán kính vòng tròn trong (km)
radius_out = 1000                       # Bán kính vòng tròn ngoài (km)
capacity_node = 100                     # Tài nguyên node

rb_bandwidth = 180e3                    # Băng thông của mỗi RBs (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                   # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10) # Công suất tại mỗi RU (mW)
noise_power_watts = 1e-10 # Công suất nhiễu (mW)   

path_loss_ref = 128.1
path_loss_exp = 37.6

D_j = 100                                 # yêu cầu tài nguyên của node DU j
D_m = 100                                 # yêu cầu tài nguyên của node CU m

R_min = 1e6                              # Data rate ngưỡng yêu cầu
epsilon = 1e-10                           # Giá trị nhỏ ~ 0

#Toạ toạ độ RU, UE
coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)                  
coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out) 

# Tính khoảng cách giữa euclid RU-UE (km)
distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

# Tạo mạng RAN
G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, capacity_node)

# Danh sách tập các liên kết trong mạng
l_ru_du, l_du_cu = RAN_topo.get_links(G)

# Tập các capacity của các node DU, CU
A_j, A_m = RAN_topo.get_node_cap(G)

gain = wireless.channel_gain(distances_RU_UE, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
#print("gain: ", gain)

#P_bi_sk = wireless.allocate_power(num_RUs, num_UEs, num_RBs, max_tx_power_mwatts)
P_bi_sk = max_tx_power_mwatts / num_RBs
#print("P_bi_sk: ", P_bi_sk)

a_sk, z_bi_sk, phi_i_sk = solving.short_term(num_UEs, num_RUs, num_RBs, P_bi_sk, max_tx_power_mwatts, rb_bandwidth, R_min, gain, epsilon)
benchmark.print_results_short_term(a_sk, z_bi_sk, phi_i_sk)

#pi_sk, phi_j_sk, phi_m_sk = solving.long_term(num_UEs, num_RUs, num_DUs, num_CUs, D_j, D_m, A_j, A_m, l_ru_du, l_du_cu, phi_i_sk)
#benchmark.print_result_long_term(pi_sk, phi_i_sk, phi_j_sk, phi_m_sk)

