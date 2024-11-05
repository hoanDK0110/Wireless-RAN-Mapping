import gen_RU_UE
import wireless
import RAN_topo
import solving
import benmark

num_RUs = 4                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 2                             # Số lượng DU
num_CUs = 2                             # Số lượng CU
num_UEs = 5                             # Số lượng user
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

D_j = 50                                 # yêu cầu tài nguyên của node DU j
D_m = 50                                 # yêu cầu tài nguyên của node CU m

R_min = 1e6                              # Data rate ngưỡng yêu cầu
epsilon = 1e-6                           # Giá trị nhỏ ~ 0

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

# Solve
pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk = solving.optimize(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon)


#short-term
for i in range(num_UEs):
    if pi_sk[i].value == 1 :
        coordinates_UE_short_term = gen_RU_UE.gen_coordinates_UE_for_short_term(i, coordinates_UE, 5) 
distances_RU_UE_short_term = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
gain_short_term = wireless.channel_gain(distances_RU_UE_short_term, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
R_sk_short_term = solving.short_term(num_RUs, num_RBs, num_UEs, rb_bandwidth, gain_short_term, R_min, max_tx_power_mwatts, pi_sk)
# Show result
benmark.print_results(pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk)
benmark.print_short_term_results(R_sk_short_term)