import gen_RU_UE
import wireless
import RAN_topo
import solving
import benmark
import time
import chart
import numpy as np

num_RUs = 4                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 2                             # Số lượng DU
num_CUs = 2                             # Số lượng CU
num_UEs = 5                             # Số lượng user
num_RBs = 3                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas
num_slice = 1

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

gain = wireless.channel_gain(distances_RU_UE, num_slice, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)

# Solve
pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk = solving.global_problem(num_slice, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon)

benmark.print_results(num_slice,num_RUs,num_DUs,num_CUs,num_UEs,pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk)
used_power = np.zeros(P_bi_sk.shape[1])
unused_power = np.zeros(P_bi_sk.shape[1])
for s in range(P_bi_sk.shape[0]):
    for i in range(P_bi_sk.shape[1]):
            temp = 0
            for k in range(P_bi_sk.shape[2]):
                for b in range(P_bi_sk.shape[3]):
                    temp += P_bi_sk[s,i,k,b].value
            used_power[i] = (temp / max_tx_power_mwatts) * 100
            unused_power[i] = 100 - used_power[i]
chart.plot_power_usage(used_power, unused_power)
used_rb = 0
unused_rb = 0
for s in range(z_bi_sk.shape[0]):
    for i in range(z_bi_sk.shape[1]):
        temp_rb = 0
        for k in range(z_bi_sk.shape[2]):
            for b in range(z_bi_sk.shape[3]):
                temp_rb += z_bi_sk[s, i, k, b].value
        used_rb += temp_rb
used_rb = (used_rb / num_RBs) * 100
unused_rb = 100 - used_rb
chart.plot_rb_usage(used_rb, unused_rb)
#short-term
pi_sk_value_for_short_term = pi_sk.value
# for i in range(num_UEs):
#     if pi_sk_value_for_short_term[i] == 1 :
#         coordinates_UE_short_term = gen_RU_UE.gen_coordinates_UE_for_short_term(i, coordinates_UE, 5) 
# distances_RU_UE_short_term = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
# gain_short_term = wireless.channel_gain(distances_RU_UE_short_term, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
# R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term = solving.short_term(num_RUs, num_RBs, num_UEs, rb_bandwidth, gain_short_term, R_min, max_tx_power_mwatts, pi_sk_value_for_short_term)
# benmark.print_short_term_results(R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term)
for s in range(num_slice):
    for i in range(num_UEs):
        if pi_sk_value_for_short_term[s,i] == 1 :
            print(f"{i}")

last_3s_time = time.time()
last_12s_time = time.time()
while True:
    current_time = time.time()
    if current_time - last_3s_time >= 3:
        for s in range(num_slice):
            for i in range(num_UEs):
                if pi_sk_value_for_short_term[s,i] == 1 :
                    coordinates_UE_short_term = gen_RU_UE.gen_coordinates_UE_for_short_term(i, coordinates_UE, 5) 
            distances_RU_UE_short_term = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
            gain_short_term = wireless.channel_gain(distances_RU_UE_short_term, num_slice, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
            R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term = solving.short_term(num_slice, num_RUs, num_RBs, num_UEs, rb_bandwidth, gain_short_term, R_min, max_tx_power_mwatts, pi_sk_value_for_short_term)
            benmark.print_short_term_results(num_slice,num_RUs,num_DUs,num_CUs,num_UEs,R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term)
            last_3s_time = current_time
    if current_time - last_12s_time >=12:
        pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk = solving.global_problem(num_slice, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon)
        benmark.print_results(num_slice,num_RUs,num_DUs,num_CUs,num_UEs,pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk)
        pi_sk_value_for_short_term = pi_sk.value
        last_12s_time = current_time
    time.sleep(1)