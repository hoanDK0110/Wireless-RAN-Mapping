import gen_RU_UE
import wireless
import RAN_topo
import solving
import benchmark
import time
import numpy as np
import csv

num_RUs = 5                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 4                             # Số lượng DU
num_CUs = 4                             # Số lượng CU
num_UEs = 5                             # Số lượng user
num_RBs = 10                             # Số lượng của RBs
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

running_mode = 1 # 0 for random_RB and 1 for nearest_RU

D_j = 50                                 # yêu cầu tài nguyên của node DU j
D_m = 50                                 # yêu cầu tài nguyên của node CU m

D_j_ramdom_list = [5]
D_m_random_list = [5]
A_j_random_list = [100]
A_m_random_list = [100] 
R_min_ramdom_list = [1e6]

R_min = 1e6                              # Data rate ngưỡng yêu cầu
epsilon = 1e-6                           # Giá trị nhỏ ~ 0

#Toạ toạ độ RU, UE
coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)                  
coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out) 

delta_coordinates = 5

# Tính khoảng cách giữa euclid RU-UE (km)
distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

# Tạo mạng RAN
G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, A_j_random_list, A_m_random_list)

# Danh sách tập các liên kết trong mạng
l_ru_du, l_du_cu = RAN_topo.get_links(G)

# Tập các capacity của các node DU, CU
A_j, A_m = RAN_topo.get_node_cap(G)

gain = wireless.channel_gain(distances_RU_UE, num_slice, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)

#random_RBs
if running_mode == 0:
# Solve
    pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk = solving.global_problem(running_mode, 0, num_slice, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon)

    benchmark.print_results(running_mode,num_slice, num_RUs, num_DUs, num_CUs, num_UEs, pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk)
    # Tính toán used_power và unused_power
    used_power = np.zeros(P_bi_sk.shape[1])
    unused_power = np.zeros(P_bi_sk.shape[1])
    for s in range(P_bi_sk.shape[0]):
        for i in range(P_bi_sk.shape[1]):
            temp = 0
            for k in range(P_bi_sk.shape[2]):
                for b in range(P_bi_sk.shape[3]):
                    temp += P_bi_sk[s, i, k, b].value
            used_power[i] = (temp / max_tx_power_mwatts) * 100
            unused_power[i] = 110 - used_power[i]

    # Tính toán used_rb và unused_rb
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
    unused_rb = 110 - used_rb

    #Tính toán used_du và unused_du
    used_du = np.zeros(phi_j_sk.shape[1])
    unused_du = np.zeros(phi_j_sk.shape[1])
    for s in range(phi_j_sk.shape[0]):
        for j in range(phi_j_sk.shape[1]):
            temp_du = 0
            for k in range(phi_j_sk.shape[2]):
                temp_du += phi_j_sk[s, j , k].value
            used_du[j] = (temp_du * D_j / capacity_node) * 100
            unused_du[j] = 110 - used_du[j]

    #Tính toán used_cu và unused_cu
    used_cu = np.zeros(phi_m_sk.shape[1])
    unused_cu = np.zeros(phi_m_sk.shape[1])
    for s in range(phi_m_sk.shape[0]):
        for m in range(phi_m_sk.shape[1]):
            temp_cu = 0
            for k in range(phi_m_sk.shape[2]):
                temp_cu += phi_m_sk[s, m , k].value
            used_cu[m] = (temp_cu * D_m / capacity_node) * 100
            unused_cu[m] = 110 - used_cu[m]

    # Lưu trữ dữ liệu vào tệp
    np.save("used_power.npy", used_power)
    np.save("unused_power.npy", unused_power)
    np.save("used_rb.npy", used_rb)
    np.save("unused_rb.npy", unused_rb)
    np.save("used_du.npy", used_du)
    np.save("unused_du.npy", unused_du)
    np.save("used_cu.npy", used_cu)
    np.save("unused_cu.npy", unused_cu)

    # Ghi pi_sk ra file CSV (nếu pi_sk là một dictionary)
    with open('pi_sk.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Accepted"])
        if pi_sk.value is not None:
            for idx, val in enumerate(pi_sk.value):
                writer.writerow([idx, val])

    # Ghi z_bi_sk ra file CSV (nếu z_bi_sk là một dictionary)
    with open('z_bi_sk.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["(RU, UE, RB)", "Value"])
        for s in range(num_slice):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    for b in range(num_RBs):  # z_bi_sk là dictionary
                        writer.writerow([s, i,k,b, z_bi_sk[(s, i, k, b)].value])

    # Ghi phi_i_sk ra file CSV (nếu phi_i_sk là một dictionary)
    with open('phi_i_sk.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Slice", "RU", "UE", "Value"])
        
        # Duyệt qua từng slice, RU, và UE trong mảng phi_i_sk
        for s in range(phi_i_sk.shape[0]):  
            for i in range(phi_i_sk.shape[1]):  
                for k in range(phi_i_sk.shape[2]):  
                    Phi_i_sk = phi_i_sk[s, i, k]  
                    if Phi_i_sk.value is not None:  
                        writer.writerow([s, i, k, Phi_i_sk.value])  

    # Ghi phi_j_sk ra file CSV (nếu phi_j_sk là một dictionary)
    with open('phi_j_sk.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Slice", "DU", "UE", "Value"])
        
        # Duyệt qua từng slice, RU, và UE trong mảng phi_i_sk
        for s in range(phi_j_sk.shape[0]):  
            for i in range(phi_j_sk.shape[1]):  
                for k in range(phi_j_sk.shape[2]):  
                    Phi_j_sk = phi_j_sk[s, i, k]  
                    if Phi_j_sk.value is not None:  
                        writer.writerow([s, i, k, Phi_j_sk.value])

    # Ghi phi_m_sk ra file CSV (nếu phi_m_sk là một dictionary)
    with open('phi_m_sk.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Slice", "CU", "UE", "Value"])
        
        # Duyệt qua từng slice, RU, và UE trong mảng phi_i_sk
        for s in range(phi_m_sk.shape[0]):  
            for i in range(phi_m_sk.shape[1]):  
                for k in range(phi_m_sk.shape[2]):  
                    Phi_m_sk = phi_m_sk[s, i, k]  
                    if Phi_m_sk.value is not None:  
                        writer.writerow([s, i, k, Phi_m_sk.value])

    # Gọi hàm chart1 từ file chart
    #chart.chart1(used_power, unused_power, used_rb, unused_rb)
    #short-term
    pi_sk_value_for_short_term = pi_sk.value
    phi_i_sk_value_for_short_term = np.zeros((num_slice, num_RUs, num_UEs))
    for s in range(num_slice):
        for i in range(num_RUs):
            for k in range(num_UEs):
                phi_i_sk_value_for_short_term[s,i,k] = phi_i_sk[s,i,k].value
    phi_j_sk_value_for_short_term = np.zeros((num_slice, num_DUs, num_UEs))
    for s in range(num_slice):
        for j in range(num_DUs):
            for k in range(num_UEs):
                phi_j_sk_value_for_short_term[s,j,k] = phi_j_sk[s,j,k].value
    phi_m_sk_value_for_short_term = np.zeros((num_slice, num_CUs, num_UEs))
    for s in range(num_slice):
        for m in range(num_CUs):
            for k in range(num_UEs):
                phi_m_sk_value_for_short_term[s,m,k] = phi_m_sk[s,m,k].value
    # for i in range(num_UEs):
    #     if pi_sk_value_for_short_term[i] == 1 :
    #         coordinates_UE_short_term = gen_RU_UE.gen_coordinates_UE_for_short_term(i, coordinates_UE, 5) 
    # distances_RU_UE_short_term = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
    # gain_short_term = wireless.channel_gain(distances_RU_UE_short_term, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
    # R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term = solving.short_term(num_RUs, num_RBs, num_UEs, rb_bandwidth, gain_short_term, R_min, max_tx_power_mwatts, pi_sk_value_for_short_term)
    # benmark.print_short_term_results(R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term)
    # for s in range(num_slice):
    #     for i in range(num_UEs):
    #         if pi_sk_value_for_short_term[s,i] == 1 :
    #             print(f"{i}")
    # print(type(phi_i_sk_value_for_short_term))
    # print(type(phi_j_sk_value_for_short_term))
    # print(type(phi_m_sk_value_for_short_term))
    # print(type(l_ru_du))
    last_3s_time = time.time()
    last_12s_time = time.time()
    while True:
        current_time = time.time()
        if current_time - last_3s_time >= 3:
            for s in range(num_slice):
                coordinates_UE_short_term = gen_RU_UE.gen_coordinates_UE_for_short_term(i, coordinates_UE, delta_coordinates) 
                distances_RU_UE_short_term = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE_short_term, num_RUs, num_UEs)
                gain_short_term = wireless.channel_gain(distances_RU_UE_short_term, num_slice, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
                R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term = solving.short_term(num_slice, num_RUs, num_DUs, num_CUs, num_RBs, num_UEs, epsilon, l_ru_du, l_du_cu, rb_bandwidth, gain_short_term, R_min, max_tx_power_mwatts, pi_sk_value_for_short_term, phi_i_sk_value_for_short_term, phi_j_sk_value_for_short_term, phi_m_sk_value_for_short_term)
                benchmark.print_short_term_results(num_slice,num_RUs,num_DUs,num_CUs,num_UEs,R_sk_short_term,mu_bi_sk_short_term,z_bi_sk_short_term)
                last_3s_time = current_time
        if current_time - last_12s_time >= 12:
            pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk = solving.global_problem(num_slice, 0, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon)
            benchmark.print_results(running_mode,num_slice,num_RUs,num_DUs,num_CUs,num_UEs,pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk)
            pi_sk_value_for_short_term = pi_sk.value
            last_12s_time = current_time
        if current_time >= 6:
            break
        time.sleep(1)




#NearestRU
if running_mode == 1:
    closest_RU_for_UE = []
    for k in range(num_UEs):
        min_distance = float('inf')
        closest_RU = -1
        for i in range(num_RUs):
            distances = distances_RU_UE[i,k]
            if distances < min_distance:
                min_distance = distances
                closest_RU = i
        closest_RU_for_UE.append(closest_RU)
    phi_i_sk_for_nearestRU = np.zeros((num_slice, num_RUs, num_UEs))
    for s in range(num_slice):
        for index, value in enumerate(closest_RU_for_UE):
            phi_i_sk_for_nearestRU[s, value, index] = 1
    for s in range(num_slice):
        print(f"\nGiá trị của phi_i_sk tại slice {s} (mối liên hệ giữa RU và UE):")
        for i in range(num_RUs):
            for k in range(num_UEs):
                print(phi_i_sk_for_nearestRU[s,i,k])
    pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk = solving.global_problem(running_mode, phi_i_sk_for_nearestRU, num_slice, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon)

    benchmark.print_results(running_mode,num_slice,num_RUs,num_DUs,num_CUs,num_UEs,pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk)
