import numpy as np
import cvxpy as cp
import wireless
import gen_RU_UE

# Các tham số đã được định nghĩa
num_RUs = 4                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 2                             # Số lượng DU
num_CUs = 2                             # Số lượng CU
num_UEs = 20                            # Số lượng user
radius_in = 100                         # Bán kính vòng tròn trong (km)
radius_out = 1000                       # Bán kính vòng tròn ngoài (km)
num_RBs = 2                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas
user_requests = np.random.uniform(0, 20, num_UEs)

rb_bandwidth = 180e3                    # Băng thông
max_tx_power_dbm = 43                   # dBm
max_tx_power_watts = 10**((max_tx_power_dbm)/10) # in mWatts  
noise_power_watts = 1e-10               # Công suất nhiễu (in mWatts)

path_loss_ref = 128.1
path_loss_exp = 37.6

D_j = 10                                # yêu cầu tài nguyên của node DU j
D_m = 10                                # yêu cầu tài nguyên của node CU m

R_min = 1e6                             # Data rate ngưỡng yêu cầu
epsilon = 1e-6

# Biến đã có sẵn
pi_sk = cp.Variable((num_RUs, num_UEs), boolean=True)

#Toạ toạ độ RU, UE
coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)                  
coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out) 

# Tính khoảng cách giữa euclid RU-UE (km)
distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

# Tính độ lợi của kênh truyền 
gain = wireless.channel_gain(distances_RU_UE, num_RUs, num_UEs, num_RBs, noise_power_watts, num_antennas, path_loss_ref, path_loss_exp)

# Tính phân bổ công suất
P_bi_sk = wireless.allocate_power(num_RUs, num_UEs, num_RBs, max_tx_power_watts, gain, user_requests)

# Hàm tối ưu hóa
def optimize(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_watts, rb_bandwidth, D_j, D_m, R_min, gain, P_bi_sk, epsilon):
    try:
        # Khởi tạo ma trận z_bi_sk (biến nhị phân)
        z_bi_sk = {}
        for i in range(num_RUs):
            for k in range(num_UEs):
                for b in range(num_RBs):
                    z_bi_sk[(i, k, b)] = cp.Variable(boolean=True)

        # Khởi tạo các biến phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = cp.Variable((num_RUs, num_UEs), boolean=True)
        phi_j_sk = cp.Variable((num_DUs, num_UEs), boolean=True)
        phi_m_sk = cp.Variable((num_CUs, num_UEs), boolean=True)

        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_UEs))

        # Hàm mục tiêu
        objective = cp.Maximize(cp.sum(pi_sk))

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc (15a)
        for k in range(num_UEs):
            R_sk = 0
            for b in range(num_RBs):
                SNR = 0
                for i in range(num_RUs): 
                    SNR += P_bi_sk[i, k, b] * z_bi_sk[(i, k, b)] * gain[i, k, b]
                R_sk += rb_bandwidth * cp.log(1 + SNR) / np.log(2)
            constraints.append(R_sk >= R_min * cp.sum(pi_sk[k]))

        # Ràng buộc (15b)
        for i in range(num_RUs):
            tổng_công_suất = 0
            for b in range(num_RBs):
                for k in range(num_UEs):
                    tổng_công_suất += P_bi_sk[i, k, b] * z_bi_sk[(i, k, b)]
            constraints.append(tổng_công_suất <= max_tx_power_watts)

        # Ràng buộc (15e)
        for k in range(num_UEs):
            constraints.append(cp.sum(phi_i_sk[:, k]) == cp.sum(pi_sk[k]))

        # Ràng buộc (15h)
        for i in range(num_RUs):
            for k in range(num_UEs):
                constraints.append(cp.sum([z_bi_sk[(i, k, b)] for b in range(num_RBs)])/num_RBs <= phi_i_sk[i, k])
                constraints.append(phi_i_sk[i, k] <= cp.sum([z_bi_sk[(i, k, b)] for b in range(num_RBs)])/num_RBs + (1 - epsilon))
                
        # Additional constraint
        for i in range(num_RUs):
            for k in range(num_UEs):
                for b in range(num_RBs):
                    constraints.append(z_bi_sk[(i, k, b)] <= pi_sk[k])

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

        # In kết quả của z_bi_sk sau khi tối ưu
        print("Giá trị của các biến z_bi_sk:")
        for i in range(num_RUs):
            for k in range(num_UEs):
                for b in range(num_RBs):
                    print(f"z_bi_sk[{i}, {k}, {b}] = {z_bi_sk[(i, k, b)].value}")

        return pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk

    except cp.SolverError:
        print('Lỗi solver: không khả thi')
        return None, None, None, None, None
    except Exception as e:
        print(f"Lỗi xảy ra: {e}")

# Gọi hàm tối ưu và in kết quả
pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk = optimize(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_watts, rb_bandwidth, D_j, D_m, R_min, gain, P_bi_sk, epsilon)

# Kiểm tra kết quả của hàm tối ưu
if pi_sk is not None and z_bi_sk is not None:
    print("Giá trị của các biến pi_sk:")
    print(pi_sk.value)

    print("Giá trị của các biến z_bi_sk:")
    for i in range(num_RUs):
        for k in range(num_UEs):
            for b in range(num_RBs):
                print(f"z_bi_sk[{i}, {k}, {b}] = {z_bi_sk[(i, k, b)].value}")
else:
    print("Quá trình tối ưu hóa không thành công hoặc không khả thi.")