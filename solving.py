import numpy as np
import cvxpy as cp

def global_problem(num_slice, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon):
    try:
        # Khởi tạo ma trận z_bi_sk (biến nhị phân)
        z_bi_sk = np.empty((num_slice, num_RUs, num_UEs, num_RBs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        z_bi_sk[s,i,k,b]= cp.Variable(boolean=True, name = f"z_bi_sk({b}, {i}, {s}, {k})")
        
        # Khởi tạo ma trận mu (biến liên tục): mu = Z_bi_sk * P_bi_sk
        mu_bi_sk = np.empty((num_slice, num_RUs, num_UEs, num_RBs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        mu_bi_sk[s,i,k,b]= cp.Variable(name = f"mu_bi_sk({b}, {i}, {s}, {k})")
        
        # Khởi tạo ma trận phân bổ công suất (biến liên tục): P_bi_sk
        P_bi_sk = np.empty((num_slice, num_RUs, num_UEs, num_RBs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        P_bi_sk[s,i,k,b]= cp.Variable(name = f"P_bi_sk({b}, {i}, {s}, {k})")

        # Khởi tạo các biến phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = np.empty((num_slice, num_RUs, num_UEs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    phi_i_sk[s,i,k]= cp.Variable(boolean=True, name = f"phi_i_sk({i}, {s}, {k})")
        phi_j_sk = np.empty((num_slice, num_DUs, num_UEs), dtype=object)
        for s in range(num_slice):
            for j in range(num_DUs):
                for k in range(num_UEs):
                    phi_j_sk[s,j,k]= cp.Variable(boolean=True, name = f"phi_j_sk({i}, {s}, {k})")
        phi_m_sk = np.empty((num_slice, num_CUs, num_UEs), dtype=object)
        for s in range(num_slice):
            for m in range(num_CUs):
                for k in range(num_UEs):
                    phi_m_sk[s,m,k]= cp.Variable(boolean=True, name = f"phi_m_sk({i}, {s}, {k})")

        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_slice, num_UEs), boolean=True, name = f"pi_sk")

        sigma = cp.Variable((num_slice, num_UEs), name = f"sigma")

        # Hàm mục tiêu
        count_P = 0
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        count_P += P_bi_sk[(s, i, k, b)]

        #objective = cp.Maximize(cp.sum(pi_sk) - count_P * 1e-3)
        #objective = cp.Maximize(cp.sum(pi_sk))
        objective = cp.Maximize(0.8*cp.sum(pi_sk) + cp.sum(sigma)*0.2)
        # Danh sách ràng buộc
        constraints = []
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(mu_bi_sk[(s, i, k, b)] >= 0)
        
        for s in range(num_slice):
            for b in range(num_RBs):
                sum_z_bi_sk = 0
                for i in range(num_RUs):
                    for k in range(num_UEs):
                        sum_z_bi_sk += z_bi_sk[s,i,k,b]
                constraints.append(sum_z_bi_sk <= 1)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(z_bi_sk[(s, i, k, b)] <= 1)
        # Ràng buộc (16a)
        for s in range(num_slice):
            for k in range(num_UEs):
                R_sk = 0
                for b in range(num_RBs):
                    SNR_sum = 0 
                    for i in range(num_RUs): 
                        SNR_sum += gain[s, i, k, b] * mu_bi_sk[(s, i, k, b)]
                    R_sk += rb_bandwidth * cp.log(1 + SNR_sum) / np.log(2)
                constraints.append(R_sk >= R_min * pi_sk[s,k])
                constraints.append(sigma[s,k] <= R_sk)

        # Ràng buộc (16b)
        for s in range(num_slice):
            for i in range(num_RUs):
                total_power = 0
                for b in range(num_RBs):
                    for k in range(num_UEs):  
                        total_power += mu_bi_sk[(s, i, k, b)]
                constraints.append(total_power <= max_tx_power_mwatts) 

        # Ràng buộc (16c)
        for s in range(num_slice):
            for j in range(num_DUs):
                temp_du = 0
                for k in range(num_UEs):
                    temp_du += phi_j_sk[s, j, k]
                count_du = temp_du * D_j
                constraints.append(count_du <= A_j[j])

        # Ràng buộc (16d)
        for s in range(num_slice):
            for m in range(num_CUs):
                temp_cu = 0
                for k in range(num_UEs):
                    temp_cu += phi_m_sk[s, m, k]
                count_cu = temp_cu * D_m
                constraints.append(count_cu <= A_m[m])

        # Ràng buộc (16e + 16f + 16g)
        for s in range(num_slice):
            for k in range(num_UEs):
                temp_ru = 0
                for i in range(num_RUs):
                    temp_ru += phi_i_sk[s,i,k]
                constraints.append(temp_ru == pi_sk[s,k])


        for s in range(num_slice):
            for k in range(num_UEs):
                temp_du = 0
                for j in range(num_DUs):
                    temp_du += phi_j_sk[s,j,k]
                constraints.append(temp_du == pi_sk[s,k])
        
        for s in range(num_slice):
            for k in range(num_UEs):
                temp_cu = 0
                for m in range(num_CUs):
                    temp_cu += phi_m_sk[s,m,k]
                constraints.append(temp_cu == pi_sk[s,k])
        
        # Ràng buộc (16h)
        for s in range(num_slice):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    count_z = 0
                    for b in range(num_RBs):
                        count_z += z_bi_sk[(s, i, k, b)]
                    constraints.append(count_z / num_RBs <= phi_i_sk[s, i, k])
                    constraints.append(phi_i_sk[s, i, k] <= count_z / num_RBs + (1 - epsilon)) 

        # Ràng buộc (16i)
        for s in range(num_slice):
            for i in range(num_RUs):
                for j in range(num_DUs):
                    for k in range(num_UEs):
                        constraints.append(phi_j_sk[s, j, k] <= l_ru_du[i, j] - phi_i_sk[s, i, k] + 1)

        # Ràng buộc (16j)
        for s in range(num_slice):
            for j in range(num_DUs):
                for m in range(num_CUs):
                    for k in range(num_UEs):
                        constraints.append(phi_m_sk[s, m, k] <= l_du_cu[j, m] - phi_j_sk[s, j, k] + 1)

        
        # Ràng buộc bổ sung: z <= y_max * x, z >= y - y_max*(1-x), z <= y
        for s in range(num_slice):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    for b in range(num_RBs):
                        constraints.append(mu_bi_sk[(s, i, k, b)] <= max_tx_power_mwatts * z_bi_sk[(s, i, k, b)])
                        constraints.append(mu_bi_sk[(s, i, k, b)] >= P_bi_sk[(s, i, k, b)] - max_tx_power_mwatts * (1- z_bi_sk[(s, i, k, b)]))
                        constraints.append(mu_bi_sk[(s, i, k, b)] <= P_bi_sk[(s, i, k, b)])

        # Các UE k khác nhau connect tới RU i 
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    count_z = 0
                    for k in range(num_UEs):
                        count_z += z_bi_sk[(s, i, k, b)]
                    constraints.append(count_z <= 1)

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

        return pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, P_bi_sk, mu_bi_sk

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None
    

def short_term(num_slice, num_RUs, num_RBs, num_UEs, rb_bandwidth, gain, R_min, max_tx_power_mwatts, pi_sk):
    try:
        # Khởi tạo ma trận z_bi_sk (biến nhị phân)
        z_bi_sk = np.empty((num_slice, num_RUs, num_UEs, num_RBs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        z_bi_sk[s,i,k,b]= cp.Variable(boolean=True, name = f"z_bi_sk({b}, {i}, {s}, {k})")

        # Khởi tạo ma trận mu (biến liên tục): mu = Z_bi_sk * P_bi_sk
        mu_bi_sk = np.empty((num_slice, num_RUs, num_UEs, num_RBs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        mu_bi_sk[s,i,k,b]= cp.Variable(name = f"mu_bi_sk({b}, {i}, {s}, {k})")
        
        # Khởi tạo ma trận phân bổ công suất (biến liên tục): P_bi_sk
        P_bi_sk = np.empty((num_slice, num_RUs, num_UEs, num_RBs), dtype=object)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        P_bi_sk[s,i,k,b]= cp.Variable(name = f"P_bi_sk({b}, {i}, {s}, {k})")
        constraints = []    
        sigma = cp.Variable((num_slice,num_UEs), name = f"sigma")
        # Ràng buộc (18a) - Mỗi RB tại một RU chỉ được sử dụng bởi tối đa một UE
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    count_z = 0
                    for k in range(num_UEs):
                        count_z += z_bi_sk[s, i, k, b]
                    constraints.append(count_z <= 1)
        for s in range(num_slice):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(mu_bi_sk[(s, i, k, b)] >= 0)
        for s in range(num_slice):
            for b in range(num_RBs):
                sum_z_bi_sk = 0
                for i in range(num_RUs):
                    for k in range(num_UEs):
                        sum_z_bi_sk += z_bi_sk[s,i,k,b]
                constraints.append(sum_z_bi_sk <= 1)
        # Ràng buộc (18c) - Tốc độ yêu cầu tối thiểu cho mỗi UE
        for s in range(num_slice):
            for k in range(num_UEs):
                R_sk = 0
                for b in range(num_RBs):
                    SNR_sum = 0 
                    for i in range(num_RUs): 
                        SNR_sum += gain[s, i, k, b] * mu_bi_sk[(s, i, k, b)]
                    R_sk += rb_bandwidth * cp.log(1 + SNR_sum) / np.log(2)
                constraints.append(R_sk >= R_min * pi_sk[s,k])
                constraints.append(sigma[s,k] <= R_sk)
                #cần xem xét
        # Ràng buộc (18d) - Giới hạn tổng công suất phát cho mỗi RU
        for s in range(num_slice):
            for i in range(num_RUs):
                total_power = 0
                for b in range(num_RBs):
                    for k in range(num_UEs):  
                        total_power += mu_bi_sk[s, i, k, b]
                constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc liên quan đến biến mu và P (đảm bảo mu = z * P và không vượt quá công suất tối đa)
        for s in range(num_slice):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    for b in range(num_RBs):
                        constraints.append(mu_bi_sk[s, i, k, b] <= max_tx_power_mwatts * z_bi_sk[s, i, k, b])
                        constraints.append(mu_bi_sk[s, i, k, b] >= P_bi_sk[s, i, k, b] - max_tx_power_mwatts * (1 - z_bi_sk[s, i, k, b]))
                        constraints.append(mu_bi_sk[s, i, k, b] <= P_bi_sk[s, i, k, b])

        # Hàm mục tiêu: Tối đa hóa tổng tốc độ của tất cả các UE
        total_rate = cp.sum(sigma)  # Tổng tốc độ của tất cả các UE

        objective = cp.Maximize(total_rate)

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

        return sigma,mu_bi_sk,z_bi_sk  # Trả về kết quả tốc độ của các UE sau khi tối ưu

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None