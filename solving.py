import numpy as np
import cvxpy as cp
import time
import traceback as tb

SOLVER = cp.MOSEK
solver_settings = {
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-5,   # Dung sai khả thi primal
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,   # Dung sai khả thi dual
    'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-3,    # Khoảng cách tương đối
    'MSK_DPAR_INTPNT_TOL_PSAFE': 1e-3,      # Độ chính xác an toàn
    'MSK_IPAR_LOG': 0,                      # Tắt log để giảm thời gian
}

def short_term(num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain, R_min, epsilon, arr_long_pi_sk, arr_long_phi_i_sk, max_tx_power_mwatts):
    try:
        # Khởi tạo ma trận nhị phân: short_z_ib_sk (biến xác phân bổ ánh xạ UE k kết nối tới RU i qua RB b tại slice s)
        short_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"short_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): short_p_ib_sk (biến xác phân bổ công suất UE k kết nối tới RU i qua RB b tại slice s)
        short_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_p_ib_sk({i}, {b}, {s}, {k})")
        
        # Khởi tạo ma trận mu (biến liên tục): short_mu_ib_sk = short_z_ib_sk * short_P_ib_sk
        short_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_mu_ib_sk({i}, {b}, {s}, {k})")
        
        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # Biến tối ưu cho việc phân bổ
        short_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="short_pi_sk")  # Tối ưu số lượng UE
        

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        short_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([bandwidth_per_RB * cp.log(1 + cp.sum([short_gain[i, b, s, k] * short_mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])
                constraints.append(R_sk >= R_min * short_pi_sk[s, k])
                short_total_R_sk += R_sk

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([short_mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs): 
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= max_tx_power_mwatts * short_z_ib_sk[i, b, s, k])  
                        constraints.append(short_mu_ib_sk[i, b, s, k] >= short_p_ib_sk[i, b, s, k] - max_tx_power_mwatts * (1 - short_z_ib_sk[i, b, s, k]))
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= short_p_ib_sk[i, b, s, k])  
                        
        # Ràng buộc: Chuyển đổi short_z_ib_sk sang short_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([short_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= short_phi_i_sk[i, s, k])
                    constraints.append(short_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Chuyển đổi short_z_ib_sk == arr_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(short_phi_i_sk[i, s, k] == arr_long_phi_i_sk[i, s, k])

        # Ràng buộc: Chuyển đổi short_pi_sk == arr_pi_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(short_pi_sk[s, k] == arr_long_pi_sk[s, k])


        # Giải bài toán tối ưu
        short_objective = cp.Maximize(short_total_R_sk / R_min)
        problem = cp.Problem(short_objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True, mosek_params=solver_settings)

        short_total_pi_sk = cp.sum(short_pi_sk).value

        return short_pi_sk, short_total_pi_sk, short_objective

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None

def long_term(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, bandwidth_per_RB, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping):
    try:
        # Khởi tạo ma trận nhị phân: z_ib_sk
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): p_ib_sk
        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận mu (biến liên tục): mu_ib_sk = z_ib_sk * p_ib_sk
        mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"phi_i_sk({i}, {s}, {k})")

        phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        for j in range(num_DUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"phi_j_sk({j}, {s}, {k})")

        phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for m in range(num_CUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"phi_m_sk({m}, {s}, {k})")
        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj") 

        
        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1) 

        # Ràng buộc: Đảm bảo QoS (Data rate)
        total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([bandwidth_per_RB * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k]])) / np.log(2) for i in range(num_RUs) for b in range(num_RBs)])
 
                constraints.append(R_sk >= R_min * pi_sk[s, k])
                total_R_sk += R_sk

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts[i])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(phi_j_sk[j, s, k] <= l_ru_du[i, j] - phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(phi_m_sk[m, s, k] <= l_du_cu[j, m] - phi_j_sk[j, s, k] + 1)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(mu_ib_sk[i, b, s, k] <= max_tx_power_mwatts[i] * z_ib_sk[i, b, s, k])  
                        constraints.append(mu_ib_sk[i, b, s, k] >= p_ib_sk[i, b, s, k] - max_tx_power_mwatts[i] * (1 - z_ib_sk[i, b, s, k]))
                        constraints.append(mu_ib_sk[i, b, s, k] <= p_ib_sk[i, b, s, k])  

        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] <= slice_mapping[s, k])

        # Giải bài toán tối ưu
        print("Solving optimization problem...")
        objective = cp.Maximize(gamma * cp.sum(pi_sk) + (1 - gamma) * total_R_sk / R_min)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)
        print("Optimization problem solved.")
        total_pi_sk = cp.sum(pi_sk).value

        return pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, objective, total_pi_sk

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None, None, None

def mapping_RU_nearest_UE(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, arr_phi_i_sk, data_rate, P_ib_sk, max_tx_power_mwatts):
    try:
        # Khởi tạo ma trận nhị phân: nearest_z_ib_sk
        nearest_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        nearest_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"nearest_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk
        nearest_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        nearest_phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        nearest_phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    nearest_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"nearest_phi_i_sk({i}, {s}, {k})")
                for j in range(num_DUs):
                    nearest_phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"nearest_phi_j_sk({j}, {s}, {k})")
                for m in range(num_CUs):
                    nearest_phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"nearest_phi_m_sk({m}, {s}, {k})")

        # Biến tối ưu cho việc phân bổ
        nearest_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj") 

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: ánh xạ UE gần RU nhất
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    constraints.append(nearest_phi_i_sk[i, s, k] <= arr_phi_i_sk[i, s, k])

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([nearest_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        nearest_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                # Tính toán data rate R_sk
                R_sk = cp.sum([data_rate[i, b, s, k] * nearest_z_ib_sk[i, b, s, k]for i in range(num_RUs)for b in range(num_RBs)])

                nearest_total_R_sk += R_sk
                # Đảm bảo tốc độ dữ liệu tối thiểu
                constraints.append(R_sk >= R_min * nearest_pi_sk[s, k])

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([nearest_z_ib_sk[i, b, s, k] * P_ib_sk for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([nearest_phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])
        
        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([nearest_phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([nearest_phi_i_sk[i, s, k] for i in range(num_RUs)]) == nearest_pi_sk[s, k])
                constraints.append(cp.sum([nearest_phi_j_sk[j, s, k] for j in range(num_DUs)]) == nearest_pi_sk[s, k])
                constraints.append(cp.sum([nearest_phi_m_sk[m, s, k] for m in range(num_CUs)]) == nearest_pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([nearest_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= nearest_phi_i_sk[i, s, k])
                    constraints.append(nearest_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(nearest_phi_j_sk[j, s, k] <= l_ru_du[i, j] - nearest_phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(nearest_phi_m_sk[m, s, k] <= l_du_cu[j, m] - nearest_phi_j_sk[j, s, k] + 1)

        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(nearest_pi_sk[s, k] <= slice_mapping[s, k])

        # Hàm tối ưu
        nearest_objective = cp.Maximize(gamma * cp.sum(nearest_pi_sk) + (1 - gamma) * nearest_total_R_sk / R_min)
        problem = cp.Problem(nearest_objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True, mosek_params=solver_settings)   

        # Tính tổng nearest_pi_sk
        nearest_total_pi_sk = cp.sum(nearest_pi_sk).value

        return nearest_pi_sk, nearest_z_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk, nearest_total_pi_sk, nearest_objective

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None



def long_term_2(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts):
    try:
        # Khởi tạo ma trận nhị phân: z_ib_sk
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"phi_i_sk({i}, {s}, {k})")
                for j in range(num_DUs):
                    phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"phi_j_sk({j}, {s}, {k})")
                for m in range(num_CUs):
                    phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"phi_m_sk({m}, {s}, {k})")

        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="pi_sk")

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] 
                                        for s in range(num_slices) 
                                        for k in range(num_UEs) 
                                        for i in range(num_RUs)]) <= 1)

        total_R_sk = 0
        # Ràng buộc: Đảm bảo QoS (Data rate)
        for s in range(num_slices):
            for k in range(num_UEs):
                # Tính toán data rate R_sk
                R_sk = cp.sum([data_rate[i, b, s, k] * z_ib_sk[i, b, s, k] 
                               for i in range(num_RUs) 
                               for b in range(num_RBs)])
                total_R_sk += R_sk
                constraints.append(R_sk >= R_min * pi_sk[s, k])

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([z_ib_sk[i, b, s, k] * P_ib_sk 
                                  for b in range(num_RBs) 
                                  for k in range(num_UEs) 
                                  for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] 
                               for s in range(num_slices) 
                               for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] 
                               for s in range(num_slices) 
                               for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(phi_j_sk[j, s, k] <= l_ru_du[i, j] - phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(phi_m_sk[m, s, k] <= l_du_cu[j, m] - phi_j_sk[j, s, k] + 1)

        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] == pi_sk[s, k] * slice_mapping[s, k])

        # Giải bài toán tối ưu
        objective = cp.Maximize(gamma * cp.sum(pi_sk) + (1 - gamma) * total_R_sk / R_min)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True, mosek_params=solver_settings)

        total_pi_sk = cp.sum(pi_sk).value

        return pi_sk, z_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, total_pi_sk, objective

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None


def short_term_1(num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain, R_min, epsilon, arr_long_pi_sk, arr_long_phi_i_sk, max_tx_power_mwatts):
    try:
        # Khởi tạo biến nhị phân short_z_ib_sk: UE k kết nối tới RU i qua RB b tại slice s
        short_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"short_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo biến phân bổ công suất short_p_ib_sk
        short_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo biến short_mu_ib_sk: đại diện cho công suất ánh xạ
        short_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo biến nhị phân short_phi_i_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")

        # Biến tối ưu short_pi_sk
        short_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="short_pi_sk")

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for k in range(num_UEs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        short_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([
                    bandwidth_per_RB * cp.log(1 + cp.sum([short_gain[i, b, s, k] * short_mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2)
                    for b in range(num_RBs)
                ])
                constraints.append(R_sk >= R_min * short_pi_sk[s, k])
                short_total_R_sk += R_sk
        

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([short_mu_ib_sk[i, b, s, k] 
                                  for b in range(num_RBs) 
                                  for k in range(num_UEs) 
                                  for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc: Chuyển đổi tích thành tổng
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= max_tx_power_mwatts * short_z_ib_sk[i, b, s, k])
                        constraints.append(short_mu_ib_sk[i, b, s, k] >= short_p_ib_sk[i, b, s, k] - max_tx_power_mwatts * (1 - short_z_ib_sk[i, b, s, k]))
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= short_p_ib_sk[i, b, s, k])

        # Ràng buộc: Liên kết short_z_ib_sk với short_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([short_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= short_phi_i_sk[i, s, k])
                    constraints.append(short_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: short_phi_i_sk khớp với arr_long_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(short_phi_i_sk[i, s, k] == arr_long_phi_i_sk[(i, s, k)])

        # Ràng buộc: short_pi_sk khớp với arr_long_pi_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(short_pi_sk[s, k] == arr_long_pi_sk[(s, k)])

        # Mục tiêu: Tối đa hóa tổng dữ liệu đạt được
        short_objective = cp.Maximize(short_total_R_sk / R_min) 

        # Giải bài toán tối ưu
        problem = cp.Problem(short_objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)

        short_total_pi_sk = cp.sum(short_pi_sk).value
        return short_pi_sk, short_total_pi_sk, short_objective

    except cp.SolverError:
        print('Solver error: Non-feasible solution.')
        return None, None, None
    except Exception as e:
        print(f'Error: {e}')
        return None, None, None


def random_mapping(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, gamma, data_rate, epsilon, slice_mapping):
    try:
        # Khởi tạo ma trận nhị phân
        random_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        random_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"random_z_ib_sk({i}, {b}, {s}, {k})")

        random_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        random_phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        random_phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    random_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"random_phi_i_sk({i}, {s}, {k})")
                for j in range(num_DUs):
                    random_phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"random_phi_j_sk({j}, {s}, {k})")
                for m in range(num_CUs):
                    random_phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"random_phi_m_sk({m}, {s}, {k})")

        random_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj")
        
        constraints = []
        
        ue_to_ru_mapping = {}

        # Danh sách trạng thái của các RU, DU, và CU
        available_RUs = {i: True for i in range(num_RUs)}
        available_DUs = {j: A_j[j] for j in range(num_DUs)}
        available_CUs = {m: A_m[m] for m in range(num_CUs)}

        # Xét từng UE
        for ue_idx in range(num_UEs):
            valid_RUs = []
            for ru_idx in range(num_RUs):
                # Kiểm tra QoS
                if available_RUs[ru_idx] and data_rate[ru_idx, ue_idx] >= R_min:
                    valid_RUs.append(ru_idx)

            if not valid_RUs:
                ue_to_ru_mapping[ue_idx] = None
                continue

            # Random chọn RU
            selected_RU = np.random.choice(valid_RUs)

            # Tìm DU khả dụng kết nối với RU
            valid_DUs = []
            for du_idx in range(num_DUs):
                if l_ru_du[selected_RU][du_idx] and available_DUs[du_idx] >= D_j[ue_idx]:
                    valid_DUs.append(du_idx)

            if not valid_DUs:
                ue_to_ru_mapping[ue_idx] = None
                continue

            selected_DU = np.random.choice(valid_DUs)

            # Tìm CU khả dụng kết nối với DU
            valid_CUs = []
            for cu_idx in range(num_CUs):
                if l_du_cu[selected_DU][cu_idx] and available_CUs[cu_idx] >= D_m[ue_idx]:
                    valid_CUs.append(cu_idx)

            if not valid_CUs:
                ue_to_ru_mapping[ue_idx] = None
                continue

            selected_CU = np.random.choice(valid_CUs)

            # Lưu kết nối thành công
            ue_to_ru_mapping[ue_idx] = selected_RU
            constraints.append(random_phi_i_sk[selected_RU, 0, ue_idx] == 1)
            constraints.append(random_phi_j_sk[selected_DU, 0, ue_idx] == 1)
            constraints.append(random_phi_m_sk[selected_CU, 0, ue_idx] == 1)

            # Cập nhật tài nguyên
            available_RUs[selected_RU] = False
            available_DUs[selected_DU] -= D_j[ue_idx]
            available_CUs[selected_CU] -= D_m[ue_idx]

        # Ràng buộc: Mỗi UE chỉ kết nối với 1 RU
        for k in range(num_UEs):
            for s in range(num_slices):
                constraints.append(cp.sum([random_phi_i_sk[i, s, k] for i in range(num_RUs)]) <= 1)

        # Ràng buộc data rate
        random_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([data_rate[i, b, s, k] * random_z_ib_sk[i, b, s, k]
                               for i in range(num_RUs) for b in range(num_RBs)])
                random_total_R_sk += R_sk
                constraints.append(R_sk >= R_min * random_pi_sk[s, k])

        # Ràng buộc tài nguyên DU, CU
        for j in range(num_DUs):
            total_du = cp.sum([random_phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        for m in range(num_CUs):
            total_cu = cp.sum([random_phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])
            
        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([random_phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])
        
        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([random_phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([random_phi_i_sk[i, s, k] for i in range(num_RUs)]) == random_pi_sk[s, k])
                constraints.append(cp.sum([random_phi_j_sk[j, s, k] for j in range(num_DUs)]) == random_pi_sk[s, k])
                constraints.append(cp.sum([random_phi_m_sk[m, s, k] for m in range(num_CUs)]) == random_pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([random_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= random_phi_i_sk[i, s, k])
                    constraints.append(random_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(random_phi_j_sk[j, s, k] <= l_ru_du[i, j] - random_phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(random_phi_m_sk[m, s, k] <= l_du_cu[j, m] - random_phi_j_sk[j, s, k] + 1)

        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(random_pi_sk[s, k] <= slice_mapping[s, k])

        # Tối ưu hóa
        random_objective = cp.Maximize(gamma * cp.sum(random_pi_sk) + (1 - gamma) * random_total_R_sk * 1e-6)
        problem = cp.Problem(random_objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)

        return random_pi_sk, random_z_ib_sk, random_phi_i_sk, random_phi_j_sk, random_phi_m_sk, random_total_R_sk, random_objective, ue_to_ru_mapping

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None
    
def random_choice(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, D_j, D_m, R_min, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, data_rate, P_ib_sk, max_tx_power_mwatts):
    try:
        # Initialize results variables
        z_ib_sk = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
        phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs))
        phi_j_sk = np.zeros((num_DUs, num_slices, num_UEs))
        phi_m_sk = np.zeros((num_CUs, num_slices, num_UEs))
        pi_sk = np.zeros((num_slices, num_UEs))

        total_R_sk = 0

        for s in range(num_slices):
            for k in range(num_UEs):
                # Randomly assign RU, DU, CU, and RB while satisfying constraints
                assigned_RU = np.random.randint(0, num_RUs)
                assigned_DU = np.random.randint(0, num_DUs)
                assigned_CU = np.random.randint(0, num_CUs)

                # Ensure at least one RB is assigned to this UE in the chosen RU
                num_assigned_RBs = np.random.randint(1, num_RBs + 1)
                assigned_RBs = np.random.choice(range(num_RBs), size=num_assigned_RBs, replace=False)

                # Calculate data rate and ensure it meets the QoS constraint
                R_sk = np.sum([data_rate[assigned_RU, b, s, k] for b in assigned_RBs])
                if R_sk >= R_min:
                    pi_sk[s, k] = 1
                    total_R_sk += R_sk

                    # Update z_ib_sk for the chosen RBs
                    for b in assigned_RBs:
                        z_ib_sk[assigned_RU, b, s, k] = 1

                    # Assign RU, DU, CU mappings
                    phi_i_sk[assigned_RU, s, k] = 1
                    phi_j_sk[assigned_DU, s, k] = 1
                    phi_m_sk[assigned_CU, s, k] = 1

                    # Ensure resource constraints are met
                    if np.sum(phi_j_sk[:, s, k] * D_j[k]) > np.min(A_j):
                        raise ValueError("DU resource constraint violated.")
                    if np.sum(phi_m_sk[:, s, k] * D_m[k]) > np.min(A_m):
                        raise ValueError("CU resource constraint violated.")
                    if np.sum(z_ib_sk[assigned_RU, :, s, k] * P_ib_sk) > max_tx_power_mwatts:
                        raise ValueError("RU power constraint violated.")

        # Return random assignment results
        return pi_sk, z_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, np.sum(pi_sk), "Random Assignment Success"

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None, None


def short_term_2(num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain, R_min, epsilon, arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk, max_tx_power_mwatts):
    try:
        # Khởi tạo biến nhị phân short_z_ib_sk: UE k kết nối tới RU i qua RB b tại slice s
        short_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"short_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo biến phân bổ công suất short_p_ib_sk
        short_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo biến short_mu_ib_sk: đại diện cho công suất ánh xạ
        short_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo biến nhị phân short_phi_i_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")

        # Biến tối ưu short_pi_sk
        short_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="short_pi_sk")

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for k in range(num_UEs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        short_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([
                    bandwidth_per_RB * cp.log(1 + cp.sum([short_gain[i, b, s, k] * arr_long_z_ib_sk[i, b, s, k] * short_p_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2)
                    for b in range(num_RBs)
                ])
                constraints.append(R_sk >= R_min * short_pi_sk[s, k])
                short_total_R_sk += R_sk
        

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([arr_long_z_ib_sk[i, b, s, k] * short_p_ib_sk[i, b, s, k] 
                                  for b in range(num_RBs) 
                                  for k in range(num_UEs) 
                                  for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc: short_z_ib_sk khớp với arr_long_z_ib_sk
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        constraints.append(short_z_ib_sk[i, b, s, k] == arr_long_z_ib_sk[(i, b, s, k)])

        # Ràng buộc: short_pi_sk khớp với arr_long_pi_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(short_pi_sk[s, k] == arr_long_pi_sk[(s, k)])

        # Mục tiêu: Tối đa hóa tổng dữ liệu đạt được
        short_objective = cp.Maximize(short_total_R_sk / R_min) 

        # Giải bài toán tối ưu
        problem = cp.Problem(short_objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)

        short_total_pi_sk = cp.sum(short_pi_sk).value
        return short_pi_sk, short_total_pi_sk, short_objective

    except cp.SolverError:
        print('Solver error: Non-fea sible solution.')
        return None, None, None
    except Exception as e:
        print(f'Error: {e}')
        return None, None, None
    
def short_term_3(num_slices, num_UEs, num_RUs, num_RBs, bandwidth_per_RB, short_gain, R_min, epsilon, arr_long_pi_sk, arr_long_z_ib_sk, arr_long_phi_i_sk, max_tx_power_mwatts):

    try:
        # Khởi tạo ma trận nhị phân: short_z_ib_sk (biến xác phân bổ ánh xạ UE k kết nối tới RU i qua RB b tại slice s)
        short_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"short_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): short_p_ib_sk (biến xác phân bổ công suất UE k kết nối tới RU i qua RB b tại slice s)
        short_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_p_ib_sk({i}, {b}, {s}, {k})")
        
        # Khởi tạo ma trận mu (biến liên tục): short_mu_ib_sk = short_z_ib_sk * short_P_ib_sk
        short_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_mu_ib_sk({i}, {b}, {s}, {k})")
        
        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # Biến tối ưu cho việc phân bổ
        short_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="short_pi_sk")  # Tối ưu số lượng UE
        

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        short_total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([bandwidth_per_RB * cp.log(1 + cp.sum([short_gain[i, b, s, k] * short_mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])
                constraints.append(R_sk >= R_min * short_pi_sk[s, k])
                short_total_R_sk += R_sk

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([short_mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs): 
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= max_tx_power_mwatts * short_z_ib_sk[i, b, s, k])  
                        constraints.append(short_mu_ib_sk[i, b, s, k] >= short_p_ib_sk[i, b, s, k] - max_tx_power_mwatts * (1 - short_z_ib_sk[i, b, s, k]))
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= short_p_ib_sk[i, b, s, k])  
                        
        # Ràng buộc: Chuyển đổi short_z_ib_sk sang short_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([short_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= short_phi_i_sk[i, s, k])
                    constraints.append(short_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Chuyển đổi short_z_ib_sk == arr_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(short_phi_i_sk[i, s, k] == arr_long_phi_i_sk[i, s, k])

        # Ràng buộc: Chuyển đổi short_pi_sk == arr_pi_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(short_pi_sk[s, k] == arr_long_pi_sk[s, k])


        # Giải bài toán tối ưu
        short_objective = cp.Maximize(short_total_R_sk / R_min)
        problem = cp.Problem(short_objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True, mosek_params=solver_settings)

        short_total_pi_sk = cp.sum(short_pi_sk).value

        return short_pi_sk, short_total_pi_sk, short_objective

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None


def global_solving(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_mwatts, bandwidth_per_RB, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping):
    try:
        # Khởi tạo ma trận nhị phân: z_ib_sk
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): p_ib_sk
        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận mu (biến liên tục): mu_ib_sk = z_ib_sk * p_ib_sk
        mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"phi_i_sk({i}, {s}, {k})")

        phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        for j in range(num_DUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"phi_j_sk({j}, {s}, {k})")

        phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for m in range(num_CUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"phi_m_sk({m}, {s}, {k})")
        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj") 
        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1) 

        # Ràng buộc: Đảm bảo QoS (Data rate)
        total_R_sk = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([bandwidth_per_RB * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])

                constraints.append(R_sk >= R_min * pi_sk[s, k])
                total_R_sk += R_sk

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= max_tx_power_mwatts)

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(phi_j_sk[j, s, k] <= l_ru_du[i, j] - phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(phi_m_sk[m, s, k] <= l_du_cu[j, m] - phi_j_sk[j, s, k] + 1)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(mu_ib_sk[i, b, s, k] <= max_tx_power_mwatts * z_ib_sk[i, b, s, k])  
                        constraints.append(mu_ib_sk[i, b, s, k] >= p_ib_sk[i, b, s, k] - max_tx_power_mwatts * (1 - z_ib_sk[i, b, s, k]))
                        constraints.append(mu_ib_sk[i, b, s, k] <= p_ib_sk[i, b, s, k])  

        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] <= slice_mapping[s, k])

        # Giải bài toán tối ưu
        print("Solving optimization problem...")
        objective = cp.Maximize(gamma * cp.sum(pi_sk) + (1 - gamma) * total_R_sk / R_min)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, warm_start=True)
        print("Optimization problem solved.")
        total_pi_sk = cp.sum(pi_sk).value
        
        return pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk, total_pi_sk, objective

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None, None, None