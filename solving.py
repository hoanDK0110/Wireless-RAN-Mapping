import numpy as np
import cvxpy as cp

# Hàm kiểm tra tính khả thi
def check_feasibility(pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_watts, rb_bandwidth, D_j, D_m, R_min, gain, P_bi_sk, A_j, A_m, l_ru_du, l_du_cu, epsilon):
    try:
        feasibility_report = {}

        # Kiểm tra ràng buộc (15b)
        for k in range(num_UEs):
            R_sk = 0
            for b in range(num_RBs):
                SNR = 0
                for i in range(num_RUs):
                    SNR += P_bi_sk[i, k, b] * z_bi_sk[(i, k, b)].value * gain[i, k, b]
                R_sk += rb_bandwidth * np.log(1 + SNR) / np.log(2)
            feasibility_report[f"Ràng buộc 15b cho UE {k}"] = (R_sk >= R_min * np.sum(pi_sk[k].value))

        # Kiểm tra ràng buộc (15c)
        for i in range(num_RUs):
            tổng_công_suất = 0
            for b in range(num_RBs):
                for k in range(num_UEs):
                    tổng_công_suất += P_bi_sk[i, k, b] * z_bi_sk[(i, k, b)].value
            feasibility_report[f"Ràng buộc 15c cho RU {i}"] = (tổng_công_suất <= max_tx_power_watts)

        # Kiểm tra ràng buộc (15d)
        for j in range(num_DUs):
            count_du = np.sum(phi_j_sk[j, :].value) * D_j
            feasibility_report[f"Ràng buộc 15d cho DU {j}"] = (count_du <= A_j[j])

        # Kiểm tra ràng buộc (15e)
        for m in range(num_CUs):
            count_cu = np.sum(phi_m_sk[m, :].value) * D_m
            feasibility_report[f"Ràng buộc 15e cho CU {m}"] = (count_cu <= A_m[m])

        # Kiểm tra ràng buộc (15f + 15g + 15h)
        for k in range(num_UEs):
            f_constraint = (np.sum(phi_i_sk[:, k].value) == np.sum(pi_sk[k].value))
            g_constraint = (np.sum(phi_j_sk[:, k].value) == np.sum(pi_sk[k].value))
            h_constraint = (np.sum(phi_m_sk[:, k].value) == np.sum(pi_sk[k].value))
            feasibility_report[f"Ràng buộc 15f-15g-15h cho UE {k}"] = (f_constraint and g_constraint and h_constraint)

        # Kiểm tra ràng buộc (15i)
        # for i in range(num_RUs):
        #     for k in range(num_UEs):
        #         for b in range(num_RBs):
        #             i_constraint_1 = (cp.sum([z_bi_sk[(i, k, b)].value for b in range(num_RBs)])/num_RBs <= phi_i_sk[i, k].value)
        #             i_constraint_2 = (phi_i_sk[i, k].value <= cp.sum([z_bi_sk[(i, k, b)].value for b in range(num_RBs)])/num_RBs + (1 - epsilon))
        #             feasibility_report[f"Ràng buộc 15i cho RU {i}, UE {k}, RB {b}"] = (i_constraint_1 and i_constraint_2)

        for i in range(num_RUs):
            for k in range(num_UEs):
                sum_z_bi_sk_value = 0
                for b in range(num_RBs):
                    sum_z_bi_sk_value += z_bi_sk[(i, k, b)].value
                sum_z_bi_sk_value = sum_z_bi_sk_value / num_RBs
                i_constraint_1 = (sum_z_bi_sk_value <= phi_i_sk[i, k].value)
                i_constraint_2 = (phi_i_sk[i, k].value <= sum_z_bi_sk_value + (1 - epsilon))
                feasibility_report[f"Ràng buộc 15i cho RU {i}, UE {k}, RB {b}"] = (i_constraint_1 and i_constraint_2)

        # Kiểm tra ràng buộc (15j)
        for i in range(num_RUs):
            for j in range(num_DUs):
                feasibility_report[f"Ràng buộc 15j cho RU {i}, DU {j}"] = (phi_j_sk[j, :].value <= l_ru_du[i, j] - phi_i_sk[i, :].value + 1).all()

        # Kiểm tra ràng buộc (15k)
        for j in range(num_DUs):
            for m in range(num_CUs):
                feasibility_report[f"Ràng buộc 15k cho DU {j}, CU {m}"] = (phi_m_sk[m, :].value <= l_du_cu[j, m] - phi_j_sk[j, :].value + 1).all()

        return feasibility_report

    except Exception as e:
        print(f"Lỗi xảy ra trong quá trình kiểm tra tính khả thi: {e}")
        return None

# Hàm tối ưu hóa
def optimize(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, max_tx_power_watts, rb_bandwidth, D_j, D_m, R_min, gain, P_bi_sk, A_j, A_m, l_ru_du, l_du_cu, epsilon):
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

        # Ràng buộc (15b)
        for k in range(num_UEs):
            R_sk = 0
            for b in range(num_RBs):
                SNR = 0
                for i in range(num_RUs): 
                    SNR += P_bi_sk[i, k, b] * z_bi_sk[(i, k, b)] * gain[i, k, b]
                R_sk += rb_bandwidth * cp.log(1 + SNR) / np.log(2)
            constraints.append(R_sk >= R_min * cp.sum(pi_sk[k]))

        # Ràng buộc (15c)
        for i in range(num_RUs):
            tổng_công_suất = 0
            for b in range(num_RBs):
                for k in range(num_UEs):
                    tổng_công_suất += P_bi_sk[i, k, b] * z_bi_sk[(i, k, b)]
            constraints.append(tổng_công_suất <= max_tx_power_watts)

        # Ràng buộc (15d)
        for j in range(num_DUs):
            count_du = cp.sum(phi_j_sk[j, :]) * D_j
            constraints.append(count_du <= A_j[j])

        # Ràng buộc (15e)
        for m in range(num_CUs):
            count_cu = cp.sum(phi_m_sk[m, :]) * D_m
            constraints.append(count_cu <= A_m[m])

        # Ràng buộc (15f + 15g + 15h)
        for k in range(num_UEs):
            constraints.append(cp.sum(phi_i_sk[:, k]) == cp.sum(pi_sk[k]))
            constraints.append(cp.sum(phi_j_sk[:, k]) == cp.sum(pi_sk[k]))
            constraints.append(cp.sum(phi_m_sk[:, k]) == cp.sum(pi_sk[k]))

        # Ràng buộc (15i)
        for i in range(num_RUs):
            for k in range(num_UEs):
        # Tính tổng của tất cả các giá trị z_bi_sk[(i, k, b)] trên miền b
                sum_z_bi_sk = cp.sum([z_bi_sk[(i, k, b)] for b in range(num_RBs)]) / num_RBs
        
        # Áp dụng ràng buộc mới sử dụng sum_z_bi_sk
                constraints.append(sum_z_bi_sk <= phi_i_sk[i, k])
                constraints.append(phi_i_sk[i, k] <= sum_z_bi_sk + (1 - epsilon))

        # Ràng buộc (15j)
        for i in range(num_RUs):
            for j in range (num_DUs):
                constraints.append(phi_j_sk[j, :] <= l_ru_du[i, j] - phi_i_sk[i, :] + 1)

        # Ràng buộc (15k)
        for j in range (num_DUs):
            for m in range (num_CUs):
                constraints.append(phi_m_sk[m, :] <= l_du_cu[j, m] - phi_j_sk[j, :] + 1)

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

        return pi_sk, z_bi_sk, phi_i_sk, phi_j_sk, phi_m_sk

    except cp.SolverError:
        print('Lỗi solver: không khả thi')
        return None, None, None, None, None
    except Exception as e:
        print(f"Lỗi xảy ra: {e}")

# Hàm long-term tối ưu hóa tại NON-RT RIC
def long_term_optimization(pi_sk, phi_j_sk, phi_i_sk, phi_m_sk, num_UEs, num_RUs, num_DUs, num_CUs, D_j, D_m, A_j, A_m, l_ru_du, l_du_cu):
    try:
        # Hàm mục tiêu: Maximize tổng số người dùng được chấp nhận
        objective = cp.Maximize(cp.sum(pi_sk))

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc (15d) trên DUs
        for j in range(num_DUs):
            count_du = cp.sum(phi_j_sk[j, :]) * D_j
            constraints.append(count_du <= A_j[j])

        # Ràng buộc (15e) trên CUs
        for m in range(num_CUs):
            count_cu = cp.sum(phi_m_sk[m, :]) * D_m
            constraints.append(count_cu <= A_m[m])

        # Ràng buộc 15f, 15g, 15h: matching constraint giữa các phi và pi
        for k in range(num_UEs):
            constraints.append(cp.sum(phi_i_sk[:, k]) == pi_sk[k])
            constraints.append(cp.sum(phi_j_sk[:, k]) == pi_sk[k])
            constraints.append(cp.sum(phi_m_sk[:, k]) == pi_sk[k])

        # Ràng buộc (15j) giữa RUs và DUs
        for i in range(num_RUs):
            for j in range(num_DUs):
                constraints.append(phi_j_sk[j, :] <= l_ru_du[i, j] - phi_i_sk[i, :] + 1)

        # Ràng buộc (15k) giữa DUs và CUs
        for j in range(num_DUs):
            for m in range(num_CUs):
                constraints.append(phi_m_sk[m, :] <= l_du_cu[j, m] - phi_j_sk[j, :] + 1)

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

        # Trả về các biến đã tối ưu
        return pi_sk, phi_i_sk, phi_j_sk, phi_m_sk

    except cp.SolverError:
        print('Lỗi solver: không khả thi')
        return None, None, None, None
    except Exception as e:
        print(f"Lỗi xảy ra: {e}")