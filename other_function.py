import numpy as np

def extract_optimization_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    def extract_values(array, dtype):
        shape = array.shape
        flat_array = np.array([x.value for x in array.flatten()], dtype=dtype)
        return flat_array.reshape(shape)

    arr_pi_sk = extract_values(pi_sk, int)
    arr_z_ib_sk = extract_values(z_ib_sk, int)
    arr_p_ib_sk = extract_values(p_ib_sk, float)
    arr_mu_ib_sk = extract_values(mu_ib_sk, float)
    arr_phi_i_sk = extract_values(phi_i_sk, int)
    arr_phi_j_sk = extract_values(phi_j_sk, int)
    arr_phi_m_sk = extract_values(phi_m_sk, int)

    return arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk


def generate_new_num_UEs(num_UEs, delta_num_UE):
    # Tính sai số ngẫu nhiên trong khoảng [-delta_num_UE, delta_num_UE]
    delta = np.random.randint(-delta_num_UE, delta_num_UE)

    # Tính số lượng UE mới
    new_num_UEs = num_UEs + delta
    # Đảm bảo số lượng UE không âm
    return max(new_num_UEs, 0)
