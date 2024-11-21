import numpy as np

def extract_optimization_results(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs,pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    """
    Trích xuất giá trị của tất cả các biến tối ưu (cvxpy.Variable) sau khi giải quyết bài toán.

    Args:
    - pi_sk (cvxpy.Variable): Ma trận pi_sk (boolean), với kích thước (num_slices, num_UEs).
    - z_ib_sk (np.array): Ma trận z_ib_sk (boolean), với kích thước (num_RUs, num_RBs, num_slices, num_UEs).
    - p_ib_sk (np.array): Ma trận p_ib_sk (liên tục), với kích thước (num_RUs, num_RBs, num_slices, num_UEs).
    - mu_ib_sk (np.array): Ma trận mu_ib_sk (liên tục), với kích thước (num_RUs, num_RBs, num_slices, num_UEs).
    - phi_i_sk (np.array): Ma trận phi_i_sk (liên tục), với kích thước (num_RUs, num_slices, num_UEs).
    - phi_j_sk (np.array): Ma trận phi_j_sk (liên tục), với kích thước (num_RUs, num_slices, num_UEs).
    - phi_m_sk (np.array): Ma trận phi_m_sk (liên tục), với kích thước (num_RUs, num_RBs, num_slices).

    Returns:
    - dict: Một từ điển chứa các mảng kết quả của tất cả các biến tối ưu.
    """

    arr_pi_sk = np.empty((num_slices, num_UEs), dtype=int) 
    for s in range(num_slices):
        for k in range(num_UEs):
            arr_pi_sk[s, k] = pi_sk[s, k].value

    # Extract z_ib_sk (binary)
    arr_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=int)
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    arr_z_ib_sk[i, b, s, k] = z_ib_sk[i, b, s, k].value

    # Extract p_ib_sk (continuous)
    arr_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=float)
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    p_ib_sk[i, b, s, k] = p_ib_sk[i, b, s, k].value

    # Extract mu_ib_sk (continuous)
    arr_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=float)
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    arr_mu_ib_sk[i, b, s, k] = mu_ib_sk[i, b, s, k].value

    # Extract phi_i_sk
    arr_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=int)
    for i in range(num_RUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                arr_phi_i_sk[i, s, k] = phi_i_sk[i, s, k].value

    # Extract phi_j_sk 
    arr_phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=int)
    for j in range(num_DUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                arr_phi_j_sk[j, s, k] = phi_j_sk[j, s, k].value

    # Extract phi_m_sk 
    arr_phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=int)
    for m in range(num_CUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                arr_phi_m_sk[m, s, k] = phi_m_sk[m, s, k].value

    return arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk


